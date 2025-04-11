# docker run -it --rm --gpus '"device=7"' -v $(pwd):/app -v $(pwd)/../cache:/app/cache --name ${USER}_lcc ${USER}_lc python -m scripts.rl_test

# tokens are 0: pad, 1, 2: actions, 3 -> 258: observations
import jax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
import jax.numpy as jnp
import flax
import datetime
import numpy as np
import optax
import time
from dataclasses import dataclass

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from typing import Sequence, NamedTuple, Any, Dict, Callable, Optional

import distrax
import gymnax
import functools
from gymnax.environments import spaces
from gymnax_exchange.jaxrl.utils import FlattenObservationWrapper, LogWrapper
from jax._src import dtypes
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig
#import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)
import dataclasses
import jax
import jax.numpy as jnp
#import optax
import distrax


from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString

j_calculate_gae = jax.jit(jax.vmap(calculate_gae, in_axes=(0, 0, 0, 0, 0, None, None)))

import wandb


wandbOn = True # False
if wandbOn:
    import wandb


##Special class to handel the flag list, Jax String.
@jax.tree_util.register_pytree_node_class
@dataclass
class JString:
    tokens: jnp.ndarray
    length: jnp.ndarray

    def __init__(self, tokens, length=None):
        self.tokens = jnp.array(tokens)
        self.length = (
            length if length is not None
            else jnp.ones_like(tokens[:, 0]) * tokens.shape[1]
        )

    def tree_flatten(self):
        # The children are the arrays that JAX can trace.
        children = (self.tokens, self.length)
        # No auxiliary static data.
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tokens, length = children
        return cls(tokens, length)

#===========Define a train function so we can call it in a sweep===#
#  
def make_train(config):
    """Make train function. 
    Input: Config
    Output: train function
    Train(rng): returns:
    return {"params": params, "info_train": info_train, "eval_info": eval_info}
    Infos and params
    """

    #Calculate the number of updates we will do
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    #Function to process the obsveration
    def handle_continuous(observation):
        return jnp.array(observation).astype(jnp.float8_e4m3b11fnuz).view(jnp.uint8).astype(jnp.int32)
    
    #========#
    # Get Keys
    #=========#
    rng = jax.random.key(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env_config=EnvironmentConfig(**config["ENV_CONFIG"])

    #===========#
    #Define Envs#
    #==========#
    env = MarketMakingEnv(
            env_config,
            key_reset,
            alphatradePath=config["ATFOLDER"]+"/train",
            window_index=config["WINDOW_INDEX"],
            episode_time=config["EPISODE_TIME"],
            ep_type=config["DATA_TYPE"],
        )

    eval_env=MarketMakingEnv(
            env_config,
            key_reset,
            alphatradePath=config["ATFOLDER"]+"/val",
            window_index=config["WINDOW_INDEX"],
            episode_time=config["EPISODE_TIME"],
            ep_type=config["DATA_TYPE"],
        )
    eval_env_params = dataclasses.replace(
            eval_env.default_params,
            episode_time=config["EPISODE_TIME"],
        )

    env_params = dataclasses.replace(
            env.default_params,
            episode_time=config["EPISODE_TIME"],
        )
    
    #====================#
    #Apply purejaxRL wrappers#
    #=========================#
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    eval_env = FlattenObservationWrapper(eval_env)
    eval_env = LogWrapper(eval_env)      
    
    #Define the update schedule
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    
   #Define the JIT functions
    jit_ppo_update = get_jit_ppo(config)

    #Train env jit fn
    v_env_step = jax.jit(jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    ))

    #Eval env jit function
    v_eval_env_step=jax.jit(jax.vmap(
        eval_env.step, in_axes=(0, 0, 0, None)
    ))


    def train(rng):
        """Train function
        input rng key
        output:
        return {"params": params, "info_train": info_train, "eval_info": eval_info}"""
        


        #===================================================#
        #Intialise our model: rwkv
        #===================================================#

        #Define the vocab
        num_tokens = 1 + env.action_space(env_params).n + 256
        config["MIN_ACTION_TOK"] = 1
        config["MAX_ACTION_TOK"] = env_config.n_actions

        #Load the RWKV
        RWKV, params = get_rand_model(0, "6", 3, 256, num_tokens, dtype=jnp.float32, rwkv_type="ScanRWKV")
        #Define the forward function and jit version
        forward, params = get_ppo_agent(RWKV, params, seed=1)
        v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))
    
        #Get init state for training
        init_state = RWKV.default_state(params)
        #returns 0s for weights, in /home/duser/AlphaTrade/jax_rwkv/src/jax_rwkv/base_rwkv.py
        if isinstance(init_state, tuple):
            init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
        else:
            init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)
        state = init_state


        #Define optimiser
        solver = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(linear_schedule, eps=1e-5)
        )
        optimizer = solver.init(params)



        #Start count
        global_timestep = 1

        #Reset training env
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        for _ in range(int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]):
            #Intialise lists
            initial_state = state
            tokens_list = []
            flags_list = []
            values_list = []
            rewards_list = []
            log_prob_list = []
            dones_list = []
            all_actions = []
            update_returns = []
            update_pnl=[]

            
            for t in range(config["NUM_STEPS"]):
                rng, _rng = jax.random.split(rng)

                #===============#
                #tokenizer the obvs space
                #=======================#
                tokenized = handle_continuous(obsv)

                #====================#
                #Evaluate policy from model
                #=====================#
                pi, value, state = v_forward_jit(tokenized, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1])
                pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
                action = pi.sample(seed=_rng)
                #log actions
                def log_action_distribution(action):
                            unique_actions, counts = jnp.unique(action, return_counts=True)
                            action_distribution = {f"action_{int(a)}": int(c) for a, c in zip(unique_actions, counts)}
                            wandb.log(action_distribution)
                if wandbOn:
                    jax.debug.callback(log_action_distribution, action)

                current_actions = jax.device_get(action)
                all_actions.extend(current_actions.flatten().tolist())
                log_prob = pi.log_prob(action)
                #============#
                #Update the state with the new action
                #============#
                _, value1, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

                #=============#
                #Process action through environmnet
                #==============#
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info_train = v_env_step(rng_step, env_state, action, env_params)

                #Reset state if done (rwkv state)
                state = jax.vmap(jax.lax.select)(done, init_state, state)


                #======Append list of actions, dones, rewards, value#

                
                tokens_list.append(tokenized)
                tokens_list.append(action[:, None] + config["MIN_ACTION_TOK"])
                
                flags_list.append(jnp.ones_like(tokenized) * OBS_FLAG)
                flags_list.append(jnp.ones_like(tokenized)[:, :1] * ACT_FLAG)

                values_list.append(value)
                values_list.append(value1)

                rewards_list.append(jnp.zeros(shape=tokenized.shape))
                rewards_list.append(reward[:, None])

                log_prob_list.append(jnp.zeros_like(value))
                log_prob_list.append(log_prob[:, None])

                dones_list.append(jnp.zeros(value.shape, dtype=jnp.bool))
                dones_list.append(done[:, None])

                return_values = info_train["returned_episode_returns"][info_train["returned_episode"]]
                episdoic_pnl=info_train["total_PnL"][info_train["returned_episode"]]
                for r in return_values:
                    update_returns.append(r)
                for p in episdoic_pnl:
                     update_pnl.append(p)                    
                global_timestep += 1

            #Form lists for adv calcs
            tokens_list = jnp.concatenate(tokens_list, axis=1)
            flags_list = jnp.concatenate(flags_list, axis=1)
            values_list = jnp.concatenate(values_list, axis=1)
            rewards_list = jnp.concatenate(rewards_list, axis=1)
            log_probs_list = jnp.concatenate(log_prob_list, axis=1)[..., 1:]
            dones_list = jnp.concatenate(dones_list, axis=1)
            buf = JString(tokens_list, jnp.ones_like(tokens_list[:, 0]) * tokens_list.shape[1])
        
            dones_list = jnp.cumsum(dones_list, axis=1, dtype=jnp.bool)
            flags_list = jnp.where(jnp.concatenate((dones_list[:, :1], dones_list[:, :-1]), axis=1), PAD_FLAG, flags_list)
          
            #Get last value from state
            _, last_value, _ = v_forward_jit(handle_continuous(obsv), state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))
            
            advantages, targets = j_calculate_gae(flags_list, dones_list, values_list, rewards_list, last_value[..., -1], config["GAMMA"], config["GAE_LAMBDA"])
            # print("value", values_list)
            # print("target", targets)
            print("UPDATING")
            if len(update_returns) > 0:
                average_return=sum(update_returns) / len(update_returns)
                print("avg returns:", sum(update_returns) / len(update_returns))
            else:
                average_return=0
                print("None ended")
            if len(update_pnl) > 0:
                average_pnl=sum(update_pnl) / len(update_pnl)
                print("avg episodic pnl:", sum(update_pnl) / len(update_pnl))
            else:
                average_pnl=0
                print("None ended")
            wandb.log( 
                data={
                     "average_return":average_return,
                     "average_pnl":average_pnl,
                     "global_timestep":global_timestep
                },
                commit=True
            )

            #Update weights
            for _ in range(config["UPDATE_EPOCHS"]):
                params, optimizer, (loss, value_loss, loss_actor, entropy, state) = jit_ppo_update(solver, v_forward_jit, params, optimizer, buf, flags_list, values_list, log_probs_list, advantages, targets, initial_state)
                print(loss, value_loss, loss_actor, entropy)
            

            #Reset state if done (rwkv state)
            state = jax.vmap(jax.lax.select)(dones_list[:, -1], init_state, state)

            ##==================Eval Steps================================================================#

            #==================#
            #Define fresh eval hidden state
            #======================#
            eval_init_h_state = RWKV.default_state(params)
            if isinstance(eval_init_h_state, tuple):
                eval_init_h_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in eval_init_h_state])
            else:
                eval_init_h_state = jnp.repeat(eval_init_h_state[None], config["NUM_ENVS"], axis=0)
            eval_h_state = eval_init_h_state

            #=============#
            #Reset eval env
            #============#
            eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)

            for t in range(config["NUM_STEPS_EVAL"]):
                rng, _rng = jax.random.split(rng)

                #Tokenize obvs
                tokenized = handle_continuous(eval_obsv)

                #Get policy from network
                eval_pi, eval_value, eval_h_state = v_forward_jit(tokenized, eval_h_state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1])
                eval_pi = distrax.Categorical(logits=eval_pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
                #Sample eval action
                eval_action = eval_pi.sample(seed=_rng)
                current_eval_actions = jax.device_get(eval_action)

                #Update h state with action
                log_prob = pi.log_prob(current_eval_actions)
                _, value1, eval_h_state = v_forward_jit(eval_action, eval_h_state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

                #Step Eval action through env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state, eval_reward, eval_done, eval_info = v_eval_env_step(rng_step, eval_env_state, eval_action, eval_env_params)
            

            ##=====================LOGGING==============#
            # Call back, log every update step as in rnn:
            #===========================================#
            return_values = info_train["returned_episode_returns"][info_train["returned_episode"]]
            wandb.log({"return_values:": return_values})
            if config.get("DEBUG"):
                        def callback(info_train, info_eval, loss=None, value_loss=None, loss_actor=None, entropy=None):
                            #------------Collect info for plotting---------------------------#
                            #Matricies, size num_envs by num_steps. Mutliplying gives an array, a value for every non 0

                            #1)Step and return info
                           # return_values = info_train["returned_episode_returns"][info_train["returned_episode"]]
                            timesteps=info_train["timestep"][info_train["returned_episode"]] * config["NUM_ENVS"]
                            
                            
                            #-----------Train info----------#
                            ##Global episodic plots
                            #episodic_PnL_train = info_train["total_PnL"][info_train["returned_episode"]]
                            #episodic_netWorth_train = info_train["netWorth"][info_train["returned_episode"]]
                            
                        
                            #Average across all envs
                            PnL_train= info_train["total_PnL"]
                            netWorth_train= info_train["netWorth"]
                            Episodic_inventories_train= info_train["inventory"][info_train["returned_episode"]]
                            inventories_train= info_train["inventory"]
                            buyQuant_train=info_train["buyQuant"]  
                            sellQuant_train=info_train["sellQuant"]  
                            reward_train=info_train["reward"]  
                            other_exec_quants_train=info_train["other_exec_quants"]  
                            averageMidprice_train=info_train["averageMidprice"]  
                            averageBestbid_train=info_train["average_best_bid"]  
                            averageBestask_train=info_train["average_best_ask"]  
                        

                            #-------------eval info------#   
                            Episodic_PnL_eval = info_eval["total_PnL"][info_eval["returned_episode"]]
                            Episodic_netWorth_eval = info_eval["netWorth"][info_eval["returned_episode"]]
                            inventories_eval = info_eval["inventory"] 
                            buyQuant_eval=info_eval["buyQuant"]
                            sellQuant_eval=info_eval["sellQuant"]
                            reward_eval=info_eval["reward"]
                            other_exec_quants_eval=info_eval["other_exec_quants"]
                            averageMidprice_eval=info_eval["averageMidprice"]
                            averageBestbid_eval=info_eval["average_best_bid"]
                            averageBestask_eval=info_eval["average_best_ask"]
                            
                        
                            #-----------------Logging-------------------#

                            if wandbOn:
                                wandb.log(
                                    data={
                                        #-----time and return------------#
                                       # "episodic_return": jnp.mean(return_values) if return_values.size > 0 else 0,  # Handle empty arrays
                                      #  "global_step": jnp.max(timesteps) if timesteps.size>0 else 0,
                                        
                                        #"windowIndextrain": jnp.mean(windowIndextrain) if windowIndextrain.size > 0 else 0,
                                        #---------Reward and error bars--------#
                                        #train average
                                        "reward_train":jnp.mean(reward_train) if reward_train.size > 0 else 0,
                                        "reward_train_plus_std": (jnp.mean(reward_train) + jnp.std(reward_train)) if reward_train.size > 0 else 0,
                                        "reward__train_minus_std": (jnp.mean(reward_train) - jnp.std(reward_train)) if reward_train.size > 0 else 0,
                        
                                        #eval
                                        "reward_eval":jnp.mean(reward_eval) if reward_eval.size > 0 else 0,
                                        "reward_eval_plus_std": (jnp.mean(reward_eval) + jnp.std(reward_eval)) if reward_eval.size > 0 else 0,
                                        "reward_eval_minus_std": (jnp.mean(reward_eval) - jnp.std(reward_eval)) if reward_eval.size > 0 else 0,
                                
                                        
                                        #---------PnL and errors bars-----------#
                                        #Average, end
                                       # "Episodic_PnL_train_mean": jnp.mean(episodic_PnL_train) if episodic_PnL_train.size > 0 else 0,
                                       # "Episodic_PnL_train_plus_std": (jnp.mean(episodic_PnL_train) + jnp.std(episodic_PnL_train)) if episodic_PnL_train.size > 0 else 0,
                                       # "Episodic_PnL_train_minus_std": (jnp.mean(episodic_PnL_train) - jnp.std(episodic_PnL_train)) if episodic_PnL_train.size > 0 else 0,
                                        #Average
                                        "PnL_train":jnp.mean(PnL_train) if PnL_train.size > 0 else 0,
                                        "PnL_train_plus_std": (jnp.mean(PnL_train) + jnp.std(PnL_train)) if PnL_train.size > 0 else 0,
                                        "PnL_train_minus_std": (jnp.mean(PnL_train) - jnp.std(PnL_train)) if PnL_train.size > 0 else 0,
                                
                                        #eval
                                        "Episodic_PnL_eval_mean": jnp.mean(Episodic_PnL_eval) if Episodic_PnL_eval.size > 0 else 0,
                                        "Episodic_PnL_eval_plus_std": (jnp.mean(Episodic_PnL_eval) + jnp.std(Episodic_PnL_eval)) if Episodic_PnL_eval.size > 0 else 0,
                                        "Episodic_PnL_eval_minus_std": (jnp.mean(Episodic_PnL_eval) - jnp.std(Episodic_PnL_eval)) if Episodic_PnL_eval.size > 0 else 0,
                                
                                        #-------------NetWorth and error bars----------#
                                        #train
                                      #  "Episodic_netWorth_train": jnp.mean(episodic_netWorth_train) if episodic_netWorth_train.size > 0 else 0,
                                       # "Episodic_netWorth_train_plus_std": (jnp.mean(episodic_netWorth_train) + jnp.std(episodic_netWorth_train)) if episodic_netWorth_train.size > 0 else 0,
                                        #"Episodic_netWorth_train_minus_st": (jnp.mean(episodic_netWorth_train) - jnp.std(episodic_netWorth_train)) if episodic_netWorth_train.size > 0 else 0,
                                        #Average
                                        "netWorth_train":jnp.mean(netWorth_train) if netWorth_train.size > 0 else 0,
                                        "netWorth_train_plus_std": (jnp.mean(netWorth_train) + jnp.std(netWorth_train)) if netWorth_train.size > 0 else 0,
                                        "netWorth_train_minus_std": (jnp.mean(netWorth_train) - jnp.std(netWorth_train)) if netWorth_train.size > 0 else 0,

                                        #eval
                                        "Episodic_netWorth_eval": jnp.mean(Episodic_netWorth_eval) if Episodic_netWorth_eval.size > 0 else 0,
                                        "Episodic_netWorth_eval_plus_std": (jnp.mean(Episodic_netWorth_eval) + jnp.std(Episodic_netWorth_eval)) if Episodic_netWorth_eval.size > 0 else 0,
                                        "Episodic_netWorth_eval_minus_std": (jnp.mean(Episodic_netWorth_eval) - jnp.std(Episodic_netWorth_eval)) if Episodic_netWorth_eval.size > 0 else 0,
                                
                                                                        
                                        #----------Iventory and error bars------------#
                                        #train
                                        #Average
                                        "Episodic_inventories_train": jnp.mean(Episodic_inventories_train) if Episodic_inventories_train.size > 0 else 0,
                                        "inventory_train": jnp.mean(inventories_train) if inventories_train.size > 0 else 0, 
                                        "inventory_train_plus_std":(jnp.mean(inventories_train) + jnp.std(inventories_train)) if inventories_train.size > 0 else 0,
                                        "inventory_train_minus_std":(jnp.mean(inventories_train) - jnp.std(inventories_train)) if inventories_train.size > 0 else 0,
                                
                                        #eval
                                        "inventory_eval": jnp.mean(inventories_eval) if inventories_eval.size > 0 else 0,
                                        "inventory_eval_plus_std":(jnp.mean(inventories_eval) + jnp.std(inventories_eval)) if inventories_eval.size > 0 else 0,
                                        "inventory_eval_minus_std":(jnp.mean(inventories_eval) - jnp.std(inventories_eval)) if inventories_eval.size > 0 else 0,
                                    
                                        
                                        #----------Buy and Sell Quant and error bars------------#
                                        #train
                                        #Average
                                        "buyQuant_train":jnp.mean(buyQuant_train) if buyQuant_train.size > 0 else 0,
                                        "sellQuant_train":jnp.mean(sellQuant_train) if sellQuant_train.size > 0 else 0,
                                        "other_exec_quants_train":jnp.mean(other_exec_quants_train) if other_exec_quants_train.size > 0 else 0,
                                        "averageMidprice_train":jnp.mean(averageMidprice_train) if averageMidprice_train.size>0 else 0,
                                        "averageBestbid_train":jnp.mean(averageBestbid_train) if averageBestbid_train.size>0 else 0,
                                        "averageBestask_train":jnp.mean(averageBestask_train) if averageBestask_train.size>0 else 0,
                                
                                        "buyQuant_eval":jnp.mean(buyQuant_eval) if buyQuant_eval.size > 0 else 0,
                                        "sellQuant_eval":jnp.mean(sellQuant_eval) if sellQuant_eval.size > 0 else 0,
                                        "other_exec_quants_eval":jnp.mean(other_exec_quants_eval) if other_exec_quants_eval.size > 0 else 0,
                                        "averageMidprice_eval":jnp.mean(averageMidprice_eval) if averageMidprice_eval.size>0 else 0,
                                        "averageBestbid_eval":jnp.mean(averageBestbid_eval) if averageBestbid_eval.size>0 else 0,
                                        "averageBestask_eval":jnp.mean(averageBestask_eval) if averageBestask_eval.size>0 else 0,                          
                                        
                                        # PPO Loss components
                                        "ppo_loss": float(loss) if loss is not None else 0,
                                        "ppo_value_loss": float(value_loss) if value_loss is not None else 0,
                                        "ppo_actor_loss": float(loss_actor) if loss_actor is not None else 0,
                                        "ppo_entropy": float(entropy) if entropy is not None else 0,
                                        # Weighted PPO Loss components
                                        "ppo_weighted_value_loss": float(config["VF_COEF"] * value_loss) if value_loss is not None else 0,
                                        "ppo_weighted_entropy": float(config["ENT_COEF"] * entropy) if entropy is not None else 0,
                                        "ppo_weighted_actor_loss": float(loss_actor) if loss_actor is not None else 0,
                                                                    },
                                    commit=True
                                )
                        jax.debug.callback(callback, info_train, eval_info, loss, value_loss, loss_actor, entropy)
        return {"params": params, "info_train": info_train, "eval_info": eval_info}
    return train

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

    
    env_config_hps = [{"observation_space":"engineered",
                            "reward_space":"portfolio_value",
                            "inv_penalty":"none",
                            "end_fn":"unwind_ref_price",
                            "fixed_quant_value":10,
                            "reference_price_portfolio_value":"near_touch",
                            "action_space":"directional_trading",
                            },
                            {"observation_space":"engineered",
                            "reward_space":"delta_netWorth",
                            "inv_penalty":"none",
                            "end_fn":"unwind_ref_price",
                            "fixed_quant_value":10,
                            "reference_price_portfolio_value":"near_touch",
                            "action_space":"directional_trading",
                            },
                            ]

    training_parameters = {
        "LR": {"values": [5e-5]},#, 3e-4, 1e-3
        "NUM_ENVS": {"values": [32,64]},
        "NUM_STEPS": {"values": [64,128]},  
        "TOTAL_TIMESTEPS": {"values": [1e6]},
        "UPDATE_EPOCHS": {"values": [4,8]},
        "NUM_MINIBATCHES": {"values": [16]},
        "GAMMA": {"values": [0.99,0.999]},
        "GAE_LAMBDA": {"values": [0.999,0.99]},
        "CLIP_EPS": {"values": [0.2]},
        "ENT_COEF": {"values": [0.1,0.01]},
        "VF_COEF": {"values": [0.0000005,0.00000005]},
        "MAX_GRAD_NORM": {"values": [0.5]},
        "ENV_NAME": {"values": ["AlphaTradeMM"]},
        "ANNEAL_LR": {"values": [True]},
        "DEBUG": {"values": [True]},

        "VERBOSE": {"values": [False]},
        "ACTION_TYPE": {"values": ["pure"]},
        "WINDOW_INDEX": {"values": [14]},
        "EPISODE_TIME": {"values": [60*2]},
        "DATA_TYPE": {"values": ["fixed_time"]},
        "NUM_STEPS_EVAL":{"values":[2]},
        "ATFOLDER": {"values": [ATFolder]},
        "ENV_CONFIG": {"values": env_config_hps},

    }    

    
    sweep_config={
        "method": "grid",
       
                        "parameters": training_parameters
    }

    def sweep_fun():
            run = wandb.init(
                project="Alphatrade_Sweeps",
                save_code=True,  # 
            )
            params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
            print(f"Results will be saved to {params_file_name}")
            # +++++ Single GPU +++++
            rng = jax.random.PRNGKey(1)
            train = (make_train(wandb.config))
            # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
            out = train(rng)
            params = out['params']
        
            # Save the params to a file using flax.serialization.to_bytes
            with open(params_file_name, 'wb') as f:
                f.write(flax.serialization.to_bytes(params))
                print(f"params saved")

            run.finish()

    sweep_id = wandb.sweep(sweep=sweep_config, project="MM_RWKV_directional_every_step_14")
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)




   