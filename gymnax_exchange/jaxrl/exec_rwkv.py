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
from gymnax_exchange.jaxen.exec_env import ExecutionEnv 
from gymnax_exchange.jaxob.jaxob_config import EnvironmentExecutionConfig
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
    env_config=EnvironmentExecutionConfig(**config["ENV_CONFIG"])

    #===========#
    #Define Envs#
    #==========#
    env = ExecutionEnv(
        env_config,
        key_reset,
        alphatradePath=config["ATFOLDER"]+"/train",
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        rewardLambda=config["REWARD_LAMDA"],
        trader_unique_id=config["TRADER_UNIQUE_ID"],
        ep_type=config["DATA_TYPE"],
        
    )

    eval_env=ExecutionEnv(
        env_config,
        key_reset,
        alphatradePath=config["ATFOLDER"]+"/val",
       window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        rewardLambda=config["REWARD_LAMDA"],
        trader_unique_id=config["TRADER_UNIQUE_ID"],
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
                for r in return_values:
                    update_returns.append(r)
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
                print("avg returns:", sum(update_returns) / len(update_returns))
            else:
                print("None ended")

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
            if config.get("DEBUG"):
                def callback(info_train,info_eval):
                    #------------Collect info for plotting---------------------------#
                    #1)Step and return info
                    return_values = info_train["returned_episode_returns"][info_train["returned_episode"]] 
                    timesteps = info_train["timestep"][info_train["returned_episode"]] * config["NUM_ENVS"] 
                    #windowIndextrain = info_train["window_index"][:, config["ENVID"]] 

                    #-----------Train info----------#
                    episodic_revenues_train = info_train["total_revenue"][info_train["returned_episode"]] 
                    reward_lam1_train=info_train["reward_lam1"]
                    quant_executed_train=info_train["quant_executed"]
                    average_price_train = info_train["average_price"]
                    current_step_train = info_train["current_step"] 
                    mkt_forced_quant_train=info_train["mkt_forced_quant"]
                    doom_quant_train=info_train["doom_quant"]
                    trade_duration_train=info_train["trade_duration"]
                    advantage_reward_train=info_train["advantage_reward"]
                    drift_reward_train = info_train["drift_reward"] 
                    
                   

                    #-------------eval info------#   
                    episodic_revenues_eval = info_eval["total_revenue"][info_eval["returned_episode"]] 
                    reward_lam1_eval=info_eval["reward_lam1"]
                    quant_executed_eval=info_eval["quant_executed"]
                    average_price_eval = info_eval["average_price"]
                    current_step_eval = info_eval["current_step"] 
                    mkt_forced_quant_eval=info_eval["mkt_forced_quant"][info_eval["returned_episode"]] 
                    doom_quant_eval=info_eval["doom_quant"][info_eval["returned_episode"]] 
                    trade_duration_eval=info_eval["trade_duration"]
                    advantage_reward_eval=info_eval["advantage_reward"]
                    drift_reward_eval = info_eval["drift_reward"]
                    
                    
                    #-----------------Logging-------------------#

                    if wandbOn:
                        wandb.log(
                            data={
                                #-----time and return------------#
                                "episodic_return": jnp.mean(return_values) if return_values.size > 0 else 0,  # Handle empty arrays
                                "global_step": jnp.max(timesteps) if timesteps.size>0 else 0,
                                #"windowIndextrain": jnp.mean(windowIndextrain) if windowIndextrain.size > 0 else 0,

                                #---------Reward and error bars--------#
                                #train
                                #advantage
                                "advantage_reward_train":jnp.mean(advantage_reward_train) if advantage_reward_train.size > 0 else 0,
                                "advantage_reward_train_plus_std": (jnp.mean(advantage_reward_train) + jnp.std(advantage_reward_train)) if advantage_reward_train.size > 0 else 0,
                                "advantage_reward_train_minus_std": (jnp.mean(advantage_reward_train) - jnp.std(advantage_reward_train)) if advantage_reward_train.size > 0 else 0,
                                #drift
                                "drift_reward_train":jnp.mean(drift_reward_train) if drift_reward_train.size > 0 else 0,
                                "drift_reward_train_plus_std": (jnp.mean(drift_reward_train) + jnp.std(drift_reward_train)) if drift_reward_train.size > 0 else 0,
                                "drift_reward_train_minus_std": (jnp.mean(drift_reward_train) - jnp.std(drift_reward_train)) if drift_reward_train.size > 0 else 0,
                                #eval
                                #advantage
                                "advantage_reward_eval":jnp.mean(advantage_reward_eval) if advantage_reward_eval.size > 0 else 0,
                                "advantage_reward_eval_plus_std": (jnp.mean(advantage_reward_eval) + jnp.std(advantage_reward_eval)) if advantage_reward_eval.size > 0 else 0,
                                "advantage_reward_eval_minus_std": (jnp.mean(advantage_reward_eval) - jnp.std(advantage_reward_eval)) if advantage_reward_eval.size > 0 else 0,
                                #drift
                                "drift_reward_eval":jnp.mean(drift_reward_eval) if drift_reward_eval.size > 0 else 0,
                                "drift_reward_eval_plus_std": (jnp.mean(drift_reward_eval) + jnp.std(drift_reward_eval)) if drift_reward_eval.size > 0 else 0,
                                "drift_reward_eval_minus_std": (jnp.mean(drift_reward_eval) - jnp.std(drift_reward_eval)) if drift_reward_eval.size > 0 else 0,
                                #Baseline
                              
                                
                                #---------Revenue and errors bars-----------#
                                #reward
                                "revenues_train": jnp.mean(episodic_revenues_train) if episodic_revenues_train.size > 0 else 0,
                                #"revenues_train_plus_std": (jnp.mean(episodic_revenues_train) + jnp.std(episodic_revenues_train)) if episodic_revenues_train.size > 0 else 0,
                                #"revenues_train_minus_std": (jnp.mean(episodic_revenues_train) - jnp.std(episodic_revenues_train)) if episodic_revenues_train.size > 0 else 0,
                                #eval
                                "revenues_eval": jnp.mean(episodic_revenues_eval) if episodic_revenues_eval.size > 0 else 0,
                                "revenues_eval_plus_std": (jnp.mean(episodic_revenues_eval) + jnp.std(episodic_revenues_eval)) if episodic_revenues_eval.size > 0 else 0,
                                "revenues_eval_minus_std": (jnp.mean(episodic_revenues_eval) - jnp.std(episodic_revenues_eval)) if episodic_revenues_eval.size > 0 else 0,


                                "reward_lam1_train":jnp.mean(reward_lam1_train) if reward_lam1_train.size > 0 else 0,
                                "reward_lam1_eval":jnp.mean(reward_lam1_eval) if reward_lam1_eval.size > 0 else 0,


                                

                                #-------------quant_executed_train and error bars----------#
                                #train
                                "quant_executed_train": jnp.mean(quant_executed_train) if quant_executed_train.size > 0 else 0,
                                "quant_executed_train_plus_std": (jnp.mean(quant_executed_train) + jnp.std(quant_executed_train)) if quant_executed_train.size > 0 else 0,
                                "quant_executed_train_minus_st": (jnp.mean(quant_executed_train) - jnp.std(quant_executed_train)) if quant_executed_train.size > 0 else 0,
                                #eval
                                "quant_executed_eval": jnp.mean(quant_executed_eval) if quant_executed_eval.size > 0 else 0,
                                "quant_executed_eval_plus_std": (jnp.mean(quant_executed_eval) + jnp.std(quant_executed_eval)) if quant_executed_eval.size > 0 else 0,
                                "quant_executed_eval_minus_st": (jnp.mean(quant_executed_eval) - jnp.std(quant_executed_eval)) if quant_executed_eval.size > 0 else 0,
                              
                                #----------average_price_eval and error bars------------#
                                #train
                                "average_price_train": jnp.mean(average_price_train) if average_price_train.size > 0 else 0,
                                "average_price_train_plus_std":(jnp.mean(average_price_train) + jnp.std(average_price_train)) if average_price_train.size > 0 else 0,
                                "average_price_train_minus_std":(jnp.mean(average_price_train) - jnp.std(average_price_train)) if average_price_train.size > 0 else 0,
                                #eval
                                "average_price_eval": jnp.mean(average_price_eval) if average_price_eval.size > 0 else 0,
                                "average_price_eval_plus_std":(jnp.mean(average_price_eval) + jnp.std(average_price_eval)) if average_price_eval.size > 0 else 0,
                                "average_price_eval_minus_std":(jnp.mean(average_price_eval) - jnp.std(average_price_eval)) if average_price_eval.size > 0 else 0,
                               
                                #----------Buy and Sell Quant and error bars------------#
                                #train
                                "mkt_forced_quant_train":jnp.mean(mkt_forced_quant_train) if mkt_forced_quant_train.size > 0 else 0,
                                "doom_quant_train":jnp.mean(doom_quant_train) if doom_quant_train.size > 0 else 0,
                                "trade_duration_train":jnp.mean(trade_duration_train) if trade_duration_train.size > 0 else 0,

                                #eval
                                "mkt_forced_quant_eval":jnp.mean(mkt_forced_quant_eval) if mkt_forced_quant_eval.size > 0 else 0,
                                "doom_quant_eval":jnp.mean(doom_quant_eval) if doom_quant_eval.size > 0 else 0,
                                "trade_duration_eval":jnp.mean(trade_duration_eval) if trade_duration_eval.size > 0 else 0,   
                            
                             
                                #-------step-----------------#
                                "current_step_train":jnp.mean(current_step_train) if current_step_train.size > 0 else 0,
                                "current_step_eval":jnp.mean(current_step_eval) if current_step_eval.size > 0 else 0,
                              
                                #----------Action prices------------#
                              
                               
                              
                            
                                                            },
                            commit=True
                        )
                       
                jax.debug.callback(callback, info_train,eval_info)
        return {"params": params, "info_train": info_train, "eval_info": eval_info}
    return train

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

    env_config_hps = [ {"task":"buy",
                        "action_type":"pure",
                        "action_space":"fixed_quants",
                        "end_fn":"unwind_FT",
                        "max_task_size":100,
                        "n_actions":8,
                        "fixed_quant_value":10,
                        "num_messages_by_agent":8,}
                        ]

    training_parameters = {
        "LR": {"values": [1e-4, 3e-4, 1e-3]},
        "NUM_ENVS": {"values": [32]},
        "NUM_STEPS": {"values": [32]},  
        "TOTAL_TIMESTEPS": {"values": [1e5]},
        "UPDATE_EPOCHS": {"values": [4,10]},#
        "NUM_MINIBATCHES": {"values": [4]},
        "GAMMA": {"values": [0.9999,0.99]},
        "GAE_LAMBDA": {"values": [0.99,0.95]},#
        "CLIP_EPS": {"values": [0.15]},
        "ENT_COEF": {"values": [0.01,0.0,0.1]},#
        "VF_COEF": {"values": [1,0.5]},#,0.5
        "MAX_GRAD_NORM": {"values": [5,1,0.5]},
        "ENV_NAME": {"values": ["AlphaTradeExec"]},
        "ANNEAL_LR": {"values": [True]},
        "DEBUG": {"values": [True]},
        "VERBOSE": {"values": [False]},
        "ACTION_TYPE": {"values": ["pure"]},
        "WINDOW_INDEX": {"values": [-1]},
        "EPISODE_TIME": {"values": [60*5]},
        "DATA_TYPE": {"values": ["fixed_time"]},
        "NUM_STEPS_EVAL":{"values":[32]},
        "ATFOLDER": {"values": [ATFolder]},
        "ENV_CONFIG": {"values": env_config_hps},
        "TRADER_UNIQUE_ID": {"values": [10]},
        "REWARD_LAMDA": {"values": [0.2]},
        "ENVID": {"values": [1]},

    }    

    
    sweep_config={
        "method": "bayes",
        "parameters": training_parameters,
        "metric": {'goal': 'maximize', 'name': 'episodic_return'},
    }

    def sweep_fun():
            run = wandb.init(
                project="Alphatrade_Sweeps",
                save_code=True,  # 
            )
            params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
            print(f"Results will be saved to {params_file_name}")
            # +++++ Single GPU +++++
            rng = jax.random.PRNGKey(0)
            train = (make_train(wandb.config))
            # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
            out = train(rng)
            params = out['params']
        
            # Save the params to a file using flax.serialization.to_bytes
            with open(params_file_name, 'wb') as f:
                f.write(flax.serialization.to_bytes(params))
                print(f"params saved")

            run.finish()

    sweep_id = wandb.sweep(sweep=sweep_config, project="EXEC_RWKV_FULL_DAY")
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)




   