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
from flax import serialization
import dataclasses
from flax.core import frozen_dict


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
def make_test(config):
    """Make train function. 
    Input: Config
    Output: train function
    Train(rng): returns:
    return {"params": params, "info_train": info_train, "eval_info": eval_info}
    Infos and params
    """



    #Function to process the obsveration
    def handle_continuous(observation):
        if config["FLOAT_TYPE"] == "float16":
            return jnp.array(observation).astype(jnp.float16).view(jnp.uint16).astype(jnp.int32)
        elif config["FLOAT_TYPE"] == "float8":
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
            alphatradePath=config["ATFOLDER"]+"/val",
            window_index=config["WINDOW_INDEX"],
            episode_time=config["EPISODE_TIME"],
            ep_type=config["DATA_TYPE"],
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

    #Train env jit fn
    v_env_step = jax.jit(jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
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
        if config["FLOAT_TYPE"] == "float16":
            num_tokens = 1 + env.action_space(env_params).n + 65536 
        elif config["FLOAT_TYPE"] == "float8":
            num_tokens = 1 + env.action_space(env_params).n + 256
        config["MIN_ACTION_TOK"] = 1
        config["MAX_ACTION_TOK"] = env_config.n_actions

        #Load the RWKV
         # Load the trained model parameters 
        params_filename = config["params_path"]
        with open(params_filename, 'rb') as f:
            params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())
        RWKV, _ = get_rand_model(0, "6", 3, 256, num_tokens, dtype=jnp.float32, rwkv_type="ScanRWKV")
        #Define the forward function and jit version
        forward, _ = get_ppo_agent(RWKV, params, seed=1)
        v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))
    
        #Get init state for training
        init_state = RWKV.default_state(params)
        #returns 0s for weights, in /home/duser/AlphaTrade/jax_rwkv/src/jax_rwkv/base_rwkv.py
        if isinstance(init_state, tuple):
            init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
        else:
            init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)
        state = init_state

        #Start count
        global_timestep = 1

        #Reset training env
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        update_count=0

        for _ in range(int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]):
            #Intialise lists
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
                pi, _, state = v_forward_jit(tokenized, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1])
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
                _ = pi.log_prob(action)
                #============#
                #Update the state with the new action
                #============#
                _, _, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

                #=============#
                #Process action through environmnet
                #==============#
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, _, done, info_train = v_env_step(rng_step, env_state, action, env_params)

                #Reset state if done (rwkv state)
                state = jax.vmap(jax.lax.select)(done, init_state, state)


                #======Append list of actions, dones, rewards, value#

                return_values = info_train["returned_episode_returns"][info_train["returned_episode"]]
                episdoic_pnl=info_train["total_PnL"][info_train["returned_episode"]]
                for r in return_values:
                    update_returns.append(r)
                for p in episdoic_pnl:
                     update_pnl.append(p)                    
                global_timestep += 1*config["NUM_ENVS"]

            #Form lists for adv calcs
            print("Steps End")
            if len(update_returns) > 0:
                average_return = sum(update_returns) / len(update_returns)
                std_return = np.std(update_returns)
                print("avg returns:", average_return)
                print("std returns:", std_return)
            else:
                average_return = 0
                std_return = 0
                print("None ended")

            if len(update_pnl) > 0:
                average_pnl = sum(update_pnl) / len(update_pnl)
                std_pnl = np.std(update_pnl)
                print("avg episodic pnl:", average_pnl)
                print("std episodic pnl:", std_pnl)
            else:
                average_pnl = 0
                std_pnl = 0
                print("None ended")
            update_count+=1
            print("global_timestep",global_timestep)
            print("update_count",update_count)
            wandb.log(
                data={
                    "average_episodic_return": average_return,
                    "std_episodic_return": std_return,
                    "average_episodic_pnl": average_pnl,
                    "std_episodic_pnl": std_pnl,
                    "global_timestep": global_timestep,
                    "update_count":update_count
                },
                commit=True
            )

        return {"params": params, "info_train": info_train}
    return train

if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

    
    env_config_hps = [{"observation_space":"engineered",
                            "reward_space":"spooner_scaled",
                            "inv_penalty":"none",
                            "end_fn":"unwind_ref_price",
                            "fixed_quant_value":10,
                            "reference_price_portfolio_value":"near_touch",
                            "action_space":"fixed_quants",
                            "inventoryPnL_lambda":0.8,
                            "asymmetrically_dampened_lambda":0.2
                            },
                            ]

    training_parameters = {
        "LR": {"values": [5e-5]},#, 3e-4, 1e-3
        "NUM_ENVS": {"values": [32]},
        "NUM_STEPS": {"values": [64]},  
        "TOTAL_TIMESTEPS": {"values": [2.5e5]},
        "UPDATE_EPOCHS": {"values": [4]},
        "NUM_MINIBATCHES": {"values": [16]},
        "GAMMA": {"values": [0.99]},
        "GAE_LAMBDA": {"values": [0.999]},
        "CLIP_EPS": {"values": [0.2]},
        "ENT_COEF": {"values": [0.01]},
        "VF_COEF": {"values": [0.1]},
        "MAX_GRAD_NORM": {"values": [0.5]},
        "ENV_NAME": {"values": ["AlphaTradeMM"]},
        "ANNEAL_LR": {"values": [True]},
        "DEBUG": {"values": [True]},
        "VERBOSE": {"values": [False]},
        "ACTION_TYPE": {"values": ["pure"]},
        "WINDOW_INDEX": {"values": [-1]},
        "EPISODE_TIME": {"values": [60*5]},
        "DATA_TYPE": {"values": ["fixed_time"]},
        "NUM_STEPS_EVAL":{"values":[2]},
        "ATFOLDER": {"values": [ATFolder]},
        "ENV_CONFIG": {"values": env_config_hps},
        "FLOAT_TYPE": {"values": ["float16"]},
        "params_path":{"values": ["/home/duser/AlphaTrade/params_file_silver-sweep-3_04-16_12-34"]}
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
            train = (make_test(wandb.config))
            # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
            out = train(rng)
            params = out['params']
        
            # Save the params to a file using flax.serialization.to_bytes
            with open(params_file_name, 'wb') as f:
                f.write(flax.serialization.to_bytes(params))
                print(f"params saved")

            run.finish()

    sweep_id = wandb.sweep(sweep=sweep_config, project="MM_RWKV_TEST")
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)




   