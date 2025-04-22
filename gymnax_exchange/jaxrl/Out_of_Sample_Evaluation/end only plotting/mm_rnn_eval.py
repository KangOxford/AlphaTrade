import jax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
import jax.numpy as jnp
import flax.linen as nn
import datetime
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Callable, Optional
import distrax
import gymnax
import functools
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
import flax
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

from purejaxrl.purejaxrl.wrappers import LogWrapper, FlattenObservationWrapper


#=============import the policy==============#
import jax
import jax.numpy as jnp
from flax import serialization


# ================= imports =================#

from dataclasses import dataclass

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"




from ast import Dict
from contextlib import nullcontext
# from email import message
# from random import sample
# from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, flatten_util
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
import pandas as pd 
import matplotlib.pyplot as plt 
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.base_env import EnvParams as BaseEnvParams
from gymnax_exchange.jaxen.base_env import EnvState as BaseEnvState
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv as MarketMakingEnv
from gymnax_exchange.utils import utils
import dataclasses
from flax.core import frozen_dict
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig


from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString




wandbOn = True # False
if wandbOn:
    import wandb
#----Wandb and parameters----#



# ============================
# Configuration
# ============================




#----Define the RNN model----#

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        # Initialize the GRUCell with the hidden size.
        gru_cell = nn.GRUCell(features=ins.shape[-1])  # Use the last dimension of `ins` as the hidden size.
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[-1]),  # Use the last dimension of `ins` as the hidden size.
            rnn_state,
        )
        new_rnn_state, y = gru_cell(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Initialize the GRUCell with the hidden_size as the features argument.
        gru_cell = nn.GRUCell(features=hidden_size)
        # Use a dummy key since the default state init fn is just zeros.
        return gru_cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )
class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


def make_test(config):
    """Make train function. 
    Input: Config
    Output: Test function
    Train(rng): returns:
    Infos and params
    """
    #=========#
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
    


    #===========================================#
    #Init the pre trained model
    #======================================#
    # Load the trained model parameters 
    # Initialize the model
    params_filename = config["params_path"]
    with open(params_filename, 'rb') as f:
        params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

    def train(rng):
        """Train function
        input rng key
        output:
        return {"params": params, "info_train": info_train, "eval_info": eval_info}"""  

        #Start count
        global_timestep = 1

        #Reset training env
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        done = jnp.zeros((config["NUM_ENVS"],), dtype=bool) #init done
        update_count=0

        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        hstate=init_hstate
        
        for _ in range(int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]):
            #Intialise lists
            all_actions = []
            update_returns = []
            update_pnl=[]
            

            for t in range(config["NUM_STEPS"]):
                rng, _rng = jax.random.split(rng)

                ac_in = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
                hstate, pi, _ = network.apply(params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                action = jnp.squeeze(action)

                #log actions
                def log_action_distribution(action):
                            unique_actions, counts = jnp.unique(action, return_counts=True)
                            action_distribution = {f"action_{int(a)}": int(c) for a, c in zip(unique_actions, counts)}
                            wandb.log(action_distribution)
                if wandbOn:
                    jax.debug.callback(log_action_distribution, action)
                
                # Take a step in the environment
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, _, done,_info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
                #======Append list of actions, dones, rewards, value#
                hstate = jax.vmap(jax.lax.select)(done, init_hstate, hstate)

                return_values = _info["returned_episode_returns"][_info["returned_episode"]]
                episdoic_pnl=_info["total_PnL"][_info["returned_episode"]]
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

        return {"params": params, "info_train": _info}
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
        "params_path":{"values": ["/home/duser/AlphaTrade/params_file_mild-sweep-1_04-15_12-35"]}
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

            run.finish()

    sweep_id = wandb.sweep(sweep=sweep_config, project="MM_RNN_TEST")
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)



        