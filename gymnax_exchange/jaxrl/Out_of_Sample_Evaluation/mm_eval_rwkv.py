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
# Initialize wandb
wandb.init(project="AlphaTrade_MM_RWKV_Eval", config={"run_type": "evaluation"})

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
    
#Function to process the obsveration
def handle_continuous(observation):
        return jnp.array(observation).astype(jnp.float8_e4m3b11fnuz).view(jnp.uint8).astype(jnp.int32)







#==================Main==================#
if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        #ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        ATFolder= "/home/duser/AlphaTrade/training_oneDay/val"

        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"

    #-----Shared configuration-----#
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": -1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*15,
        "NUM_ENVS": 256, 
        "TRADERID": 10,
    }
    
    
    #-----Define testing configuration-----#
    test_env_config_hps = [{"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"linear",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"fixed_quants"
                         }]
   
    test_env_cfg=EnvironmentConfig(**test_env_config_hps[0])
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    test_env = MarketMakingEnv(
        cfg = test_env_cfg,
        key = key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        trader_unique_id=config["TRADERID"],
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    test_env_params = dataclasses.replace(
        test_env.default_params,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )

    
# Load the trained model parameters 

    params_filename = "/home/duser/AlphaTrade/params_file_upbeat-sweep-1_04-08_11-16"
    with open(params_filename, 'rb') as f:
        params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())
        
    # Initialize the model
    num_tokens = 1 + test_env.action_space(test_env_params).n + 256
    config["MIN_ACTION_TOK"] = 1
    config["MAX_ACTION_TOK"] = test_env_cfg.n_actions

    #Load the RWKV
    RWKV, _ = get_rand_model(0, "6", 3, 256, num_tokens, dtype=jnp.float32, rwkv_type="ScanRWKV")
    #Define the forward function and jit version
    forward, params = get_ppo_agent(RWKV, params, seed=1)
    v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))

    #Get init state
    init_state = RWKV.default_state(params)
    #returns 0s for weights, in /home/duser/AlphaTrade/jax_rwkv/src/jax_rwkv/base_rwkv.py
    if isinstance(init_state, tuple):
        init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
    else:
        init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)
    state = init_state
    


    #-'''''Define baseline configuration-------#
    baseline_config_hps = {"observation_space":"engineered",
                            "reward_space":"portfolio_value",
                            "inv_penalty":"linear",
                            "n_actions":8,
                            "end_fn":"unwind_ref_price",
                            "fixed_quant_value":10,
                            "reference_price_portfolio_value":"best_bid_ask",
                            "action_space":"AvSt",
                            },
  
    baseline_env_config = EnvironmentConfig(**baseline_config_hps[0])
    baseline_env=MarketMakingEnv(  
        cfg = baseline_env_config,
        key = key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        trader_unique_id=config["TRADERID"],
        ep_type=config["EP_TYPE"],
    )

    baseline_env_params = dataclasses.replace(
        baseline_env.default_params,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )




    reset_rng = jax.random.split(key_reset, config["NUM_ENVS"])

    total_rewards = []
    total_revenues = []
    total_executed = []

    j_calculate_gae = jax.jit(jax.vmap(calculate_gae, in_axes=(0, 0, 0, 0, 0, None, None)))

    episodes = 1 # Run for multiple episodes
    for episode in range(episodes):
        # Reset the environments
        #test
        test_obsv, test_env_state = jax.vmap(test_env.reset, in_axes=(0, None))(reset_rng, test_env_params)
        test_done = jnp.array([False])
        #baseline
        baseline_obsv, baseline_env_state = baseline_env.reset(key_reset, baseline_env_params)
        baseline_done = jnp.array([False])

        test_episode_reward = 0
        baseline_episode_reward = 0
        # ============================
        # Run the test loop
        # ============================
        for step in range(100000000): 
            rng, _rng = jax.random.split(rng)
            #test action
            tokenized = handle_continuous(test_obsv)

            #====================#
            #Evaluate policy from model
            #=====================#
            pi, value, state = v_forward_jit(tokenized, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1])
            pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
            action = pi.sample(seed=_rng)
            current_actions = jax.device_get(action)
            log_prob = pi.log_prob(action)

            #============#
            #Update the state with the new action
            #============#
            _, value1, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

            # Take a step in the environment
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            test_obsv, test_env_state, test_reward, test_done, test_info = jax.vmap(test_env.step, in_axes=(0, 0, 0, None))(rng_step, test_env_state, action, test_env_params)
            
            test_episode_reward += test_reward.sum()

            #baseline action
            baseline_action = 5
            baseline_obsv, baseline_env_state, baseline_reward, baseline_done, baseline_info = baseline_env.step(key_step, baseline_env_state, baseline_action, baseline_env_params)
            baseline_episode_reward += baseline_reward

            # Logging every N steps (to avoid spamming WandB)
        
        
            # Log results
            #-----------Train info----------#
            PnL_test = test_info["total_PnL"]
            inventories_test = test_info["inventory"] 
            buyQuant_test=test_info["buyQuant"]
            sellQuant_test=test_info["sellQuant"]
            reward_test=test_info["reward"]
            other_exec_quants_test=test_info["other_exec_quants"]
            netWorth_test = test_info["netWorth"]
            averageMidprice_test=test_info["averageMidprice"]
            averageBestbid_test=test_info["average_best_bid"]
            averageBestask_test=test_info["average_best_ask"]
            

            #-------------Baseline info------#   
            PnL_baseline = baseline_info["total_PnL"]
            inventories_baseline = baseline_info["inventory"]
            buyQuant_baseline=baseline_info["buyQuant"]
            sellQuant_baseline=baseline_info["sellQuant"]
            reward_baseline=baseline_info["reward"]
            other_exec_quants_baseline=baseline_info["other_exec_quants"]
            netWorth_baseline = baseline_info["netWorth"]
            averageMidprice_baseline=baseline_info["averageMidprice"]
            averageBestbid_baseline=baseline_info["average_best_bid"]
            averageBestask_baseline=baseline_info["average_best_ask"]
            
            
            
            #-----------------Logging-------------------#
            wandb.log(
                    data={
                        #-----time and return------------#

                        "global_step": step,
                        #---------Reward and error bars--------#
                        #train
                        "reward_test":jnp.mean(reward_test) if reward_test.size > 0 else 0,
                        "reward_test_plus_std": (jnp.mean(reward_test) + jnp.std(reward_test)) if reward_test.size > 0 else 0,
                        "reward__test_minus_std": (jnp.mean(reward_test) - jnp.std(reward_test)) if reward_test.size > 0 else 0,

                    
                        #baseline
                        "reward_baseline":jnp.mean(reward_baseline) if reward_baseline.size > 0 else 0,
                        "reward_baseline_plus_std": (jnp.mean(reward_baseline) + jnp.std(reward_baseline)) if reward_baseline.size > 0 else 0,
                        "reward_baseline_minus_std": (jnp.mean(reward_baseline) - jnp.std(reward_baseline)) if reward_baseline.size > 0 else 0,
                        
                        #---------PnL and errors bars-----------#
                        #reward
                        "PnL_test_mean": jnp.mean(PnL_test) if PnL_test.size > 0 else 0,
                        "PnL_test_plus_std": (jnp.mean(PnL_test) + jnp.std(PnL_test)) if PnL_test.size > 0 else 0,
                        "PnL_test_minus_std": (jnp.mean(PnL_test) - jnp.std(PnL_test)) if PnL_test.size > 0 else 0,
            
                        #baseline
                        "PnL_baseline": PnL_baseline,
                

                        #-------------NetWorth and error bars----------#
                        #train
                        "netWorth_test": jnp.mean(netWorth_test) if netWorth_test.size > 0 else 0,
                        "netWorth_test_plus_std": (jnp.mean(netWorth_test) + jnp.std(netWorth_test)) if netWorth_test.size > 0 else 0,
                        "netWorth_test_minus_st": (jnp.mean(netWorth_test) - jnp.std(netWorth_test)) if netWorth_test.size > 0 else 0,
                    
                        #baseline
                        "netWorth_baseline": netWorth_baseline,
                                                        
                        #----------Iventory and error bars------------#
                        #train
                        "inventory_test": jnp.mean(inventories_test) if inventories_test.size > 0 else 0, 
                        "inventory_test_plus_std":(jnp.mean(inventories_test) + jnp.std(inventories_test)) if inventories_test.size > 0 else 0,
                        "inventory_test_minus_std":(jnp.mean(inventories_test) - jnp.std(inventories_test)) if inventories_test.size > 0 else 0,
                        #eval

                        #baseline
                        "inventory_baseline": inventories_baseline,
                        
                        #----------Buy and Sell Quant and error bars------------#
                        #train
                        "buyQuant_test":jnp.mean(buyQuant_test) if buyQuant_test.size > 0 else 0,
                        "sellQuant_test":jnp.mean(sellQuant_test) if sellQuant_test.size > 0 else 0,
                        "other_exec_quants_test":jnp.mean(other_exec_quants_test) if other_exec_quants_test.size > 0 else 0,
                        "averageMidprice_test":jnp.mean(averageMidprice_test) if averageMidprice_test.size>0 else 0,
                        "averageBestbid_test":jnp.mean(averageBestbid_test) if averageBestbid_test.size>0 else 0,
                        "averageBestask_test":jnp.mean(averageBestask_test) if averageBestask_test.size>0 else 0,
                        
                        
                        #baseline
                        "buyQuant_baseline":jnp.mean(buyQuant_baseline) if buyQuant_baseline.size > 0 else 0,
                        "sellQuant_baseline":jnp.mean(sellQuant_baseline) if sellQuant_baseline.size > 0 else 0,
                        "other_exec_quants_baseline":jnp.mean(other_exec_quants_baseline) if other_exec_quants_baseline.size > 0 else 0,
                        "averageMidprice_baseline":jnp.mean(averageMidprice_baseline) if averageMidprice_baseline.size>0 else 0,
                        "averageBestbid_baseline":jnp.mean(averageBestbid_baseline) if averageBestbid_baseline.size>0 else 0,
                        "averageBestask_baseline":jnp.mean(averageBestask_baseline) if averageBestask_baseline.size>0 else 0,

                        "reward_histogram": wandb.Histogram(reward_test),
                        "PnL_histogram": wandb.Histogram(PnL_test),
                        "networth_histogram": wandb.Histogram(netWorth_test),
                        },
                                                                                
                    commit=True,
                )
            print(f"on step {step} of episode {episode} out of {episodes}")         
            
            if test_done.all():
                break

    print("Done epsiode", episode)   
            
            

        