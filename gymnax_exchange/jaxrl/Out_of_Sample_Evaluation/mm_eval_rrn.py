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


import wandb  # Import Weights & Biases for logging
#----Wandb and parameters----#
# Initialize wandb
wandb.init(project="AlphaTrade_MM_RNN_Eval", config={"run_type": "evaluation"})

# Load the trained model parameters 

params_filename = "/home/duser/AlphaTrade/params_file_balmy-sweep-6_03-20_00-48"
with open(params_filename, 'rb') as f:
    params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())


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
    



#==================Main==================#
if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        #ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        ATFolder= "/home/duser/AlphaTrade/training_oneDay/test"

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
    
    # Initialize the model
    network = ActorCriticRNN(test_env.action_space(test_env_params).n, config=config)
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)


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
            test_ac_in = (test_obsv[jnp.newaxis, :], test_done[jnp.newaxis, :])
            init_hstate, pi, value = network.apply(params, init_hstate, test_ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                        value.squeeze(0),
                        action.squeeze(0),
                        log_prob.squeeze(0),
                    )
            
            # Take a step in the environment
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
            
            

        