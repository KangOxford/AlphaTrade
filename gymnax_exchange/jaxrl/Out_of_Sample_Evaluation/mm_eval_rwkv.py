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
        "EPISODE_TIME": 60*5, 
        "TRADERID": 10,
        "NUM_EPS":20,
        "NUM_ENVS": 64,
    }
    
    
    #-----Define testing configuration-----#
    test_env_config_hps = [{"observation_space":"engineered",
                         "reward_space":"spooner_scaled",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"mid",
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
    params_filename = "/home/duser/AlphaTrade/params_file_faithful-sweep-1_04-12_21-08"
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
    v_env_step = jax.jit(jax.vmap(
        test_env.step, in_axes=(0, 0, 0, None)
    ))

    #Get init state for training
    init_state = RWKV.default_state(params)
    #returns 0s for weights, in /home/duser/AlphaTrade/jax_rwkv/src/jax_rwkv/base_rwkv.py
    if isinstance(init_state, tuple):
        init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
    else:
        init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)
    state = init_state

    returns_mean_per_ep = []
    returns_std_per_ep = []
    pnls_mean_per_ep = []
    pnls_std_per_ep = []

    episodes = config["NUM_EPS"] # Run for multiple episodes
    for episode in range(episodes):
        print(f"=== Starting Episode {episode} ===")
        # Reset the environments
        reset_rng = jax.random.split(key_reset, config["NUM_ENVS"])
        test_obsv, test_env_state = jax.vmap(test_env.reset, in_axes=(0, None))(reset_rng, test_env_params)
  
        # ============================
        # Run the test loop
        # ============================
        # Initialize per-env tracking arrays
        test_episode_reward = jnp.zeros(config["NUM_ENVS"])
        episode_returns = jnp.full(config["NUM_ENVS"], jnp.nan)
        episode_pnls = jnp.full(config["NUM_ENVS"], jnp.nan)
        test_done_mask = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        for step in range(100000000): 
            # =========================== #
            # Skip completed envs by masking observations
            # =========================== #
            masked_obsv = jax.tree_util.tree_map(
                lambda x: jnp.where(test_done_mask[:, None], jnp.zeros_like(x), x), 
                test_obsv
            )

            # Tokenize observation
            tokenized = handle_continuous(masked_obsv)
            #jax.debug.print("tokenized shape: {}", tokenized.shape)

            # Evaluate policy
            pi, value, state = v_forward_jit(
                tokenized, state, params,
                jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1]
            )
            pi = distrax.Categorical(
                logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1]
            )
            action = pi.sample(seed=key_policy)
            log_prob = pi.log_prob(action)

            # Mask action for done envs
            action = jnp.where(test_done_mask, 0, action)  # or any no-op action

            # Update state
            _, value1, state = v_forward_jit(
                action, state, params,
                jnp.ones(config["NUM_ENVS"], dtype=jnp.int32)
            )

            # Step environments
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, test_env_state, test_reward, test_done, info_train = v_env_step(
                rng_step, test_env_state, action, test_env_params
            )

            # Accumulate rewards only for active envs
            test_episode_reward += jnp.where(test_done_mask, 0, test_reward)

            # Identify envs that just finished this step
            newly_done = jnp.logical_and(test_done, jnp.logical_not(test_done_mask))

            # Log final return and PnL for newly finished envs
            def get_final_pnl(info, default=0.0):
                return info.get("total_PnL", default)

            extract_pnl_vmap = jax.vmap(get_final_pnl)
            final_pnls = extract_pnl_vmap(info_train)

            # Update logs where newly done
            episode_returns = jnp.where(newly_done, test_episode_reward, episode_returns)
            episode_pnls = jnp.where(newly_done, final_pnls, episode_pnls)

            # Update done mask
            test_done_mask = jnp.logical_or(test_done_mask, test_done)

            # Update observation
            test_obsv = obsv

            print(f"on step {step} of episode {episode} out of {episodes}")

            if test_done_mask.all():
                break   
        print(f"\n=== Episode {episode} Results ===")
        print("Returns:", episode_returns)
        print("PnLs:   ", episode_pnls)
    
        mean_return = jnp.nanmean(episode_returns)
        std_return = jnp.nanstd(episode_returns)

        mean_pnl = jnp.nanmean(episode_pnls)
        std_pnl = jnp.nanstd(episode_pnls)

        wandb.log({
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "mean_pnl": float(mean_pnl),
            "std_pnl": float(std_pnl),
            "episode": episode,
        })
        wandb.log({
            "returns_distribution": wandb.Histogram(episode_returns),
            "pnls_distribution": wandb.Histogram(episode_pnls),
            "episode": episode,
        })

        # Store them
        returns_mean_per_ep.append(mean_return)
        returns_std_per_ep.append(std_return)
        pnls_mean_per_ep.append(mean_pnl)
        pnls_std_per_ep.append(std_pnl)

        print(f"Mean Return: {mean_return:.2f}, Std Return: {std_return:.2f}")
        print(f"Mean PnL: {mean_pnl:.2f}, Std PnL: {std_pnl:.2f}")
        
            

  
            
            

        