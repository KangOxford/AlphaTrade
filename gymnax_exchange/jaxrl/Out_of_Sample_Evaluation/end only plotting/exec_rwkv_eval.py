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
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import flax
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
# Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False)  # finds a whole assortment of leaks if true... bizarre.
import datetime
import dataclasses

from gymnax_exchange.jaxob.jaxob_config import  EnvironmentExecutionConfig
from flax.core import frozen_dict
from flax import serialization

import wandb  # Import Weights & Biases for logging


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



from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString




wandbOn = True # False
if wandbOn:
    import wandb
#----Wandb and parameters----#
# Initialize wandb
wandb.init(project="AlphaTrade_EXEC_RWKV_Eval", config={"run_type": "evaluation"})


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
        ATFolder= "/home/duser/AlphaTrade/training_oneDay/val"


    #-----Shared configuration-----#
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 13,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 5,
        "NUM_EPS":10
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env_config_hps = [ {"task":"buy",
                        "action_type":"pure",
                        "action_space":"fixed_quants",
                        "end_fn":"unwind_FT",
                        "max_task_size":100,
                        "n_actions":8,
                        "fixed_quant_value":10,
                        "num_messages_by_agent":8,
                        "debug_mode":False}
                        ]
    trader_id=10
    env_cfg=EnvironmentExecutionConfig(**env_config_hps[0])
    env = ExecutionEnv(
        cfg=env_cfg,
        key=key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        trader_unique_id=trader_id,
        ep_type=config["EP_TYPE"],
    )

    env_params = dataclasses.replace(
        env.default_params,
        episode_time=config["EPISODE_TIME"],
    )

    obs, env_state = env.reset(key_reset, env_params)
    # Load the trained model parameters 
    params_filename = "/home/duser/AlphaTrade/params_file_polished-sweep-1_04-15_14-52"
    with open(params_filename, 'rb') as f:
        params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())
        
    # Initialize the model
    num_tokens = 1 + env.action_space(env_params).n + 256
    config["MIN_ACTION_TOK"] = 1
    config["MAX_ACTION_TOK"] = env_cfg.n_actions

    #Load the RWKV
    RWKV, _ = get_rand_model(0, "6", 3, 256, num_tokens, dtype=jnp.float32, rwkv_type="ScanRWKV")
    #Define the forward function and jit version
    forward, params = get_ppo_agent(RWKV, params, seed=1)
    v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))
    v_env_step = jax.jit(jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
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
    revenue_mean_per_ep = []
    revenue_std_per_ep = []

    episodes = config["NUM_EPS"] # Run for multiple episodes
    for episode in range(episodes):
        print(f"=== Starting Episode {episode} ===")
        # Reset the environments
        reset_rng = jax.random.split(key_reset, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
  
        # ============================
        # Run the test loop
        # ============================
        # Initialize per-env tracking arrays
        episode_reward = jnp.zeros(config["NUM_ENVS"])
        episode_returns = jnp.full(config["NUM_ENVS"], jnp.nan)
        episode_revenues = jnp.full(config["NUM_ENVS"], jnp.nan)
        done_mask = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        for step in range(100000000): 
            # =========================== #
            # Skip completed envs by masking observations
            # =========================== #
            masked_obsv = jax.tree_util.tree_map(
                lambda x: jnp.where(done_mask[:, None], jnp.zeros_like(x), x), 
                obsv
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
            action = jnp.where(done_mask, 0, action)  # or any no-op action

            # Update state
            _, value1, state = v_forward_jit(
                action, state, params,
                jnp.ones(config["NUM_ENVS"], dtype=jnp.int32)
            )

            # Step environments
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = v_env_step(
                rng_step, env_state, action, env_params
            )

            # Accumulate rewards only for active envs
            test_episode_reward += jnp.where(done_mask, 0, reward)

            # Identify envs that just finished this step
        
            newly_done = jnp.logical_and(done, jnp.logical_not(done_mask))

            # Log final return and PnL for newly finished envs
            def get_revenue_pnl(info, default=0.0):
                return info.get("total_revenue", default)

            extract_revenue_vmap = jax.vmap(get_revenue_pnl)
            final_revenue= extract_revenue_vmap(info)

            # Update logs where newly done
            episode_returns = jnp.where(newly_done, episode_reward, episode_returns)
            episode_revenues = jnp.where(newly_done, final_revenue, episode_revenues)

            # Update done mask
            done_mask = jnp.logical_or(done_mask, done)

            # Update observation
            obsv = obsv

            print(f"on step {step} of episode {episode} out of {episodes}")

            if done_mask.all():
                break   
        print(f"\n=== Episode {episode} Results ===")
        print("Returns:", episode_returns)
        print("PnLs:   ", episode_revenues)
    
        mean_return = jnp.nanmean(episode_returns)
        std_return = jnp.nanstd(episode_returns)

        mean_revenue = jnp.nanmean(episode_revenues)
        std_revenue = jnp.nanstd(episode_revenues)

        wandb.log({
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "mean_revenue": float(mean_revenue),
            "std_revenue": float(std_revenue),
            "episode": episode,
        })
        wandb.log({
            "returns_distribution": wandb.Histogram(episode_returns),
            "revenue_distribution": wandb.Histogram(episode_revenues),
            "episode": episode,
        })

        # Store them
        returns_mean_per_ep.append(mean_return)
        returns_std_per_ep.append(std_return)
        revenue_mean_per_ep.append(mean_revenue)
        revenue_std_per_ep.append(std_revenue)

        print(f"Mean Return: {mean_return:.2f}, Std Return: {std_return:.2f}")
        print(f"Mean PnL: {revenue_mean_per_ep:.2f}, Std PnL: {revenue_std_per_ep:.2f}")
        
        
            

  
            
            

        