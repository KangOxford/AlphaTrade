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
wandb.init(project="AlphaTrade_EXEC_RNN_Eval", config={"run_type": "evaluation"})


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


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/val"


    # Initialize wandb
    wandb.init(project="AlphaTrade_EXEC_Eval", config={"run_type": "EXEC_evaluation"})

    # ===== Load trained model parameters =====
    params_filename = "/home/duser/AlphaTrade/gymnax_exchange/jaxrl/Out_of_Sample_Evaluation/params_exec/params_file_quiet-sweep-12_03-31_16-51"  # Execution model
    
    print(f"Loading EXE params from: {params_filename}")
    with open(params_filename, 'rb') as f:
        params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

    # ===== Environment configuration =====
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 5,
        "REWARD_LAMBDA": 1.0,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 50, # 60 seconds
        "trader_unique_id": 10,
    }
    
    # ===== Setup RNG keys =====
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # ===== Initialize Exec environment =====

    
    env_config = EnvironmentExecutionConfig(
        task="random",
        action_type="pure",
        end_fn="unwind_FT",
        max_task_size=50,
        n_actions=8,  #  use 8 as in training config
        action_space="fixed_quants"
    )

    
    env = ExecutionEnv(
        cfg = env_config,
        key = key_reset,
        alphatradePath = config["ATFOLDER"],
        window_index = config["WINDOW_INDEX"],
        episode_time = config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        trader_unique_id=config["trader_unique_id"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=1,
    )
    

    # Initialize the environment state
    start = time.time()
    obs, env_state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {env_state.start_index}")
    print("Time for reset: \n", time.time() - start)
    print("Inventory after reset: \n", env_state.inventory)
    print(f"Number of available windows: {env.n_windows}")


    
    #===========================================#
    #Init the pre trained model
    #======================================#
    # Load the trained model parameters 
    # Initialize the model

    network = ActorCriticRNN(env.action_space(env_params).n, config=config)
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

    reset_rng = jax.random.split(key_reset, config["NUM_ENVS"])


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
            rng, _rng = jax.random.split(rng)
            #test action
            ac_in = (masked_obsv[jnp.newaxis, :], done_mask[jnp.newaxis, :])
            init_hstate, pi, value = network.apply(params, init_hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                        value.squeeze(0),
                        action.squeeze(0),
                        log_prob.squeeze(0),
                    )
            
            # Mask action for done envs
            action = jnp.where(done_mask, 0, action)  # or any no-op action
            
            # Take a step in the environment
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
        
            # Accumulate rewards only for active envs
            episode_reward += jnp.where(done_mask, 0, reward)

           # Identify envs that just finished this step
            newly_done = jnp.logical_and(done, jnp.logical_not(done_mask))

            # Log final return and PnL for newly finished envs
            def get_revenue_pnl(info, default=0.0):
                return info.get("total_revenue", default)

            extract_revenue_vmap = jax.vmap(get_revenue_pnl)
            final_revenue= extract_revenue_vmap(info)

            # Update logs where newly done
            episode_returns = jnp.where(newly_done, episode_reward, episode_returns)
            episode_revenue = jnp.where(newly_done, final_revenue, episode_revenues)

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
        
            

  
            
            

        