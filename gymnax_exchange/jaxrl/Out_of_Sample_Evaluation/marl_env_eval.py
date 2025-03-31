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
from gymnax_exchange.jaxen.marl_env import MARLEnv
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

from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig, EnvironmentExecutionConfig
from flax.core import frozen_dict
from flax import serialization

import wandb  # Import Weights & Biases for logging

# ===== Define RNN models for both agents =====
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
        
        # Make sure the reset behavior preserves the original batch size
        # This is critical to avoid shape mismatches
        batch_size = rnn_state.shape[0]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(batch_size, ins.shape[-1]),  # Use same batch size as input
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

# ============ Main function ==============
if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/test"
        print("Using default folder:", ATFolder)

    # Initialize wandb
    wandb.init(project="AlphaTrade_MARL_Eval", config={"run_type": "marl_evaluation"})

    # ===== Load trained model parameters =====
    mm_params_filename = "/home/duser/AlphaTrade/gymnax_exchange/jaxrl/Out_of_Sample_Evaluation/params_mm/params_file_icy-sweep-5_03-31_17-37"  # MM model
    exe_params_filename = "/home/duser/AlphaTrade/gymnax_exchange/jaxrl/Out_of_Sample_Evaluation/params_exec/params_file_quiet-sweep-12_03-31_16-51"  # Execution model
    
    
    print(f"Loading MM params from: {mm_params_filename}")
    with open(mm_params_filename, 'rb') as f:
        mm_params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())
    
    print(f"Loading EXE params from: {exe_params_filename}")
    with open(exe_params_filename, 'rb') as f:
        exe_params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

    # ===== Environment configuration =====
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": -1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*10,  # 10 minutes
        "NUM_ENVS": 128,  # Use a larger batch size to match training
        "MM_TRADER_ID": -4999991,
        "EXE_TRADER_ID": -9999992,
        "EXE_REWARD_LAMBDA": 1.0,
    }
    
    # ===== Setup RNG keys =====
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # ===== Initialize MARL environment =====
    # Create configs matching the ones used in training
    mm_config = EnvironmentConfig(
        observation_space="engineered",
        reward_space="portfolio_value",
        inv_penalty="linear",
        n_actions=8,
        end_fn="unwind_ref_price",
        fixed_quant_value=10,
        reference_price_portfolio_value="best_bid_ask",
        action_space="spread_skew"
    )
    
    exe_config = EnvironmentExecutionConfig(
        task="random",
        action_type="pure",
        end_fn="unwind_FT",
        max_task_size=50,
        n_actions=8,  #  use 8 as in training config
        action_space="fixed_quants"
    )
    
    env = MARLEnv(
        key=key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        mm_trader_id=config["MM_TRADER_ID"],
        exe_trader_id=config["EXE_TRADER_ID"],
        exe_reward_lambda=config["EXE_REWARD_LAMBDA"]
    )
    
    # Override the environment's configs to match training
    env.mm_env.cfg = mm_config
    env.exe_env.cfg = exe_config
    
    # Get the default environment parameters
    env_params = env.default_params
    env_params = dataclasses.replace(env_params, episode_time=config["EPISODE_TIME"])
    
    # ===== Initialize the models =====
    # MM model -  action space to 6 for spread_skew
    mm_action_dim = 6  # spread_skew has 6 actions (2 spreads × 3 skews)
    mm_network = ActorCriticRNN(mm_action_dim, config=config)
    # Initialize with consistent batch size 1 for evaluation
    mm_init_hstate = ScannedRNN.initialize_carry(1, 128)
    
    # Execution model - use 8 actions as in training
    exe_action_dim = 8  # Based on training config
    exe_network = ActorCriticRNN(exe_action_dim, config=config)
    # Initialize with consistent batch size 1 for evaluation
    exe_init_hstate = ScannedRNN.initialize_carry(1, 128)
    
    # ===== Run evaluation episodes =====
    episodes = 1
    for episode in range(episodes):
        # Reset the environment
        obs, state = env.reset(key_reset, env_params)
        done = {"__all__": False}
        
        # Episode tracking metrics
        episode_rewards = {"market_maker": 0, "execution": 0}
        step_count = 0
        
        # Main evaluation loop
        while not done["__all__"] and step_count < 10000:  # Safety limit
            # Split the RNG key
            rng, _rng = jax.random.split(rng)
            key_mm, key_exe = jax.random.split(_rng)
            
            # Get market maker action
            mm_obs = obs["market_maker"]
            mm_done = jnp.array([done.get("market_maker", False)])
            # Create batch dimension of size 1
            mm_ac_in = (mm_obs[jnp.newaxis, :], mm_done[jnp.newaxis, :])
            
            # Apply the model and get action
            mm_init_hstate, mm_pi, mm_value = mm_network.apply(mm_params, mm_init_hstate, mm_ac_in)
            # For spread_skew, we need to sample a scalar action value from the distribution (0-5)
            mm_action = mm_pi.sample(seed=key_mm)
            # Remove the batch dimension and ensure it's a scalar integer
            mm_action = jnp.asarray(mm_action.squeeze(), dtype=jnp.int32)
            
            # Get execution agent action
            exe_obs = obs["execution"]
            exe_done = jnp.array([done.get("execution", False)])
            # Create batch dimension of size 1  
            exe_ac_in = (exe_obs[jnp.newaxis, :], exe_done[jnp.newaxis, :])
            
            # Apply the model and get action
            exe_init_hstate, exe_pi, exe_value = exe_network.apply(exe_params, exe_init_hstate, exe_ac_in)
            exe_action = exe_pi.sample(seed=key_exe)
            # Remove the batch dimension and ensure it's a scalar integer
            exe_action = jnp.asarray(exe_action.squeeze(), dtype=jnp.int32)
            
            # Combine actions for both agents
            actions = {"market_maker": mm_action, "execution": exe_action}
            
            # Step the environment
            rng, key_step = jax.random.split(rng)
            obs, state, rewards, done, info = env.step(key_step, state, actions, env_params)
            
            # Update episode metrics
            episode_rewards["market_maker"] += rewards["market_maker"]
            episode_rewards["execution"] += rewards["execution"]
            step_count += 1
            
            # Extract key metrics for logging
            mm_info = info["market_maker"]
            exe_info = info["execution"]
            
            # Log results to wandb
            wandb.log({
                "step": step_count,
                "episode": episode,
                
                # Market maker metrics
                "mm_reward": rewards["market_maker"],
                "mm_total_pnl": mm_info["total_PnL"],
                "mm_inventory": mm_info["inventory"],
                "mm_netWorth": mm_info["netWorth"],
                "mm_buyQuant": mm_info["buyQuant"],
                "mm_sellQuant": mm_info["sellQuant"],
                
                # Execution metrics
                "exe_reward": rewards["execution"],
                "exe_total_revenue": exe_info["total_revenue"],
                "exe_quant_executed": exe_info["quant_executed"],
                "exe_average_price": exe_info.get("average_price", 0),
                "exe_task_to_execute": exe_info["task_to_execute"],
                "exe_is_sell_task": exe_info["is_sell_task"],
                
                # Environment metrics
                "done": done["__all__"],
                "mm_done": done["market_maker"],
                "exe_done": done["execution"],
            })
            
            print(f"Step {step_count}, MM reward: {rewards['market_maker']}, EXE reward: {rewards['execution']}")
            
            if done["__all__"]:
                break
        
        # Episode summary
        print(f"Episode {episode} completed in {step_count} steps")
        print(f"Market Maker total reward: {episode_rewards['market_maker']}")
        print(f"Execution total reward: {episode_rewards['execution']}")
        
        wandb.log({
            "episode_complete": True,
            "episode_steps": step_count,
            "episode_mm_reward": episode_rewards["market_maker"],
            "episode_exe_reward": episode_rewards["execution"]
        })
    
    # Close wandb
    wandb.finish()
