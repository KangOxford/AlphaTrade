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

# ===== Define RNN models =====
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
    
    
    # ===== Initialize the model=====   
    # Execution model - use 8 actions as in training
    exe_action_dim = 8  # Based on training config
    network = ActorCriticRNN(exe_action_dim, config=config)
    # Initialize with consistent batch size 1 for evaluation
    init_hstate = ScannedRNN.initialize_carry(1, 128)
    
    # ===== Run evaluation episodes =====
    episodes = 1
    for episode in range(episodes):
        # Reset the environment
        obs,state=env.reset(key_reset, env_params)
        done=False
        
        
        # Episode tracking metrics
        episode_rewards = 0
        step_count = 0
        
        # Main evaluation loop
        while not done and step_count < 10000:  # Safety limit
            # Split the RNG key          
            # Get execution agent action
    
            # Create batch dimension of size 1  
            ac_in = (obs[jnp.newaxis, :], done[jnp.newaxis, :])
            
            init_hstate, pi, value = network.apply(params, init_hstate, ac_in)
            action = pi.sample(seed=rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                        value.squeeze(0),
                        action.squeeze(0),
                        log_prob.squeeze(0),
                    )
            
            # Step the environment
            rng, key_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step(key_step, state, action, env_params)
            
            # Update episode metrics
            
            episode_rewards += reward
            step_count += 1
            
            
            
            # Log results to wandb
            wandb.log({
                "step": step_count,
                "episode": episode,
                
                
                
                # Execution metrics
                "reward": reward["execution"],
                "total_revenue": info["total_revenue"],
                "quant_executed": info["quant_executed"],
                "average_price": info.get("average_price", 0),
                "task_to_execute": info["task_to_execute"],
                "is_sell_task": info["is_sell_task"],
                
                
            })
            
            print(f"Step {step_count},  EXE reward: {reward['execution']}")
            
            if done["__all__"]:
                break
        
        # Episode summary
        print(f"Episode {episode} completed in {step_count} steps")
        
        wandb.log({
            "episode_complete": True,
            "episode_steps": step_count,
            "episode_exe_reward": episode_rewards
        })
    
    # Close wandb
    wandb.finish()
