
import os, sys, time, dataclasses
from typing import Tuple, Optional, Dict
import sys
import time
import dataclasses
import jax
from jax import vmap
import jax.numpy as jnp
import chex
from flax import struct
import jax.tree_util as jtu
from functools import partial
from typing import Any
from gymnax_exchange.utils import utils as util
import pandas as pd
#from typing import List, Tuple

# for debugging
jax.config.update('jax_disable_jit', False)
jax.config.update("jax_traceback_in_locations_limit", -1)
jax.config.update("jax_log_compiles", False)

from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.from_JAXMARL.multi_agent_env import MultiAgentEnv
#from gymnax_exchange.jaxen.from_JAXMARL.spaces import Box, MultiDiscrete, Discrete

from gymnax_exchange.jaxen.StatesandParams import MultiAgentState, MultiAgentParams, LoadedEnvParams, LoadedEnvState, WorldState


from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import MarketMaking_EnvironmentConfig
from gymnax_exchange.jaxob.jaxob_config import Execution_EnvironmentConfig
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig
from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig

from gymnax_exchange.jaxen.marl_env import MARLEnv

# ----------------------------------------------
# New VMAP rollout script + timing statistics
# ----------------------------------------------

def main():

    output_file_path = "/home/myuser/gymnax_exchange/jaxen/Timing_speed/vmap_timing_results.txt"
    #with open(output_file_path, "w") as f:
    #    f.write(f"Running with {10} envs and {10} steps\n")

    rng = jax.random.PRNGKey(30) # TODO i think this should be changed to the new key function in JAX .key()
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)


    agent_type_options = [
        [1, 1],
        [5, 5],
        [10, 10],
    ]
    n_data_msg_options = [1, 10, 50, 100]
    num_envs_options = [1000, 3000, 5000, 8000, 10000]
    num_steps_options = [1000, 5000]

    results = []

    with open(output_file_path, "w") as f: 
        for number_of_agents_per_type in agent_type_options:
            for n_data_msg_per_step in n_data_msg_options:
                for num_envs in num_envs_options:
                    for num_steps in num_steps_options:
                        
                            # print(f"Running with {num_envs} envs and {num_steps} steps")

                            # print("\n" + "="*60)
                            # print("Starting VMAP timing test loop for MARL")
                            # print("="*60)
                            # Create a new world config for this run
                            world_config = World_EnvironmentConfig(
                                n_data_msg_per_step=n_data_msg_per_step,
                            )

                            list_of_agents_configs = [
                                MarketMaking_EnvironmentConfig(),
                                Execution_EnvironmentConfig(),
                            ]
                            # Create a new MultiAgentConfig for this run
                            multi_agent_config = MultiAgentConfig()
                            # Set the fields (dataclasses are frozen, so use object.__setattr__)
                            object.__setattr__(multi_agent_config, 'world_config', world_config)
                            object.__setattr__(multi_agent_config, 'list_of_agents_configs', list_of_agents_configs)
                            object.__setattr__(multi_agent_config, 'number_of_agents_per_type', number_of_agents_per_type)

                            # Re-instantiate the environment and params for this config
                            env = MARLEnv(key=key_reset, multi_agent_config=multi_agent_config)
                            env_params = env.default_params

                            # Write config info to file
                            config_str = (
                                f"\n{'#'*20}\n"
                                f"Agents per type: {number_of_agents_per_type}\n"
                                f"n_data_msg_per_step: {n_data_msg_per_step}\n"
                                f"{'#'*20}\n"
                            )
                            print(config_str)
                            f.write(config_str)
                            f.flush()

                            status_str = (
                                f"Running with {num_envs} envs and {num_steps} steps\n"
                                + "="*60 + "\n"
                                + "Starting VMAP timing test loop for MARL\n"
                                + "="*60 + "\n"
                            )
                            print(status_str)
                            f.write(status_str)
                            f.flush()


                            NUM_ENVS   = num_envs         # number of parallel environments
                            NUM_STEPS  = num_steps     # total steps per environment
                            MASTER_KEY = jax.random.PRNGKey(0)

                            # -------------------------------------------------
                            # 1) Initial reset of all envs (batched)
                            # -------------------------------------------------
                            master_key, *reset_keys = jax.random.split(MASTER_KEY, NUM_ENVS + 1)
                            batched_reset = jax.vmap(env.reset_env, in_axes=(0, None))

                            reset_start = time.time()
                            obs, state  = batched_reset(jnp.stack(reset_keys), env_params)
                            # force execution to finish before timing
                            jax.block_until_ready(state)
                            reset_time  = time.time() - reset_start

                            # -------------------------------------------------
                            # 2) Helper: one step for a single env
                            # -------------------------------------------------
                            def single_step(state, key, env_params):
                                # one sub-key per agent type
                                subkeys = jax.random.split(key, len(env.action_spaces))
                                # sample random actions for every agent of each type
                                actions = [
                                    jax.vmap(space.sample)(
                                        jax.random.split(sk, n_agents)
                                    )
                                    for sk, space, n_agents in zip(
                                        subkeys,
                                        env.action_spaces,
                                        env.multi_agent_config.number_of_agents_per_type,
                                    )
                                ]
                                # env.step auto-resets when done
                                return env.step(key, state, actions, env_params)

                            # JIT & vmap
                            @jax.jit
                            def batched_step(state_batch, key_batch):
                                return jax.vmap(single_step, in_axes=(0, 0, None))(
                                    state_batch, key_batch, env_params
                                )

                            # -------------------------------------------------
                            # 3) Scan across a fixed number of steps
                            # -------------------------------------------------
                            def scan_body(carry, _):
                                state_batch, rng = carry
                                rng, *step_keys = jax.random.split(rng, NUM_ENVS + 1)
                                obs, state_batch, rew, done, _ = batched_step(
                                    state_batch, jnp.stack(step_keys)
                                )
                                return (state_batch, rng), (obs, rew, done)

                            rollout_start = time.time()
                            (final_state, _), (traj_obs, traj_rew, traj_done) = jax.lax.scan(
                                scan_body,
                                (state, master_key),
                                None,
                                length=NUM_STEPS,
                            )
                            # ensure all work is finished
                            jax.block_until_ready(final_state)
                            rollout_time = time.time() - rollout_start

                            # -------------------------------------------------
                            # 4) Timing statistics
                            # -------------------------------------------------
                            total_steps       = NUM_STEPS * NUM_ENVS          # every env took NUM_STEPS steps
                            avg_steps_per_env = NUM_STEPS
                            avg_time_per_step = rollout_time / total_steps
                            avg_steps_per_sec = total_steps / rollout_time

                            result_str = (
                            f"\n[4] Timing Results\n"
                            f"{'-' * 60}\n"
                            f"Total Envs:           {NUM_ENVS}\n"
                            f"Reset time:           {reset_time:.4f} seconds\n"
                            f"Rollout (steps) time: {rollout_time:.4f} seconds\n"
                            f"Total steps:          {total_steps}\n"
                            f"Avg steps per env:    {avg_steps_per_env:.2f}\n"
                            f"Avg time per step:    {avg_time_per_step:.6f} seconds\n"
                            f"Avg steps per sec:    {avg_steps_per_sec:.2f}\n"
                            f"{'=' * 60}\n"
                            )

                            print(result_str)
                            f.write(result_str)
                            f.flush()


                            results.append({
                                "num_envs": NUM_ENVS,
                                "num_steps": NUM_STEPS,
                                "agents_per_type": number_of_agents_per_type,
                                "n_data_msg_per_step": n_data_msg_per_step,
                                "reset_time": reset_time,
                                "rollout_time": rollout_time,
                                "total_steps": total_steps,
                                "avg_steps_per_env": avg_steps_per_env,
                                "avg_time_per_step": avg_time_per_step,
                                "avg_steps_per_sec": avg_steps_per_sec,
                            })

                            # print("\n[4] Timing Results")
                            # print("-" * 60)
                            # print(f"Total Envs:           {NUM_ENVS}")
                            # print(f"Reset time:           {reset_time:.4f} seconds")
                            # print(f"Rollout (steps) time: {rollout_time:.4f} seconds")
                            # print(f"Total steps:          {total_steps}")
                            # print(f"Avg steps per env:    {avg_steps_per_env:.2f}")
                            # print(f"Avg time per step:    {avg_time_per_step:.6f} seconds")
                            # print(f"Avg steps per sec:    {avg_steps_per_sec:.2f}")
                            # print("=" * 60)

    df = pd.DataFrame(results)
    df.to_csv("/home/myuser/gymnax_exchange/jaxen/Timing_speed/timing_results.csv", index=False)


if __name__ == "__main__":
    main()