"""
Based on PureJaxRL Implementation of PPO
"""

import os

from git import Union
from humanize import metric
import pandas as pd
import csv

from sympy import plot
from torch import le

from docs.source import conf
# from lobgen.tgci.tgci import train
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import time
import jax # type: ignorepip 
jax.config.update('jax_disable_jit', False)

import jax.numpy as jnp # type: ignore
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal # type: ignore
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import distrax
import orbax.checkpoint as oxcp
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

#from jaxmarl.wrappers.baselines import SMAXLogWrapper
#from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig,MarketMaking_EnvironmentConfig

import wandb


import functools
import matplotlib.pyplot as plt

import sys
import os
import pickle
from datetime import datetime

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
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        # obs, dones, avail_actions = x
        obs, dones = x

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)

        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )

        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0) # type: ignore
        )(actor_mean)
        # Avail actions are not used in the current implementation, but can be added if needed.
        # unavail_actions = 1 - avail_actions
        action_logits = actor_mean # - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class RandomPolicy(nn.Module):
    action_dim: Sequence[int]
    @nn.compact
    def __call__(self, hidden, x):
        obs,done= x
        pi = distrax.Categorical(probs=jnp.ones((obs.shape[1], self.action_dim)) / self.action_dim, dtype=jnp.int32)
        critic=np.array(0)
        return hidden, pi, critic

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jnp.zeros((batch_size,hidden_size), dtype=jnp.float32)

class FixedAction(nn.Module):
    action_dim: Sequence[int]
    action: Sequence[int]  # Default action to return
    """A fixed action policy that always returns the same action."""
    @nn.compact
    def __call__(self, hidden, x):
        obs,done= x
        probs = jnp.zeros((obs.shape[1], self.action_dim), dtype=jnp.float32)
        for i in self.action:
            probs = probs.at[:, i].set(1.0)
        pi = distrax.Categorical(probs=probs)

        critic=np.array(0)
        return hidden, pi, critic

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jnp.zeros((batch_size,hidden_size), dtype=jnp.float32)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # avail_actions: jnp.ndarray


def batchify(x: jnp.ndarray, num_actors):
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray,num_envs, num_agents):
    return  x.reshape((num_envs, num_agents, -1))


def make_sim(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
    init_key = jax.random.PRNGKey(config["SEED"])
    print("init_key: ", init_key)




    env : MARLEnv = MARLEnv(key=init_key,
                            multi_agent_config=MultiAgentConfig(
                                    number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
                                    list_of_agents_configs=
                                        [MarketMaking_EnvironmentConfig(), Execution_EnvironmentConfig()], 
                                    world_config=World_EnvironmentConfig(timePeriod=config["EvalTimePeriod"])
                            ))

    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]


    # config["CLIP_EPS"] = (
    #     config["CLIP_EPS"] / env.num_agents
    #     if config["SCALE_CLIP_EPS"]
    #     else config["CLIP_EPS"]
    # )

    print("Config:")
    for k, v in config.items():
        print(f"{k}: {v}")
    # env = SMAXLogWrapper(env)

    def linear_schedule(lr,count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    def run(rng):
        # IF APPLICABLE, LOAD NETWORK FROM CHECKPOINT OR FILE HERE
        def load_network_from_checkpoint(config,env,rng: jax.random.PRNGKey,step=None):
            hstates = []
            network_params_list = []
            train_states = []
            num_agents_of_instance_list = []
            init_dones_agents = []
            for i, instance in enumerate(env.instance_list):
                # print("Action space dimension for network i ",env.action_spaces[i].n)
                network = ActorCriticRNN(env.action_spaces[i].n, config=config)
                rng, _rng = jax.random.split(rng)
                init_x = (
                    jnp.zeros(
                        (1, config["NUM_ENVS"], env.observation_spaces[i].shape[0])
                    ), # obs
                    jnp.zeros((1, config["NUM_ENVS"])), # dones
                    # jnp.zeros((1, config["NUM_ENVS"], env.action_spaces[i].n)), #     avail_actions
                )

                init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
                network_params = network.init(_rng, init_hstate, init_x)
                if config["ANNEAL_LR"]:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=functools.partial(linear_schedule,config["LR"]), eps=1e-5),
                    )
                else:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(config["LR"], eps=1e-5),
                    )
                train_state = TrainState.create(
                    apply_fn=network.apply,
                    params=network_params,
                    tx=tx,
                )
                init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"])

                # Instead of appending dicts, maintain separate lists for each attribute
                hstates.append(init_hstate)
                network_params_list.append(network_params)
                train_states.append(train_state)
                num_agents_of_instance_list.append(env.multi_agent_config.number_of_agents_per_type[i])
                init_dones_agents.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))

            target_ckpt= {
                'model': train_states,  # train_states
                # 'config': {} ,
                'metrics': {
                    'train_rewards': [np.nan,np.nan],
                    'eval_rewards': [np.nan,np.nan],
                    }
            }
            orbax_checkpointer = oxcp.PyTreeCheckpointer()
            checkpoint_manager = oxcp.CheckpointManager(
             f'/home/myuser/data/checkpoints/MARLCheckpoints/{config["RESTORE_PROJECT"]}/{config["RESTORE_RUN"]}', orbax_checkpointer
                )
            if step is None:
                step=checkpoint_manager.latest_step()

            restored_state = checkpoint_manager.restore(
                step,
                items=target_ckpt,
                restore_kwargs={'restore_args': orbax_utils.restore_args_from_target(target_ckpt)}
            )


            restored_train_states = restored_state['model']

            return hstates, restored_train_states, init_dones_agents

        # BASELINE POLICY ONLY
        def init_baseline_policies(config: Dict, env: MARLEnv,rng :jax.random.PRNGKey) -> tuple[list, list, list]:
            hstates: list[jnp.ndarray] = []
            network_params_list: list = []
            train_states: list[TrainState] = []
            init_dones_agents: list[jnp.ndarray] = []
            for i, instance in enumerate(env.instance_list):
                # print("Action space dimension for network i ",env.action_spaces[i].n)
                network = FixedAction(env.action_spaces[i].n, config["FIXED_ACTIONS"][i])
                # network = RandomPolicy(env.action_spaces[i].n)
                rng, _rng = jax.random.split(rng)
                init_x = (
                    jnp.zeros(
                    (1, config["NUM_ENVS"], env.observation_spaces[i].shape[0])
                    ), # obs
                    jnp.zeros((1, config["NUM_ENVS"])), # dones
                    # jnp.zeros((1, config["NUM_ENVS"], env.action_spaces[i].n)), #     avail_actions
                )

                init_hstate = FixedAction.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], 1)
                network_params = network.init(_rng, init_hstate, init_x)
                print("Params", network_params)
                train_state = TrainState.create(
                    apply_fn=network.apply,
                    params=network_params,
                    tx=optax.adam(1000000),
                )
                init_hstate = FixedAction.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], 1)
                # Instead of appending dicts, maintain separate lists for each attribute
                hstates.append(init_hstate)
                network_params_list.append(network_params)
                train_states.append(train_state)
                init_dones_agents.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))
            return hstates, train_states, init_dones_agents

        bl_init_hiddens , bl_train_states, bl_init_dones_agents = init_baseline_policies(config, env,rng)
        lrn_hstates, lrn_train_states, lrn_init_dones_agents = load_network_from_checkpoint(config, env,rng)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_params=env.default_params

        # env_params=jax.device_put(env_params)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng,env_params)
        # TRAIN LOOP

        # env_params.loaded_params.message_data
        # window_index=38847
        # s=env.base_env.start_indeces[window_index]
        # e=env.base_env.end_indeces[window_index]
        # print("Start and end indices for window:", s, e)
        # print(env_params.loaded_params.message_data[s:e].shape)
        # print(env_params.loaded_params.message_data[s:s+10])
        # print(env_params.loaded_params.message_data[e-10:e])
        # theoretical_end_time = env_params.loaded_params.message_data[s, -2] + env.multi_agent_config.world_config.episode_time
        # print("Datamessages outside",env.base_env._get_data_messages(
        #                                     env_params.loaded_params.message_data,
        #                                     s,
        #                                     0,
        #                                     theoretical_end_time
        # ))


        def callback(metric, combo_desc=None):
            action_distribution = {}
            episodes_complete =[]
            for i, tr in enumerate(metric["traj_batch"]):
                actions = np.array(tr.action).flatten()
                unique_actions, counts = np.unique(actions, return_counts=True)
                tot_counts=sum(counts)
                # Add each action count to the dictionary with a unique key
                for a, c in zip(unique_actions, counts):
                    action_distribution[f"action_{i}_{int(a)}"] = c/tot_counts*100
                episodes_complete.append(tr.global_done.sum())
            print(f"Completed Episodes from global dones: {episodes_complete}")


            logging_dict = {
                    # TODO: Log the quantities of interest. Keep it trivial for now.
                    "env_step": metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    **{f"avg_reward_{i}": metric["avg_reward"][i] for i in range(len(metric["avg_reward"]))},
                    **action_distribution
                }
            if config["CALC_EVAL"]:
                logging_dict.update({
                    **{f"avg_eval_reward_{i}": metric["avg_reward_eval"][i] for i in range(len(metric["avg_reward_eval"]))},
                })
            if config["WANDB_MODE"]!= "disabled":
                wandb.log(logging_dict)

            for i in range(len(metric["avg_reward"])):
                print(f"avg_reward_{i} {metric["avg_reward"][i]}")

            print(f"Completed Episodes: {metric['total_dones']}")

            # Save trajectory batch to a pickle file

            # Create trajectories directory if it doesn't exist
            os.makedirs("trajectories", exist_ok=True)

            # Create an informative filename with timestamp and action type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if combo_desc is None:
                combo_desc = "default"
            filename = f"trajectories/traj_batch_{combo_desc}__{timestamp}.pkl"

            # Save the trajectory batch
            with open(filename, "wb") as f:
                pickle.dump(metric["traj_batch"], f)

            print(f"Saved trajectory batch to {filename}")

            # if config["TINY_RUN"]:
            #     plot_episode_features(metric["traj_batch"])

        def _update_step(update_runner_state,env_params):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done,h_states, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Ignore getting the available actions for now, assume all actions are available.
                # avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                # avail_actions = jax.lax.stop_gradient(
                #     batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                # )
                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actions=[]
                values=[]
                log_probs=[]

                for i, train_state in enumerate(train_states):
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i])  # Reshape to match the input shape of the network
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    h_states[i], pi, value = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                    values.append(value)
                    action = pi.sample(seed=_rng)
                    # jax.debug.print(f"Pi: {pi._probs}")
                    log_probs.append(pi.log_prob(action))
                    action=unbatchify(action, config["NUM_ACTORS_PERTYPE"][i], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                    actions.append(action.squeeze())
                    print(actions)
                    # env_act = unbatchify(
                    #     action, env.agents, config["NUM_ENVS"], env.num_agents
                    # )
                    # env_act = {k: v.squeeze() for k, v in env_act.items()}
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0,None)
                )(rng_step, env_state, actions,env_params)

                # info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                
                done_batch=done
                transitions=[]
                for i, train_state in enumerate(train_states):
                    done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]).squeeze()
                    obs_batch = batchify(obsv[i],config["NUM_ACTORS_PERTYPE"][i])
                    action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i])
                    value = values[i]
                    log_prob = log_probs[i]

                    info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i],-1).squeeze(),info["agents"][i])}
                    # print(f"info for agenttype {i}:", info_i)


                    transitions.append(Transition(
                        jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                        last_done[i],
                        action_batch.squeeze(),
                        value.squeeze(),
                        batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                        log_prob.squeeze(),
                        obs_batch,
                        info_i,
                        # avail_actions,
                    ))
                runner_state = (train_states, env_state, obsv, done_batch['agents'], h_states, rng)
                return runner_state, transitions

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_states, env_state, last_obs, last_dones, hstates_new, rng = runner_state
            total_dones = []
            for tr in traj_batch:
                total_dones.append(jax.tree.map(lambda x: x.sum(), tr.done))


            metrics= {}
            metrics['agents'] = [jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS_PER_TYPE"][i],-1)
                ).squeeze(),
                trjbtch.info['agent']) for i, trjbtch in enumerate(traj_batch)]
            metrics['world'] = [traj_batch.info['world'] for i, traj_batch in enumerate(traj_batch)]

            metrics['avg_reward'] = [jnp.mean(tr.reward) for tr in traj_batch]
            metrics["traj_batch"] = traj_batch
            metrics["total_dones"] = total_dones   



           
            metrics["update_steps"] = update_steps
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_dones, hstates_new, rng)

            # jax.profiler.save_device_memory_profile(f"memory_{update_steps}.prof")
            return (runner_state, update_steps), metrics


        jitted_update_step = jax.jit(_update_step)
        
        def eval_policies(rng,env_params, config):
            """
            Run evaluation with different policy combinations:
            - Learned vs. Baseline
            - Baseline vs. Learned
            - Baseline vs. Baseline
            - Learned vs. Learned
            
            Generalizes to n agents per type.
            """

            n_agent_types = len(config["NUM_AGENTS_PER_TYPE"])
            
            # All possible policy combinations
            policy_combinations = []
            
            # For n agent types, we have 2^n possible combinations (each type can be either learned or baseline)
            for i in range(2**n_agent_types):
                # Convert i to binary, padded to n_agent_types digits
                # '1' means learned policy, '0' means baseline policy
                binary = format(i, f'0{n_agent_types}b')
                policy_choices = [int(bit) for bit in binary]
                policy_combinations.append(policy_choices)
            
            results = {}
            
            for combo_idx, policy_choice in enumerate(policy_combinations):
                # Create a description of this combination (e.g., "L-B" for Learned-Baseline)
                combo_desc = ''.join(['L' if choice == 1 else 'B' for choice in policy_choice])
                print(f"\nEvaluating policy combination {combo_idx+1}/{len(policy_combinations)}: {combo_desc}")
                
                # Reset environment
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
                
                # Initialize hidden states and dones
                hstates_eval = []
                dones_eval = []
                
                # For each agent type, choose either learned or baseline policy
                train_states_eval = []
                
                for i in range(n_agent_types):
                    if policy_choice[i] == 1:  # Use learned policy
                        hstates_eval.append(lrn_hstates[i])
                        train_states_eval.append(lrn_train_states[i])
                        dones_eval.append(lrn_init_dones_agents[i])
                    else:  # Use baseline policy
                        hstates_eval.append(bl_init_hiddens[i])
                        train_states_eval.append(bl_train_states[i])
                        dones_eval.append(bl_init_dones_agents[i])
                
                # Run evaluation
                eval_runner_state = (
                    train_states_eval,
                    env_state,
                    obsv,
                    dones_eval,
                    hstates_eval,
                    rng,
                )
                
                (eval_runner_state, _), eval_metrics = jitted_update_step((eval_runner_state, 0), env_params)
                callback(eval_metrics, combo_desc)

                # Store results
                results[combo_desc] = {
                    'avg_reward': eval_metrics['avg_reward'],
                    'total_dones': eval_metrics['total_dones'],
                    'traj_batch': eval_metrics['traj_batch']
                }
                
                print(f"Results for {combo_desc}:")
                for i in range(n_agent_types):
                    agent_type = 'L' if policy_choice[i] == 1 else 'B'
                    print(f"  {agent_type} (Agent type {i}): avg_reward = {eval_metrics['avg_reward'][i]:.4f}")
                # callback(eval_metrics)
                del eval_metrics
                gc.collect()

            
            return results

        # Run all policy combinations
        print("Running evaluations with all possible policy combinations...")
        eval_results = eval_policies(rng, env_params, config)

        # # Define which combination is the main training combination - this will be used for callbacks
        # train_states = runner_state[0]  # Get current train states

        # rng, _rng = jax.random.split(rng)
        # runner_state = (
        #     train_states,
        #     env_state,
        #     obsv,
        #     init_dones_agents, # last_done
        #     hstates,  # initial hidden states for RNN
        #     _rng,
        # )
        # updates=0
        # for i in range(1):
        #     print(f"Update step {i+1}/{100}")
        #     # Run the update step:
        #     if i>2 and i<4:
        #         jax.profiler.start_trace("/tmp/profile-data")
        #     (runner_state,updates),metrics=jitted_update_step((runner_state,updates),env_params)
        #     if i>2 and i<4:
        #         jax.block_until_ready((runner_state,updates,metrics))
        #         jax.profiler.stop_trace()

            # env_params.loaded_params.message_data
            # window_index=metrics["traj_batch"][0].info['world']['window_index'][0,0]
            # print(f"Window index: {window_index}")
            # s=env.base_env.start_indeces[window_index]
            # e=env.base_env.end_indeces[window_index]
            # print("Start and end indices for window:", s, e)
            # print(env_params.loaded_params.message_data[s:e].shape)
            # print(env_params.loaded_params.message_data[s:s+10])
            # print(env_params.loaded_params.message_data[e-10:e])
            # theoretical_end_time = env_params.loaded_params.message_data[s, -2] + env.multi_agent_config.world_config.episode_time
            # print("Datamessages outside",env.base_env._get_data_messages(
            #                                     env_params.loaded_params.message_data,
            #                                     s,
            #                                     0,
            #                                     theoretical_end_time
            # ))
            # callback(metrics)
            # del metrics
            # gc.collect()
        


        # runner_state, metrics = jax.lax.scan(
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )
        
        
        return {"results": eval_results}

    return run


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_JAXMARL_2player")
def main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)


    print(config)

    def sweep_fun():
        print(f"WANDB CONFIG PRIOR {wandb.config}")


        run=wandb.init(
            entity=config["ENTITY"], # type: ignore
            project=config["PROJECT"], # type: ignore
            tags=["IPPO", "RNN"], # type: ignore
            config=config, # type: ignore
            mode=config["WANDB_MODE"], # type: ignore
            allow_val_change=True,
        )
        # params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        
        
        # print(f"WANDB CONFIG {wandb.config}")
        # +++++ Single GPU +++++
        

        rng = jax.random.PRNGKey(0)

        print("wandb.config", wandb.config)

        if config["Timing"]:
            start_time = time.time()


        train_fun = make_train(config)
        out = train_fun(rng)
        # train_state = out['runner_state'][0] # runner_state.train_state
        # params = train_state.params

        if config["Timing"]:
            end_time = time.time()
            elapsed = end_time - start_time
            total_steps = config["TOTAL_TIMESTEPS"]
            agents_per_type = config["NUM_AGENTS_PER_TYPE"]
            num_data_msgs = config.get("n_data_msg_per_step", None)
            num_envs = config["NUM_ENVS"]

            # Print results
            print(f"Total steps: {total_steps}")
            print(f"Elapsed time: {elapsed} seconds")
            print(f"Steps per second: {total_steps / elapsed}")
            print(f"Agents per type: {agents_per_type}")
            print(f"Num data messages: {num_data_msgs}")
            print(f"Num envs: {num_envs}")

            # Save to CSV
            results = {
                "total_steps": [total_steps],
                "elapsed_seconds": [elapsed],
                "steps_per_second": [total_steps / elapsed],
                "agents_per_type": [str(agents_per_type)],
                "num_data_msgs": [num_data_msgs],
                "num_envs": [num_envs],
            }
            # df = pd.DataFrame(results)
            # csv_path = "timing_results.csv"
            # # Append if file exists, else write header
            # try:
            #     with open(csv_path, "x", newline="") as f:
            #         df.to_csv(f, index=False)
            # except FileExistsError:
            #     with open(csv_path, "a", newline="") as f:
            #         df.to_csv(f, index=False, header=False)

        
        # # Save the params to a file using flax.serialization.to_bytes
        # with open(params_file_name, 'wb') as f:
        #     f.write(flax.serialization.to_bytes(params))
        #     print(f"params saved")

        # Load the params from the file using flax.serialization.from_bytes
        # with open(params_file_name, 'rb') as f:
        #     restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        #     print(f"params restored")

        run.finish()

    # NOTE: Sweep Parameters will override the config file, but cannot be used to override any environment params currently. 
    # This latter option will require some careful thought on how best to implement - due to to variable number of agent types.
    sweep_parameters = {
        # "LR": {"values": [config["LR"]]},
        "NUM_STEPS": {"values": [config["NUM_STEPS"], 512,32,]}
        #"GAMMA": {"values": [config["GAMMA"], [0.99,0.99]]},
        #"LR": {"values": [config["LR"], [0.004,0.004], [0.00004,0.00004]]},
        #"ENT_COEF": {"values": [config["ENT_COEF"], [0.1,0.1], [0.05,0.05]]},
        #"NUM_STEPS": {"values": [config["NUM_STEPS"], 2048 ,512]},
        #"CLIP_EPS": {"values": [config["CLIP_EPS"], 0.3, 0.1]},
        #"VF_COEF": {"values": [config["VF_COEF"], [1e-6,1e-7], [1e-9,1e-8]]},
        #"FC_DIM_SIZE": {"values": [config["FC_DIM_SIZE"], 256]},
       # "NUM_AGENTS_PER_TYPE": {"values": [config["NUM_AGENTS_PER_TYPE"], [2,2], [10,10]]},
       #"SEED": {"values": [2,3,4,5,6,7,8,9,10]},
       #"NUM_ENVS": {"values": [config["NUM_ENVS"]]},
       #"NUM_STEPS": {"values": [config["NUM_STEPS"], 128, 32, 8]},
       
        
        # "env_params" : {"parameters": {
        #                 "world_params" : {"parameters":
        #                                 {"n_data_msg_per_step": {"values":[50,150]},
        #                                 }
        #                                 },
        #                 }},
    }

    sweep_config={
        "method": "grid",
        "parameters": sweep_parameters,
    }
    print(sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project=config["PROJECT"],entity=config["ENTITY"])
    print(sweep_id)
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)

@hydra.main(version_base=None, config_path="config", config_name="base_config")
def seperate_main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)

    # jax.profiler.start_trace("/tmp/profile-data")

    
    rng = jax.random.PRNGKey(0)

    run_fn = make_sim(config)
    # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
    out = run_fn(rng)
    # out=jax.block_until_ready(out)  # Ensure the computation is complete before proceeding
    # (dummy * dummy).block_until_ready()
    # jax.profiler.stop_trace()







    

        




if __name__ == "__main__":
    seperate_main()
