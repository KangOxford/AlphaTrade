"""
Based on PureJaxRL Implementation of PPO
"""

import os

from git import Union
import pandas as pd


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
from dataclasses import replace,fields
from gymnax_exchange.jaxob.config_io import load_config_from_file, save_config_to_file

import datetime


#from jaxmarl.wrappers.baselines import SMAXLogWrapper
#from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig,MarketMaking_EnvironmentConfig,CONFIG_OBJECT_DICT

import wandb


import functools
import matplotlib.pyplot as plt

import sys
import os
import pickle
from datetime import datetime


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

def create_agent_configs(config):
    """
    Create agent configs with three layers of precedence (lowest to highest):
    1. Default attributes from the EnvironmentConfig classes
    2. Values from the JSON config
    3. Sweep parameters from BASELINE_CONFIGS
    
    Args:
        config: The full config dict containing both JSON config and sweep parameters
        config_dict: Dict mapping agent type names to their config classes
                    e.g., {"MarketMaking": MarketMaking_EnvironmentConfig, ...}
    
    Returns:
        Dict of agent configs keyed by agent type name
    """
    agent_configs = {}
    if "BASELINE_CONFIGS" in config:
        for agent_type, agent_cfg in config["BASELINE_CONFIGS"].items():
            # Start with defaults from the config class
            agent_config_class = CONFIG_OBJECT_DICT[agent_type]
            
            # First apply config values (from JSON) to override defaults
            config_overrides = {}
            field_names = {f.name for f in fields(agent_config_class)}
            for key, value in config["dict_of_agents_configs"].items():
                if  isinstance(value, dict) and key == agent_type:
                    for key, value in value.items():
                        if key in field_names:
                            config_overrides[key] = value
            
            # Then apply sweep parameters which take highest precedence
            sweep_overrides = {k.lower(): v for k, v in agent_cfg.items()}
            
            # Merge: sweep_overrides will override config_overrides
            all_overrides = {**config_overrides, **sweep_overrides}
            
            # Create the agent config with all overrides
            agent_configs[agent_type] = agent_config_class(**all_overrides)
    
    return agent_configs


def make_sim(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
    init_key = jax.random.PRNGKey(config["SEED"])


    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = sum(config["NUM_ACTORS_PERTYPE"])

    def linear_schedule(lr,count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    def run(rng):
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
                # print("Params", network_params)
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

            # for i in range(len(metric["avg_reward"])):
            #     print(f"avg_reward_{i} {metric["avg_reward"][i]}")
            #     # print(metric["traj_batch"][i].info['agent'].keys())
            #     for main_metric in ["total_PnL","revenue_direction_normalised"]:
            #         if main_metric in metric["traj_batch"][i].info['agent'].keys():
            #             print(f"avg_PNL_{i} {metric["traj_batch"][i].info['agent'][main_metric].mean()}")

            # print(f"Completed Episodes: {metric['total_dones']}")

            # Save trajectory batch to a pickle file

            # Create trajectories directory if it doesn't exist
            os.makedirs("trajectories", exist_ok=True)

            # Create an informative filename with timestamp and action type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if combo_desc is None:
                combo_desc = "default"
            filename = f"trajectories/traj_batch_{combo_desc}__{timestamp}.pkl"


            if config["TINY_RUN"]:
                # Save the trajectory batch
                with open(filename, "wb") as f:
                    pickle.dump(metric["traj_batch"], f)

                print(f"Saved trajectory batch to {filename}")

            # if config["TINY_RUN"]:
            #     plot_episode_features(metric["traj_batch"])

        def _update_step(update_runner_state,env_params,env):
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
                # for i, train_state in enumerate(train_states):

                #     # print("Observation space for agent type{}, {} and actual array shape {}:",i,env.observation_spaces[i].shape,last_obs[i].shape)

                    # Print all dimensions of train state pytree leaves
                    # print(f"Train state dimensions for agent type {i}:")
                    # flat_params = jax.tree_util.tree_leaves(train_state)
                    # for j, param in enumerate(flat_params):
                    #     if hasattr(param, "shape"):
                    #         print(f"  Leaf {j}: shape={param.shape}, dtype={param.dtype}")



                    # jax.debug.print("Action space for agent type{}, {}:",i,env.action_spaces[i].n)
                    # print(i)



                for i, train_state in enumerate(train_states):
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i])  # Reshape to match the input shape of the network
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    # print(i, " ac_in shape:", ac_in[0].shape, "last_done shape:", ac_in[1].shape)
                    # flat_params = jax.tree_util.tree_leaves(train_state)
                    # for j, param in enumerate(flat_params):
                    #     if hasattr(param, "shape"):
                    #         print(f"  Leaf {j}: shape={param.shape}, dtype={param.dtype}")

                    # print(train_state.apply_fn)
                    h_states[i], pi, value = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                    values.append(value)
                    action = pi.sample(seed=_rng)
                    # jax.debug.print(f"Pi: {pi._probs}")
                    log_probs.append(pi.log_prob(action))
                    action=unbatchify(action, config["NUM_ACTORS_PERTYPE"][i], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                    actions.append(action.squeeze())
                    # print(actions)
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


        jitted_update_step = jax.jit(_update_step,static_argnums=(2,))
        
        def eval_policies(rng, config):
            """
            Run evaluation with different policy combinations:
            - Learned vs. Baseline
            - Baseline vs. Learned
            - Baseline vs. Baseline
            - Learned vs. Learned
            
            Generalizes to n agents per type.
            """

            
            
            # All possible policy combinations
            

            
            results = {}

            

            baselinetuple=((),(),())
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            env_params = None


            def eval_policy_choice(env_params,results,rng,baselinetuple):
                n_agent_types = len(config["NUM_AGENTS_PER_TYPE"])    
                # policy_choice=[1,1]
                # Create a description of this combination (e.g., "L-B" for Learned-Baseline)
                

                # Create a dictionary of agent configs based on policy choice
                agent_configs = create_agent_configs(config)
                # print("The config items are: \n \t ",config["world_config"].items())
                # print("The world config overrides are: \n \t ",{k: v for k, v in config["world_config"].items() 
                #         if hasattr(World_EnvironmentConfig(), k) and k != "SEED"})

                ma_config = MultiAgentConfig(
                    number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
                    dict_of_agents_configs=agent_configs,
                    world_config=World_EnvironmentConfig(
                        seed=config["SEED"],
                        timePeriod=config["EvalTimePeriod"],
                        save_raw_observations=True,
                        # Only override parameters that exist in both config and World_EnvironmentConfig
                        **{k: v for k, v in config["world_config"].items() 
                        if hasattr(World_EnvironmentConfig(), k) and k not in ["seed",
                                                                               "timePeriod",
                                                                               "save_raw_observations"]}
                    ))


                env : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config)
                env_params=env.default_params
                baselinetuple = init_baseline_policies(config, env,rng)

                bl_init_hiddens , bl_train_states, bl_init_dones_agents = baselinetuple
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
                    print("Appending Baseline policy for agent type", i)
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

                (eval_runner_state, _), eval_metrics = jitted_update_step((eval_runner_state, 0), env_params,env)
                callback(eval_metrics)

                # Store results
                results = {
                    'avg_reward': eval_metrics['avg_reward'],
                    'total_dones': eval_metrics['total_dones'],
                    'traj_batch': eval_metrics['traj_batch']
                }
                
                for i in range(n_agent_types):
                    print(f"   (Agent type {i}): avg_reward = {eval_metrics['avg_reward'][i]:.4f}")
                    for main_metric in ["reward_portfolio_value","revenue_direction_normalised","end_of_ep_pv"]:
                        if main_metric in eval_metrics["traj_batch"][i].info['agent'].keys():
                            print(f"     (Agent type {i}): {main_metric} = {eval_metrics['traj_batch'][i].info['agent'][main_metric].mean()}")
                            print(f"     (Agent type {i}): Dimensions = {eval_metrics['traj_batch'][i].info['agent'][main_metric].std()}")
                            # if main_metric == "reward_portfolio_value":
                            #     print(f"     (Agent type {i}): Dimensions = {eval_metrics['traj_batch'][i].info['agent'][main_metric].shape}")
                            #     print(f"     (Agent type {i}): PNL END = {eval_metrics['traj_batch'][i].info['agent'][main_metric][63::64,:].mean()}")
                # callback(eval_metrics)
                del eval_metrics
                gc.collect()
                return results,env_params,baselinetuple

            results,env_params,baselinetuple, = eval_policy_choice(env_params, results,rng=rng,baselinetuple=baselinetuple)

            
            return results

        # Run all policy combinations
        print("Running evaluations...")
        eval_results = eval_policies(rng, config)

        
        
        return {"results": eval_results}

    return run



@hydra.main(version_base=None, config_path="config", config_name="baseline_exec_config")
def main(config):
    try:
        if config["ENV_CONFIG"] is not None:
            print(f"Loading the env config from file \n\t{config['ENV_CONFIG']} ")
            env_config=load_config_from_file(config["ENV_CONFIG"])
            print("********* DEBUG ********** \n Loaded env_config: ", env_config)
        else:
            print("Using default MultiAgentConfig as defined in jaxob_config.py file.")
            env_config=MultiAgentConfig()
            save_config_to_file(env_config,f"config/env_configs/default_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    except Exception as e:
        print(f"Error loading env config: {e}")
        print("Reverting to default MultiAgentConfig as defined in jaxob_config.py file.")
        env_config=MultiAgentConfig()
        save_config_to_file(env_config,f"config/env_configs/default_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    print("Note: The BASELINE_CONFIG parameters in yaml will override these settings.")
    env_config=OmegaConf.structured(env_config)
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)


    
    rng = jax.random.PRNGKey(0)

    run_fn = make_sim(config)
    out = run_fn(rng)








    

        




if __name__ == "__main__":
    main()
