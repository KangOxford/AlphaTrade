"""
Based on PureJaxRL Implementation of PPO
"""

import os

from git import Union
from humanize import metric
import pandas as pd
import csv


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
#from jaxmarl.wrappers.baselines import SMAXLogWrapper
#from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig,MarketMaking_EnvironmentConfig,CONFIG_OBJECT_DICT
from gymnax_exchange.jaxob.config_io import load_config_from_file, save_config_to_file

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
    
class MultiActionOutputIndependant(nn.Module):
    action_dims: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Create multiple output heads
        if isinstance(self.action_dims, (list, tuple)):
            # Multi-output case: create separate heads for each output
            action_logits_list = []
            for dim in self.action_dims:
                logits = nn.Dense(
                    dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
                )(x)
                action_logits_list.append(logits)

            pi = MultiCategorical(action_logits_list)
        else:
            raise ValueError("action_dims must be a list or tuple for MultiActionOutputIndependant.")

        return pi

class MultiActionOutputAutoregressive(nn.Module):
    action_dims: Sequence[int]
    config: Dict
    embed_dim: int = 32

    def get_logits_for_action(self, x, action_idx, prev_actions):
        """
        Compute logits for action_idx conditioned on prev_actions.
        
        Args:
            x: actor features (batch, feature_dim)
            action_idx: which action we're computing logits for (0, 1, 2, ...)
            prev_actions: list of previously sampled actions [action_0, action_1, ...]
        """
        if action_idx == 0:
            # First action: no conditioning
            logits = nn.Dense(
                self.action_dims[0], 
                kernel_init=orthogonal(0.01), 
                bias_init=constant(0.0),
                name=f'action_{action_idx}_head'
            )(x)
            return logits
        
        # Subsequent actions: condition on previous actions
        embeddings = []
        for i, prev_action in enumerate(prev_actions):
            # Embed each previous action
            embed = nn.Embed(
                num_embeddings=self.action_dims[i],
                features=self.embed_dim,
                name=f'action_{i}_embed'
            )(prev_action)
            embeddings.append(embed)
        
        # Concatenate features with all previous action embeddings
        combined = jnp.concatenate([x] + embeddings, axis=-1)
        
        # Process through hidden layer
        hidden = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name=f'action_{action_idx}_hidden'
        )(combined)
        hidden = nn.relu(hidden)
        
        # Output logits
        logits = nn.Dense(
            self.action_dims[action_idx],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name=f'action_{action_idx}_head'
        )(hidden)
        
        return logits

    @nn.compact
    def __call__(self, x, given_actions=None):
        """
        Compute autoregressive action distribution.
        
        Args:
            x: actor features from the network
            given_actions: Optional. If provided (during training), use these for conditioning.
                          Shape: (..., num_actions). Otherwise sample autoregressively.
        
        Returns:
            AutoregressiveMultiCategorical distribution object
        """
        if not isinstance(self.action_dims, (list, tuple)):
            raise ValueError("action_dims must be a list or tuple for MultiActionOutputAutoregressive.")
        
        # Return a distribution that can sample autoregressively or compute log_prob
        return AutoregressiveMultiCategorical(
            actor_features=x,
            action_dims=self.action_dims,
            logits_fn=self.get_logits_for_action,
            given_actions=given_actions
        )

class SingleActionOutput(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, x):
        # Create multiple output heads
        if isinstance(self.action_dim, int):
            actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0) # type: ignore
            )(x)
            # Avail actions are not used in the current implementation, but can be added if needed.
            # unavail_actions = 1 - avail_actions
            action_logits = actor_mean # - (unavail_actions * 1e10)
            pi = distrax.Categorical(logits=action_logits)
        else:
            raise ValueError("action_dims must be a list or tuple for MultiActionOutputIndependant.")

        return pi

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

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )

        actor_mean = nn.relu(actor_mean)

        # Option 1: Single action output (current behavior)
        if isinstance(self.action_dim, int):
            pi = SingleActionOutput(action_dim=self.action_dim, config=self.config)(actor_mean)

        # Option 2: Multiple independent actions
        elif isinstance(self.action_dim, (list, tuple)):
            pi = MultiActionOutputIndependant(action_dims=self.action_dim, config=self.config)(actor_mean)

        # Option 3: Multiple autoregressive actions
        elif isinstance(self.action_dim, (list, tuple)) and self.config.get("AUTOREGRESSIVE", True):
            pi = MultiActionOutputAutoregressive(
                action_dims=self.action_dim,  # e.g., [10, 10, 5]
                config=self.config
            )(actor_mean)
        else:
            raise ValueError("action_dims must be int or list/tuple for ActorCriticRNN.")

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class MultiCategorical():
    """Wrapper for multiple independent categorical distributions.
    NOTE: The correct thing would be to let it inherit from distrax.Distribution but
    this requires additional thought to implement all abstract methods, many of which are not 
    needed for this use case. """
    
    def __init__(self, logits_list):
        self.categoricals = [distrax.Categorical(logits=logits) for logits in logits_list]

    
    def sample(self, seed):
        keys = jax.random.split(seed, len(self.categoricals))
        samples = [cat.sample(seed=key) for cat, key in zip(self.categoricals, keys)]
        return jnp.stack(samples, axis=-1)  # Shape: (..., num_outputs)
    
    def log_prob(self, actions):
        # actions should have shape (..., num_outputs)
        log_probs = [cat.log_prob(actions[...,i]) for i, cat in enumerate(self.categoricals)]
        return jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)  # Sum log probs for independence
    
    def entropy(self):
        entropies = [cat.entropy() for cat in self.categoricals]
        return jnp.sum(jnp.stack(entropies, axis=-1), axis=-1)  # Sum entropies for independence


class AutoregressiveMultiCategorical():
    """
    Wrapper for multiple categorical distributions where later actions 
    are conditioned on previously sampled actions.
    
    During sampling: samples actions sequentially, feeding each into the next.
    During training: computes conditional log probabilities using given actions.
    """
    
    def __init__(self, actor_features, action_dims, logits_fn, given_actions=None):
        """
        Args:
            actor_features: base features from the network (batch, feature_dim)
            action_dims: list of action space sizes, e.g., [10, 10, 5]
            logits_fn: function(x, action_idx, prev_actions) -> logits
            given_actions: optional actions to condition on (for training)
                          Shape: (..., num_actions)
        """
        self.actor_features = actor_features
        self.action_dims = action_dims
        self.logits_fn = logits_fn
        self.given_actions = given_actions
    
    def sample(self, seed):
        """Sample actions autoregressively."""
        keys = jax.random.split(seed, len(self.action_dims))
        samples = []
        
        for i, key in enumerate(keys):
            # Get logits conditioned on previously sampled actions
            logits = self.logits_fn(self.actor_features, i, samples)
            action = distrax.Categorical(logits=logits).sample(seed=key)
            samples.append(action)
        
        return jnp.stack(samples, axis=-1)  # Shape: (..., num_actions)
    
    def log_prob(self, actions):
        """
        Compute log probability of action sequence.
        Uses chain rule: log p(a1,a2,a3) = log p(a1) + log p(a2|a1) + log p(a3|a1,a2)
        
        Args:
            actions: action sequence, shape (..., num_actions)
        """
        log_probs = []
        
        for i in range(len(self.action_dims)):
            # Get previous actions for conditioning
            prev_actions = [actions[..., j] for j in range(i)]
            
            # Get conditional logits
            logits = self.logits_fn(self.actor_features, i, prev_actions)
            
            # Compute log prob of this action given previous ones
            log_p = distrax.Categorical(logits=logits).log_prob(actions[..., i])
            log_probs.append(log_p)
        
        # Sum log probabilities (chain rule)
        return jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)
    
    def entropy(self):
        """
        Compute entropy of the autoregressive distribution.
        For autoregressive models: H = sum of conditional entropies
        """
        entropies = []
        
        # For entropy, we need to marginalize over previous actions
        # Simplified: compute entropy of each conditional separately
        # (This is an approximation - true entropy requires marginalization)
        for i in range(len(self.action_dims)):
            if self.given_actions is not None and i > 0:
                # Use given actions for conditioning
                prev_actions = [self.given_actions[..., j] for j in range(i)]
            else:
                # For first action or when no given actions, use empty list
                prev_actions = []
            
            logits = self.logits_fn(self.actor_features, i, prev_actions)
            entropy = distrax.Categorical(logits=logits).entropy()
            entropies.append(entropy)
        
        return jnp.sum(jnp.stack(entropies, axis=-1), axis=-1)


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

def create_agent_configs(config,override_str:str = "BASELINE_CONFIGS") -> Dict[str, Any]:
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
    if override_str in config:
        for agent_type, agent_cfg in config[override_str].items():
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
    print("init_key: ", init_key)

    print("init_key: ", init_key)
    




    # env_baseline : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config_baseline)


    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = sum(config["NUM_ACTORS_PERTYPE"])


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

                # print("Observation spaces at init:", env.observation_spaces[i].shape)

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


            train_states.reverse()  # Reverse the list to match the order of instances
            target_ckpt= {
                'model': train_states,  # train_states
                # 'config': {} ,
                'metrics': {
                    'train_rewards': [np.nan],
                    'eval_rewards': [np.nan],
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

            # print(isinstance(restored_state["model"], list))
            restored_train_states = restored_state['model']
            restored_train_states.reverse()
            # print(len(restored_train_states), " restored train states")

            # for i,ts in enumerate(restored_train_states):
            #     # Print all dimensions of train state pytree leaves
            #     flat_params = jax.tree_util.tree_leaves(ts)
            #     for j, param in enumerate(flat_params):
            #         if hasattr(param, "shape") and j==2:
            #             print(f"  Leaf {j}: shape={param.shape}, dtype={param.dtype}")
            #     print("Apply function for agent type", i, ":", ts.apply_fn)

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
            # if config["CALC_EVAL"]:
            #     logging_dict.update({
            #         **{f"avg_eval_reward_{i}": metric["avg_reward_eval"][i] for i in range(len(metric["avg_reward_eval"]))},
            #     })
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
            policy_combinations = []
            
            # For n agent types, we have 2^n possible combinations (each type can be either learned or baseline)
            n_combos= 2 ** len(config["NUM_AGENTS_PER_TYPE"])
            for i in range(n_combos):
                # Convert i to binary, padded to n_agent_types digits
                # '1' means learned policy, '0' means baseline policy
                binary = format(i, f'0{len(config["NUM_AGENTS_PER_TYPE"])}b')
                policy_choices = [int(bit) for bit in binary]
                policy_combinations.append(policy_choices)
            
            results = {}

            policy_choice = policy_combinations[0]
            
            bl_init_hiddens , bl_train_states, bl_init_dones_agents = None, None, None
            lrn_hstates, lrn_train_states, lrn_init_dones_agents = None, None, None
            baselinetuple=((),(),())
            learnedtuple=((),(),())
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            env_params = None


            def eval_policy_choice(env_params,results,combo_idx,policy_choice,rng,baselinetuple,learnedtuple):
                n_agent_types = len(config["NUM_AGENTS_PER_TYPE"])    
                # policy_choice=[1,1]
                # Create a description of this combination (e.g., "L-B" for Learned-Baseline)
                combo_desc = ''.join(['L' if choice == 1 else 'B' for choice in policy_choice])
                
                print(f"\nEvaluating policy combination {combo_idx+1}/{len(policy_combinations)}: {combo_desc}")

                # Create a dictionary of agent configs based on policy choice
                ma_config = get_ma_config(config, policy_choice, combo_desc)
                env : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config)
                if combo_idx == 0:
                    print("Initializing baseline policies for the first combination...")
                    baselinetuple = init_baseline_policies(config, env,rng)
                    env_params=env.default_params
                if combo_idx == n_combos-1:
                    print("Loading learned policies for the last combination...")
                    learnedtuple = load_network_from_checkpoint(config, env,rng)
                bl_init_hiddens , bl_train_states, bl_init_dones_agents = baselinetuple
                lrn_hstates, lrn_train_states, lrn_init_dones_agents = learnedtuple
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
                        print("appending Learned policy for agent type", i)
                        hstates_eval.append(lrn_hstates[i])
                        train_states_eval.append(lrn_train_states[i])
                        dones_eval.append(lrn_init_dones_agents[i])
                    else:  # Use baseline policy
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
                    for main_metric in ["reward_portfolio_value","revenue_direction_normalised"]:
                        if main_metric in eval_metrics["traj_batch"][i].info['agent'].keys():
                            print(f"    {agent_type} (Agent type {i}): PNL = {eval_metrics["traj_batch"][i].info['agent'][main_metric].mean()}")
                            if main_metric == "reward_portfolio_value":
                                print(f"    {agent_type} (Agent type {i}): Dimensions = {eval_metrics["traj_batch"][i].info['agent'][main_metric].shape}")
                                print(f"    {agent_type} (Agent type {i}): PNL std = {eval_metrics["traj_batch"][i].info['agent'][main_metric][63::64,:].mean()}")
                # callback(eval_metrics)
                del eval_metrics
                gc.collect()
                return results,env_params,baselinetuple,learnedtuple

            results,env_params,baselinetuple,learnedtuple = eval_policy_choice(env_params, results, 0, policy_combinations[0],rng=rng,baselinetuple=baselinetuple,learnedtuple=learnedtuple)
            results,env_params,baselinetuple,learnedtuple = eval_policy_choice(env_params, results, n_combos-1, policy_combinations[-1],rng=rng,baselinetuple=baselinetuple,learnedtuple=learnedtuple)

            for combo_idx, policy_choice in enumerate(policy_combinations[1:-1], start=1):
                print("COMBOD INDEX",combo_idx)
                results,env_params,baselinetuple,learnedtuple = eval_policy_choice(env_params, results, combo_idx, policy_choice,rng=rng,baselinetuple=baselinetuple,learnedtuple=learnedtuple)

            return results

        # Run all policy combinations
        print("Running evaluations with all possible policy combinations...")
        eval_results = eval_policies(rng, config)

    
        
        return {"results": eval_results}

    return run

def get_ma_config(config, policy_choice, combo_desc):
    agent_configs = {}
    print(policy_choice)
    for i, use_learned in enumerate(policy_choice):
        agent_type = list(config["AGENT_CONFIGS"].keys())[i]
        # Start with common agent config
        
        if use_learned == 0:  # Use baseline policy - apply baseline overrides
            override_str="BASELINE_CONFIGS"
        elif use_learned == 1:  # Use learned policy - apply learned overrides
            override_str="AGENT_CONFIGS"
        else:
            raise ValueError(f"In get_ma_config: \n\t Invalid policy choice {use_learned} for agent type {agent_type}")
        
        # Convert all keys to lowercase for the environment config
        agent_configs=create_agent_configs(config,override_str=override_str)

    # Print agent configs with decorative formatting to make it stand out
    print("\n" + "="*80)
    print("🚀 POLICY COMBINATION: " + combo_desc + " 🚀")
    print("="*80)
    print("📊 AGENT CONFIGURATIONS:")
    for agent_type, config_obj in agent_configs.items():
        print(f"\n{'*'*40}")
        print(f"🤖 AGENT TYPE: {agent_type}")
        print(f"{'*'*40}")
        for param_name, param_value in vars(config_obj).items():
            print(f"  • {param_name}: {param_value}")
    print("="*80 + "\n")

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
    print("MultiAgentConfig for Learned Agents \n","%"*50,ma_config)
    return ma_config



@hydra.main(version_base=None, config_path="config", config_name="baseline_exec_config.yaml")
def seperate_main(config):
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
    env_config=OmegaConf.structured(env_config)    
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
