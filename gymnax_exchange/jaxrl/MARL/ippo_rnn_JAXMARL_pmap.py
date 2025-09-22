"""
Based on PureJaxRL Implementation of PPO
"""

import os

import pandas as pd
import csv

from docs.source import conf
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import time
import jax # type: ignorepip 
jax.config.update('jax_disable_jit', False)
import flax
import jax.numpy as jnp # type: ignore
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal # type: ignore
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
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


def make_train(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
    init_key = jax.random.PRNGKey(config["SEED"])
    config_dict={"MarketMaking": MarketMaking_EnvironmentConfig,"Execution": Execution_EnvironmentConfig}
    print("init_key: ", init_key)
    ###############CLAUDE##############
    # Create a MultiAgentConfig object with parameters from the config
    agent_configs = {}
    if "AGENT_CONFIGS" in config:
        agent_configs = {
            agent_type: config_dict[agent_type](**{k.lower(): v for k, v in agent_cfg.items()})
            for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
        }
    else:
        agent_configs = {
            agent_type: config_dict[agent_type]()
            for agent_type, agent_cfg in config_dict.items()
        }
    print("agent_configs:", agent_configs)
    


    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            # Only override parameters that exist in both config and World_EnvironmentConfig
            **{k.lower(): v for k, v in config.items() 
               if hasattr(World_EnvironmentConfig(), k.lower()) and k != "SEED"}
        )
    )
    print(ma_config)

    print("MultiAgentInventoryPenalty",ma_config.dict_of_agents_configs["MarketMaking"].inv_penalty)

    # For evaluation, create a separate config with evaluation-specific parameters
        # Reuse agent_configs from above if it exists
    eval_agent_configs = {}
    if "AGENT_CONFIGS" in config:
        eval_agent_configs = {
            agent_type: config_dict[agent_type](**{k.lower(): v for k, v in agent_cfg.items()})
            for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
        }
    else:
        eval_agent_configs = {
            agent_type: config_dict[agent_type]()
            for agent_type, agent_cfg in config_dict.items()
        }
        
    eval_ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=eval_agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            timePeriod=config["EvalTimePeriod"],
            # Only override parameters that exist in both config and World_EnvironmentConfig
            **{k.lower(): v for k, v in config.items() 
                if hasattr(World_EnvironmentConfig(), k.lower()) and k not in ["SEED", "EvalTimePeriod"]}
        )
    )
   



    env : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config)
    eval_env: MARLEnv = MARLEnv(key=init_key,multi_agent_config=eval_ma_config)
    
    agent_type_names = list(env.type_names)

    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZES"] = [
        nact * config["NUM_STEPS"] // config["NUM_MINIBATCHES"] for i,nact in enumerate(config["NUM_ACTORS_PERTYPE"])
    ]
    # config["CLIP_EPS"] = (
    #     config["CLIP_EPS"] / env.num_agents
    #     if config["SCALE_CLIP_EPS"]
    #     else config["CLIP_EPS"]
    # )

    def linear_schedule(lr,count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    def train(rng):
        # INIT NETWORK


        # For a given agent type (instance) we need the following inputs:
        # Action space, obs space, 

        # The outputs that depends on these and are kept seperate are;
        # - network, init_x, init_hstate, network_params, train_state
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

            # FIXME: very unsure about this, why is it NUM_ENVS and not NUM_ACTORS?
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            network_params = network.init(_rng, init_hstate, init_x)
            if config["ANNEAL_LR"][i]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(learning_rate=functools.partial(linear_schedule,config["LR"][i]), eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(config["LR"][i], eps=1e-5),
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
        train_states=flax.jax_utils.replicate(train_states)
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_params=env.default_params
        eval_env_params=eval_env.default_params # type: ignore

        # env_params=jax.device_put(env_params)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng,env_params)
        # TRAIN LOOP

        def reshape_pytree_leading_dim(pytree, devices):
            """
            Reshape each leaf of a pytree by splitting the leading dimension M into (N, M/N).
            
            Args:
                pytree: A PyTree with arrays that have a leading dimension to reshape.
                num_envs: The number of environments (N) for the first dimension.
                num_agents: Optional, the number of agents per environment. If None, 
                           calculated as M / num_envs.
            
            Returns:
                A PyTree with the same structure but with leaves reshaped from (M, ...) 
                to (N, M/N, ...).
            """
            def _reshape_leaf(leaf):
                if not isinstance(leaf, jnp.ndarray):
                    return leaf
                
                if leaf.ndim == 0:
                    return leaf
                
                leading_dim = leaf.shape[0]
                # If num_agents is provided, use it; otherwise calculate it
                env_per_device = leading_dim // devices
                
                # Check that the reshape is valid
                if leading_dim != devices * env_per_device:
                    raise ValueError(
                        f"Leading dimension {leading_dim} cannot be reshaped to "
                        f"({devices}, {env_per_device})"
                    )
                    
                new_shape = (devices, env_per_device) + leaf.shape[1:]
                return leaf.reshape(new_shape)
            
            return jax.tree_util.tree_map(_reshape_leaf, pytree)

        env_state=reshape_pytree_leading_dim(env_state, config["N_DEVICES"])
        obsv=reshape_pytree_leading_dim(obsv, config["N_DEVICES"])
        init_dones_agents=reshape_pytree_leading_dim(init_dones_agents, config["N_DEVICES"]) # last_done
        hstates=reshape_pytree_leading_dim(hstates, config["N_DEVICES"])

        print(jax.tree_util.tree_map(lambda x: x.shape, (env_state,obsv,init_dones_agents,hstates,_rng)))


        def callback(metric):
            print("Update step:", metric["update_steps"])
            action_distribution = {}
            for i, tr in enumerate(metric["traj_batch"]):
                actions = np.array(tr.action).flatten()
                unique_actions, counts = np.unique(actions, return_counts=True)
                tot_counts=sum(counts)
                # Add each action count to the dictionary with a unique key
                for a, c in zip(unique_actions, counts):
                    action_distribution[f"action_{i}_{int(a)}"] = c/tot_counts*100
            logging_dict = {
                    # TODO: Log the quantities of interest. Keep it trivial for now.
                    "env_step": (metric["update_steps"].sum()+1)
                    * config["NUM_ENVS"]// config["N_DEVICES"]
                    * config["NUM_STEPS"],
                    **{f"network_{i}": m for i,m in enumerate(metric["loss"])},
                    **{f"avg_reward_{i}": metric["avg_reward"][i].mean() for i in range(len(metric["avg_reward"]))},
                    **action_distribution
                }
            if config["CALC_EVAL"]:
                logging_dict.update({
                    **{f"avg_eval_reward_{i}": metric["avg_reward_eval"][i].mean() for i in range(len(metric["avg_reward_eval"]))},
                })
            if config["WANDB_MODE"]!= "disabled":
                wandb.log(logging_dict)

            for i in range(len(metric["avg_reward"])):
                print(f"avg_reward_{i} {metric["avg_reward"][i]}")
                if config["CALC_EVAL"]:
                    print(f"avg_eval_reward_{i} {metric["avg_reward_eval"][i]}")

        def speed_only_callback(metric):
            logging_dict = {
                    "env_step": (metric["update_steps"][0]+1)
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]}
            print(metric["update_steps"],config["NUM_ENVS"],config["NUM_STEPS"])
            print(logging_dict["env_step"])
            if config["WANDB_MODE"]!= "disabled":
                wandb.log(logging_dict) 


        def _update_step(update_runner_state,env_params,eval_env_params):
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
                    rng_, _rng = jax.random.split(_rng)
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])  # Reshape to match the input shape of the network
                    ac_in = (
                        obs_i[jnp.newaxis, :],
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    h_states[i], pi, value = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                    values.append(value)
                    action = pi.sample(seed=_rng)
                    log_probs.append(pi.log_prob(action))
                    action=unbatchify(action, config["NUM_ENVS"]//config["N_DEVICES"], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                    actions.append(action.squeeze())
                    # env_act = unbatchify(
                    #     action, env.agents, config["NUM_ENVS"], env.num_agents
                    # )
                    # env_act = {k: v.squeeze() for k, v in env_act.items()}
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"]//config["N_DEVICES"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0,None)
                )(rng_step, env_state, actions,env_params)

                # info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                
                done_batch=done
                transitions=[]
                for i,train_state in enumerate(train_states):
                    done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze()
                    obs_batch = batchify(obsv[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                    action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                    value = values[i]
                    log_prob = log_probs[i]

                    info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]),info["agents"][i])}
                    # print(f"info for agenttype {i}:", info_i)


                    transitions.append(Transition(
                        jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                        last_done[i],
                        action_batch.squeeze(),
                        value.squeeze(),
                        batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze(),
                        log_prob.squeeze(),
                        obs_batch,
                        info_i,
                        # avail_actions,
                    ))
                runner_state = (train_states, env_state, obsv, done_batch['agents'], h_states, rng)
                return runner_state, transitions

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )


            

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_dones, hstates_new, rng = runner_state

            def _calculate_gae(gamma,gae_lambda,traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.global_done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + gamma * next_value * (1 - done) - value
                        gae = (
                            delta
                            + gamma * gae_lambda * (1 - done) * gae
                        )
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

            advantages=[]
            targets=[]
            for i, train_state in enumerate(train_states):
                last_obs_batch = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                # avail_actions = jnp.ones(
                #     (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
                # )
                ac_in = (
                    last_obs_batch[jnp.newaxis, :],
                    last_dones[i][jnp.newaxis, :],
                    # avail_actions,
                )
                _, _, last_val = train_state.apply_fn(train_state.params, hstates_new[i], ac_in)
                last_val = last_val.squeeze()

                advantages_i, targets_i = _calculate_gae(config["GAMMA"][i],config["GAE_LAMBDA"][i],traj_batch[i], last_val)
                advantages.append(advantages_i)
                targets.append(targets_i)

            # UPDATE NETWORKS
            loss_infos = []
            for i, train_state in enumerate(train_states):
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                            # RERUN NETWORK
                            _, pi, value = train_state.apply_fn(
                                params,
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob = pi.log_prob(traj_batch.action)

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()

                            # CALCULATE ACTOR LOSS
                            logratio = log_prob - traj_batch.log_prob
                            ratio = jnp.exp(logratio)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            # debug
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"][i] * value_loss
                                - config["ENT_COEF"][i] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, init_hstate, traj_batch, advantages, targets
                        )
                        total_loss=jax.lax.pmean(total_loss, axis_name="device_batch")
                        grads = jax.lax.pmean(grads, axis_name="device_batch")
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss
                    (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    # adding an additional "fake" dimensionality to perform minibatching correctly
                    init_hstate = jnp.reshape(
                        init_hstate, (1, config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"], -1)
                    )
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])

                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        init_hstate.squeeze(),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss

                update_state = (
                    train_state,
                    initial_hstates[i],
                    traj_batch[i],
                    advantages[i],
                    targets[i],
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_states[i] = update_state[0]
                loss_infos.append(loss_info)



            metrics= {}
            metrics['agents'] = [jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"]//config["N_DEVICES"], config["NUM_AGENTS_PER_TYPE"][i])
                ),
                trjbtch.info['agent']) for i, trjbtch in enumerate(traj_batch)]
            metrics['world'] = [traj_batch.info['world'] for i, traj_batch in enumerate(traj_batch)]
            metrics["loss"]=[]
            for i,loss_info in enumerate(loss_infos):
                ratio_0 = loss_info[1][3].at[0,0].get().mean()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
                metrics["loss"].append({
                    "total_loss": loss_info[0],
                    "value_loss": loss_info[1][0],
                    "actor_loss": loss_info[1][1],
                    "entropy": loss_info[1][2],
                    "ratio": loss_info[1][3],
                    "ratio_0": ratio_0,
                    "approx_kl": loss_info[1][4],
                    "clip_frac": loss_info[1][5],
                    "weighted_entropy_loss": loss_info[1][2] * config["ENT_COEF"][i],
                    "weighted_value_loss": loss_info[1][0] * config["VF_COEF"][i],
                })
            metrics['avg_reward'] = [jnp.mean(tr.reward) for tr in traj_batch]
            metrics["traj_batch"] = traj_batch



            if config["CALC_EVAL"]:
                def _eval_step(eval_runner_state, unused):
                    train_states, eval_env_state, last_obs, last_done,hstates, rng = eval_runner_state
                    rng, _rng = jax.random.split(rng)
                
                    actions=[]
                    values=[]
                    log_probs=[]

                    for i, train_state in enumerate(train_states):
                        obs_i= last_obs[i]
                        obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])  # Reshape to match the input shape of the network
                        ac_in = (
                            obs_i[jnp.newaxis, :],
                            last_done[i][jnp.newaxis, :],
                            # avail_actions,
                        )
                        hstates[i], pi, value = train_state.apply_fn(train_state.params, hstates[i], ac_in)
                        values.append(value)
                        action = pi.sample(seed=_rng)
                        log_probs.append(pi.log_prob(action))
                        action=unbatchify(action, config["NUM_ENVS"]//config["N_DEVICES"], env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                        actions.append(action.squeeze())

                        rng, _rng = jax.random.split(rng)

                





                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"]//config["N_DEVICES"])
                    obsv, eval_env_state, reward, done, info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0, None) # type: ignore
                    )(rng_step, eval_env_state, actions, eval_env_params)
                    done_batch=done
                    transitions=[]    

                    for i,train_state in enumerate(train_states):
                        done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze()
                        obs_batch = batchify(obsv[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                        action_batch = batchify(actions[i],config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"])
                        value = values[i]
                        log_prob = log_probs[i]

                        info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]),info["agents"][i])}
                        # print(f"info for agenttype {i}:", info_i)


                        transitions.append(Transition(
                            jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                            last_done[i],
                            action_batch.squeeze(),
                            value.squeeze(),
                            batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]).squeeze(),
                            log_prob.squeeze(),
                            obs_batch,
                            info_i,
                            # avail_actions,
                        ))
                    eval_runner_state = (train_states, eval_env_state, obsv, done_batch['agents'], hstates, rng)
                    return eval_runner_state, transitions

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(rng, config["NUM_ENVS"]//config["N_DEVICES"])
                eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params) # type: ignore
                jax.debug.print("WINDOW INDECES: {}",eval_env_state.world_state.window_index)

                eval_hstates=[]
                init_dones_agents_eval=[]
                for i,train_state in enumerate(train_states):
                    eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]//config["N_DEVICES"]), dtype=bool))


                
                eval_runner_state = (
                train_states,
                eval_env_state,
                eval_obsv,
                init_dones_agents_eval,
                eval_hstates,
                _rng,
                )
                eval_runner_state, eval_traj_batch = jax.lax.scan(
                    _eval_step, eval_runner_state, None,  config["NUM_STEPS_EVAL"]
                )
                metrics['agents_eval'] = [jax.tree.map(
                    lambda x: x.reshape(
                        (config["NUM_STEPS_EVAL"], config["NUM_ENVS"]//config["N_DEVICES"], config["NUM_AGENTS_PER_TYPE"][i])
                    ),
                    trjbtch.info['agent']) for i, trjbtch in enumerate(eval_traj_batch)]
                metrics['world_eval'] = [trjbtch.info['world'] for i, trjbtch in enumerate(eval_traj_batch)]
                if config["CALC_EVAL"]:
                    metrics['avg_reward_eval'] = [jnp.mean(tr.reward) for tr in eval_traj_batch]
                    metrics["traj_batch_eval"] = eval_traj_batch

            
            metrics["update_steps"] = update_steps
            # jax.experimental.io_callback(callback, None, metrics)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_dones, hstates_new, rng)

            # jax.profiler.save_device_memory_profile(f"memory_{update_steps}.prof")
            return (runner_state, update_steps), metrics

        rng, _rng = jax.random.split(rng)
        device_rng=jax.random.split(_rng, config["N_DEVICES"])  # Split the RNG for each device
        runner_state = (
            train_states,
            env_state,
            obsv,
            init_dones_agents, # last_done
            hstates,  # initial hidden states for RNN
            device_rng,
        )


        jitted_update_step = jax.jit(_update_step,)
        pmapped_update_step = jax.pmap(
            jitted_update_step,
            axis_name="device_batch",
            in_axes=(((0, 0, 0, 0, 0, 0),None),None,None),
            out_axes=(((0, 0, 0, 0, 0, 0),None), 0),
        )
        updates=0
        # compiled_update_step = jax.jit(pmapped_update_step).trace((runner_state,updates),env_params,eval_env_params).lower().compile()  # type: ignore



        # Print details about the runner state components before training starts
        print("========== Runner State Components ==========")
        print(f"Number of train states: {len(train_states)}")
        for i, ts in enumerate(train_states):
            print(f"Train state {i} structure: {jax.tree_util.tree_structure(ts)}")

        print(f"\nEnvironment state structure: {jax.tree_util.tree_structure(env_state)}")

        print("\nObservation shapes:")
        for i, obs in enumerate(obsv):
            print(f"Agent type {i} observation shape: {obs.shape}")

        print("\nInitial done flags:")
        for i, done in enumerate(init_dones_agents):
            print(f"Agent type {i} done flags shape: {done.shape}, dtype: {done.dtype}")

        print("\nHidden state details:")
        for i, h in enumerate(hstates):
            print(f"Agent type {i} hidden state shape: {h.shape}")

        
        for i in range(config["NUM_UPDATES"]):
            print(f"Update step {i+1}/{config['NUM_UPDATES']}")
            # Run the update step:
            # if i>2 and i<4:
            #     jax.profiler.start_trace("/tmp/profile-data")
            (runner_state,updates),metrics=pmapped_update_step((runner_state,updates),env_params,eval_env_params)
            speed_only_callback(metrics)

            # if i>2 and i<4:
            #     jax.block_until_ready((runner_state,updates,metrics))
            #     jax.profiler.stop_trace()


            # callback(metrics)
            

            
            del metrics
            gc.collect()
        


        # runner_state, metrics = jax.lax.scan(
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )
        
        
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="PMAP_ippo_rnn_JAXMARL_2player")
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
        

        rng = jax.random.PRNGKey(config["SEED"])

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
        "LR": {"values": [config["LR"]]},
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

@hydra.main(version_base=None, config_path="config", config_name="PMAP_ippo_rnn_JAXMARL_2player")
def seperate_main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)

    # jax.profiler.start_trace("/tmp/profile-data")

    
    rng = jax.random.PRNGKey(0)

    train_fun = make_train(config)
    # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
    out = train_fun(rng)
    # out=jax.block_until_ready(out)  # Ensure the computation is complete before proceeding
    # (dummy * dummy).block_until_ready()
    # jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
