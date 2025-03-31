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
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
from gymnax_exchange.jaxrl.utils import FlattenObservationWrapper, LogWrapper
from jax._src import dtypes
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
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig



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



class ScannedRNN_old(nn.Module):
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
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y
  
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

wandbOn = True # False
if wandbOn:
    import wandb


def make_train(config):
    env_config=EnvironmentConfig(**config["ENV_CONFIG"])
    baseline_env_config=EnvironmentConfig(**config["BASELINE_ENV_CONFIG"])

    # env_config = dataclasses.replace(env_config, **config["ENV_CONFIG"])

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    #env, env_params = gymnax.make(config["ENV_NAME"])
    rng = jax.random.key(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env = MarketMakingEnv(
        env_config,
        key_reset,
        alphatradePath=config["ATFOLDER"]+"/train",
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["DATA_TYPE"],
    )

    eval_env=MarketMakingEnv(
        env_config,
        key_reset,
        alphatradePath=config["ATFOLDER"]+"/val",
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["DATA_TYPE"],
    )

    #Add an AvSt baseline, compare to eval env
    baseline_env=MarketMakingEnv(
        baseline_env_config,
        key_reset,
        alphatradePath=config["ATFOLDER"]+"/val",
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["DATA_TYPE"],
    )

    eval_env_params = dataclasses.replace(
        eval_env.default_params,
        episode_time=config["EPISODE_TIME"],
    )

    env_params = dataclasses.replace(
        env.default_params,
        episode_time=config["EPISODE_TIME"],
    )
    baseline_env_params = dataclasses.replace(
        baseline_env.default_params,
        episode_time=config["EPISODE_TIME"],
    )
    baseline_action=config["BASELINE_FIXED_ACTION"]

    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    eval_env = FlattenObservationWrapper(eval_env)
    eval_env = LogWrapper(eval_env)

    baseline_env = FlattenObservationWrapper(baseline_env)
    baseline_env = LogWrapper(baseline_env) 

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )

        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            update_count=runner_state[-1]
            runner_state=runner_state[:-1]
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)
                

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])


                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                def log_action_distribution(action):
                    unique_actions, counts = jnp.unique(action, return_counts=True)
                    action_distribution = {f"action_{int(a)}": int(c) for a, c in zip(unique_actions, counts)}
                    wandb.log(action_distribution)
                if wandbOn:
                 jax.debug.callback(log_action_distribution, action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )

                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward 
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
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
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
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
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            #-----Evaluation------#

            def _eval_step(eval_runner_state, unused):
                train_state, eval_env_state, last_obs, last_done, hstate, rng = eval_runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                def log_action_distribution(action):
                    unique_actions, counts = jnp.unique(action, return_counts=True)
                    action_distribution = {f"action_{int(a)}": int(c) for a, c in zip(unique_actions, counts)}
                    wandb.log(action_distribution)
                if wandbOn:
                 jax.debug.callback(log_action_distribution, action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, eval_env_state, reward, done, info = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(rng_step, eval_env_state, action, eval_env_params)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                eval_runner_state = (train_state, eval_env_state, obsv, done, hstate, rng)
                return eval_runner_state, transition

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)
            initial_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
            
            eval_runner_state = (
            train_state,
            eval_env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            initial_hstate,
            _rng,
            )
            eval_runner_state, eval_traj_batch = jax.lax.scan(
                _eval_step, eval_runner_state, None, config["NUM_STEPS"]
            )
            eval_metric=eval_traj_batch.info
            #-----Baseline evaluation------#
            def _baseline_step(baseline_runner_state,unused ):
                baseline_action,baseline_env_state, last_obs, last_done, rng = baseline_runner_state

                # SELECT ACTION
                action=jnp.full((config["NUM_ENVS"],), baseline_action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, baseline_env_state, reward, done, info = jax.vmap(
                    baseline_env.step, in_axes=(0, 0, 0, None)
                )(rng_step, baseline_env_state, action, baseline_env_params)
                value=jnp.array([0])
                log_prob=jnp.array([0])
                transition= Transition(
                    last_done, action, value,reward,log_prob, last_obs, info
                )
      
                baseline_runner_state = (baseline_action,baseline_env_state, obsv, done, rng)
                return baseline_runner_state,transition
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, baseline_env_state = jax.vmap(baseline_env.reset, in_axes=(0, None))(reset_rng, baseline_env_params)
            
            baseline_runner_state = (
            baseline_action,  
            baseline_env_state,   
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            _rng,
            )
            baseline_runner_state, baseline_traj_batch = jax.lax.scan(
                _baseline_step, baseline_runner_state, None, config["NUM_STEPS"]
            )
            baseline_metric=baseline_traj_batch.info

            if config.get("DEBUG"):
                def callback(info_train,info_eval,baseline_metric,update_count):
                    #------------Collect info for plotting---------------------------#
                    #1)Step and return info
                    return_values = info_train["returned_episode_returns"][info_train["returned_episode"]]
                    #Returns for anything done in any of the last N_steps steps. Size=N_steps by N_envs
                    #jax.debug.print("Returned episode size:{}",info_train["returned_episode"].shape) 
                    #jax.debug.print("Inventory size:{}",info_train["inventory"].shape)


                   # timesteps = info_train["timestep"]


                    #-----------Train info----------#
                    episodic_PnL_train = info_train["total_PnL"][info_train["returned_episode"]]
                    episodic_netWorth_train = info_train["netWorth"][info_train["returned_episode"]]
                    #Return episode ending PnL

                    inventories_train = info_train["inventory"][:, config["ENVID"]]  
                    buyQuant_train=info_train["buyQuant"][:, config["ENVID"]]  
                    sellQuant_train=info_train["sellQuant"][:, config["ENVID"]]  
                    reward_train=info_train["reward"][:, config["ENVID"]]  
                    other_exec_quants_train=info_train["other_exec_quants"][:, config["ENVID"]]  
                    averageMidprice_train=info_train["averageMidprice"][:, config["ENVID"]]  
                    averageBestbid_train=info_train["average_best_bid"][:, config["ENVID"]]  
                    averageBestask_train=info_train["average_best_ask"][:, config["ENVID"]]  
                   

                    #-------------eval info------#   
                    PnL_eval = info_eval["total_PnL"]
                    inventories_eval = info_eval["inventory"] 
                    buyQuant_eval=info_eval["buyQuant"]
                    sellQuant_eval=info_eval["sellQuant"]
                    reward_eval=info_eval["reward"]
                    other_exec_quants_eval=info_eval["other_exec_quants"]
                    netWorth_eval = info_eval["netWorth"]
                    averageMidprice_eval=info_eval["averageMidprice"]
                    #averageBestbid_eval=info_eval["average_best_bid"]
                    #averageBestask_eval=info_eval["average_best_ask"]
                    
                    #-------------baseline info------#
                    PnL_baseline = baseline_metric["total_PnL"]
                    inventories_baseline = baseline_metric["inventory"]
                    buyQuant_baseline=baseline_metric["buyQuant"]
                    sellQuant_baseline=baseline_metric["sellQuant"]
                    reward_baseline=baseline_metric["reward"]
                    other_exec_quants_baseline=baseline_metric["other_exec_quants"]
                    netWorth_baseline = baseline_metric["netWorth"]
                    averageMidprice_baseline=baseline_metric["averageMidprice"]
                    #averageBestbid_baseline=baseline_metric["average_best_bid"]
                    #averageBestask_baseline=baseline_metric["average_best_ask"]
                   
                    #-----------------Logging-------------------#

                    if wandbOn:
                        wandb.log(
                            data={
                                #-----time and return------------#
                                "episodic_return": jnp.mean(return_values) if return_values.size > 0 else 0,  # Handle empty arrays
                               # "global_step": jnp.sum(timesteps) if timesteps.size > 0 else 0,
                                #"time":jnp.mean(time) if time.size>0 else 0,

                                #---------Reward and error bars--------#
                                #train
                                "reward_train":jnp.mean(reward_train) if reward_train.size > 0 else 0,
                               # "reward_train_plus_std": (jnp.mean(reward_train) + jnp.std(reward_train)) if reward_train.size > 0 else 0,
                               # "reward__train_minus_std": (jnp.mean(reward_train) - jnp.std(reward_train)) if reward_train.size > 0 else 0,
                                #eval
                                "reward_eval":jnp.mean(reward_eval) if reward_eval.size > 0 else 0,
                                "reward_eval_plus_std": (jnp.mean(reward_eval) + jnp.std(reward_eval)) if reward_eval.size > 0 else 0,
                                "reward_eval_minus_std": (jnp.mean(reward_eval) - jnp.std(reward_eval)) if reward_eval.size > 0 else 0,
                                #baseline
                                "reward_baseline":jnp.mean(reward_baseline) if reward_baseline.size > 0 else 0,
                                "reward_baseline_plus_std": (jnp.mean(reward_baseline) + jnp.std(reward_baseline)) if reward_baseline.size > 0 else 0,
                                "reward_baseline_minus_std": (jnp.mean(reward_baseline) - jnp.std(reward_baseline)) if reward_baseline.size > 0 else 0,
                                
                                #---------PnL and errors bars-----------#
                                #reward
                                "Episodic_PnL_train_mean": jnp.mean(episodic_PnL_train) if episodic_PnL_train.size > 0 else 0,
                                "Episodic_PnL_train_plus_std": (jnp.mean(episodic_PnL_train) + jnp.std(episodic_PnL_train)) if episodic_PnL_train.size > 0 else 0,
                                "Episodic_PnL_train_minus_std": (jnp.mean(episodic_PnL_train) - jnp.std(episodic_PnL_train)) if episodic_PnL_train.size > 0 else 0,
                                #eval
                                "PnL_eval_mean": jnp.mean(PnL_eval) if PnL_eval.size > 0 else 0,
                                "PnL_eval_plus_std": (jnp.mean(PnL_eval) + jnp.std(PnL_eval)) if PnL_eval.size > 0 else 0,
                                "PnL_eval_minus_std": (jnp.mean(PnL_eval) - jnp.std(PnL_eval)) if PnL_eval.size > 0 else 0,
                                #baseline
                                "PnL_baseline_mean": jnp.mean(PnL_baseline) if PnL_baseline.size > 0 else 0,
                                "PnL_baseline_plus_std": (jnp.mean(PnL_baseline) + jnp.std(PnL_baseline)) if PnL_baseline.size > 0 else 0,
                                "PnL_baseline_minus_std": (jnp.mean(PnL_baseline) - jnp.std(PnL_baseline)) if PnL_baseline.size > 0 else 0,

                                #-------------NetWorth and error bars----------#
                                #train
                                "Episodic_netWorth_train": jnp.mean(episodic_netWorth_train) if episodic_netWorth_train.size > 0 else 0,
                                "Episodic_netWorth_train_plus_std": (jnp.mean(episodic_netWorth_train) + jnp.std(episodic_netWorth_train)) if episodic_netWorth_train.size > 0 else 0,
                                "Episodic_netWorth_train_minus_st": (jnp.mean(episodic_netWorth_train) - jnp.std(episodic_netWorth_train)) if episodic_netWorth_train.size > 0 else 0,
                                #eval
                                "netWorth_eval": jnp.mean(netWorth_eval) if netWorth_eval.size > 0 else 0,
                                "netWorth_eval_upper": (jnp.mean(netWorth_eval) + jnp.std(netWorth_eval)) if netWorth_eval.size > 0 else 0,
                                "netWorth_eval_lower": (jnp.mean(netWorth_eval) - jnp.std(netWorth_eval)) if netWorth_eval.size > 0 else 0,
                                #baseline
                                "netWorth_baseline": jnp.mean(netWorth_baseline) if netWorth_baseline.size > 0 else 0,
                                "netWorth_baseline_upper": (jnp.mean(netWorth_baseline) + jnp.std(netWorth_baseline)) if netWorth_baseline.size > 0 else 0,
                                "netWorth_baseline_lower": (jnp.mean(netWorth_baseline) - jnp.std(netWorth_baseline)) if netWorth_baseline.size > 0 else 0,
                                                                
                                #----------Iventory and error bars------------#
                                #train
                                "inventory_train": jnp.mean(inventories_train) if inventories_train.size > 0 else 0, 
                               # "inventory_train_plus_std":(jnp.mean(inventories_train) + jnp.std(inventories_train)) if inventories_train.size > 0 else 0,
                               # "inventory_train_minus_std":(jnp.mean(inventories_train) - jnp.std(inventories_train)) if inventories_train.size > 0 else 0,
                                #eval
                                "inventory_eval": jnp.mean(inventories_eval) if inventories_eval.size > 0 else 0,
                                "inventory_eval_plus_std":(jnp.mean(inventories_eval) + jnp.std(inventories_eval)) if inventories_eval.size > 0 else 0,
                                "inventory_eval_minus_std":(jnp.mean(inventories_eval) - jnp.std(inventories_eval)) if inventories_eval.size > 0 else 0,
                                #baseline
                                "inventory_baseline": jnp.mean(inventories_baseline) if inventories_baseline.size > 0 else 0,
                                "inventory_baseline_plus_std":(jnp.mean(inventories_baseline) + jnp.std(inventories_baseline)) if inventories_baseline.size > 0 else 0,
                                "inventory_baseline_minus_std":(jnp.mean(inventories_baseline) - jnp.std(inventories_baseline)) if inventories_baseline.size > 0 else 0,
                                
                                #----------Buy and Sell Quant and error bars------------#
                                #train
                                "buyQuant_train":jnp.mean(buyQuant_train) if buyQuant_train.size > 0 else 0,
                                "sellQuant_train":jnp.mean(sellQuant_train) if sellQuant_train.size > 0 else 0,
                                "other_exec_quants_train":jnp.mean(other_exec_quants_train) if other_exec_quants_train.size > 0 else 0,
                                "averageMidprice_train":jnp.mean(averageMidprice_train) if averageMidprice_train.size>0 else 0,
                                #"averageBestbid_train":jnp.mean(averageBestbid_train) if averageBestbid_train.size>0 else 0,
                                #"averageBestask_train":jnp.mean(averageBestask_train) if averageBestask_train.size>0 else 0,
                                #eval
                                "buyQuant_eval":jnp.mean(buyQuant_eval) if buyQuant_eval.size > 0 else 0,
                                "sellQuant_eval":jnp.mean(sellQuant_eval) if sellQuant_eval.size > 0 else 0,
                                "other_exec_quants_eval":jnp.mean(other_exec_quants_eval) if other_exec_quants_eval.size > 0 else 0,
                                "averageMidprice_eval":jnp.mean(averageMidprice_eval) if averageMidprice_eval.size>0 else 0,
                                #"averageBestbid_eval":jnp.mean(averageBestbid_eval) if averageBestbid_eval.size>0 else 0,
                                #"averageBestask_eval":jnp.mean(averageBestask_eval) if averageBestask_eval.size>0 else 0,
                               
                                #baseline
                                "buyQuant_baseline":jnp.mean(buyQuant_baseline) if buyQuant_baseline.size > 0 else 0,
                                "sellQuant_baseline":jnp.mean(sellQuant_baseline) if sellQuant_baseline.size > 0 else 0,
                                "other_exec_quants_baseline":jnp.mean(other_exec_quants_baseline) if other_exec_quants_baseline.size > 0 else 0,
                                "averageMidprice_baseline":jnp.mean(averageMidprice_baseline) if averageMidprice_baseline.size>0 else 0,
                                #"averageBestbid_baseline":jnp.mean(averageBestbid_baseline) if averageBestbid_baseline.size>0 else 0,
                               # "averageBestask_baseline":jnp.mean(averageBestask_baseline) if averageBestask_baseline.size>0 else 0,
                                #----------Action prices------------#
                              
                               
                                 "update_count": update_count,
                            
                                                            },
                            commit=True
                        )
                        
                        # Additionally log histograms for full distributions
                        if reward_train.size > 0:
                            wandb.log({"reward_histogram": wandb.Histogram(reward_train)}, commit=False)
                        if return_values.size > 0:
                            wandb.log({"episodic_return_histogram": wandb.Histogram(return_values)}, commit=False)
                        if episodic_PnL_train.size > 0:
                            wandb.log({"PnL_histogram": wandb.Histogram(episodic_PnL_train)}, commit=False)
                        # Add networth histogram
                        if episodic_netWorth_train.size > 0:
                            wandb.log({"networth_histogram": wandb.Histogram(episodic_netWorth_train)}, commit=False)
                    print("Update step is",update_count, "of",config["NUM_UPDATES"])
                   # if config["VERBOSE"]:
                    #    for t in range(len(timesteps)):
                     #       print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric,eval_metric,baseline_metric,update_count)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng,update_count+1)

            return runner_state, (metric,eval_metric)

        rng, _rng = jax.random.split(rng)
        update_count=0
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            update_count, 
        )
        runner_state, (metric,eval_metric) = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric, "eval_metric": eval_metric}

    return train


if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

    # Need only to add deviations from the default environment config.
    env_config_hps = []
    #for o in ["engineered"]:
    #    for r in ["portfolio_value","pnl","complex"]:
    #        for i in ["none", "linear", "quadratic"]:
    #            for a in ["fixed_quants"]:
    #                for e in ["unwind_ref_price","do_nothing"]:
    #                    for ref in ["mid", "best_bid_ask"]:
    #                        for n in [4,8]:
    #                            for q in [1,10]:
    #                                env_config_hps.append({"observation_space":o,
    #                                                    "reward_space":r,
    #                                                    "inv_penalty":i,
    #                                                    "action_space":a,
    #                                                    "end_fn":e,
    #                                                    "reference_price_portfolio_value":ref,
    #                                                    "n_actions":n,
    #                                                    "fixed_quant_value":q})  
    env_config_hps = [  {"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"linear",
                         "n_actions":6,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"fixed_quants"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"AvSt"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"spread_skew"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"spooner",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"fixed_quants"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"spooner",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"AvSt"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"spooner",
                         "inv_penalty":"none",
                         "n_actions":6,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"spread_skew"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"linear",
                         "n_actions":6,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"fixed_quants"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"AvSt"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"portfolio_value",
                         "inv_penalty":"none",
                         "n_actions":6,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"spread_skew"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"spooner",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"fixed_quants"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"spooner",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"AvSt"
                          },
                          {"observation_space":"engineered",
                         "reward_space":"spooner",
                         "inv_penalty":"none",
                         "n_actions":6,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"best_bid_ask",
                         "action_space":"spread_skew"
                          }]
    baseline_env_config_hps = [{"observation_space":"engineered",
                            "reward_space":"portfolio_value",
                            "inv_penalty":"none",
                            "n_actions":8,
                            "end_fn":"unwind_ref_price",
                            "fixed_quant_value":10,
                            "reference_price_portfolio_value":"best_bid_ask",
                            "action_space":"spread_skew"
                            }]      
    
    # Model & Training parameters, should be independant of the environment config
    # TODO: Some adjustment needed, some of these are effectively environment parameters
    training_parameters = {
        "LR": {"values": [2.5e-4]},
        "NUM_ENVS": {"values": [256]},
        "NUM_STEPS": {"values": [32]},
        "TOTAL_TIMESTEPS": {"values": [8e5]},
        "UPDATE_EPOCHS": {"values": [2]},
        "NUM_MINIBATCHES": {"values": [16]},
        "GAMMA": {"values": [0.999]},
        "GAE_LAMBDA": {"values": [0.99]},
        "CLIP_EPS": {"values": [0.2]},
        "ENT_COEF": {"values": [0.0,0.01]},
        "VF_COEF": {"values": [0.5]},
        "MAX_GRAD_NORM": {"values": [0.5]},
        "ENV_NAME": {"values": ["AlphaTradeMM"]},
        "ANNEAL_LR": {"values": [True]},
        "DEBUG": {"values": [True]},
        "VERBOSE": {"values": [False]},
        "ACTION_TYPE": {"values": ["pure"]},
        "WINDOW_INDEX": {"values": [-1]},
        "EPISODE_TIME": {"values": [60*10]},
        "DATA_TYPE": {"values": ["fixed_time"]},

        "ATFOLDER": {"values": [ATFolder]},
        "ENV_CONFIG": {"values": env_config_hps},
        "BASELINE_ENV_CONFIG": {"values": baseline_env_config_hps},
        "BASELINE_FIXED_ACTION": {"values": [7]},
        "ENVID":{"values":[1]}
    }    

    sweep_config={
        "method": "grid",
        "parameters": training_parameters
    }

    def sweep_fun():
        run = wandb.init(
            project="Alphatrade_Sweeps",
            save_code=True,  # 
        )
        params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        print(f"Results will be saved to {params_file_name}")
        # +++++ Single GPU +++++
        rng = jax.random.PRNGKey(0)
        train_jit = jax.jit(make_train(wandb.config))
        # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
        out = train_jit(rng)
        train_state = out['runner_state'][0] # runner_state.train_state
        params = train_state.params
    
        # Save the params to a file using flax.serialization.to_bytes
        with open(params_file_name, 'wb') as f:
            f.write(flax.serialization.to_bytes(params))
            print(f"params saved")

        # Load the params from the file using flax.serialization.from_bytes
        # with open(params_file_name, 'rb') as f:
        #     restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        #     print(f"params restored")

        run.finish()

    sweep_id = wandb.sweep(sweep=sweep_config, project="MM_RNN_action_space_skewing_2")
    wandb.agent(sweep_id, function=sweep_fun, count=500)


    sys.exit(0)


    # rng = jax.random.PRNGKey(0)
    # for r in ["pnl"]: 
    #     if wandbOn:
    #         run = wandb.init(
    #             project="AlphaTradeJAX_Train",
    #             config=config,
    #             save_code=True,  # 
    #         )
    #         params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
    #     else:
    #         params_file_name = f'params_file_{timestamp}'

    #     print(f"Results will be saved to {params_file_name}")
    #     config["ENV_CONFIG"]=dataclasses.replace(config["ENV_CONFIG"],reward_space=r)
    #     train_jit = jax.jit(make_train(config))
    #     out = train_jit(rng)
  
    #     # +++++ Single GPU +++++

    #     # # +++++ Multiple GPUs +++++
    #     # num_devices = 4F
    #     # rng = jax.random.PRNGKey(30)
    #     # rngs = jax.random.split(rng, num_devices)
    #     # train_fn = lambda rng: make_train(ppo_config)(rng)
    #     # start=time.time()
    #     # out = jax.pmap(train_fn)(rngs)
    #     # print("Time: ", time.time()-start)
    #     # # +++++ Multiple GPUs +++++
    
    
    #     # # ---------- Save Output ----------
    #     train_state = out['runner_state'][0] # runner_state.train_state
    #     params = train_state.params
    


    #     params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'

    #     # Save the params to a file using flax.serialization.to_bytes
    #     with open(params_file_name, 'wb') as f:
    #         f.write(flax.serialization.to_bytes(params))
    #         print(f"params saved")

    #     # Load the params from the file using flax.serialization.from_bytes
    #     with open(params_file_name, 'rb') as f:
    #         restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
    #         print(f"params restored")

    #     if wandbOn:
    #         run.finish()