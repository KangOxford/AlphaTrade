import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import os
import sys
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))  # Absolute path to AlphaTrade
sys.path.append('.')
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
from purejaxrl.purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper
from gymnax.environments import spaces


# from jax import config
# config.update("jax_enable_x64",True)
import dataclasses

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import flax.linen as nn
# import numpy as np
import optax
import time
# from flax.linen.initializers import constant, orthogonal
from typing import Optional, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex



sys.path.append(os.path.abspath('/home/duser/AlphaTrade/purejaxrl'))
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))  # Absolute path to AlphaTrade
sys.path.append('.')

#sys.path.append('../AlphaTrade/purejaxrl')
#sys.path.append('../AlphaTrade')
from purejaxrl.purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
from purejaxrl.purejaxrl.experimental.s5.s5 import StackedEncoderModel#, init_S5SSM, make_DPLR_HiPPO
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
#from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
from gymnax_exchange.jaxrl.networks.actorCritic import ActorCriticRNN, ScannedRNN
from gymnax_exchange.jaxrl.networks import actorCriticS5mm
import os
import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
from gymnax_exchange.jaxen.jaxen_config import EnvironmentConfig
import gymnax_exchange.jaxrl.training_config as tcfg


config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)

wandbOn = True # False
if wandbOn:
    import wandb


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):

    env_config=EnvironmentConfig(**config["ENV_CONFIG"])

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
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

    eval_env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=config["REWARD_LAMBDA"],
        episode_time=config["EPISODE_TIME"],
    )

    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=config["REWARD_LAMBDA"],
        episode_time=config["EPISODE_TIME"],
    )
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    eval_env = FlattenObservationWrapper(eval_env)
    eval_env = LogWrapper(eval_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = actorCriticS5mm.ActorCriticS5(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = actorCriticS5mm.ActorCriticS5.initialize_carry(
                    config["NUM_ENVS"], actorCriticS5mm.ssm_size, actorCriticS5mm.n_layers)
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
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
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

            init_hstate = initial_hstate # TBH
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
            obsv, env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)
            initial_hstate = actorCriticS5mm.ActorCriticS5.initialize_carry(
                    config["NUM_ENVS"], actorCriticS5mm.ssm_size, actorCriticS5mm.n_layers)
            
            eval_runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            initial_hstate,
            _rng,
            )
            eval_runner_state, eval_traj_batch = jax.lax.scan(
                _eval_step, eval_runner_state, None, config["NUM_STEPS"]
            )
            eval_metric=eval_traj_batch.info
            if config.get("DEBUG"):
                def callback(info_train,info_eval,update_count):
                    return_values = info_train["returned_episode_returns"][info_train["returned_episode"]]
                    timesteps = info_train["timestep"][info_train["returned_episode"]] * config["NUM_ENVS"]
                    PnL = info_train["total_PnL"]
                    inventories = info_train["inventory"] 
                    buyQuant=info_train["buyQuant"]
                    sellQuant=info_train["sellQuant"]
                    reward=info_train["reward"]
                    other_exec_quants=info_train["other_exec_quants"]
                    reward_eval=info_eval["reward"]
                    PnL_eval=info_eval["total_PnL"]

                    if wandbOn:
                        wandb.log(
                            data={
                                "global_step": jnp.max(timesteps) if timesteps.size > 0 else 0, # timesteps[t],
                                "reward":jnp.mean(reward) if reward.size > 0 else 0,
                                "episodic_return": jnp.mean(return_values) if return_values.size > 0 else 0,  # Handle empty arrays
                                "PnL": jnp.mean(PnL) if PnL.size > 0 else 0,  # Handle empty arrays
                                "PnL_eval": jnp.mean(PnL_eval) if PnL_eval.size > 0 else 0,  # Handle empty arrays
                                "reward_eval":jnp.mean(reward_eval) if reward_eval.size > 0 else 0,
                                "inventory": jnp.mean(inventories) if inventories.size > 0 else 0, 
                                "buyQuant":jnp.mean(buyQuant) if buyQuant.size > 0 else 0,
                                "sellQuant":jnp.mean(sellQuant) if sellQuant.size > 0 else 0,
                                "other_exec_quants":jnp.mean(other_exec_quants) if other_exec_quants.size > 0 else 0,
                                "update_count": update_count,
                            },
                            commit=True
                        )
                    print("Update step is",update_count, "of",config["NUM_UPDATES"])
                    if config["VERBOSE"]:
                        for t in range(len(timesteps)):
                            print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric,eval_metric,update_count)

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

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
    training_parameters = tcfg.acs5_training_parameters
    training_parameters["ATFOLDER"] = {"values" : [ATFolder]}
    training_parameters["ENV_CONFIG"]= {"values" : tcfg.mm_env_config_hps}

    
    sweep_config={
        "method": "grid",
        "parameters": training_parameters
    }

    def sweep_fun():
        run = wandb.init(
            project="Alphatrade_Sweeps",
            save_code=True,  # 
        )
        # Create the 'params' folder if it doesn't already exist
        os.makedirs(f'{wandb.config["ATFOLDER"]}/params', exist_ok=True)
        params_file_name = f'{wandb.config["ATFOLDER"]}/params/params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
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

    sweep_id = wandb.sweep(sweep=sweep_config, project="TEST_SWEEPS")
    wandb.agent(sweep_id, function=sweep_fun, count=10)
