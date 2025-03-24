import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
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

from purejaxrl.purejaxrl.wrappers import LogWrapper, FlattenObservationWrapper

from gymnax_exchange.jaxen.jaxen_config import EnvironmentConfig
import gymnax_exchange.jaxrl.training_config as tcfg

wandbOn = True # False
if wandbOn:
    import wandb

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


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
        rewardLambda=config["REWARD_LAMBDA"],
        ep_type=config["DATA_TYPE"],
    )

    eval_env=MarketMakingEnv(
        env_config,
        key_reset,
        alphatradePath=config["ATFOLDER"]+"/val",
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        rewardLambda=config["REWARD_LAMBDA"],
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
         
        network = ActorCritic(
            env.action_space(env_params).n
           , activation=config["ACTIVATION"] #
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
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
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                pi, value = network.apply(train_state.params, last_obs)
                rng, _rng = jax.random.split(rng)

                action = pi.sample(seed=_rng)
                #jax.debug.print("action:{}",action)
                log_prob = pi.log_prob(action)
                #jax.debug.print("log_prob:{}",log_prob)
                # Track actions over time
            
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
                            # Debugging: Print Inventory
               # def print_inventory(info):
                #    inventory = info["inventory"]
                 #   print(f"Inventory at step: {inventory}")
                #jax.debug.print("reward:{}",reward)
                #jax.debug.callback(print_inventory, info)  # Ensures printing during execution
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                        #jax.debug.print("ratio:{}",ratio)
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
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            def _eval_step(eval_runner_state, unused):
                train_state, eval_env_state, last_obs, rng = eval_runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                #jax.debug.print("action:{}",action)
                log_prob = pi.log_prob(action)
                #jax.debug.print("log_prob:{}",log_prob)
                # Track actions over time
                log_prob = pi.log_prob(action)

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
                    done, action, value, reward, log_prob, last_obs, info
                )
                eval_runner_state = (train_state, eval_env_state, obsv, rng)
                return eval_runner_state, transition

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params)
            
            eval_runner_state = (
            train_state,
            env_state,
            obsv,
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

            runner_state = (train_state, env_state, last_obs, rng,update_count+1)
            return runner_state, (metric,eval_metric)

        rng, _rng = jax.random.split(rng)
        update_count=0
        runner_state = (train_state, env_state, obsv, _rng,update_count)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    try:
        ATFolder = sys.argv[1]
        print("ATFFolder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"
    training_parameters = tcfg.nn_training_parameters
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

    # sweep_id = wandb.sweep(sweep=sweep_config, project="TEST_SWEEPS")
    wandb.agent("0m1aiyxt", function=sweep_fun, count=40,project="TEST_SWEEPS")
