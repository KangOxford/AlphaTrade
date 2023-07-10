import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
from gymnax_exchange.jaxen.exec_env import ExecutionEnv


#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
#config.update("jax_disable_jit", True)

config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.

import wandb


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

        #pi = distrax.Categorical(logits=actor_mean)
        #Old version^^

        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        #New version ^^

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


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["DEBUG"])
    env_params = env.default_params
    env = LogWrapper(env)
    
    #FIXME : Uncomment normalisation.
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])
    

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
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
        
        # jax.debug.breakpoint()
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            """
            Pseudocode
            if i%50 ==0:
                envparam.message_data.reshuffled()
                envparam.book_data.reshuffled()
            
            reshuffled():
                0-30,30-60.... --> 5-35,35-65
                                    ...or 40-70,70-100
            
            """

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                # jax.debug.print('Step')
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng) # 4*1, should be (4*4: 4actions * 4envs)
                # Guess to be 4 actions. caused by ppo_rnn is continuous. But our action space is discrete
                log_prob = pi.log_prob(action)

                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done_step, action, value, reward_step, log_prob, last_obs, info_step
                )
                runner_state = (train_state, env_state_step, obsv_step, done_step, hstate, rng)
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
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

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
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    
                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]

                    
                    for t in range(len(timesteps)):
                        # print(
                        # f"global step={timesteps[t]}, episodic return={return_values[t]}, episodic revenue={revenues[t]}"
                        # # f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        # )    
                        # print(
                        #     f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        # )      
                        wandb.log(
                            {
                                "global_step": timesteps[t],
                                "episodic_return": return_values[t],
                                "episodic_revenue": revenues[t],
                                "quant_executed":quant_executed[t],
                            }
                        )                     

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1] 
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/home/duser/AlphaTrade'
    print("AlphaTrade folder:",ATFolder)

    ppo_config = {
        "LR": 2.5e-4,
        #"NUM_ENVS": 1,
        "NUM_ENVS": 1000,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        #"NUM_MINIBATCHES": 1,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "alphatradeExec-v0",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "NORMALIZE_ENV": True,
        "ATFOLDER": ATFolder,
        "TASKSIDE":'buy'
    }
    
    run = wandb.init(
        project="AlphaTradeJAX",
        config=ppo_config,
        # sync_tensorboard=True,  # auto-upload  tensorboard metrics
        save_code=True,  # optional
    )
    
    
    # +++++ Single GPU +++++
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(ppo_config))
    start=time.time()
    out = train_jit(rng)
    print("Time: ", time.time()-start)
    # +++++ Single GPU +++++

    # # +++++ Multiple GPUs +++++
    # num_devices = 4
    # rng = jax.random.PRNGKey(30)
    # rngs = jax.random.split(rng, num_devices)
    # train_fn = lambda rng: make_train(ppo_config)(rng)
    # start=time.time()
    # out = jax.pmap(train_fn)(rngs)
    # print("Time: ", time.time()-start)
    # # +++++ Multiple GPUs +++++s
    
    

    # '''
    # # ---------- Save Output ----------
    import flax

    train_state = out['runner_state'][0] # runner_state.train_state
    params = train_state.params
    
    import datetime;params_file_name = 'params_file_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Save the params to a file using flax.serialization.to_bytes
    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"pramas saved")

    # Load the params from the file using flax.serialization.from_bytes
    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"pramas restored")
        
    # jax.debug.breakpoint()
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, restored_params))
    # print(">>>")
    # '''
    
    run.finish()
