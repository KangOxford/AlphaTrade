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


import logging
logging.getLogger("jax").setLevel(logging.INFO)


#Code snippet to disable all jitting.
from jax import config
# config.update("jax_disable_jit", False)
#config.update("jax_disable_jit", True)


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
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"])
    env_params = env.default_params
    env = LogWrapper(env)
    env=ClipAction(env,low=0, high=100)
    
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
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )


        
        # jax.debug.breakpoint()
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                env_state, last_obs, last_done, rng = runner_state
                rng, _rng = jax.random.split(rng)


                action=jnp.zeros((1,2,4,),dtype=jnp.int32)
                jax.debug.print('Action: {}',action)
                # Guess to be 4 actions. caused by ppo_rnn is continuous. But our action space is discrete
                # jax.debug.breakpoint()
                action =  action.squeeze(0)

                jax.debug.print('Action: {}',action)

                if config.get("DEBUG"):
                    jax.debug.print("About to take step")
                    #jax.debug.breakpoint()
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, 0, reward, 0, last_obs, info
                )
                runner_state = (env_state, obsv, done, rng)
                if config.get("DEBUG"):
                    pass
                    #jax.debug.breakpoint()

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            jax.debug.print("Ran {} steps",config["NUM_STEPS"])

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
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
    print("AlphaTrade folder:",ATFolder)

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 2,
        "NUM_STEPS": 2,
        "TOTAL_TIMESTEPS": 4,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "NORMALIZE_ENV": False,
        "ATFOLDER": ATFolder,
        "TASKSIDE":'buy'
    }

    rng = jax.random.PRNGKey(30)
    # jax.debug.breakpoint()
    train_jit = jax.jit(make_train(config))
    train = make_train(config)
    # jax.debug.breakpoint()
    out = train_jit(rng)
    #out = train(rng)
    jax.debug.breakpoint()
