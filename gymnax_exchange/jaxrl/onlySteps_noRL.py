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
from gymnax_exchange.jaxen.exec_env_old import ExecutionEnv


#Code snippet to disable all jitting.
from jax import config
# config.update("jax_disable_jit", False)
#config.update("jax_disable_jit", True)

config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.






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
    #if config["NORMALIZE_ENV"]:
         #env = NormalizeVecObservation(env)
         #env = NormalizeVecReward(env, config["GAMMA"])
    

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
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # jax.debug.breakpoint()
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # jax.debug.breakpoint()
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                env_state, last_obs, last_done, rng = runner_state
                rng, _rng = jax.random.split(rng)

                rng_action=jax.random.split(_rng, config["NUM_ENVS"])
                action = jax.vmap(env.action_space().sample, in_axes=(0))(rng_action)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                transition = Transition(
                    done_step, action, 0, reward_step, 0, last_obs, info_step
                )
                runner_state = (env_state_step, obsv_step, done_step, rng)
                return runner_state,transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            env_state, last_obs, last_done, rng = runner_state

            # CALCULATE ADVANTAGE
            metric = traj_batch.info

            # UPDATE NETWORK
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (env_state, last_obs, last_done, rng)
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

    ppo_config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1000,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
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
        "NORMALIZE_ENV": False,
        "ATFOLDER": ATFolder,
        "TASKSIDE":'buy'
    }

    rng = jax.random.PRNGKey(30)
    # jax.debug.breakpoint()
    train_jit = jax.jit(make_train(ppo_config))

    if ppo_config["DEBUG"]:
        pass
        #chexify the function
        #NOTE: use chex.asserts inside the code, under a if DEBUG. 

    # train = make_train(ppo_config)
    # jax.debug.breakpoint()
    start=time.time()
    out = train_jit(rng)
    print("Time: ", time.time()-start)

