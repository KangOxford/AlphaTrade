# from jax import config
# config.update("jax_enable_x64",True)

import sys
import time

import chex
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Any, Dict, NamedTuple, Sequence
import distrax
from gymnax.environments import spaces

sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
#Code snippet to disable all jitting.
from jax import config

from gymnax_exchange.jaxen.exec_env import ExecutionEnv

config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.
np.set_printoptions(suppress=True)

@jax.jit
def hamilton_apportionment_permuted_jax(votes, seats, key):
    init_seats, remainders = jnp.divmod(votes, jnp.sum(votes) / seats) # std_divisor = jnp.sum(votes) / seats
    remaining_seats = jnp.array(seats - init_seats.sum(), dtype=jnp.int32) # in {0,1,2,3}
    def f(carry,x):
        key,init_seats,remainders=carry
        key, subkey = jax.random.split(key)
        chosen_index = jax.random.choice(subkey, remainders.size, p=(remainders == remainders.max())/(remainders == remainders.max()).sum())
        return (key,init_seats.at[chosen_index].add(jnp.where(x < remaining_seats,1,0)),remainders.at[chosen_index].set(0)),x
    (key,init_seats,remainders), x = jax.lax.scan(f,(key,init_seats,remainders),xs=jnp.arange(votes.shape[0]))
    return init_seats.astype(jnp.int32)


def VWAP_Scheduling(state, env, forcasted_volume, key):
    best_asks, best_bids=state.best_asks[:,0], state.best_bids[:,0]
    best_ask_qtys, best_bid_qtys = state.best_asks[:,1], state.best_bids[:,1]
    obs = {
        # "is_buy_task": params.is_buy_task,
        "p_aggr": best_bids if env.task=='sell' else best_asks,
        "q_aggr": best_bid_qtys if env.task=='sell' else best_ask_qtys, 
        "p_pass": best_asks if env.task=='sell' else best_bids,
        "q_pass": best_ask_qtys if env.task=='sell' else best_bid_qtys, 
        "p_mid": (best_asks+best_bids)//2//env.tick_size*env.tick_size, 
        "p_pass2": best_asks+env.tick_size*env.n_ticks_in_book if env.task=='sell' else best_bids-env.tick_size*env.n_ticks_in_book, # second_passives
        "spread": best_asks - best_bids,
        "shallow_imbalance": state.best_asks[:,1]- state.best_bids[:,1],
        "time": state.time,
        "episode_time": state.time - state.init_time,
        "init_price": state.init_price,
        "task_size": state.task_to_execute,
        "executed_quant": state.quant_executed,
        "step_counter": state.step_counter,
        "max_steps": state.max_steps_in_episode,
    }    
    start_idx_array = env.start_idx_array
    forcasted_volume = hamilton_apportionment_permuted_jax(forcasted_volume, env.task_size, key)
    print(forcasted_volume.sum(),env.task_size)
    
    allocation_array_full = jnp.concatenate([start_idx_array,forcasted_volume.reshape(-1,1)],axis=1)
    allocation_array_breif = allocation_array_full[:,[0,-1]].astype(jnp.int32)
    allocation_array_breif = jnp.concatenate([allocation_array_breif, np.insert(np.diff(allocation_array_breif[:, 0]),0,allocation_array_breif[0, 0]).reshape(-1,1)],axis=1)
    
    lst = []
    key = jax.random.PRNGKey(100)
    for i in range(allocation_array_breif.shape[0]):
        print(i)
        key, subkey = jax.random.split(key)
        lst.append(hamilton_apportionment_permuted_jax(jnp.ones(allocation_array_breif[i,2]), allocation_array_breif[i,1], key))
    allocation_array_final = np.concatenate(lst)
    return allocation_array_final

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay/"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
        
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell",
        "TASK_SIZE": 8000, #100, # 500,
        "WINDOW_INDEX": -1,
        "ACTION_TYPE": "delta", # "pure",
        "REWARD_LAMBDA": 1.0,
        "FORECASTED_VOLUME": jax.random.permutation(jax.random.PRNGKey(0),jnp.arange(1,27)),
        }
    
    
    
    
    
    
    
    
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=ExecutionEnv(ATFolder,"sell",1)
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["WINDOW_INDEX"],config["ACTION_TYPE"],config["TASK_SIZE"],config["REWARD_LAMBDA"])
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
    allocation_array_final = VWAP_Scheduling(state, env, config["FORECASTED_VOLUME"], key_reset)
    # jax.debug.breakpoint()
    # breakpoint()
    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ =  jax.random.split(key_policy, 2)
        key_step, _ =  jax.random.split(key_step, 2)
        # test_action=env.action_space().sample(key_policy)
        test_action=allocation_array_final[state.step_counter] # TODO not sure step_counter or step_counter-1, prefer to be step_counter
        print(state.step_counter)
        # test_action=env.action_space().sample(key_policy)//10 # CAUTION not real action
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        for key, value in info.items():
            print(key, value)
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
            break
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================
        
        
        

    # # ####### Testing the vmap abilities ########
    
    enable_vmap=True
    if enable_vmap:
        # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
