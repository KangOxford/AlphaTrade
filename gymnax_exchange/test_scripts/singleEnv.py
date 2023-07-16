# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import sys
from re import L
import time 
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
import chex
import faulthandler; faulthandler.enable()
chex.assert_gpu_available(backend=None)
from jax import config # Code snippet to disable all jitting.
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
from gymnax_exchange.jaxen.exec_env import *
import json
# ============== testing scripts ===============

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing'
        # ATFolder = '/homes/80/kang/AlphaTrade/testing_small'
        
    ppo_config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 1,
            "NUM_STEPS": 1,
            "TOTAL_TIMESTEPS": 5e5,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
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

    env=ExecutionEnv(ATFolder,ppo_config['TASKSIDE'])
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    rng = jax.random.PRNGKey(3)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print(state.window_index)
    print("Time for reset: \n",time.time()-start)
    # print(env_params.message_data.shape, env_params.book_data.shape)
    
    action_arr = jnp.array(
        [[0, 1, 2, 0],
        [2, 0, 0, 0],
        [0, 0, 1, 3],
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 2, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 2, 2, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 2],
        [0, 0, 0, 2],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [2, 0, 0, 0],
        [1, 2, 1, 0],
        [1, 1, 1, 2],
        [0, 0, 1, 0],
        [1, 2, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 2],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 2],
        [1, 0, 3, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 3, 1, 0],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 2],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [2, 1, 1, 0],
        [2, 1, 0, 0],
        [0, 3, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 2, 2],
        [0, 0, 0, 2],
        [0, 2, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 3, 0, 0],
        [2, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [2, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 2],
        [0, 0, 0, 0],
        [0, 1, 3, 0],
        [3, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [2, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 3],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 2, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [2, 0, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 2, 1],
        [2, 0, 2, 2],
        [0, 1, 2, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 2],
        [0, 0, 1, 0],
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 2],
        [0, 0, 0, 0],
        [0, 1, 1, 2],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 2, 1],
        [1, 1, 1, 1],
        [2, 1, 0, 1],
        [0, 1, 0, 0],
        [2, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 2, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 3],
        [1, 3, 1, 2],
        [0, 0, 2, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 3, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]
    );assert state.window_index == 1
    
    action_arr.shape


    
    i = 0
    for i in range(0,10000):
        # ==================== ACTION ====================
        action = action_arr[i,:]
        # ==================== ACTION ====================    
        print(f"-------------\nPPO {i}th actions are: {action} with sum {action.sum()}")
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,action, env_params)
        print(f"Time for {i} step: \n",time.time()-start)
        # done, state.quant_executed
        print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
        if done:
            break
    # print(info)
