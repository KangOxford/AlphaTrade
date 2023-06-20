# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import time

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
# ============== testing scripts ===============



from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxen.exec_env import EnvParams


    

class RedisualExecutionEnv(ExecutionEnv):
    def __init__(self,alphatradePath,task):
        super().__init__(alphatradePath,task)

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Dict:
        """Action space of the environment."""
        return spaces.Dict({
                "aggressive": spaces.Discrete(100),
                "mid": spaces.Discrete(100),
                "passive": spaces.Discrete(100),
                "ppassive": spaces.Discrete(100),
            })

    # #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    # def observation_space(self, params: EnvParams):
    #     """Observation space of the environment."""
    #     space = spaces.Box(-10000,99999999,(608,),dtype=jnp.int32)
    #     return space

    # #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    # #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    # def state_space(self, params: EnvParams) -> spaces.Dict:
    #     """State space of the environment."""
    #     return spaces.Dict(
    #         {
    #             "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
    #             "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
    #             "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int32),
    #             "time": spaces.Discrete(params.max_steps_in_episode),
    #         }
    #     )

# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=DiscExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)

    for i in range(1,100):
        test_action=env.action_space().sample(key_policy)
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        print(f"Time for {i} step: \n",time.time()-start)

    # ####### Testing the vmap abilities ########
    
    # enable_vmap=False
    # if enable_vmap:
    #     vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    #     vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    #     vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

    #     num_envs = 10
    #     vmap_keys = jax.random.split(rng, num_envs)

    #     test_actions=vmap_act_sample(vmap_keys)
    #     print(test_actions)

    #     start=time.time()
    #     obs, state = vmap_reset(vmap_keys, env_params)
    #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

    #     start=time.time()
    #     n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
    #     print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
