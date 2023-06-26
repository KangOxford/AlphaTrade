import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
sys.path.append('../purejaxrl')

from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import time
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

#Code snippet to disable all jitting.
#from jax import config
#config.update("jax_disable_jit", True)

print("#"*20+"Output Log File"+"#"*20,file=open('output.txt','w'))

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1] 
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'
    print("AlphaTrade folder:",ATFolder)

    rng = jax.random.PRNGKey(0)
    def step_thru_env(rng):
        env=ExecutionEnv(ATFolder,'buy')
        env_params=env.default_params
        env = LogWrapper(env)
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, 0.99)
        vmap_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
        vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        vmap_act_sample=jax.jit(jax.vmap(env.action_space().sample, in_axes=(0)))

        num_envs = 2000
        vmap_keys = jax.random.split(rng, num_envs)

        obs, state = vmap_reset(vmap_keys, env_params)


        def step_wrap_vmap(runner_state,unused):
            jax.debug.print('step')
            env_state, obsv, done, rng=runner_state
            rng,_rng=jax.random.split(rng)
            vmap_keys = jax.random.split(_rng, num_envs)
            test_actions=vmap_act_sample(vmap_keys).astype(jnp.float32)
            obsv,env_state,reward,done,info=vmap_step(vmap_keys, env_state, test_actions, env_params)
            runner_state = (env_state, obsv, done, rng)
            return runner_state,None
    
        r_state=(state, obs, jnp.zeros((num_envs), dtype=bool), rng)

        def scan_func_vmap(r_state):
            r_state,_=jax.lax.scan(step_wrap_vmap,r_state,None,10)
            return r_state
        
        r_state=jax.jit(scan_func_vmap)(r_state)
        return r_state

    out=jax.jit(step_thru_env)(rng)


    


    
    

    

    

    