import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import time

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

    enable_vmap=False 


    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,'buy')
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state,file=open('output.txt','a'))
    print("Observation after reset: \n",obs,file=open('output.txt','a'))
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)


    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for 2nd reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)

    #print(job.get_data_messages(env_params.message_data,state.window_index,state.step_counter+1))

    #print(env.action_space().sample(key_policy))
    #print(env.state_space(env_params).sample(key_policy))


    """test_action={"sides":jnp.array([1,1,1]),
                 "quantities":jnp.array([10,10,10]),
                 "prices":jnp.array([2154900,2154000,2153900]),
                 }"""
    
    test_action=env.action_space().sample(key_policy)
    print("Sampled actions are: ",test_action)

    start=time.time()
    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    print("State after one step: \n",state,done,file=open('output.txt','a'))
    print("Done after one step: ",done,file=open('output.txt','a'))
    print("Observation after one step: \n",obs,file=open('output.txt','a'))
    print("Time for one step: \n",time.time()-start)

    test_action=env.action_space().sample(key_policy)
    print("Sampled actions are: \n",test_action)
    
    start=time.time()
    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    print("Time for 2nd step: \n",time.time()-start)
    with jnp.printoptions(threshold=jnp.inf):
        print("State after 2 steps: \n",state,done,file=open('output.txt','a'))
        print("Done after 2 steps: ",done,file=open('output.txt','a'))
        print("Observation after 2 steps: \n",obs,file=open('output.txt','a'))
    
    
    start=time.time()
    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    print("Time for 3rd step: \n",time.time()-start)
    with jnp.printoptions(threshold=jnp.inf):
        print("State after 3 steps: \n",state,done,file=open('output.txt','a'))
        print("Done after 3 steps: ",done,file=open('output.txt','a'))
        print("Observation after 3 steps: \n",obs,file=open('output.txt','a'))
    
    #comment



    ####### Testing the vmap abilities ########
    

    if enable_vmap:
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 1000
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
