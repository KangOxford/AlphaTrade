import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())


#from jax import config
#config.update("jax_disable_jit", True)


if __name__ == "__main__":
    ATFolder = sys.argv[1]
    print("AlphaTrade folder:",ATFolder)

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=BaseLOBEnv(ATFolder) 
    env_params=env.default_params

    obs,state=env.reset(key_reset,env_params)
    print(obs)
    print(state)

    print(env_params.message_data.shape, env_params.book_data.shape)

    #print(job.get_data_messages(env_params.message_data,state.window_index,state.step_counter+1))

    #print(env.action_space().sample(key_policy))
    #print(env.state_space(env_params).sample(key_policy))

    print(job.get_data_messages(env_params.message_data,state.window_index,state.step_counter))

    test_action={"sides":jnp.array([1,1,1]),
                 "quantities":jnp.array([10,10,10]),
                 "prices":jnp.array([2154900,2154000,2153900]),
                 }

    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    #env.action_space().sample(key_policy)
    print("State after step: \n",state)

    #obs,state,reward,done,info=env.step(key_step, state,env.action_space().sample(key_policy), env_params)
    #print(state,done)
    #comment