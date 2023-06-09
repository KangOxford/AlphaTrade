import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
from gymnax_exchange.jaxen.base_env import BaseLOBEnv



print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())


rng = jax.random.PRNGKey(0)
rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

env=BaseLOBEnv()    
env_params=env.default_params
print(env_params)

obs,state=env.reset(key_reset,env_params)


#print(env.action_space().sample(key_policy))
#print(env.state_space(env_params).sample(key_policy))

obs,state,reward,done,info=env.step(key_step, state,env.action_space().sample(key_policy), env_params)
print(obs)

"""obs,state,reward,done,info=env.step(key_step, state,env.action_space().sample(key_policy), env_params)
print(done)"""