import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"   
import os
from collections import defaultdict
import time

import gym
import stable_baselines3
# from sbx import PPO
import numpy as np
import jax.random as random
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp

from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_env

from tools.ppo_jit import PPO
from tools.transformer_jit import TransformerPolicy

TF_CPP_MIN_LOG_LEVEL = 0

cfg.ENV.WALKER_DIR = 'data/train_mutate_1000'
cfg.MODEL.MAX_JOINTS = 16
cfg.MODEL.MAX_LIMBS = 12
cfg.TERMINATE_ON_FALL = False

class SimpleEnv(gym.Env):

    def __init__(self, agent):
        self.env = make_env(cfg.ENV_NAME, 0, 0, xml_file=agent)()
        self.observation_space = self.env.observation_space["proprioceptive"]
        self.action_space = self.env.action_space
    
    def reset(self):
        obs_dict = self.env.reset()
        self.mask = obs_dict['obs_padding_mask']
        return obs_dict["proprioceptive"]

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        info['mask'] = obs_dict['obs_padding_mask']
        return obs_dict["proprioceptive"], reward, done, info


agent = os.listdir(cfg.ENV.WALKER_DIR + '/xml')[0][:-4]
env = SimpleEnv(agent)

# policy_kwargs = dict(
#     net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024])
# )
# model = PPO(TransformerPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
model = PPO(TransformerPolicy, env, verbose=1)
rng = jax.random.PRNGKey(0)
params = model.policy.actor.init(rng, jnp.ones([1, 12, 52]), jnp.ones([1, 2, 12, 12]))
params = model.policy.vf.init(rng, jnp.ones([1, 12, 52]), jnp.ones([1, 2, 12, 12]))
f_params = freeze(params)
# model.learn(total_timesteps=10_000, progress_bar=True)

# simulate rollout
buffer = defaultdict(list)
# N_env = 256
# N_step = 2560
N_env = 256
N_step = 20
obs = env.reset()
for i in range(N_step):
    if (i % 100 == 0):
        print (i)
    mask = env.mask
    batch_obs = np.stack([obs.reshape(12, -1) for _ in range(N_env)]) + np.random.randn(N_env, 12, 52)
    batch_mask = np.tile(mask, (N_env, 2, 12, 1))
    actions, log_probs, values = model.policy.predict_all(batch_obs, batch_mask, random.PRNGKey(0))
    obs, reward, done, info = env.step(actions.reshape(N_env, -1)[0])
    buffer['obs'].append(batch_obs)
    buffer['action'].append(actions)
    buffer['advantage'].append(np.ones([N_env, 1]) * reward)
    buffer['return'].append(np.ones([N_env, 1]) * reward)
    buffer['old_log_prob'].append(log_probs)
    buffer['mask'].append(batch_mask)

# simulate PPO training
training_time = 0.
for key in buffer:
    buffer[key] = np.concatenate(buffer[key], axis=0)
    print (buffer[key].shape)
for epoch in range(8):
    for batch_id in range(int(N_env * N_step / 5120)):
        batch_idx = list(range(batch_id * 5120, (batch_id + 1) * 5120))
        batch_obs = buffer['obs'][batch_idx]
        batch_action = buffer['action'][batch_idx]
        batch_adv = buffer['advantage'][batch_idx]
        batch_return = buffer['return'][batch_idx]
        batch_old_log_prob = buffer['old_log_prob'][batch_idx]
        batch_mask = buffer['mask'][batch_idx]
        start = time.time()
        model._one_update(
            model.policy.actor_state, 
            model.policy.vf_state, 
            batch_obs, 
            batch_action, 
            batch_adv, 
            batch_return, 
            batch_old_log_prob, 
            batch_mask, 
            0.2, 
            0., 
            0.5, 
        )
        end = time.time()
        training_time += (end - start)
print (training_time)
