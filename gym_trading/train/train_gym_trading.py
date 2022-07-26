# =============================================================================
# # %% 
# # =============================================================================
# import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # ----------------------------------------------------------------------------
# from gym_trading.envs import base_environment
# # =============================================================================
# 
# max_episode = int(1e6)
# 
# 
# # >>> 01.Initializes a trading environment.
# env = base_environment.BaseEnv()
# 
# for i_episode in range(1, max_episode + 1):
#     episode_reward = 0
#     observation = env.reset()
#     running_reward = torch.tensor(0.0)
#     for t in range(env.num_steps):
#         pass
# 
# =============================================================================
# import time

# %%
import gym

from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env

# env = gym.make("CartPole-v1")
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
Flow = ExternalData.get_sample_order_book_data()

# env = gym.make("GymTrading-v1",Flow) ## TODO
env = BaseEnv(Flow)


check_env(env)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    # time.sleep(1)
    if done:
      obs = env.reset()
env.close()