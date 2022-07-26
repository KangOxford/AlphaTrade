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

# clear warnings
import warnings
warnings.filterwarnings("ignore")


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
# model.learn(total_timesteps=int(1e6)) ## setiting for Console 65
model.learn(total_timesteps=int(1e5))
# %%
model.save("gym_trading-v1")
del model 
model = PPO.load("gym_trading-v1")
# %%
obs = env.reset()
running_reward = 0
for i in range(int(1e6)):
    if i//int(1e5) == i/int(1e5):
        print("Epoch ",i)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    running_reward += reward
    if done:
      obs = env.reset()
env.close()







