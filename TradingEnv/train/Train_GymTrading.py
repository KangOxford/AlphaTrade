from TradingEnv.envs.gym_trading import *
import torch


max_episode = int(1e6)


# >>> 01.Initializes a trading environment.
env = BaseEnvironment()

for i_episode in range(1, max_episode + 1):
    episode_reward = 0
    observation = env.reset()
    
