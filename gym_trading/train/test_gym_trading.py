# %% =============================================================================
import time
import warnings
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()

env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
model = PPO.load("gym_trading-v1")

start = time.time()
obs = env.reset()
done = False
running_reward = 0
for i in range(int(1e8)):
    if i//int(1e5) == i/int(1e5):
        print("Epoch {}, training time {}".format(i,time.time()-start))
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    running_reward += reward
    if done:
        running_reward += reward
        obs = env.reset()
        break 
