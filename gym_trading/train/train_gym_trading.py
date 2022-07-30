# %% ==========================================================================

# clear warnings
import warnings
warnings.filterwarnings("ignore")


import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
Flow = ExternalData.get_sample_order_book_data()

# env = BaseEnv(Flow)
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
# env = gym.DummyVecEnv([lambda: gym.make("GymTrading-v1",Flow = Flow)])


check_env(env)

# %%
model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=int(1e6)) ## setiting for Console 65
model.learn(total_timesteps=int(1e5))

# model.save("gym_trading-v1") # del model 
# model = PPO.load("gym_trading-v1")
# %% =============================================================================
import time
start = time.time()
obs = env.reset()
done = False
running_reward = 0
for i in range(int(1e6)):
    if i//int(1e5) == i/int(1e5):
        print("Epoch {}, training time {}".format(i,time.time()-start))
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    running_reward += reward
    if done:
        running_reward += reward
        obs = env.reset()
        break 








