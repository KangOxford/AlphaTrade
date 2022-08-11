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
model = PPO.load("/Users/kang/GitHub/NeuralLOB/gym_trading-v1")

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

obs = env.reset()
done = False
for i in range(int(1e6)):
    action, _states = model.predict(obs)
    # action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print(">"*20+" timestep: "+str(i))
        env.reset()
        
        
   

def get_result(name):
    import re  
    fp = open("/Users/kang/GitHub/NeuralLOB/gym_trading/train/test.log")   
    lst = []
    for line in fp.readlines():
        try:
            m = re.search(name, line)
            result = line[m.end()+3:-1]
            lst.append(float(result))
        except:
            pass
    fp.close()
    return lst

Init = get_result('Init')
Diff = get_result('Diff')
RL = [x+y for x,y in zip(Init,Diff)]

import numpy as np
np.mean(Init)
np.mean(Diff)
np.mean(RL)
