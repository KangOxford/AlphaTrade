# IPython log file

get_ipython().run_line_magic('logstart', '-o Out[n]')
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
    fp = open("/Users/kang/Desktop/FINALREMAINING(RL)")   
    # fp = open("/Users/kang/GitHub/NeuralLOB/gym_trading/train/test.log")   
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
#[Out]# 79.42032647084883
np.mean(Diff)
#[Out]# -1.4351415405102463
np.mean(RL)
#[Out]# 77.98518493033858


# runfile('/Users/kang/GitHub/NeuralLOB/gym_trading/train/train_gym_trading.py', wdir='/Users/kang/GitHub/NeuralLOB/gym_trading/train')
# runfile('/Users/kang/GitHub/NeuralLOB/gym_trading/train/train_gym_trading.py', wdir='/Users/kang/GitHub/NeuralLOB/gym_trading/train')
