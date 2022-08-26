# %% ==========================================================================
# clear warnings
from email import policy
import warnings
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData

warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
check_env(env)

# num_cpu = 10 
# venv = DummyVecEnv([lambda: gym.make("GymTrading-v1",Flow = Flow)] * num_cpu)
# monitor_venv = DummyVecEnv([lambda: Monitor(gym.make("GymTrading-v1",Flow = Flow))] * num_cpu)
# %%
# model = SAC("MultiInputPolicy", 
#             monitor_venv, 
#             verbose=1, 
#             tensorboard_log="/Users/kang/GitHub/NeuralLOB/ppo_gymtrading_tensorboard9/")

model = PPO("MultiInputPolicy", 
            env, 
            # monitor_venv, 
            verbose=1, 
            tensorboard_log="/Users/kang/GitHub/NeuralLOB/ppo_gymtrading_tensorboard21/")

# %time model.learn(total_timesteps=int(1e7), n_eval_episodes = int(1e5))
# model.learn(total_timesteps=int(3e6), n_eval_episodes = int(1e5))
model.learn(total_timesteps=int(1e10), tb_log_name="SparseReward_SAC_ShortHorizon128")
model.save("gym_trading-v1") 

# %% test the train result
import time
start = time.time()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO

for i in range(int(1e3)):
    obs = env.reset()
    # ;print(obs)
    done = False
    running_reward = 0
    for i in range(int(1e8)):
        if i//int(1e5) == i/int(1e5):
            print("Epoch {}, training time {}".format(i,time.time()-start))
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        running_reward += reward 
        if done:
            running_reward += reward
            obs = env.reset()
            break 


# %% get the result
def get_result(name):
    import re  
    fp = open("/Users/kang/Desktop/FINALREMAINING(RL)")   
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
print("Diff, ",np.mean(Diff))
print("Init, ",np.mean(Init))
print("RL, ",np.mean(RL))


# %%

import signal

def handler(signum, frame):
    print("Forever is over!")
    raise Exception("end of time")
signal.signal(signal.SIGALRM, handler)
signal.alarm(10)
def loop_forever():
    import time
    while 1:
        print("sec")
        time.sleep(1)
         
try:
    loop_forever()
except Exception, exc: 
    print(exc)

# %%



    
def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    # from __future__ import print_function
    import sys
    import threading
    from time import sleep
    try:
        import thread
    except ImportError:
        import _thread as thread
    def quit_function(fn_name):
        # print to stderr, unbuffered in Python 2.
        print('{0} took too long'.format(fn_name), file=sys.stderr)
        sys.stderr.flush() # Python 3 stderr is likely buffered.
        thread.interrupt_main() # raises KeyboardInterrupt
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

@exit_after(5)
def countdown(n):
    print('countdown started', flush=True)
    for i in range(n, -1, -1):
        print(i, end=', ', flush=True)
        sleep(1)
    print('countdown finished')
    
    
countdown(10)
    
    
    
    
    
    
    
    
    