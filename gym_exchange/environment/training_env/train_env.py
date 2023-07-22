# from copy import deepcopy
# from typing import List, Any, Type, Optional, Union, Callable, Sequence
#
# import gym
import numpy as np
# from stable_baselines3.common.vec_env import VecEnv
# from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices
#
#
#
# class StableBaselinesTradingEnvironment(VecEnv):
#     pass
from gym_exchange import Config
import sys; sys.path.append('/Users/kang/AlphaTrade/')
from gym_exchange.environment.basic_env.basic_env import BasicEnv
from gym_exchange.environment.base_env.base_env import BaseEnv
from gym_exchange.environment.timewindow_env.timewindow_env import TimewindowEnv
from stable_baselines3.common.env_checker import check_env
import time

# *************************** 2 *************************** #
class TrainEnv(TimewindowEnv):

    # ========================== 03 ==========================
    def state(self, action):
        action[0] = 1 # 1 means sell stocks, 0 means buy stocks
        # action[2] = 0 # passive orders
        state = super().state(action)
        return state

if __name__ == "__main__":
    import numpy as np
    arr = np.array([
        [1,Config.quantity_size_one_side,0],
    ])
    # arr = np.array([
    #     [1,2,0],
    #     # [1,1,0],
    # ])
    # arr = np.array([
    #     [0,1,0],
    #     [0,1,0]
    # ])
    arr = np.repeat(arr, 2000, axis=0)
    arr[::2,2] = 1
    arr[1::2,2] = 0
    env = TrainEnv()

    # check_env(env) #$

    init_state = env.reset();print("="*20+" ENV RESTED "+"="*20)
    sum_reward = []
    # state, reward, done, info = env.step([0,1,0])# for testing
    # state, reward, done, info = env.step([1,1,0])# for testing
    startTime = time.time()
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        # print(f"reward: {reward}") #$
        # print(f"info: {info}") #$
        sum_reward += [reward]
        # env.render()
        if done:
            env.reset()
            break #$
    # print(f"sum_reward:{sum(sum_reward)}")
    print(f"time: {time.time() - startTime} for {i+1} steps, with average {(time.time() - startTime)/(i+1)}")
    print(f"sum_reward:{np.sum(sum_reward)}, mean_reward:{np.mean(sum_reward)}, std_reward:{np.std(sum_reward)}")
