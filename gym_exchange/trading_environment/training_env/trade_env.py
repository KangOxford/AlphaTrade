# from copy import deepcopy
# from typing import List, Any, Type, Optional, Union, Callable, Sequence
#
# import gym
# import numpy as np
# from stable_baselines3.common.vec_env import VecEnv
# from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices
#
#
#
# class StableBaselinesTradingEnvironment(VecEnv):
#     pass


import sys; sys.path.append('/Users/kang/AlphaTrade/')
from gym_exchange.trading_environment.basic_env.basic_env import BasicEnv


# *************************** 2 *************************** #
class TradeEnv(BasicEnv):

    # ========================== 03 ==========================
    def state(self, action):
        action[0] = 1
        state = super().state(action)
        return state

if __name__ == "__main__":
    import numpy as np
    arr = np.array([
        [1,1,0],
        [1,1,0]
    ])
    # arr = np.array([
    #     [0,1,0],
    #     [0,1,0]
    # ])
    arr = np.repeat(arr, 2000, axis=0)
    env = BasicEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    sum_reward = 0
    state, reward, done, info = env.step([0,1,0])# for testing
    # state, reward, done, info = env.step([1,1,0])# for testing
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        print(f"reward: {reward}") #$
        print(f"info: {info}") #$
        sum_reward += reward
        # env.render()
        if done:
            env.reset()
            break #$
    print(sum_reward)
