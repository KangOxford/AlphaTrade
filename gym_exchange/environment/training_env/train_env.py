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
import sys; sys.path.append('../AlphaTrade/')
from gym_exchange.environment.basic_env.basic_env import BasicEnv
from gym_exchange.environment.base_env.base_env import BaseEnv
from gym_exchange.environment.timewindow_env.timewindow_env import TimewindowEnv
from stable_baselines3.common.env_checker import check_env


# *************************** 2 *************************** #
class TrainEnv(TimewindowEnv):

    # ========================== 03 ==========================
    def state(self, action):
        action[0] = 1 # 1 means sell stocks, 0 means buy stocks, "Execution-2FreeDegrees"
        # action[2] = 0 # passive orders
        # print(f"{action[0]} {action[1]} {action[2]}")  #$ less memory use
        state = super().state(action)
        return state




# from gym.envs.registration import register
# from gym_exchange import Config
# register(
#     id = "GymExchange-v1",
#     # id = "GymExchange-v1",
#     # path to the class for creating the env
#     # Note: entry_point also accept a class as input (and not only a string)
#     entry_point="gym_exchange.environment.training_env.train_env:TrainEnv",
#     # Max number of steps per episode, using a `TimeLimitWrapper`
#     max_episode_steps=Config.max_horizon,
#     )


if __name__ == "__main__":
    '''
    import numpy as np
    arr = np.array([
        [1,Config.quantity_size_negative_side,0],
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
    '''

    # arrs = np.load('/Users/kang/AlphaTrade/gym_exchange/outputs/action_array_25_apr.npy')
    # arr = arrs[0]
    # sum_reward:0.009999165185103229, mean_reward:2.0406459561435162e-05, std_reward:0.0003187590918861088

    # arr = arrs[1]
    # sum_reward:0.009999948073714555, mean_reward:2.0408057293295012e-05, std_reward:0.0003187840724970272

    # arr = arrs[2]
    # sum_reward: 0.015028457601588784, mean_reward: 3.067032163589548e-05, std_reward: 0.0005051537908412706
    # --------------------= > 371 <= --------------------
    # 1 - 16
    # 1
    # reward: 0.010021421589910963
    # info: {}

    import numpy as np
    arr = np.array([
        [1, 0, 0],
        # [1, 0, 1],

    ])
    arr = np.tile(arr, (20000,1))

    env = TrainEnv()
    # check_env(env) #$
    init_state = env.reset();print("="*20+" ENV RESTED "+"="*20)
    sum_reward = []
    # state, reward, done, info = env.step([0,1,0])# for testing
    # state, reward, done, info = env.step([1,1,0])# for testing
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        print(f"reward: {reward}") #$
        print(f"info: {info}") #$
        sum_reward += [reward]
        # env.render()
        if done:
            env.reset()
            break #$
    # print(f"sum_reward:{sum(sum_reward)}")
    print(f"sum_reward={np.sum(sum_reward)}; mean_reward={np.mean(sum_reward)}; std_reward={np.std(sum_reward)}")
