from gym_exchange import Config
from gym_exchange.environment.base_env.assets.action import Action

from gym_exchange.environment.base_env.interface_env import State  # types
# from gym_exchange.environment.env_interface import State, Observation # types
from gym_exchange.exchange.timewindow_exchange import TimewindowExchange
from gym_exchange.environment.basic_env.basic_env import BasicEnv
from gym_exchange.environment.base_env.base_env import BaseEnv
from copy import deepcopy
# *************************** 2 *************************** #
class TimewindowEnv(locals()[Config.train_env]):
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        self.exchange = TimewindowExchange()

    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        state, reward, done, info = super().step(action)
        step_memo = self.exchange.state_memos
        step_memo_arr = np.array(step_memo)
        step_memo_arr = step_memo_arr.reshape(-1, 20 , 2).astype(np.float32) # 100, 20, 2
        step_memo_arr[:,:,0] = (step_memo_arr[:,:,0]-Config.price_mean)/ Config.price_std
        step_memo_arr[:,:,1] = (step_memo_arr[:,:,1]-Config.qty_mean)/ Config.qty_std
        return step_memo_arr, reward, done, info
    # --------------------- 03.01 ---------------------
    def state(self, action: Action) -> State:
        state = super().state(action)
        return state



if __name__ == "__main__":
    import numpy as np
    arr = np.array([
        [1,1,0],
    ])
    arr = np.repeat(arr, 3000, axis=0)

    env = TimewindowEnv()
    env.reset()
    print("=" * 20 + " ENV RESTED " + "=" * 20)
    sum_reward = 0
    for i in range(int(1e6)):
        print("-" * 20 + f'=> {i} <=' + '-' * 20)  # $
        machine_code = arr[i]
        state, reward, done, info = env.step(machine_code)
        print(f"info: {info}")  # $
        sum_reward += reward
        # env.render()
        if done:
            env.reset()
            break  # $
    print(sum_reward)
