from gym_exchange import Config
from gym_exchange.environment.base_env.assets.action import Action

from gym_exchange.environment.base_env.interface_env import State  # types
# from gym_exchange.environment.env_interface import State, Observation # types
from gym_exchange.exchange.timewindow_exchange import TimewindowExchange
from gym_exchange.environment.basic_env.basic_env import BasicEnv
from gym_exchange.environment.base_env.base_env import BaseEnv
from copy import deepcopy
import numpy as np
from gym import spaces
# *************************** 2 *************************** #
class TimewindowEnv(locals()[Config.train_env]):
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        self.exchange = TimewindowExchange()
        self.state_space =  spaces.Box(
              low   = -10,
              high  = 10,
              shape = (100,4),
              dtype = np.float32,
        )

    # ========================== 02 ==========================
    # ========================= RESET ========================
    # ------------------------- 02.01 ------------------------
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""
        # ···················· 02.01.01 ····················
        # ob = np.array(self.exchange.order_book.get_L2_state()).T.reshape(20,2)
        ask_price = self.exchange.order_book.get_best_ask()
        ask_qty = self.exchange.order_book.asks.get_price_list(ask_price).volume
        bid_price = self.exchange.order_book.get_best_bid()
        bid_qty = self.exchange.order_book.bids.get_price_list(bid_price).volume
        ob = np.array([[ask_price,ask_qty],[bid_price,bid_qty]]).flatten()
        state = np.tile(ob,(100,1)).astype(np.float32)
        state[:, ::2] = (state[:,::2]-Config.price_mean)/ Config.price_std
        state[:, 1::2] = (state[:,1::2]-Config.qty_mean)/ Config.qty_std
        assert state.shape == (100,4)
        return state

    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        state, reward, done, info = super().step(action)
        return state, reward, done, info
    # --------------------- 03.01 ---------------------
    def state(self, action: Action) -> State:
        state = super().state(action)
        step_memo = self.exchange.state_memos
        step_memo_arr = np.array(step_memo)
        best_bids = step_memo_arr[:,1,0,:]
        best_asks = step_memo_arr[:,0,-1,:]
        best_prices = np.concatenate([best_asks,best_bids],axis=1)
        step_memo_arr = best_prices.astype(np.float32) # 100, 4
        step_memo_arr[:,::2] = (step_memo_arr[:,::2]-Config.price_mean)/ Config.price_std
        step_memo_arr[:,1::2] = (step_memo_arr[:,1::2]-Config.qty_mean)/ Config.qty_std
        assert step_memo_arr.shape == (100,4)
        return step_memo_arr



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
