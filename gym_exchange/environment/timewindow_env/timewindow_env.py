from gym_exchange import Config
from gym_exchange.exchange.timewindow_exchange import TimewindowExchange
from gym import spaces

# *************************** 2 *************************** #
import numpy as np
from gym_exchange.environment.base_env.base_env import BaseEnv # DO NOT DELETE
from gym_exchange.environment.basic_env.basic_env import BasicEnv # DO NOT DELETE

class TimewindowEnv(locals()[Config.train_env]):
    # ========================== 01 ==========================
    def __init__(self):
        if 'exchange' not in dir(self):
            self.exchange = TimewindowExchange()
        super().__init__()
        self.state_space =  spaces.Box(
              low   = -10,
              high  = 10,
              # shape = (100,4),
              shape = (101,4),
              dtype = np.float64,
        )
        self.observation_space = self.state_space


    # ========================== 02 ==========================
    # ========================= RESET ========================
    # def reset(self):
    #     super().reset()
    def initial_state(self):
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

        task_info = np.array([0, Config.max_horizon, Config.num2liquidate, Config.num2liquidate])
        task_info[[0,1]] = task_info[[0,1]]/task_info[1]
        task_info[[2,3]] = task_info[[2,3]]/task_info[3]

        state = np.vstack((state, task_info))

        assert state.shape == (101,4)
        return state

    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        state, reward, done, info = super().step(action)
        return state, reward, done, info
    # --------------------- 03.01 ---------------------
    def state(self, action):
        # ---------------- 01 ----------------
        _ = super().state(action)
        # ---------------- 02 ----------------
        step_memo = self.exchange.state_memos
        step_memo_arr = np.array(step_memo)
        best_bids = step_memo_arr[:,1,0,:]
        best_asks = step_memo_arr[:,0,-1,:]
        best_prices = np.concatenate([best_asks,best_bids],axis=1)
        # ---------------- 03 ----------------
        step_memo_arr = best_prices.astype(np.float32) # 100, 4
        step_memo_arr[:,::2] = (step_memo_arr[:,::2]-Config.price_mean)/ Config.price_std
        step_memo_arr[:,1::2] = (step_memo_arr[:,1::2]-Config.qty_mean)/ Config.qty_std
        # ---------------- 04 ----------------
        task_info = self.task_info[:,0]
        task_info[[0,1]] = task_info[[0,1]]/task_info[1]
        task_info[[2,3]] = task_info[[2,3]]/task_info[3]
        task_info = task_info.reshape(1, 4)
        state = np.vstack((step_memo_arr, task_info))
        # ---------------- 05 ----------------
        assert state.shape[1] == 4 # (101, 4)
        return state

if __name__ == "__main__":
    # Config.max_horizon = horizon_length
    # Config.raw_horizon = int(Config.max_horizon * Config.window_size * 1.01)

    import numpy as np
    arr = np.array([
        [1,Config.quantity_size_one_side,0],
        # [1,1,0],
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
