# ========================= 01 =========================
# import numpy as np
# from gym_exchange.exchange import Debugger
import pandas as pd

from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange
from gym_exchange.more_features.features_exc.utils import get_state_memo
from gym_exchange.exchange.basic_exc.utils import latest_timestamp, timestamp_increase
from gym_exchange.exchange.basic_exc.assets.auto_cancels import AutoCancels
from gym_exchange.exchange.basic_exc.assets.order_flow import OrderFlow


# from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order
# from gym_exchange.trading_environment.env_interface import State, Observation, Action # types

time_window = 300
'''
function:
01 only step after a timewindow
02 remember all the (level 10) brief order_book data during the past timewindow
'''

# ========================= 01 =========================
class TimewindowExchange(Exchange):
    def __init__(self):
        super().__init__()

    # -------------------------- 01.01 ----------------------------
    def step(self, action=None):  # action : Action(for the definition of type)
        # ···················· 01.01.01 ····················
        for i in range(time_window-1):
            print(f"innerloop step {i}") #$
            if i == 2:
                print() #$
            super(TimewindowExchange, self).step()
            self.update_state_memos()
        super().step(action)
        self.update_state_memos()
        # ···················· 01.01.02 ····················
        return self.order_book
    def update_state_memos(self):
        state_memo = get_state_memo(self.order_book)
        self.state_memos.append(state_memo)

    # -------------------------- 01.02 ----------------------------
    def reset(self):
        super(TimewindowExchange, self).reset()
        self.state_memos = []

if __name__ == "__main__":
    exchange = TimewindowExchange()
    exchange.reset()
    for i in range(2048):
        print(f">>> outerloop step {i}")
        exchange.step()
















































