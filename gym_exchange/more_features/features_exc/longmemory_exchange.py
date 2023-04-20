# ========================= 01 =========================
# import numpy as np
# from gym_exchange.exchange import Debugger
from gym_exchange.exchange.timewindow_exchange import TimewindowExchange

# from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order
# from gym_exchange.environment.env_interface import State, Observation, Action # types

time_window = 300


# ========================= 02 =========================
class LongmemoExchange(TimewindowExchange):
    def __init__(self):
        super().__init__()

    # -------------------------- 03.02 ----------------------------
    def step(self, action=None):  # action : Action(for the definition of type)
        for i in range(time_window-1):
            # print(f"innerloop step {i}") #$
            # if i == 248:
            #     print() #$
            super(TimewindowExchange, self).step()
        super().step(action)
        return self.order_book



if __name__ == "__main__":
    exchange = TimewindowExchange()
    exchange.reset()
    for i in range(2048):
        # print(f">>> outerloop step {i}")
        exchange.step()


