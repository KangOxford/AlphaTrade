# ========================= 01 =========================
# import numpy as np
# from gym_exchange.exchange import Debugger
from gym_exchange.exchange.basic_exc.base_exchange import BaseExchange
from gym_exchange.exchange.basic_exc.utils import latest_timestamp, timestamp_increase
from gym_exchange.exchange.basic_exc.assets.auto_cancels import AutoCancels
from gym_exchange.exchange.basic_exc.assets.order_flow import OrderFlow
# from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order
# from gym_exchange.environment.env_interface import State, Observation, Action # types

# ========================= 03 =========================
class Exchange(BaseExchange):
    def __init__(self):
        super().__init__()
        
    # -------------------------- 03.01 ----------------------------
    def reset(self):
        super().reset()
        # self.mid_prices = [(self.order_book.get_best_ask() + self.order_book.get_best_bid())/2] #$ origin should it be []???? TODO
        '''self.auto_cancels = AutoCancels()'''


    # -------------------------- 03.02 ----------------------------
    def step(self, action = None): # action : Action(for the definition of type)
        super().step(action)
        return self.order_book
    

    # ···················· 03.02.01 ···················· 
    def update_task_list(self, action = None):# action : Action(for the definition of type)
        self.task_list = []
        '''
        auto_cancels = self.auto_cancels.step()
        auto_cancels = [self.time_wrapper(auto_cancel) for auto_cancel in auto_cancels] # used for auto cancel
        self.task_list += auto_cancels
        '''
        super().update_task_list(action)
    # ···················· 03.02.02 ····················
    def time_wrapper(self, order_flow: OrderFlow) -> OrderFlow:
        # timestamp = latest_timestamp(self.order_book)
        timestamp = self.latest_timestamp
        # print(self.index) #$$
        # print(f"right: {timestamp}, my: {self.latest_timestamp}, my is {'early' if timestamp>self.latest_timestamp else 'late'}") #$$
        # assert float(timestamp) - float(self.latest_timestamp) <= 0.000000020 #$$
        return timestamp_increase(timestamp, order_flow) 
    
if __name__ == "__main__":
    exchange = Exchange()
    exchange.reset()
    for _ in range(2048):
        exchange.step()
        















































