# ========================= 01 =========================
# import numpy as np
from gym_exchange.exchange import Debugger
from gym_exchange.exchange.exchange_interface import Exchange_Interface
from gym_exchange.exchange.utils import latest_timestamp, timestamp_increase
from exchange.utils.auto_cancels import AutoCancels
from exchange.utils.deletion_handler import PartDeletionHandler, TotalDeletionHandler
from gym_exchange.exchange.utils.executed_pairs import ExecutedPairs
from gym_exchange.data_orderbook_adapter import Configuration
# from gym_exchange.data_orderbook_adapter import Debugger 
from gym_exchange.orderbook import OrderBook
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order
# from gym_exchange.trading_environment.env_interface import State, Observation, Action # types

# ========================= 03 =========================
class Exchange(Exchange_Interface):
    def __init__(self):
        super().__init__()
        
    # -------------------------- 03.01 ----------------------------
    def reset(self):
        self.order_book = OrderBook()
        self.flow_generator = self.generate_flow()
        self.initialize_orderbook()
        self.executed_pairs = ExecutedPairs()
        self.auto_cancels = AutoCancels()
        self.part_deletion_handler = PartDeletionHandler()
        self.total_deletion_handler = TotalDeletionHandler()
        
    def initialize_orderbook(self):
        for _ in range(2*Configuration.price_level):
            flow = next(self.flow_generator)
            self.order_book.process_order(flow.to_message, True, False)
            self.index += 1
            
    def generate_flow(self):
        for flow in self.flow_list: yield flow
            
    # -------------------------- 03.02 ----------------------------
    def step(self, action = None): # action : Action(for the definition of type)
        flow = next(self.flow_generator)#used for historical data
        auto_cancels = self.auto_cancels.step(); auto_cancels = [self.time_wrapper(auto_cancel) for auto_cancel in auto_cancels] # used for auto cancel
        for index, item in enumerate([action, flow] + auto_cancels): # advantange for ask limit order (in liquidation problem)
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    trades, order_in_book = self.order_book.process_order(message, True, False)
                    self.executed_pairs.step(trades, 'agent' if index == 0 else 'market') # 2nd para: kind
                elif item.type == 2:
                    print(f'>>> type2 activated')#$ 
                    pass #TODO, not implemented!!

                elif item.type == 3:
                    print(f'>>> type3 activated')#$ 
                    pass #TODO, should be partly cancel
                    # order_book.cancel_order(side = message['side'], 
                    #                         order_id = message['order_id'],
                    #                         time = message['timestamp'], 
                    #                         # order_id = order.order_id,
                    #                         # time = order.timestamp, 
                    #                         )
                if Debugger.on:
                    print(self.order_book)#$
                    print(utils.brief_order_book(self.order_book, side = 'bid'))#$
                    print(utils.brief_order_book(self.order_book, side = 'ask'))#$
        return self.order_book
    
    # ···················· 03.02.01 ···················· 
    def time_wrapper(self, order_flow: OrderFlow) -> OrderFlow:
        timestamp = latest_timestamp(self.order_book)
        return timestamp_increase(timestamp, order_flow) 
    
if __name__ == "__main__":
    exchange = Exchange()
    exchange.reset()
    for _ in range(1000):
        exchange.step()
        















































