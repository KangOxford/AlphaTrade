# ========================= 01 =========================
# import numpy as np
from gym_exchange.exchange import Debugger
from gym_exchange.exchange.utils import latest_timestamp, timestamp_increase
from gym_exchange.exchange.utils.futures import Futures
from gym_exchange.exchange.utils.executed_pairs import ExecutedPairs
from gym_exchange.data_orderbook_adapter import Configuration
# from gym_exchange.data_orderbook_adapter import Debugger 
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.orderbook import OrderBook
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order
# from gym_exchange.trading_environment.env_interface import State, Observation, Action # types
# ========================= 02 =========================
import abc; from abc import abstractclassmethod
class Exchange_Interface(abc.ABC):
    def __init__(self):
        self.index = 0
        self.encoder, self.flow_list = self.initialization()

    def initialization(self):
        decoder  = Decoder(**DataPipeline()())
        encoder  = Encoder(decoder)
        flow_list= encoder.process()
        flow_list = self.to_order_flow_list(flow_list)
        return encoder, flow_list
    
    def to_order_flow_list(self, flow_list):
        for item in flow_list:
            side = -1 if item.side == 'ask' else 1
            item.side = side
        return flow_list
    
    @abstractclassmethod
    def reset(self):
        pass
    @abstractclassmethod
    def step(self):
        pass

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
        self.futures = Futures()
        
    def initialize_orderbook(self):
        for _ in range(2*Configuration.price_level):
            flow = next(self.flow_generator)
            self.order_book.process_order(flow.to_message, True, False)
            self.index += 1
            # if self.index >= 2*Configuration.price_level:break #TODO not sure about the num
        # for index in range():
        #     order_book.process_order(self.flow_list[index])
        print()#$
            
    def generate_flow(self):
        for flow in self.flow_list: yield flow
            
    # -------------------------- 03.02 ----------------------------
    def step(self, action = None): # action : Action(for the definition of type)
        flow = next(self.flow_generator)#used for historical data
        futures = self.futures.step(); auto_cancels = [self.time_wrapper(future) for future in futures] # used for auto cancel
        for index, item in enumerate([action, flow] + auto_cancels): # advantange for ask limit order (in liquidation problem)
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    trades, order_in_book = self.order_book.process_order(message, True, False)
                    kind = 'agent' if index == 0 else 'market'
                    self.executed_pairs.step(trades, kind)
                elif item.type == 2:
                    pass #TODO, not implemented!!
                elif item.type == 3:
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
        















































