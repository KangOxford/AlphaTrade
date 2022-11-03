# ========================= 01 =========================
# import numpy as np
from gym_exchange.exchange import Debugger
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
        return encoder, flow_list
    
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
        for flow in self.flow_list:
            yield flow
            
# -------------------------- 03.02 ----------------------------
    def step(self, action = None):
        # for flow in self.flow_list:
        # order_book.process(flow.to_order)
        flow = next(self.flow_generator)
        
        # # used for auto cancel
        # future = self.futures.step()
        # auto_cancel = self.time_wrapper(future)
        # for index, item in enumerate([action, flow, auto_cancel]): # advantange for ask limit order (in liquidation problem)
        self.prev_trades = [] #prev_trades reset
        for index, item in enumerate([action, flow]): # advantange for ask limit order (in liquidation problem)
        # for item in [flow, action]:
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    trades, order_in_book = self.order_book.process_order(message, True, False)
                    self.prev_trades += trades; assert len(trades) == 1 # record prev_trades
                    kind = 'agent' if index == 0 else 'market'
                    self.executed_pairs.step(trades, kind)
                    # if len(trades) != 0:
                    #     print(trades)#$
                    # print("+++trades+++")#$
                    # print(trades)#$
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
        timestamp = self.latest_timestamp
        str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
        order_flow.timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
        return order_flow 
    
    @property
    def latest_timestamp(self):
        '''ought to return the latest timestamp in the exchange.prev_trades 
           and exchange.prev_orders_in_book'''
        timestamp_list = []
        for trade in self.prev_trades:
            timestamp_list += trade['timestamp']
        max_timestamp = max(timestamp_list)
        return max_timestamp #!TODO  exchange.prev_orders_in_book not included
        # return '34200.000000000' #!TODO
    
    
    
if __name__ == "__main__":
    exchange = Exchange()
    exchange.reset()
    for _ in range(1000):
        exchange.step()
        















































