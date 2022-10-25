# -------------------------- 01 ----------------------------
# import numpy as np
from gym_exchange.exchange import Debugger
from gym_exchange.data_orderbook_adapter import Configuration
# from gym_exchange.data_orderbook_adapter import Debugger 
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
# from gym_exchange.order_flow import OrderFlow
from gym_exchange.orderbook import OrderBook
from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order

# -------------------------- 02 ----------------------------
import abc; from abc import abstractclassmethod
class Exchange_Interface(abc.ABC):
    @abstractclassmethod
    def reset(self):
        pass
    @abstractclassmethod
    def step(self):
        pass

# -------------------------- 03 ----------------------------
@Exchange_Interface.register
class Exchange():
    def __init__(self):
        self.index = 0
        self.encoder, self.flow_list = self.initialization()
        self.reset()
    def initialize_orderbook(self):
        for _ in range(2*Configuration.price_level):
            flow = next(self.flow_generator)
            self.order_book.process_order(flow.to_message, True, False)
            self.index += 1
            # if self.index >= 2*Configuration.price_level:break #TODO not sure about the num
        # for index in range():
        #     order_book.process_order(self.flow_list[index])
        print()#$
            
    def initialization(self):
        # -------------------------- 03.01 ----------------------------
        decoder  = Decoder(**DataPipeline()())
        encoder  = Encoder(decoder)
        flow_list= encoder.process()
        # print(ofs)#$
        return encoder, flow_list
    def reset(self):
        self.order_book = OrderBook()
        self.flow_generator = self.generate_flow()
        self.initialize_orderbook()
    def generate_flow(self):
        for flow in self.flow_list:
            yield flow
    def step(self, action = None):
        # -------------------------- 03.02 ----------------------------
        # for flow in self.flow_list:
        # order_book.process(flow.to_order)
        flow = next(self.flow_generator)
        for item in [flow, action]:
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    self.order_book.process_order(message, True, False)
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


if __name__ == "__main__":
    exchange = Exchange()
    for _ in range(1000):
        exchange.step()
    













































