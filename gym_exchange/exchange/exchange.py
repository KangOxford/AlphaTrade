# import abc; from abc import abstractclassmethod
# class Exchange_Interface(abc.ABC):
#     @abstractclassmethod
#     def 

# @Exchange_Interface.register

# class Exchange():
#     def __init__(self):
#         self.
#     def 
    
# import numpy as np
from gym_exchange.data_orderbook_adapter import Configuration
# from gym_exchange.data_orderbook_adapter import Debugger 
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
# from gym_exchange.order_flow import OrderFlow
from gym_exchange.orderbook import OrderBook
from gym_exchange.data_orderbook_adapter import utils
# from gym_exchange.orderbook.order import Order


# -------------------------- 01 ----------------------------
decoder  = Decoder(**DataPipeline()())
encoder  = Encoder(decoder)
flow_list= encoder.process()
# print(ofs)#$
# -------------------------- 02 ----------------------------
order_book = OrderBook()
def initialize_orderbook(order_book):
    for index in range(2*Configuration.price_level):
        order_book.process_order(flow_list[index])
        
# -------------------------- 03 ----------------------------
for flow in flow_list:
    # order_book.process(flow.to_order)
    message = flow.to_message
    if flow.type == 1:
        order_book.process_order(message, True, False)
    elif flow.type == 2:
        pass #TODO, not implemented!!
    elif flow.type == 3:
        pass #TODO, should be partly cancel
        # order_book.cancel_order(side = message['side'], 
        #                         order_id = message['order_id'],
        #                         time = message['timestamp'], 
        #                         # order_id = order.order_id,
        #                         # time = order.timestamp, 
        #                         )
print(order_book)#$
print(utils.brief_order_book(order_book, side = 'bid'))#$
print(utils.brief_order_book(order_book, side = 'ask'))#$

















































