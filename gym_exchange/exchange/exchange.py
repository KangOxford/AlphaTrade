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
from gym_exchange.data_orderbook_adapter import Debugger 
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
# from gym_exchange.order_flow import OrderFlow
from gym_exchange.orderbook import OrderBook



decoder = Decoder(**DataPipeline()())
encoder = Encoder(decoder)
ofs     = encoder()
order_book = OrderBook()

def initialize_orderbook(order_book):
    for index in range(2*Configuration.price_level):
        order_book.process_order(ofs[index])
        





















































