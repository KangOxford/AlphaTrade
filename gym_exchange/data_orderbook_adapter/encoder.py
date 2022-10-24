# -*- coding: utf-8 -*-
# import abc; from abc import abstractclassmethod
# class Order_Flow_Interface(abc.ABC):
#     @abstractclassmethod
#     def 

# @Order_Flow_Interface.register
import numpy as np
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
class OrderFlow():
    length = 7
    def __init__(self,Type,direction,size,price,trade_id,order_id,time):
        '''Parameters
        -------------
        time : str
        Type : int
            1: submission of a new limit order
            2: cancellation(partial deletion)
            3: deletion(total deletion)
        order_id : int
        size : int
        price : int
        direction : int
            -1 : limit order at ask side
            +1 : limit order at sell side
        trade_id : int
        '''
        self.type = Type
        self.side = direction
        self.quantity = size
        self.price = price
        self.trade_id = trade_id
        self.order_id = order_id
        self.timestamp = time
    def __call__(self):
        return np.array([
        self.type, 
        self.side, 
        self.quantity, 
        self.price, 
        self.trade_id, 
        self.order_id, 
        self.timestamp 
            ])
    # def __str___(self):
    
# class Encoder():
#     def __init__(self):
    
def initialize_order_flows():
    order_flows = np.array([])
    for side in ['bid','ask']:
        List = decoder.initiaze_orderbook_message(side)
        for Dict in List:
            order_flows = np.append(
                order_flows,
                OrderFlow(
                time = Dict['timestamp'],
                Type = 1 ,
                order_id = Dict['order_id'],
                size = Dict['quantity'],
                price = Dict['price'],
                direction = Dict['side'],
                trade_id= Dict['trade_id']
                )()).reshape([-1, OrderFlow.length])
    return order_flows
    
if __name__ == "__main__":
    decoder = Decoder(**DataPipeline()())
    ofs = initialize_order_flows()
    
    
    
