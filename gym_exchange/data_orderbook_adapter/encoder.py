# -*- coding: utf-8 -*-
# import abc; from abc import abstractclassmethod
# class Order_Flow_Interface(abc.ABC):
#     @abstractclassmethod
#     def 

# @Order_Flow_Interface.register
import numpy as np
from gym_exchange.data_orderbook_adapter.decoder import Decoder
class Order_Flow():
    def __init__(self,time,Type,order_id,size,price,direction,trade_id):
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
        self.time = time
        self.type = Type
        self.order_id = order_id
        self.size = size
        self.price = price
        self.direction = direction
        self.trade_id = trade_id
    def __call__(self):
        return np.array([
            self.time,
            self.type,
            self.order_id,
            self.size,
            self.price,
            self.direction
            ])
    # def __str___(self):
    
# class Encoder():
#     def __init__(self):
    
    
    
