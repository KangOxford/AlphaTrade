# -*- coding: utf-8 -*-
# ========================= 01 =========================
from gym_exchange.orderbook import OrderBook
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.data_orderbook_adapter import Configuration

# ========================= 02 =========================
import abc; from abc import abstractclassmethod
class Exchange_Interface(abc.ABC):
    def __init__(self):
        self.index = 0
        self.encoder, self.flow_list = self.initialization()
    
    # -------------------------- 02.01 ----------------------------
    def initialization(self):
        decoder   = Decoder(**DataPipeline()())
        encoder   = Encoder(decoder)
        flow_lists= encoder()
        flow_lists= self.to_order_flow_lists(flow_lists)
        return encoder, flow_lists
    
    def to_order_flow_lists(self, flow_lists):
        for flow_list in flow_lists:
            for item in flow_list:
                side = -1 if item.side == 'ask' else 1
                item.side = side
        return flow_lists
    
    # -------------------------- 02.02 ----------------------------
    def initialize_orderbook(self):
        flow_list = next(self.flow_generator)
        for flow in flow_list:
            self.order_book.process_order(flow.to_message, True, False)
        self.index += 1
        
    def reset(self):
        self.order_book = OrderBook()
        self.flow_generator = (flow for flow in self.flow_list)
        self.initialize_orderbook()
        
        
    # -------------------------- 02.03 ----------------------------
    def update_task_list(self, action = None):# action : Action(for the definition of type)
        flow_list = next(self.flow_generator)#used for historical data
        self.task_list = [action] + [flow for flow in flow_list]
    
    @abstractclassmethod
    def process_tasks(self):
        pass
        
    def accumulating(self):
        self.index += 1

    def step(self, action = None): # action : Action(for the definition of type)
        self.update_task_list(action)
        self.process_tasks()
        self.accumulating()
        return self.order_book

