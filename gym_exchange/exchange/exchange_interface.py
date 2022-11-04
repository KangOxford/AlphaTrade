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
    

    def reset(self):
        self.order_book = OrderBook()
        self.flow_generator = self.generate_flow()
        self.initialize_orderbook()
        
    def initialize_orderbook(self):
        for _ in range(2*Configuration.price_level):
            flow = next(self.flow_generator)
            self.order_book.process_order(flow.to_message, True, False)
            self.index += 1
            
    def generate_flow(self):
        for flow in self.flow_list: yield flow
        
    @abstractclassmethod
    def step(self):
        pass
