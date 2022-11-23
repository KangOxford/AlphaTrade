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
    # -------------------------- 02.01 ----------------------------
    '''init'''
    def __init__(self):
        self.flow_list = self.initialization()
    
    def initialization(self):
        decoder   = Decoder(**DataPipeline()())
        encoder   = Encoder(decoder)
        flow_lists= encoder() 
        flow_lists= self.to_order_flow_lists(flow_lists)
        return flow_lists
    
    def to_order_flow_lists(self, flow_lists):
        '''change side format from bid/ask to 1/-1
        side = -1 if item.side == 'ask' else 1'''
        for flow_list in flow_lists:
            for item in flow_list:
                side = -1 if item.side == 'ask' else 1
                item.side = side
        return flow_lists
    
    # -------------------------- 02.02 ----------------------------
    '''reset'''
        
    def reset(self):
        self.index = 0
        self.flow_generator = (flow for flow in self.flow_list)
        self.order_book = OrderBook()
        self.initialize_orderbook()
        
    def initialize_orderbook(self):
        '''only take the index0, the first one to init the lob'''
        flow_list = next(self.flow_generator)
        for flow in flow_list:
            self.order_book.process_order(flow.to_message, True, False)
        self.index += 1 
        '''for this step is index0, for next step is index1'''
        
    # -------------------------- 02.03 ----------------------------
    '''step'''

    def step(self, action = None): # action : Action(for the definition of type)
        self.update_task_list(action)
        self.process_tasks()
        self.accumulating()
        return self.order_book

    def update_task_list(self, action = None):# action : Action(for the definition of type)
        flow_list = next(self.flow_generator)#used for historical data
        self.task_list = [action] + [flow for flow in flow_list]
    
    @abstractclassmethod
    def process_tasks(self):
        pass
        
    def accumulating(self):
        self.index += 1

if __name__ == "__main__":
    
    
    # # -------------------------- 03.01 ----------------------------
    # ''' flow_lists= encoder() '''
    # Ofs = flow_lists
    # with open("/Users/kang/GitHub/NeuralLOB/gym_exchange/log_ofs_exchange_interface.txt","w") as f:
    #     for i in range(len(Ofs)):
    #         f.write(f"------ {i} ------\n")
    #         f.write(Ofs[i].__str__())
    
    
    # # -------------------------- 03.02 ----------------------------
    # ''' flow_lists = self.to_order_flow_lists(flow_lists) '''
    # Ofs = flow_lists
    # with open("/Users/kang/GitHub/NeuralLOB/gym_exchange/log_ofs_exchange_interface2.txt","w") as f:
    #     for i in range(len(Ofs)):
    #         f.write(f"------ {i} ------\n")
    #         f.write(Ofs[i].__str__())
    
    # # -------------------------- 03.03 ----------------------------
    # ''' self.flow_generator = (flow for flow in self.flow_list) '''
    # Ofs = self.flow_generator
    # with open("/Users/kang/GitHub/NeuralLOB/gym_exchange/log_ofs_exchange_interface3.txt","w") as f:
    #     for i in range(2049):
    #         f.write(f"------ {i} ------\n")
    #         f.write(next(Ofs).__str__())
    
    
    pass