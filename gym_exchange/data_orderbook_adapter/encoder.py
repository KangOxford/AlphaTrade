# -*- coding: utf-8 -*-
# import abc; from abc import abstractclassmethod
# class Order_Flow_Interface(abc.ABC):
#     @abstractclassmethod
#     def 

# @Order_Flow_Interface.register
import numpy as np
from gym_exchange.data_orderbook_adapter import Configuration, Debugger 
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.exchange.order_flow_list import FlowList

    
class Encoder():
    def __init__(self, decoder):
        self.decoder = decoder
        self.flow_lists = [] # [flow_list, flow_list, ... ], for each step we get a flow_list
    
    # -------------------------- 01 ----------------------------
    def initialize_order_flows(self):
        flow_list = FlowList()
        for side in ['ask','bid']:
            List = self.decoder.initiaze_orderbook_message(side)
            for Dict in List:
                order_flow = OrderFlow(
                time = Dict['timestamp'],
                Type = 1 ,
                order_id = Dict['order_id'],
                size = Dict['quantity'],
                price = Dict['price'],
                direction = Dict['side'],
                trade_id= Dict['trade_id']
                )
                flow_list.append(order_flow)
        self.flow_lists.append(flow_list)
        return self.flow_lists
    
    # -------------------------- 02 ----------------------------    
    def inside_signal_encoding(self, inside_signal):
        if inside_signal['sign'] in (1,2,3,):
            order_flow = OrderFlow(
            time = inside_signal['timestamp'],
            Type = inside_signal['sign'],
            order_id = inside_signal['order_id'],
            size = inside_signal['quantity'],
            price = inside_signal['price'],
            direction = inside_signal['side'],
            trade_id= inside_signal['trade_id']
            )
        elif inside_signal['sign'] in (4,):
            order_flow = OrderFlow(
            time = inside_signal['timestamp'],
            Type = 1 ,
            order_id = inside_signal['order_id'],
            size = inside_signal['quantity'],
            price = inside_signal['price'],
            direction = inside_signal['side'], 
            trade_id= inside_signal['trade_id']
            )
        elif inside_signal['sign'] in (5,6,): order_flow = None
        else: raise NotImplementedError
        return order_flow
    
    def outside_signal_encoding(self, signal):
        if signal['sign'] in (10,11):
            order_flow = OrderFlow(
            time = signal['timestamp'],
            Type = signal['sign']//10,
            order_id = signal['order_id'],
            size = signal['quantity'],
            price = signal['price'],
            direction = signal['side'],
            trade_id= signal['trade_id']
            )
        # elif signal['sign'] in (20,): #TODO
        #     '''signal
        #     {'sign': 20, 'right_order_price': 31210000, 'wrong_order_price': 31209000, 'side': 'ask'}'''
        elif signal['sign'] in (60,):  order_flow = None
        else: order_flow = None # !not implemented yet
        return order_flow
    
    def get_all_running_order_flows(self):
        for index in range(Configuration.horizon):
            _ = self.step(index)
        return self.flow_lists
    
    def step(self, index = None): # get_single_running_order_flows
        inside_signal, outside_signals = self.decoder.step() # the decoder return single data in step()
        inside_order_flow = self.inside_signal_encoding(inside_signal)
        # ···················· 02.01 ···················· 
        flow_list = FlowList()
        if inside_order_flow is not None:
            flow_list.append(inside_order_flow)
        for signal in outside_signals:
            if type(signal) is list: 
                for s in signal:
                    outside_order_flow = self.outside_signal_encoding(s)
                    if outside_order_flow is not None:
                        flow_list.append(outside_order_flow)
            else:
                outside_order_flow = self.outside_signal_encoding(signal)
                if outside_order_flow is not None:
                    flow_list.append(outside_order_flow)
        # ···················· 02.02 ···················· 
        self.flow_lists.append(flow_list)
        # ···················· 02.03 ···················· 
        if Debugger.Encoder.on:
            try:
                print("="*10+' '+str(index)+" "+"="*10)
                print(">>> inside_signal");print(inside_signal)
                print(">>> outside_signal");[print(signal) for signal in outside_signals]
                print("-"*23)
            except: pass
        return flow_list
        
    
    # -------------------------- 03 ----------------------------
    def __call__(self):
        self.initialize_order_flows()
        self.get_all_running_order_flows()
        return self.flow_lists
        
        

if __name__ == "__main__":
    decoder = Decoder(**DataPipeline()())
    encoder = Encoder(decoder)
    Ofs     = encoder()
    
    # import pandas as pd
    # pd.DataFrame(Ofs).to_csv("Ofs.csv", header=None, index=None)

    # count = 0
    # for of in Ofs:
    #     if of.length != 0:
    #         count += 1
    
    
    with open("/Users/kang/GitHub/NeuralLOB/gym_exchange/log_ofs.txt","w") as f:
        for i in range(len(Ofs)):
            f.write(f"------ {i} ------\n")
            f.write(Ofs[i].__str__())
            
            
            
            