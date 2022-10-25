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
from gym_exchange.order_flow import Order_Flow

    
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
    
def inside_signal_encoding(inside_signal):
    if inside_signal['sign'] in (1,2,3,):
        order_flow = OrderFlow(
        time = inside_signal['timestamp'],
        Type = inside_signal['sign'],
        order_id = inside_signal['order_id'],
        size = inside_signal['quantity'],
        price = inside_signal['price'],
        direction = inside_signal['side'],
        trade_id= inside_signal['trade_id']
        )()
    elif inside_signal['sign'] in (4,):
        order_flow = OrderFlow(
        time = inside_signal['timestamp'],
        Type = 1 ,
        order_id = inside_signal['order_id'],
        size = inside_signal['quantity'],
        price = inside_signal['price'],
        direction = (set(['bid','ask'])-set([inside_signal['side']])).pop(),
        trade_id= inside_signal['trade_id']
        )()
    elif inside_signal['sign'] in (5,6,): order_flow = None
    else: raise NotImplementedError
    return order_flow

def outside_signal_encoding(signal):
    if signal['sign'] in (10,11):
        order_flow = OrderFlow(
        time = signal['timestamp'],
        Type = signal['sign']//10,
        order_id = signal['order_id'],
        size = signal['quantity'],
        price = signal['price'],
        direction = signal['side'],
        trade_id= signal['trade_id']
        )()
    # elif signal['sign'] in (20,): #TODO
    #     '''signal
    #     {'sign': 20, 'right_order_price': 31210000, 'wrong_order_price': 31209000, 'side': 'ask'}'''
    elif signal['sign'] in (60,):  order_flow = None
    else: order_flow = None # !not implemented yet
    return order_flow

def get_running_order_flows():
    order_flows = np.array([])
    for index in range(Configuration.horizon):
        inside_signal, outside_signals = decoder.step()
        inside_order_flow = inside_signal_encoding(inside_signal)
        if inside_order_flow is not None:
            order_flows = np.append(order_flows, inside_order_flow).reshape([-1, OrderFlow.length])
        # if index == 1614: breakpoint()#$
        for signal in outside_signals:
            if type(signal) is list: 
                for s in signal:
                    outside_order_flow = outside_signal_encoding(s)
                    if outside_order_flow is not None:
                        order_flows = np.append(order_flows, outside_order_flow).reshape([-1, OrderFlow.length])
            else:
                outside_order_flow = outside_signal_encoding(signal)
                if outside_order_flow is not None:
                    order_flows = np.append(order_flows, outside_order_flow).reshape([-1, OrderFlow.length])
        if Debugger.Encoder.on:
            print("="*10+' '+str(index)+" "+"="*10)
            print(">>> inside_signal");print(inside_signal)
            print(">>> outside_signal")
            for signal in outside_signals:
                print(signal)
            print("-"*23)
            #     if signal['sign'] != 60:#$
            #         breakpoint()#$
            # print()#$
    return order_flows

if __name__ == "__main__":
    decoder = Decoder(**DataPipeline()())
    ofs = initialize_order_flows()
    ofs2 = get_running_order_flows()
    Ofs = np.append(ofs, ofs2).reshape([-1, OrderFlow.length])
    
    import pandas as pd
    pd.DataFrame(Ofs).to_csv("Ofs.csv", header=None, index=None)
