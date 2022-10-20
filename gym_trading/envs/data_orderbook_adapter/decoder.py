# -*- coding: utf-8 -*-
# used as Level_N_Adapter
# =============================================================================
# 01 IMPORT PACKAGES
# =============================================================================
import numpy as np
import pandas as pd
from copy import copy
from gym_trading.envs.data_orderbook_adapter import Debugger, Configuration 
from gym_trading.envs.data_orderbook_adapter import utils
from gym_trading.envs.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_trading.envs.data_orderbook_adapter.utils.InsideSignalProducer import InsideSignalProducer
# from gym_trading.envs.data_orderbook_adapter.utils.OutsideSingalProducer import OutsideSingalProducer
from gym_trading.envs.data_orderbook_adapter.adjust_data_drift import DataAdjuster
from gym_trading.envs.orderbook import OrderBook

class Decoder:
    def __init__(self, price_level, horizon, historical_data, data_loader): 
        self.historical_data = historical_data
        self.price_level = price_level
        self.horizon = horizon
        self.data_loader = data_loader
        self.index = 0
        # --------------- NEED ACTIONS --------------------
        self.column_numbers = [i for i in range(price_level * 4) if i%4==2 or i%4==3]
        self.bid_sid_historical_data = historical_data.iloc[:,self.column_numbers]
        self.order_book = self.initialize_orderbook()
        self.data_adjuster = DataAdjuster(self.bid_sid_historical_data)
        
    def initialize_orderbook(self):
        order_book = OrderBook()
        l2 = self.historical_data.iloc[0,:].iloc[self.column_numbers].reset_index().drop(['index'],axis = 1)
        limit_orders = []
        order_id_list = [15000000 + i for i in range(self.price_level)]
        for i in range(self.price_level):
            trade_id = 90000
            # timestamp = datetime(34200.000000001)
            timestamp = str(34200.000000001)
            item = {'type' : 'limit', 
                'side' : 'bid', 
                'quantity' : l2.iloc[2 * i + 1,0], 
                'price' : l2.iloc[2 * i,0],
                'trade_id' : trade_id,
                'order_id' : order_id_list[i],
                "timestamp": timestamp}
            limit_orders.append(item)
        # Add orders to order book
    
        for order in limit_orders:
            # breakpoint()
            trades, order_id = order_book.process_order(order, True, False)   
        # The current book may be viewed using a print
        if Debugger.on: print(order_book)
        return order_book
    
    def step(self):
        # -------------------------- 01 ----------------------------
        if Debugger.on: 
            print("=="*10 + " " + str(self.index) + " "+ "=="*10)
            print("The order book used to be:"); print(self.order_book)
        historical_message = self.data_loader.iloc[self.index,:]
        timestamp = historical_message[0]

        # -------------------------- 02 ----------------------------
        self.order_book = SignalProcessor(self.order_book)(signal = InsideSignalProducer(self.order_book, historical_message)())
        self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, timestamp, self.index)
        
        # -------------------------- 03 ----------------------------
        assert utils.is_right_answer(self.order_book, self.index, self.bid_sid_historical_data), "the orderbook if different from the data"
        self.index += 1
        if Debugger.on: 
            print("brief_self.order_book(self.order_book)")
            print(utils.brief_order_book(self.order_book))
            # self.order_book.asks = None # remove the ask side
            print("=="*10 + "=" + "=====" + "="+ "=="*10+'\n')
        
    def modify(self):
        for index in range(self.horizon): # size : self.horizon
            self.step()
                    
if __name__ == "__main__":
    # =============================================================================
    # 02 READ DATA
    # =============================================================================
    df2 = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)


    df = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
    df.columns = ["timestamp",'type','order_id','quantity','price','side','remark']
    df["timestamp"] = df["timestamp"].astype(str)


    # =============================================================================
    # 03 REVISING OF ORDERBOOK
    # =============================================================================
    
    decoder =  Decoder(price_level = Configuration.price_level, horizon = Configuration.horizon, historical_data = df2, data_loader = df)
    decoder.modify()
    # breakpoint() # tbd
