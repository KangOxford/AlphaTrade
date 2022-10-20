# -*- coding: utf-8 -*-
# used as Level_N_Adapter
# =============================================================================
# 01 IMPORT PACKAGES
# =============================================================================
import pandas as pd
from gym_exchange.data_orderbook_adapter import Debugger, Configuration 
from gym_exchange.data_orderbook_adapter import utils
from gym_exchange.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_exchange.data_orderbook_adapter.utils.InsideSignalProducer import InsideSignalProducer
from gym_exchange.data_orderbook_adapter.data_adjuster import DataAdjuster
from gym_exchange.orderbook import OrderBook

class Decoder:
    def __init__(self, price_level, horizon, historical_data, data_loader): 
        self.historical_data = historical_data
        self.price_level = price_level
        self.horizon = horizon
        self.data_loader = data_loader
        self.index = 0
        # --------------- NEED ACTIONS --------------------
        self.column_numbers_bid = [i for i in range(price_level * 4) if i%4==2 or i%4==3]
        self.column_numbers_ask = [i for i in range(price_level * 4) if i%4==0 or i%4==1]
        self.bid_sid_historical_data = historical_data.iloc[:,self.column_numbers_bid]
        self.ask_sid_historical_data = historical_data.iloc[:,self.column_numbers_ask]
        self.order_book = OrderBook()
        self.initialize_orderbook('bid')
        # self.initialize_orderbook('ask')
        self.data_adjuster = DataAdjuster(self.bid_sid_historical_data)
        
    def initialize_orderbook(self, side):
        columns = self.column_numbers_bid if side == 'bid' else self.column_numbers_ask
        l2 = self.historical_data.iloc[0,:].iloc[columns].reset_index().drop(['index'],axis = 1)
        limit_orders = []
        order_id_list = [15000000 + 100*(side == 'bid') + i for i in range(self.price_level)]
        for i in range(self.price_level):
            trade_id = 90000 + + 100*(side == 'bid')
            # timestamp = datetime(34200.000000001)
            timestamp = str(34200.000000002) if (side == 'bid') else str(34200.000000001)
            item = {'type' : 'limit', 
                'side' : side, 
                'quantity' : l2.iloc[2 * i + 1,0], 
                'price' : l2.iloc[2 * i,0],
                'trade_id' : trade_id,
                'order_id' : order_id_list[i],
                "timestamp": timestamp}
            limit_orders.append(item)
        # Add orders to order book
    
        for order in limit_orders:
            # breakpoint()
            trades, order_id = self.order_book.process_order(order, True, False)   
        # The current book may be viewed using a print
        if Debugger.on: print(self.order_book)
    
    
    
    def step(self):
        # -------------------------- 01 ----------------------------
        if Debugger.on: 
            print("##"*25 + '###' + "##"*25)
            print("=="*25 + " " + str(self.index) + " "+ "=="*25)
            print("##"*25 + '###' + "##"*25+'\n')
            
            print("The order book used to be:"); print(self.order_book)
        historical_message = self.data_loader.iloc[self.index,:]
        timestamp = historical_message[0]
        side = 'bid' if historical_message[5] == 1 else 'ask'

        # -------------------------- 02 ----------------------------
        signal          = InsideSignalProducer(self.order_book, historical_message)()
        self.order_book = SignalProcessor(self.order_book)(signal)
        self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, timestamp, self.index, side)
        
        # -------------------------- 03 ----------------------------
        if Debugger.on: print(">>> Right_order_book"); print(utils.get_right_answer(self.index, self.bid_sid_historical_data))
        assert utils.is_right_answer(self.order_book, self.index, self.bid_sid_historical_data), "the orderbook if different from the data"
        self.index += 1
        if Debugger.on: 
            # print("The order book now is:"); print(self.order_book)
            print(">>> Brief_self.order_book(self.order_book)")
            print(utils.brief_order_book(self.order_book))
            # self.order_book.asks = None # remove the ask side
            print("The orderbook is right!\n")
            # print("=="*10 + "=" + "=====" + "="+ "=="*10+'\n')
        
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
