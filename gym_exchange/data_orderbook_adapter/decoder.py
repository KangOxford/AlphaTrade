# -*- coding: utf-8 -*-
# used as Level_N_Adapter
# =============================================================================
# 01 IMPORT PACKAGES
# =============================================================================
import pandas as pd
from gym_exchange.data_orderbook_adapter import Debugger, Configuration 
from gym_exchange.data_orderbook_adapter import utils
from gym_exchange.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_exchange.data_orderbook_adapter.utils.InsideSignalEncoder import InsideSignalEncoder
from gym_exchange.data_orderbook_adapter.data_adjuster import DataAdjuster
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
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
        self.initialize_orderbook()
        self.data_adjuster = DataAdjuster(d2 = self.bid_sid_historical_data, l2 = self.ask_sid_historical_data)
        
    def initiaze_orderbook_message(self, side):
        columns = self.column_numbers_bid if side == 'bid' else self.column_numbers_ask
        l2 = self.historical_data.iloc[0,:].iloc[columns].reset_index().drop(['index'],axis = 1)
        limit_orders = []
        order_id_list = [15000000 + 100*(side == 'bid') + i for i in range(self.price_level)]
        for i in range(self.price_level):
            trade_id = 9000000 + + 10000*(side == 'bid')
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
        return limit_orders
        
    def initialize_orderbook_with_side(self, side): # Add orders to order book
        limit_orders = self.initiaze_orderbook_message(side)
        for order in limit_orders:  trades, order_id = self.order_book.process_order(order, True, False) # The current book may be viewed using a print 
        if Debugger.on: print(self.order_book)
    
    def initialize_orderbook(self):
        self.initialize_orderbook_with_side('bid')
        self.initialize_orderbook_with_side('ask')
    
    def get_inside_signal(self):
        signal = InsideSignalEncoder(self.order_book, self.historical_message)()
        return signal
        
    
    
    def step(self):
        # -------------------------- 01 ----------------------------
        if Debugger.on: 
            print("##"*25 + '###' + "##"*25);print("=="*25 + " " + str(self.index) + " "+ "=="*25)
            print("##"*25 + '###' + "##"*25+'\n');print("The order book used to be:"); print(self.order_book)
        self.historical_message = self.data_loader.iloc[self.index,:]
        historical_message = list(self.historical_message) # tbd 
        inside_signal = self.get_inside_signal()
        self.order_book = SignalProcessor(self.order_book)(inside_signal)
        
        
        if self.order_book.bids.depth != 0:
            outside_signal_bid, self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, self.historical_message[0], self.index, side = 'bid') # adjust only happens when the side of lob is existed(initialised)
        if self.order_book.asks.depth != 0:
            outside_signal_ask, self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, self.historical_message[0], self.index, side = 'ask') # adjust only happens when the side of lob is existed(initialised)
            
        
        if Debugger.on: 
            # -------------------------- 04.01 ----------------------------
            if self.order_book.bids.depth != 0:
                single_side_historical_data = self.bid_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data, side = 'bid'), "the orderbook if different from the data"
            if self.order_book.asks.depth != 0:
                single_side_historical_data = self.ask_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data, side = 'ask'), "the orderbook if different from the data"
            print(">>> Right_order_book"); print(utils.get_right_answer(self.index, single_side_historical_data))
            # -------------------------- 04.02 ----------------------------
            print(">>> Brief_self.order_book(self.order_book)")
            side = 'bid' if self.historical_message[5] == 1 else 'ask'
            print(utils.brief_order_book(self.order_book, side))
            print("The orderbook is right!\n")
        self.index += 1
        outside_signals = [outside_signal_bid, outside_signal_ask]
        return inside_signal, outside_signals
        
    def process(self):
        
        for index in range(self.horizon): # size : self.horizon
            # inside_signal, outside_signal = self.step()
            _, _ = self.step()
                    
if __name__ == "__main__":
    # =============================================================================
    # 02 REVISING OF ORDERBOOK
    # =============================================================================
    decoder = Decoder(**DataPipeline()())
    decoder.process()
    # breakpoint() # tbd
