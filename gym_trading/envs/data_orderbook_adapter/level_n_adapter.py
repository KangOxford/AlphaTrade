# -*- coding: utf-8 -*-
# =============================================================================
# 01 IMPORT PACKAGES
# =============================================================================
import numpy as np
import pandas as pd
from copy import copy
from gym_trading.envs.data_orderbook_adapter import utils
from gym_trading.envs.data_orderbook_adapter.adjust_data_drift import DataAdjuster
from gym_trading.envs.orderbook import OrderBook

class Debugger: 
    on = True

class SignalProducer:
    def __init__(self, order_book, historical_message):
        self.historical_message = historical_message
        self.order_book = order_book
        self.best_bid = self.order_book.get_best_bid()
    def __call__(self):
        # ---------------------------- 01 ---------------------------- 
        ttype = self.historical_message[1] 
        side = 'bid' if self.historical_message[5] == 1 else 'ask'
        quantity = self.historical_message[3]
        price = self.historical_message[4]
        trade_id = self.historical_message[2] # not sure, in the data it is order id
        order_id = self.historical_message[2]
        timestamp = self.historical_message[0]
        
        # ---------------------------- 02 ---------------------------- 
        message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
        if Debugger.on:  print(self.historical_message)#tbd
        
        # ---------------------------- 03 ---------------------------- 
        sign = ttype
        if side == 'bid':
            if ttype == 1: pass
            elif ttype == 2: # cancellation (partial deletion of a limit order)
                origin_quantity = self.order_book.bids.get_order(order_id).quantity # origin_quantity is the quantity in the order book
                adjusted_quantity = origin_quantity - quantity # quantity is the delta quantity
                message['quantity'] = adjusted_quantity
            elif ttype == 3: # deletion (total deletion of a limit order) inside orderbook
                if price > self.best_bid:
                    sign = 6
                else: pass
            elif ttype == 4 or ttype == 5: # not sure???
                if side == 'bid' and price <= self.best_bid:
                    message['side'] = 'ask' 
                else:
                    sign = 6
            elif ttype == 6: pass
            else: raise NotImplementedError
        else:
            sign = 6
        signal = dict({'sign': sign},**message)   
        return signal
        

class SignalProcessor:
    def __init__(self, order_book):
        self.order_book = order_book
        
    def delete_order(self, message):
        # para: self.order_book, message
        try: self.order_book.cancel_order(message['side'], message['trade_id'], message['timestamp'])
        except: 
            order_list = self.order_book.bids.get_price_list(message['price'])
            assert len(order_list) == 1
            order = order_list.head_order
            self.order_book.cancel_order(side = 'bid', 
                                    order_id = order.order_id,
                                    time = order.timestamp, 
                                    )
    def __call__(self, signal):
        message = signal.copy().pop("sign")
        if signal['sign'] in (1, 4, 5):
            trades, order_in_book = self.order_book.process_order(message, True, False)
            if Debugger.on: 
                print("Trade occurs as follows:"); print(trades)
                print("The order book now is:");   print(self.order_book)
        elif signal['sign'] in (2, ): # cancellation (partial deletion of a limit order)
            self.order_book.bids.update_order(message) 
        elif signal['sign'] in (3, ):# deletion (total deletion of a limit order)
            self.delete_order(message)
        elif signal['sign'] in (6, ): pass # do nothing
        else: raise NotImplementedError
        return self.order_book
    
class Decoder:
    def __init__(self, order_book, price_level, horizon, historical_data, data_loader): 
        self.order_book = order_book
        self.price_level = price_level
        self.horizon = horizon
        self.historical_data = historical_data
        self.data_adjuster = DataAdjuster(historical_data)
        self.data_loader = data_loader
        self.index = 0
        
    def reset(self):
        pass
    
    def step(self):
        # -------------------------- 01 ----------------------------
        if Debugger.on: 
            print("=="*10 + " " + str(self.index) + " "+ "=="*10)
            print("The order book used to be:"); print(self.order_book)
        historical_message = self.data_loader.iloc[self.index,:]
        timestamp = historical_message[0]

        # -------------------------- 02 ----------------------------
        self.order_book = SignalProcessor(self.order_book)(signal = SignalProducer(self.order_book, historical_message)())
        self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, timestamp, self.index)
        
        # -------------------------- 03 ----------------------------
        assert utils.is_right_answer(self.order_book, self.index, d2), "the orderbook if different from the data"
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
    order_book = OrderBook()
    price_level = 10 # wont be changed during running
    horizon = 2048

    # =============================================================================
    # 02 READ DATA
    # =============================================================================
    df2 = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)

    column_numbers=[i for i in range(price_level * 4) if i%4==2 or i%4==3]
    l2 = df2.iloc[0,:].iloc[column_numbers].reset_index().drop(['index'],axis = 1)

    # -------------------------------------------------------------------------
    column_numbers=[i for i in range(price_level * 4) if i%4==2 or i%4==3]
    d2 = df2.iloc[:,column_numbers]  
    # -------------------------------------------------------------------------


    df = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
    df.columns = ["timestamp",'type','order_id','quantity','price','side','remark']
    df["timestamp"] = df["timestamp"].astype(str)

    # =============================================================================
    # 03 CONFIGURARION OF ORDERBOOK
    # =============================================================================
    
    def initialize_orderbook(l2):
        limit_orders = []
        order_id_list = [15000000 + i for i in range(price_level)]
        for i in range(price_level):
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
    initialized_orderbook = initialize_orderbook(l2)

    # =============================================================================
    # 04 REVISING OF ORDERBOOK
    # =============================================================================
    
    decoder =  Decoder(order_book = initialized_orderbook, price_level = 10, horizon = 2048, historical_data = d2, data_loader = df)
    decoder.modify()

        
    
    # breakpoint()    
    # # tbd
    # =============================================================================
