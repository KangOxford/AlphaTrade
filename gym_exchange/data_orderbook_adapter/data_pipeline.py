# -*- coding: utf-8 -*-
import itertools
# from tkinter.messagebox import NO
import pandas as pd
import numpy as np
from gym_exchange import Config
def normalization_config(Config,historical_data):
    Config.price_mean = historical_data.iloc[:, ::2].to_numpy().mean()
    Config.price_std = historical_data.iloc[:, ::2].to_numpy().std()
    Config.qty_mean = historical_data.iloc[:, 1::2].to_numpy().mean()
    Config.qty_std = historical_data.iloc[:, 1::2].to_numpy().std()
def horizon_config(Config, message_data):
    time = message_data.iloc[:,0]/3600
    open_interval = time[time <= 10.0000]
    horizon_length = open_interval.size//100 + 5
    # Config.max_horizon = 5 #$ for easy testing
    Config.max_horizon = horizon_length
    Config.raw_horizon = int(Config.max_horizon * Config.window_size * 1.01)
    print(f"*** horizon_length: {Config.max_horizon}")
    print(f"*** raw_horizon: {Config.raw_horizon}")
def plot_summary(historical_data):
    # length = 49490
    # length = 500
    length = 100
    tobeplotted = historical_data.iloc[:length, [0, 2]]
    tobeplotted.columns = ['best_ask', 'best_bid']
    tobeplotted.best_ask/=10000
    tobeplotted.best_bid/=10000
    tobeplotted['mid_price'] = (tobeplotted.best_ask + tobeplotted.best_bid) / 2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(36, 9))
    # fig, ax = plt.subplots(figsize=(120, 30))
    width = 1.5 if length //1000 == 0 else 0.5
    ax.plot(tobeplotted.best_ask,linewidth=width)
    ax.plot(tobeplotted.best_bid,linewidth=width)
    ax.plot(tobeplotted.mid_price,linewidth=width)
    ax_y_range = ax.get_ylim()
    ax_y = ax_y_range[1] - ax_y_range[0]
    if  3<= length //100 <= 10:
        scaling = 1
        ax.set_ylim([ax_y_range[0], ax_y_range[1] + ax_y * scaling])
    elif length //100 == 49:
        scaling = 0
        ax.set_ylim([ax_y_range[0], ax_y_range[1] + ax_y * scaling])
    elif 0<= length //100 <3:
        scaling1 = 1
        scaling2 = 1
        ax.set_ylim([ax_y_range[0] - ax_y * scaling1, ax_y_range[1] + ax_y * scaling2])
        q_ylim = ax.get_ylim()[1] - ax.set_ylim()[0]
    else: raise NotImplementedError


    q = ax.twinx()
    tobeplotted['spread'] = tobeplotted.best_ask - tobeplotted.best_bid
    q.bar(tobeplotted.index, tobeplotted.spread, color = "red")
    # q.spines['right'].set_position(('axes',1.15))
    # q = plt.gca()
    if 3<= length //100 <= 10 or length //100 == 49:
        q.set_ylim([0, ax_y * (1 + scaling)])
    elif 0 <= length // 100 < 3:
        # q.set_ylim([0, ax_y * (1 + scaling1 + scaling2)])
        q.set_ylim([0, q_ylim])
        print("***")
    else: raise NotImplementedError

    # q.set_ylim([0, 24])
    q.invert_yaxis()
    q.xaxis.tick_top()

    p = ax.twinx()
    qty = historical_data.iloc[:length, [1, 3]]
    tobeplotted['qty'] = qty.iloc[:, 0] + qty.iloc[:, 1]
    p.bar(tobeplotted.index, tobeplotted.qty)
    try:
        p.set_ylim([0, p.get_ylim()[1] * (1 + scaling1 + scaling2)])
    except:
        pass

    # plt.tight_layout()
    import time
    now = time.time()
    plt.savefig("plot_" + str(now)+".png")
    plt.show()

    # tobeplotted['spread'] = tobeplotted.best_ask - tobeplotted.best_bid
    # plt.plot(tobeplotted.spread, color = "red")
    # plt.show()


class DataPipeline:
    def __init__(self):
        if Config.raw_price_level == 10:
            self.historical_data = pd.read_csv(
                Config.AlphaTradeRoot+"data/" + Config.symbol + "_" + Config.date + "_34200000_57600000_orderbook_10.csv", header=None)
            self.data_loader = pd.read_csv(
                Config.AlphaTradeRoot+"data/" + Config.symbol + "_" + Config.date + "_34200000_57600000_message_10.csv", header=None)
            normalization_config(Config, self.historical_data)
            horizon_config(Config, message_data = self.data_loader)
            # plot_summary(self.historical_data)


        elif Config.raw_price_level == 50:
            self.historical_data = pd.read_csv("/Users/sasrey/AlphaTrade/data/TSLA_2015-01-02_34200000_57600000_orderbook_50.csv", header = None)
            self.data_loader = pd.read_csv("/Users/sasrey/AlphaTrade/data/TSLA_2015-01-02_34200000_57600000_message_50.csv", header=None)
        else: raise NotImplementedError    

        self.data_loader.dropna(axis = 1,inplace=True);assert len(self.data_loader.columns) == len(["timestamp",'type','order_id','quantity','price','side'])
        self.data_loader.columns = ["timestamp",'type','order_id','quantity','price','side']
        self.data_loader["timestamp"] = self.data_loader["timestamp"].astype(str)
        
    def __call__(self):
        # return self.historical_data, self.data_loader
        return {'price_level':Config.raw_price_level, 
                'horizon':Config.raw_horizon, 
                'historical_data':self.historical_data, 
                'data_loader':self.data_loader}
    
if __name__ == "__main__":
    ob = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)
    bid_columns =list(itertools.chain(*[[4*i+2, 4*i+3] for i in range(ob.shape[1]//4)])) 
    ask_columns =list(itertools.chain(*[[4*i+0, 4*i+1] for i in range(ob.shape[1]//4)]))
    bid = ob.iloc[:,bid_columns]
    ask = ob.iloc[:,ask_columns]    
    bid.to_csv("/Users/kang/Data/bid.csv", header = None, index=False)
    ask.to_csv("/Users/kang/Data/ask.csv", header = None, index=False)
    
"""
Message File:		(Matrix of size: (Nx6))
-------------	
        
Name: 	TICKER_Year-Month-Day_StartTime_EndTime_message_LEVEL.csv 	
    
    StartTime and EndTime give the theoretical beginning 
    and end time of the output file in milliseconds after 		
    mid night. LEVEL refers to the number of levels of the 
    requested limit order book.


Columns:

    1.) Time: 		
            Seconds after midnight with decimal 
            precision of at least milliseconds 
            and up to nanoseconds depending on 
            the requested period
    2.) Type:
            1: Submission of a new limit order
            2: Cancellation (Partial deletion 
               of a limit order)
            3: Deletion (Total deletion of a limit order)
            4: Execution of a visible limit order			   	 
            5: Execution of a hidden limit order
            6: Indicates a cross trade, e.g. auction trade
            7: Trading halt indicator 				   
               (Detailed information below)
    3.) Order ID: 	
            Unique order reference number 
            (Assigned in order flow)
    4.) Size: 		
            Number of shares
    5.) Price: 		
            Dollar price times 10000 
            (i.e., A stock price of $91.14 is given 
            by 911400)
    6.) Direction:
            -1: Sell limit order
            1: Buy limit order
            
            Note: 
            Execution of a sell (buy) limit
            order corresponds to a buyer (seller) 
            initiated trade, i.e. Buy (Sell) trade.
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
