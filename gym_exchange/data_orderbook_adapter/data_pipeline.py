# -*- coding: utf-8 -*-
import itertools
from tkinter.messagebox import NO
import pandas as pd
import numpy as np
from gym_exchange import Config
class DataPipeline:
    def __init__(self):
        if Config.raw_price_level == 10:
            symbol = "TSLA";date = "2015-01-02"
            self.historical_data = pd.read_csv("/Users/sasrey/AlphaTrade/data/TSLA_2015-01-02_34200000_57600000_orderbook_10.csv", header = None)
            self.data_loader = pd.read_csv("/Users/sasrey/AlphaTrade/data/TSLA_2015-01-02_34200000_57600000_message_10.csv", header=None)


            # symbol = "AMZN";date = "2021-04-01"
            # self.historical_data = pd.read_csv(
            #     "/Users/kang/Data/" + symbol + "_" + date + "_34200000_57600000_orderbook_10.csv", header=None)
            # self.data_loader = pd.read_csv(
            #     "/Users/kang/Data/" + symbol + "_" + date + "_34200000_57600000_message_10.csv", header=None)

            # self.historical_data = pd.read_csv("/Users/sasrey/project-RL4ABM/data_tqap/TSLA_2015-01-01_2015-01-31_10/TSLA_2015-01-02_34200000_57600000_orderbook_10.csv", header = None)
            # self.data_loader = pd.read_csv("/Users/sasrey/project-RL4ABM/data_tqap/TSLA_2015-01-01_2015-01-31_10/TSLA_2015-01-02_34200000_57600000_message_10.csv", header=None)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
