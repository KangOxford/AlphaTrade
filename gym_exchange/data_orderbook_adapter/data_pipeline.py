# -*- coding: utf-8 -*-
import itertools
import pandas as pd
import numpy as np
from gym_exchange import Config
class DataPipeline:
    def __init__(self):
        if Config.raw_price_level == 10:
            # symbol = "TSLA";date = "2015-01-02"
            symbol = "AMZN";date = "2021-04-01"
            self.historical_data = pd.read_csv("/Users/kang/Data/"+symbol+"_"+date+"_34200000_57600000_orderbook_10.csv", header = None)
            self.data_loader = pd.read_csv("/Users/kang/Data/"+symbol+"_"+date+"_34200000_57600000_message_10.csv", header=None)
        elif Config.raw_price_level == 50:
            self.historical_data = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_50.csv", header = None)
            self.data_loader = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_50.csv", header=None)
        else: raise NotImplementedError    
        
        self.data_loader.columns = ["timestamp",'type','order_id','quantity','price','side','remark']
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
