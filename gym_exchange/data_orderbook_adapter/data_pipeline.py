# -*- coding: utf-8 -*-

import pandas as pd
from gym_exchange import Config
class DataPipeline:
    def __init__(self):
        if Config.raw_price_level == 10:
            self.historical_data = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)
            self.data_loader = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
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