# -*- coding: utf-8 -*-

import pandas as pd
from gym_exchange.data_orderbook_adapter import Configuration 
class DataPipeline:
    def __init__(self):
        self.historical_data = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)
        # column_numbers_ask = [i for i in range(Configuration.price_level * 4) if i%4==0 or i%4==1]
        # l2 = historical_data.iloc[0,:].iloc[column_numbers_ask].reset_index().drop(['index'],axis = 1)
        # column_numbers_bid = [i for i in range(Configuration.price_level * 4) if i%4==2 or i%4==3]
        # r2 = historical_data.iloc[0,:].iloc[column_numbers_bid].reset_index().drop(['index'],axis = 1)
    
        self.data_loader = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
        self.data_loader.columns = ["timestamp",'type','order_id','quantity','price','side','remark']
        self.data_loader["timestamp"] = self.data_loader["timestamp"].astype(str)
        
    def __call__(self):
        # return self.historical_data, self.data_loader
        return {'price_level':Configuration.price_level, 
                'horizon':Configuration.horizon, 
                'historical_data':self.historical_data, 
                'data_loader':self.data_loader}