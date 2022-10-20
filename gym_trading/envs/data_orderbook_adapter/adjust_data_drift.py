# -*- coding: utf-8 -*-
import numpy as np
from gym_trading.envs.data_orderbook_adapter import Debugger

from gym_trading.envs.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_trading.envs.data_orderbook_adapter.utils.OutsideSignalProducer import OutsideSignalProducer 
        
class DataAdjuster():
    def __init__(self, d2):
        self.d2 = d2
        self.adjust_data_drift_id = 10000
        
    def get_message_auxiliary_info(self, timestamp):
        self.adjust_data_drift_id += 1
        trade_id = self.adjust_data_drift_id
        order_id = self.adjust_data_drift_id
        str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
        timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
        return timestamp, order_id, trade_id           
    
    def adjust_data_drift(self, order_book, timestamp, index):
        timestamp, order_id, trade_id  = self.get_message_auxiliary_info(timestamp)
        historical_message = [index, self.d2, timestamp, order_id, trade_id]
        signal = OutsideSignalProducer(order_book, historical_message)()
        order_book = SignalProcessor(order_book)(signal)
        return order_book

    
    # =============================================================================
    

