# -*- coding: utf-8 -*-
import numpy as np
from gym_exchange.data_orderbook_adapter import Debugger

from gym_exchange.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_exchange.data_orderbook_adapter.utils.OutsideSignalProducer import OutsideSignalProducer 
        
class DataAdjuster():
    def __init__(self, d2):
        self.d2 = d2
        self.adjust_data_drift_id_bid = 10000  # caution about the volumn for valid numbers
        self.adjust_data_drift_id_ask = 50000  # caution about the volumn for valid numbers
        
    def get_message_auxiliary_info(self, timestamp, side):
        if side == 'bid': self.adjust_data_drift_id_bid += 1; adjust_data_drift_id = self.adjust_data_drift_id_bid
        elif side=='ask': self.adjust_data_drift_id_ask += 1; adjust_data_drift_id = self.adjust_data_drift_id_ask
        
        trade_id = adjust_data_drift_id
        order_id = adjust_data_drift_id
        str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
        timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
        return timestamp, order_id, trade_id           
    
    def adjust_data_drift(self, order_book, timestamp, index, side):
        timestamp, order_id, trade_id  = self.get_message_auxiliary_info(timestamp, side)
        historical_message = [index, self.d2, timestamp, order_id, trade_id]
        signal = OutsideSignalProducer(order_book, historical_message)(side)
        order_book = SignalProcessor(order_book)(signal)
        return order_book

    
    # =============================================================================
    

