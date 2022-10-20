# -*- coding: utf-8 -*-
import numpy as np
from gym_trading.envs.data_orderbook_adapter import Debugger
from gym_trading.envs.data_orderbook_adapter import utils

    
        
class DataAdjuster():
    def __init__(self, d2):
        self.d2 = d2

    def adjust_data_drift(self, order_book, timestamp, index):
        signal = OutsideSingalProducer(order_book, historical_message = [index, self.d2])()
        order_book = SignalProcessor(order_book)(signal)
        return order_book

    
    # =============================================================================
    

