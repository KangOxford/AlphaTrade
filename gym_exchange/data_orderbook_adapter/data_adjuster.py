# -*- coding: utf-8 -*-
import numpy as np
from gym_exchange.data_orderbook_adapter import Debugger

from gym_exchange.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_exchange.data_orderbook_adapter.utils.OutsideSignalEncoder import OutsideSignalEncoder
        
class DataAdjuster():
    def __init__(self, d2 = None, l2 = None):
        self.d2 = d2 # bid right_order_book_data
        self.l2 = l2 # ask right_order_book_data
        self.adjust_data_drift_id_bid = 1000000  # caution about the volumn for valid numbers
        self.adjust_data_drift_id_ask = 5000000  # caution about the volumn for valid numbers
        
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
        right_order_book_data = self.d2 if side == 'bid' else self.l2
        historical_message = [index, right_order_book_data, timestamp, order_id, trade_id]

        signal = OutsideSignalEncoder(order_book, historical_message)(side)
        try: order_book = SignalProcessor(order_book)(signal)
        except: 
            for item in signal: order_book = SignalProcessor(order_book)(item)  # EXAMPLE2        
        return order_book
        # ============================== EXAMPLE2 =====================================
        #  | There should be two signal produced, with given sequence: [10 => 20]
        #  | 10 to submit new order outside pricelevel.
        #  | The new submitted price can be used as anchor to cancel orders between 
        #  | Wrong price and Anchor price
        #  | my_array
        #  | [31178200      200 31180000        4 31190000      100 31200000     1014
        #  |  31200800        1 31201200        5 31201900        1 31202000        3
        #  |  31205100      200 31205600       18]
        #  | right_array
        #  | [31178200      200 31180000        4 31190000      100 31200000     1014
        #  |  31200800        1 31201200        5 31201900        1 31202000        3
        #  |  31205100      200 31207200       10]
        #  | order_book.asks.price_map 
        #  | 31210000
        #  * 31208000
        # => 31207200
        #  > 31205600
        #  | 31205100
        #  | 31202000
        #  | 31201900
        #  | 31201200
        #  | 31200800
        #  | 31200000
        #  | 31190000
        #  | 31180000
        #  | 31178200
        # =============================================================================    
