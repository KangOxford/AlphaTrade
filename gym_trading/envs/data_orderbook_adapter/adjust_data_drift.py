# -*- coding: utf-8 -*-
import numpy as np
from gym_trading.envs.data_orderbook_adapter import Debugger
from gym_trading.envs.data_orderbook_adapter import utils

    
class OutsideSingalProducer:
    def __init__(self, order_book, historical_message):
        self.historical_message = historical_message
        self.order_book = order_book
        self.my_array, self.right_array = self.pre_process(historical_message)
    def pre_process_historical_message(self, historical_message):
        index, d2 = historical_message[0], historical_message[1]
        my_list, right_list = utils.get_two_list4compare(self.order_book, index, d2)
        my_array, right_array = np.array(my_list), np.array(right_list)
        return my_array, right_array
    def __call__(self,):    
        if np.sum(self.my_array != self.right_array) == 0:
            signal = {'sign':60}# do nothing
        else:
            if np.sum(self.my_array != self.right_array) == 1:
                signal = one_difference_signal_producer(self.order_book, self.my_array, self.right_array)
            elif np.sum(self.my_array != self.right_array) == 2:
                signal = two_difference_signal_producer(self.order_book, self.my_array, self.right_array)
            else: raise NotImplementedError       
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
            
    def delete_order_ouside(self, message):
        # para: self.order_book, message
        # func: search quanity and delete order ouside lob
        # return: orderbook
        order_list = message['order_list']
        quantity = message['quantity']
        for order in order_list:
            if order.quantity == quantity:
                self.order_book.cancel_order(side = 'bid', 
                                        order_id = order.order_id,
                                        time = order.timestamp, 
                                        )
                return self.order_book # here just cancel single order, not combined order
        raise NotImplementedError
    
    def __call__(self, signal):
        message = signal.copy(); message.pop("sign")
        if signal['sign'] in ((1, 4, 5) + (10, 11)):
            trades, order_in_book = self.order_book.process_order(message, True, False)
            if Debugger.on: 
                print("Trade occurs as follows:"); print(trades)
                print("The order book now is:");   print(self.order_book)
        elif signal['sign'] in ((2, ) + ()): # cancellation (partial deletion of a limit order)
            self.order_book.bids.update_order(message) 
        elif signal['sign'] in (20, ): 
            self.order_book = utils.partly_cancel(self.order_book, message['right_order_price'], message['wrong_order_price'])
        elif signal['sign'] in ((3, ) + ()):# deletion (total deletion of a limit order)
            self.delete_order(message)
        elif signal['sign'] in (30, ): 
            return self.delete_order_ouside(message)
        elif signal['sign'] in ((6, ) + (60, )): pass # do nothing
        else: raise NotImplementedError
        return self.order_book        
 
def one_difference_signal_producer(order_book, my_array, right_array):
    message = {}
    if my_array[-2] == right_array[-2] :
        price = right_array[-2]
        if my_array[-1] < right_array[-1]:
            # Submission of a new limit order, price exist(outside lob)
            # =============================================================================
            # my_array
            # [31170000      176 31169900        1 31169800        1 31167000        3
            #  31161600        3 31160800        1 31160000       37 31158000        7
            #  31155500       70 31155100       50]
            # ----------------------------------------------------------------------------
            # right_array
            # [31170000      176 31169900        1 31169800        1 31167000        3
            #  31161600        3 31160800        1 31160000       37 31158000        7
            #  31155500       70 31155100       51]
            # =============================================================================
            quantity = right_array[-1] - my_array[-1]
            side = 'bid'
            sign= 10
            
        elif my_array[-1] > right_array[-1]:
            # deletion of a limit order(outside lob). Might be partly deleted for the price
            # =============================================================================
            # part of order_list at this price has been partly cancelled outside the order book
            # Quantity    20  |  Price 31155100  |  Trade_ID   15227277  |  Time 34200.290719105
            # Quantity     1  |  Price 31155100  |  Trade_ID      10003  |  Time 34204.721258569
            # ----------------------------------------------------------------------------
            # my_array
            # [31171400, 5, 31171000, 200, 31167100, 4, 31160000, 4, 31159800, 20, 
            #  31158100, 10, 31158000, 7, 31157700, 1, 31155500, 70, 31155100, 21]
            # ----------------------------------------------------------------------------
            # right_array
            # [31171400, 5, 31171000, 200, 31167100, 4, 31160000, 4, 31159800, 20, 
            #  31158100, 10, 31158000, 7, 31157700, 1, 31155500, 70, 31155100, 1]
            # =============================================================================
            message['quantity'] = my_array[-1] - right_array[-1] 
            message['order_list'] =  order_book.bids.get_price_list(price)
            sign = 30

    elif my_array[-1] == right_array[-1]:
        # Submission of a new limit order, price does not exist(outside lob)
        # =============================================================================
        # my_array
        # [31169900        1 31169800        1 31167000        3 31161600        3
        #  31160800        1 31160000       37 31158000        7 31155500       70
        #  31155100       51 31152200       16]
        # right_array
        # [31169900        1 31169800        1 31167000        3 31161600        3
        #  31160800        1 31160000       37 31158000        7 31155500       70
        #  31155100       51 31154300       16]
        # =============================================================================
        price = right_array[-2]
        quantity = right_array[-1]
        side = 'bid'
        sign = 11
    elif my_array[-1] != right_array[-1] and  my_array[-2] != right_array[-2]: raise NotImplementedError # two actions needs to be taken in this step
    else: raise NotImplementedError
        
    timestamp, order_id, trade_id = self.get_message_auxiliary_info()
    message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
    signal = dict({'sign': sign},**message)  
    return signal 

def two_difference_signal_producer(order_book, my_array, right_array):
    if right_array[-2] >  my_array[-2]:
        # Submission of a new limit order, price does not exist(outside lob)                    
        # =============================================================================
        # my_array
        # [31177500       57 31175000        5 31170000      178 31169900        1
        #  31169800        1 31167000        3 31161600        3 31160800        1
        #  31160000       37 31155100       50]
        # right_array
        # [31177500       57 31175000        5 31170000      178 31169900        1
        #  31169800        1 31167000        3 31161600        3 31160800        1
        #  31160000       37 31158000        7]
        # =============================================================================
        side = 'bid'
        price = right_array[-2]
        quantity = right_array[-1]
        sign = 11
        
        timestamp, order_id, trade_id = self.get_message_auxiliary_info()
        message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
    elif right_array[-2] <  my_array[-2]:
        # Cancellation (Partial deletion of a limit order), (outside lob)
        # =============================================================================
        # part of order_list at this price has been partly cancelled outside the order book              
        # ----------------------------------------------------------------------------
        # my_array
        # [31209700, 210, 31204400, 1, 31203500, 100, 31201000, 8, 31200500, 1, 
        # 31200000, 4, 31194000, 8, 31187600, 8, 31187400, 13, 31187200, 10]
        # ----------------------------------------------------------------------------
        # my_array
        # [31209700, 210, 31204400, 1, 31203500, 100, 31201000, 8, 31200500, 1, 
        # 31200000, 4, 31194000, 8, 31187600, 8, 31187400, 13, 31187100, 1]
        # =============================================================================
        sign = 20
        message = {"right_order_price": right_array[-2], "wrong_order_price":my_array[-2]}
        # order_book = utils.partly_cancel(order_book, right_order_price, wrong_order_price)
    else: raise NotImplementedError
    signal = dict({'sign': sign},**message)  
    return signal

 
 
        
class DataAdjuster():
    def __init__(self, d2):
        self.adjust_data_drift_id = 10000
        self.d2 = d2
        
    def get_message_auxiliary_info(self):
        self.adjust_data_drift_id += 1
        trade_id = self.adjust_data_drift_id
        order_id = self.adjust_data_drift_id
        str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
        timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
        return timestamp, order_id, trade_id    
    
    def adjust_data_drift(self, order_book, timestamp, index):
            
        

        signal = OutsideSingalProducer(order_book, historical_message = [index, self.d2])()
        order_book = SignalProcessor(order_book)(signal)
        return order_book

    
    # =============================================================================
    

