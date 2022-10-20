# -*- coding: utf-8 -*-
import numpy as np
from gym_trading.envs.data_orderbook_adapter import Debugger
from gym_trading.envs.data_orderbook_adapter import utils


class DataAdjuster():
    def __init__(self, d2):
        self.adjust_data_drift_id = 10000
        self.d2 = d2

    def adjust_data_drift(self, order_book, timestamp, index):
            
        if Debugger.on: print(order_book)#tbd
        
        my_list, right_list = utils.get_two_list4compare(order_book, index, self.d2)
        my_array, right_array = np.array(my_list), np.array(right_list)
        
        if Debugger.on: print("my_array")
        if Debugger.on: print(my_array)
        if Debugger.on: print("right_array")
        if Debugger.on: print(right_array)
        
        right_order = list(set(right_list) - set(my_list))
        wrong_order = list(set(my_list) -set(right_list))
        if len(right_order) == 0 and len(wrong_order) == 0:
            if Debugger.on: print("no data drift: no incomming new limit order outside the 10 price levels")
            message = None
        else:
            # global adjust_data_drift_id
            if  np.sum(my_array != right_array) == 1:
            # if len(right_order) == 1 and len(wrong_order) == 1:
                if my_array[-2] == right_array[-2] :
                    price = right_array[-2]
                    if my_array[-1] < right_array[-1]:
                        # =============================================================================
                        # my_array
                        # [31170000      176 31169900        1 31169800        1 31167000        3
                        #  31161600        3 31160800        1 31160000       37 31158000        7
                        #  31155500       70 31155100       50]
                        # ----------------------------------------------------------------------------
                        # my_array
                        # [31170000      176 31169900        1 31169800        1 31167000        3
                        #  31161600        3 31160800        1 31160000       37 31158000        7
                        #  31155500       70 31155100       51]
                        # =============================================================================
                        
                        quantity = right_order[0] - wrong_order[0]
                        side = 'bid'
                        
                        # elif my_array[-1] > right_array[-1]:

                        
                    elif my_array[-1] > right_array[-1]:
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
                        
                        
                        quantity = -right_order[0] + wrong_order[0]
                        # side = 'ask' # cancel order here
                        # return cancel_by_price(order_book, price)
                        order_list =  order_book.bids.get_price_list(price)
                        
                        #search quanity
                        for order in order_list:
                            if order.quantity == quantity:
                                order_book.cancel_order(side = 'bid', 
                                                        order_id = order.order_id,
                                                        time = order.timestamp, 
                                                        )
                                return order_book # here just cancel single order, not combined order
                        raise NotImplementedError
                        
# =============================================================================
#                         price = right_array[-2]
#                         quantity = right_array[-1] # right quantity
#                         quantity_list = [] # wrong quantity list
#                         order_id_list = []
#                         timestamp_list = []
#                         for item in order_book.bids.get_price_list(price):
#                             quantity_list.append(item.quantity)
#                             order_id_list.append(item.order_id)
#                             timestamp_list.append(item.timestamp)
#                         try:quantity_index = quantity_list.index(quantity)
#                         except: raise NotImplementedError
#                         difference_index = np.array([i for i in range(len(quantity_list)) if i != quantity_index])
#                         # breakpoint()
#                         returned_timestamp_array = np.array(timestamp_list)[difference_index]
#                         returned_order_id_array = np.array(order_id_list)[difference_index] # returned_order_id_array for cancel order
#                         assert len(returned_order_id_array) == 1, "NotImplemented, only implement the single order situation"
#                         order_book.cancel_order(side = 'bid', 
#                                                 order_id = returned_order_id_array[0],
#                                                 time = returned_timestamp_array[0], 
#                                                 )
#                         message = None
#                         if Debugger.on: print('\n'+'-'*15)
#                         if Debugger.on: print(">>> ADJUSTED <<<")
#                         if Debugger.on: print('-'*15+'\n')
# =============================================================================
                elif my_array[-1] == right_array[-1]:
                    price = right_array[-2]
                    quantity = right_array[-1]
                    side = 'bid'
                elif my_array[-1] != right_array[-1] and  my_array[-2] != right_array[-2]:
                    # two actions needs to be taken in this step
                    pass
                else:
                    raise NotImplementedError
                self.adjust_data_drift_id += 1
                trade_id = self.adjust_data_drift_id
                order_id = self.adjust_data_drift_id
                
                str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
                timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
                
                message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
                if Debugger.on: print('\n'+'-'*15)
                if Debugger.on: print(">>> ADJUSTED <<<")
                if Debugger.on: print('-'*15+'\n')   
            elif np.sum(my_array != right_array) == 2:
                if right_array[-2] >  my_array[-2]:
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
                    self.adjust_data_drift_id += 1
                    trade_id = self.adjust_data_drift_id
                    order_id = self.adjust_data_drift_id
                    
                    str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
                    timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
                    
                    message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
                    if Debugger.on: print('\n'+'-'*15)
                    if Debugger.on: print(">>> ADJUSTED <<<")
                    if Debugger.on: print('-'*15+'\n')
                elif right_array[-2] <  my_array[-2]:
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
                    right_order_price =  right_array[-2]
                    wrong_order_price =  my_array[-2]
                    order_book = utils.partly_cancel(order_book, right_order_price, wrong_order_price)
                    message = None
                else: raise NotImplementedError
            else: raise NotImplementedError
        if message is not None:
            trades, order_in_book = order_book.process_order(message, True, False)
        if Debugger.on: print(order_book) # tbd
        return order_book
        # print("Adjusted Order Book")
        # print(brief_order_book(order_book))
    
    # =============================================================================

        