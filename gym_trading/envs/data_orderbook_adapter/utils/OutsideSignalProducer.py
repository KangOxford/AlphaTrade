
#......................................................................................
#........CCC............LL.................AAA.............SSSSSS...........SSSSSS.....
#......CCCCCCCC........LLLL...............AAAAA...........SSSSSSSS.........SSSSSSSS....
#.....CCCCCCCCCC.......LLLL...............AAAAA..........SSSSSSSSSS.......SSSSSSSSSS...
#....CCCCCCCCCCC.......LLLL..............AAAAAAA.........SSSSSSSSSS.......SSSSSSSSSS...
#....CCCC...CCCCC......LLLL..............AAAAAAA........SSSS...SSSSS..... SSS...SSSSS..
#...CCCCC....CCCC......LLLL..............AAAAAAA........SSSSSS........... SSSSS........
#...CCCC...............LLLL.............AAAAAAAAA........SSSSSSSSS........SSSSSSSSS....
#...CCCC...............LLLL.............AAAA.AAAA........SSSSSSSSSS.......SSSSSSSSSS...
#...CCCC...............LLLL.............AAAAAAAAAA.........SSSSSSSSS........SSSSSSSSS..
#...CCCCC....CCCC......LLLL............AAAAAAAAAAA......SSSS..SSSSSS..... SSS..SSSSSS..
#....CCCC...CCCCC......LLLL............AAAAAAAAAAA......SSSS....SSSS..... SSS....SSSS..
#....CCCCCCCCCCC.......LLLLLLLLLL......AAAAAAAAAAAA.....SSSSSSSSSSSS..... SSSSSSSSSSS..
#.....CCCCCCCCCC.......LLLLLLLLLL.....AAAAA....AAAA......SSSSSSSSSS.......SSSSSSSSSS...
#......CCCCCCCC........LLLLLLLLLL.....AAAA.....AAAA.......SSSSSSSSS........SSSSSSSSS...
#.......CCCCC..............................................SSSSSS...........SSSSSS.....
#......................................................................................

import numpy as np
from gym_trading.envs.data_orderbook_adapter import Debugger
from gym_trading.envs.data_orderbook_adapter import utils
 
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