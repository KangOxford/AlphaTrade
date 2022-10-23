
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
from gym_exchange.data_orderbook_adapter import Debugger, Configuration 
from gym_exchange.data_orderbook_adapter.utils import get_two_list4compare

class OutsideSignalEncoder:
    def __init__(self, order_book, historical_message):
        self.historical_message = historical_message
        self.order_book = order_book
        
    
    def pre_process_historical_message(self, historical_message, side):
        index, d2 = historical_message[0], historical_message[1]
        my_list, right_list = get_two_list4compare(self.order_book, index, d2, side)
        my_array, right_array = np.array(my_list), np.array(right_list)
        return my_array, right_array, historical_message[2], historical_message[3], historical_message[4]
        
    def __call__(self, side):   
        self.my_array, self.right_array, self.timestamp, self.order_id, self.trade_id = self.pre_process_historical_message(self.historical_message, side)
        if Debugger.on: print(">>> my_array");print(self.my_array)
        if Debugger.on: print(">>> right_array");print(self.right_array)
        if np.sum(self.my_array != self.right_array) == 0:
            signal = {'sign':60}# do nothing
        else:
            if Debugger.on: print('\n'+"ADJUSTING"+'\n')
            # if Debugger.on: print(self.my_array); print(self.right_array); print('\n'+"ADJUSTED"+'\n')
            if np.sum(self.my_array != self.right_array) == 1:
                signal = self.one_difference_signal_producer(self.order_book, self.my_array, self.right_array, side)
            elif np.sum(self.my_array != self.right_array) == 2:
                signal = self.two_difference_signal_producer(self.order_book, self.my_array, self.right_array, side)
            else: 
                raise NotImplementedError       
        return signal 

    def one_difference_signal_producer(self, order_book, my_array, right_array, side):
        timestamp, order_id, trade_id = self.timestamp, self.order_id, self.trade_id
        message = {'type': 'limit', 'timestamp': timestamp, 'order_id': order_id, 'trade_id': trade_id}
        if my_array.size == Configuration.price_level * 2:
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
                    # side = 'bid'
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
                    quantity = my_array[-1] - right_array[-1] 
                    # side = 'bid'
                    message['order_list'] =  order_book.bids.get_price_list(price)
                    sign = 30
                    # breakpoint()#tbd
        
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
                # side = 'bid'
                sign = 11
            elif my_array[-1] != right_array[-1] and  my_array[-2] != right_array[-2]: raise NotImplementedError # two actions needs to be taken in this step
            else: raise NotImplementedError
        elif my_array.size == (Configuration.price_level - 1) * 2:
            # =============================================================================
            # my_array
            # [31190000        2 31200000       18 31210000        3 31214000        2
            #  31220000        4 31229800      100 31230000       24 31237900        1
            #  31240000        4]
            # right_array
            # [31190000        2 31200000       18 31210000        3 31214000        2
            #  31220000        4 31229800      100 31230000       24 31237900        1
            #  31240000        4 31240800        5], ask price
            # =============================================================================
            price, quantity = right_array[-2],  right_array[-1]
            sign = 11
        else: raise NotImplementedError
        message['side'], message['quantity'], message['price'] = side, quantity, price
        signal = dict({'sign': sign},**message)  
        return signal 

    def two_difference_signal_producer(self, order_book, my_array, right_array, side):
        if ~((right_array[-2] >  my_array[-2])^(side == 'bid')):
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
            # side = 'bid'
            price = right_array[-2]
            quantity = right_array[-1]
            sign = 11
            
            timestamp, order_id, trade_id = self.timestamp, self.order_id, self.trade_id
            message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
        elif ~((right_array[-2] <  my_array[-2])^(side == 'bid')):
            # !! caution haven't tested for all ask situations 
            # breakpoint()
            sign = 20
            message = {"right_order_price": right_array[-2], "wrong_order_price":my_array[-2],'side':side}            
            
            # ================================== EXAMPLE 1 ================================
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
            
            # ================================== EXAMPLE 2 ================================
            # | Quantity   100  |  Price 31171000  |  Trade_ID   17152985  |  Time 34216.637352306
            # | Quantity   100  |  Price 31171000  |  Trade_ID   17218081  |  Time 34220.295996111
            # | Quantity     4  |  Price 31167100  |  Trade_ID   17313633  |  Time 34220.61156594 
            # | Quantity     4  |  Price 31160000  |  Trade_ID   17177301  |  Time 34217.174425019
            # | Quantity    20  |  Price 31159800  |  Trade_ID   17110257  |  Time 34215.711125741
            # | Quantity    10  |  Price 31158100  |  Trade_ID   16720913  |  Time 34209.347077682
            # | Quantity     7  |  Price 31158000  |  Trade_ID      10086  |  Time 34201.238593006
            # | Quantity     1  |  Price 31157700  |  Trade_ID   16782961  |  Time 34210.304185033
            # | Quantity    70  |  Price 31155500  |  Trade_ID      10089  |  Time 34201.238727592
            # | Quantity     1  |  Price 31155100  |  Trade_ID      10604  |  Time 34204.721258569
            # > Quantity    16  |  Price 31154300  |  Trade_ID      10655  |  Time 34204.879222266
            # > Quantity   100  |  Price 31154200  |  Trade_ID      11832  |  Time 34216.905286670
            # | Quantity     5  |  Price 31153300  |  Trade_ID      10659  |  Time 34204.880820915
            # | Quantity    16  |  Price 31152200  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity   100  |  Price 31151600  |  Trade_ID   16551229  |  Time 34205.787263469
            # | Quantity     2  |  Price 31151000  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity    15  |  Price 31151000  |  Trade_ID      10707  |  Time 34205.043944893
            # | Quantity     2  |  Price 31150100  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity   506  |  Price 31150000  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity   121  |  Price 31150000  |  Trade_ID      10714  |  Time 34205.044381822
            # | Quantity    28  |  Price 31144200  |  Trade_ID   15563997  |  Time 34200.950954186
            # | Quantity     4  |  Price 31140000  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity     2  |  Price 31130000  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity    35  |  Price 31120300  |  Trade_ID      90100  |  Time 34200.000000002
            # | Quantity    35  |  Price 31120200  |  Trade_ID      90100  |  Time 34200.000000002
            # | ====================================== Bids ======================================
            # | my_array
            # | [31171000      200 31167100        4 31160000        4 31159800       20
            # |  31158100       10 31158000        7 31157700        1 31155500       70
            # |  31155100        1 31154300       16]
            # | right_array
            # | [31171000      200 31167100        4 31160000        4 31159800       20
            # |  31158100       10 31158000        7 31157700        1 31155500       70
            # |  31155100        1 31153300        5]
            # ====================================================================================
            
            # ================================== EXAMPLE 3 ================================
            # | ask index 524
            # | ====================================== Asks ======================================
            # | Quantity     2  |  Price 31246700  |  Trade_ID    5000166  |  Time 34201.446177486
            # | Quantity     5  |  Price 31240800  |  Trade_ID    5000003  |  Time 34200.012480375
            # | Quantity     4  |  Price 31240000  |  Trade_ID      90000  |  Time 34200.000000001
            # | Quantity     1  |  Price 31237900  |  Trade_ID      90000  |  Time 34200.000000001
            # | Quantity    18  |  Price 31231300  |  Trade_ID   15815645  |  Time 34201.538229675
            # | Quantity    24  |  Price 31230000  |  Trade_ID      90000  |  Time 34200.000000001
            # | Quantity    40  |  Price 31230000  |  Trade_ID   15818437  |  Time 34201.54624381 
            # | Quantity   100  |  Price 31229800  |  Trade_ID      90000  |  Time 34200.000000001
            # > Quantity     8  |  Price 31227600  |  Trade_ID   16007705  |  Time 34201.926715564
            # > Quantity    16  |  Price 31224700  |  Trade_ID   16170369  |  Time 34202.251699862
            # | Quantity     4  |  Price 31220000  |  Trade_ID      90000  |  Time 34200.000000001            
            # my_array
            # [31190000      100 31200000      725 31200500      200 31209000      300
            #  31209500      100 31210000        3 31214000        2 31218400        4
            #  31220000       54 31224700       16]
            # right_array
            # [31190000      100 31200000      725 31200500      200 31209000      300
            #  31209500      100 31210000        3 31214000        2 31218400        4
            #  31220000       54 31229800      100]
            # ====================================================================================
            
            # ================================== tbd ================================
            # if ~((right_array[-2] < my_array[-2])^(side == 'bid')):
            #     # deletion of a limit order(outside lob). Might be partly deleted for the price
            #     # =============================================================================
            #     # Quantity     3  |  Price 31210000  |  Trade_ID      90000  |  Time 34200.000000001
            #     # Quantity    20  |  Price 31209000  |  Trade_ID   15605909  |  Time 34201.0434255  
            #     # Quantity    18  |  Price 31200000  |  Trade_ID      90000  |  Time 34200.000000001
            #     # ----------------------------------------------------------------------------
            #     # my_array
            #     # [31182000        1 31183300        1 31185500        1 31185600       10
            #     #  31187300       50 31189000        1 31190000      102 31192300        5
            #     #  31200000       18 31209000       20]
            #     # right_array
            #     # [31182000        1 31183300        1 31185500        1 31185600       10
            #     #  31187300       50 31189000        1 31190000      102 31192300        5
            #     #  31200000       18 31210000        3], ask prices
            #     # =============================================================================
            #     quantity = my_array[-1] - right_array[-1] 
            #     message['order_list'] =  order_book.bids.get_price_list(price)
            #     sign = 30
            # else:
            


        else: raise NotImplementedError
        signal = dict({'sign': sign},**message)  
        return signal


    