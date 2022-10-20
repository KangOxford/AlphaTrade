
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

from gym_exchange.data_orderbook_adapter import Debugger
from gym_exchange.data_orderbook_adapter.utils import partly_cancel

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
            print(order)#tbd
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
            self.order_book = partly_cancel(self.order_book, message['right_order_price'], message['wrong_order_price'])
        elif signal['sign'] in ((3, ) + ()):# deletion (total deletion of a limit order)
            self.delete_order(message)
        elif signal['sign'] in (30, ): 
            return self.delete_order_ouside(message)
        elif signal['sign'] in ((6, ) + (60, )): pass # do nothing
        else: raise NotImplementedError
        return self.order_book      