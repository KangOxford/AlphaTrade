
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

class InsideSignalProducer:
    def __init__(self, order_book, historical_message):
        self.historical_message = historical_message
        self.order_book = order_book
        self.best_bid = self.order_book.get_best_bid()
    def __call__(self):
        # ---------------------------- 01 ---------------------------- 
        ttype = self.historical_message[1] 
        side = 'bid' if self.historical_message[5] == 1 else 'ask'
        quantity = self.historical_message[3]
        price = self.historical_message[4]
        trade_id = self.historical_message[2] # not sure, in the data it is order id
        order_id = self.historical_message[2]
        timestamp = self.historical_message[0]
        
        # ---------------------------- 02 ---------------------------- 
        message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
        if Debugger.on:  print(self.historical_message)#tbd
        
        # ---------------------------- 03 ---------------------------- 
        sign = ttype
        if side == 'bid':
            if ttype == 1: pass
            elif ttype == 2: # cancellation (partial deletion of a limit order)
                origin_quantity = self.order_book.bids.get_order(order_id).quantity # origin_quantity is the quantity in the order book
                adjusted_quantity = origin_quantity - quantity # quantity is the delta quantity
                message['quantity'] = adjusted_quantity
            elif ttype == 3: # deletion (total deletion of a limit order) inside orderbook
                if price > self.best_bid:
                    sign = 6
                else: pass
            elif ttype == 4 or ttype == 5: # not sure???
                if side == 'bid' and price <= self.best_bid:
                    message['side'] = 'ask' 
                else: sign = 6
            elif ttype == 6: pass
            else: raise NotImplementedError
        else: sign = 6
        signal = dict({'sign': sign},**message)   
        return signal