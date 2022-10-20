# -*- coding: utf-8 -*-
from gym_trading.envs.data_orderbook_adapter import Debugger, Configuration 

#................................................................................................
#...............UU.....UU....NN......N.......CCC......TTTTTTTTT...I......OOOOOO......NN......N...
#...FFFFFFFFFF.UUUU...UUUU..NNNN....NNN....CCCCCCCC..CTTTTTTTTTT.III....OOOOOOOOO...NNNN....NNN..
#...FFFFFFFFFF.UUUU...UUUU..NNNNN...NNN...CCCCCCCCCC.CTTTTTTTTTT.III...OOOOOOOOOO...NNNNN...NNN..
#...FFFFFFFFFF.UUUU...UUUU..NNNNN...NNN..CCCCCCCCCCC.CTTTTTTTTTT.III..OOOOOOOOOOOO..NNNNN...NNN..
#...FFFF.......UUUU...UUUU..NNNNNN..NNN..CCCC...CCCCC....TTTT....III..OOOO....OOOO..NNNNNN..NNN..
#...FFFF.......UUUU...UUUU..NNNNNNN.NNN.NCCCC....CCCC....TTTT....III..OOOO....OOOOO.NNNNNNN.NNN..
#...FFFFFFFFF..UUUU...UUUU..NNNNNNN.NNN.NCCC.............TTTT....III.IOOO......OOOO.NNNNNNN.NNN..
#...FFFFFFFFF..UUUU...UUUU..NNN.NNNNNNN.NCCC.............TTTT....III.IOOO......OOOO.NNN.NNNNNNN..
#...FFFFFFFFF..UUUU...UUUU..NNN.NNNNNNN.NCCC.............TTTT....III.IOOO......OOOO.NNN.NNNNNNN..
#...FFFF.......UUUU...UUUU..NNN..NNNNNN.NCCCC....CCCC....TTTT....III..OOOO....OOOOO.NNN..NNNNNN..
#...FFFF.......UUUU...UUUU..NNN..NNNNNN..CCCC...CCCCC....TTTT....III..OOOO....OOOO..NNN..NNNNNN..
#...FFFF.......UUUUUUUUUUU..NNN...NNNNN..CCCCCCCCCCC.....TTTT....III..OOOOOOOOOOOO..NNN...NNNNN..
#...FFFF.......UUUUUUUUUUU..NNN...NNNNN...CCCCCCCCCC.....TTTT....III...OOOOOOOOOO...NNN...NNNNN..
#...FFFF........UUUUUUUUU...NNN....NNNN....CCCCCCCC......TTTT....III....OOOOOOOOO...NNN....NNNN..
#.................UUUUU.....................CCCCC........................OOOOOO..................
#................................................................................................


def cancel_by_price(order_book, Price):
    side = 'bid'
    order_list =  order_book.bids.get_price_list(Price)
    order = order_list.get_head_order()
    order_id = order.order_id
    trade_id = order.trade_id
    timestamp = order.timestamp
    order_book.cancel_order(side, trade_id, time = timestamp)
    return order_book

def partly_cancel(order_book, right_order_price, wrong_order_price):
    for price, order_list in reversed(order_book.bids.price_map.items()):
        print(right_order_price, price , wrong_order_price)
        if right_order_price < price  and price <= wrong_order_price:
            for order in order_list:
                order_book.cancel_order(side = 'bid', 
                                        order_id = order.order_id,
                                        time = order.timestamp, 
                                        )
    return order_book

def get_two_list4compare(order_book, index, d2):
    my_list = brief_order_book(order_book)[0:2*Configuration.price_level]
    right_list = d2.iloc[index,:].reset_index().drop(['index'],axis= 1).iloc[:,0].to_list() 
    return my_list, right_list
    
def is_right_answer(order_book, index, d2):
    my_list, right_list = get_two_list4compare(order_book, index, d2)
    return len(list(set(right_list) - set(my_list))) == 0
    
    
def brief_order_book(order_book):
    my_list = []
    count = 0
    for key, value in reversed(order_book.bids.price_map.items()):
        count +=1 
        quantity = value.volume
        price = value.head_order.price
        my_list.append(price)
        my_list.append(quantity)
        if count == Configuration.price_level:
            break
    return my_list



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




class SignalProducer:
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
    def __call__(self, signal):
        message = signal.copy(); message.pop("sign")
        if signal['sign'] in (1, 4, 5):
            trades, order_in_book = self.order_book.process_order(message, True, False)
            if Debugger.on: 
                print("Trade occurs as follows:"); print(trades)
                print("The order book now is:");   print(self.order_book)
        elif signal['sign'] in (2, ): # cancellation (partial deletion of a limit order)
            self.order_book.bids.update_order(message) 
        elif signal['sign'] in (3, ):# deletion (total deletion of a limit order)
            self.delete_order(message)
        elif signal['sign'] in (6, ): pass # do nothing
        else: raise NotImplementedError
        return self.order_book
    
    
    
    
    
