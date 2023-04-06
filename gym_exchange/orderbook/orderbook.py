import sys
import math
from collections import deque # a faster insert/pop queue
from six.moves import cStringIO as StringIO  # pyright: ignore
from decimal import Decimal

from .ordertree import OrderTree


INITID=90000000

class OrderBook(object):
    def __init__(self, tick_size = 0.0001):
        self.tape = deque(maxlen=None) # Index[0] is most recent trade
        self.bids = OrderTree()
        self.asks = OrderTree()
        self.last_tick = None
        self.last_timestamp = 0
        self.tick_size = tick_size
        self.time = 0
        self.next_order_id = 0

    def update_time(self):
        self.time += 1

    def process_order(self, quote, from_data, verbose):
        order_type = quote['type']
        order_in_book = None
        if from_data:
            self.time = quote['timestamp']
        else:
            self.update_time()
            quote['timestamp'] = self.time
        if quote['quantity'] <= 0:
            sys.exit('process_order() given order of quantity <= 0')
        if not from_data:
            self.next_order_id += 1
        if order_type == 'market':
            trades = self.process_market_order(quote, verbose)
        elif order_type == 'limit':
            quote['price'] = int(quote['price'])
            # quote['price'] = Decimal(quote['price'])
            trades, order_in_book = self.process_limit_order(quote, from_data, verbose)
        else:
            sys.exit("order_type for process_order() is neither 'market' or 'limit'")
        return trades, order_in_book

    def processOrder(self,quote,from_data,verbose):
        """This function assumes that the "type" field in the quote is an integer, and follows the LOBSTER convention of
        order types."""
        type=quote['type']
        if type=='limit' or type=='market': #Normal Limit Order
            trades,order_in_book=self.process_order(quote,from_data=from_data,verbose=verbose)
        elif type=='cancel': #Cancellation order (partial deletion): simply update quantity
            if quote['side']=='bid':
                if self.bids.order_exists(quote['order_id']):
                    if self.bids.get_order(quote['order_id']).quantity<=quote['quantity']:
                            self.bids.remove_order_by_id(quote['order_id'])
                    else:
                        #remove as normal
                        origin_quantity = self.bids.get_order(quote['order_id']).quantity # origin_quantity is the quantity in the order book
                        adjusted_quantity = origin_quantity - quote['quantity'] # quantity is the delta quantity
                        quote['quantity']=adjusted_quantity
                        self.bids.update_order(quote)
                    
                elif self.bids.price_exists(quote['price']):
                    #Try to find price to cancel and check if it contains an initial order. 
                    orderlist=self.bids.get_price_list(quote['price'])
                    if orderlist.get_head_order().order_id>=INITID: #assumes INITID is the start of a sequence of integers that grow - better convention might be to use continuous INITID for all. 
                        if orderlist.get_head_order().quantity<=quote['quantity']:
                            self.bids.remove_order_by_id(orderlist.get_head_order().order_id)
                        else:
                            quote['order_id']=orderlist.get_head_order().order_id
                            origin_quantity = self.bids.get_order(quote['order_id']).quantity # origin_quantity is the quantity in the order book
                            adjusted_quantity = origin_quantity - quote['quantity'] # quantity is the delta quantity
                            quote['quantity']=adjusted_quantity
                            self.bids.update_order(quote)
                else:
                    #IGNORE cancel order
                    pass
                
            elif quote['side']=='ask':
                if self.asks.order_exists(quote['order_id']):
                    if self.asks.get_order(quote['order_id']).quantity<=quote['quantity']:
                            self.asks.remove_order_by_id(quote['order_id'])
                    else:
                        #remove as normal
                        origin_quantity = self.asks.get_order(quote['order_id']).quantity # origin_quantity is the quantity in the order book
                        adjusted_quantity = origin_quantity - quote['quantity'] # quantity is the delta quantity
                        quote['quantity']=adjusted_quantity
                        self.asks.update_order(quote)
                elif self.asks.price_exists(quote['price']):
                    #Try to find price to cancel and check if it contains an initial order. 
                    orderlist=self.asks.get_price_list(quote['price'])
                    if orderlist.get_head_order().order_id>=INITID: #assumes INITID is the start of a sequence of integers that grow - better convention might be to use continuous INITID for all. 
                        if orderlist.get_head_order().quantity<=quote['quantity']:
                            self.asks.remove_order_by_id(orderlist.get_head_order().order_id)
                        else:
                            quote['order_id']=orderlist.get_head_order().order_id
                            origin_quantity = self.asks.get_order(quote['order_id']).quantity # origin_quantity is the quantity in the order book
                            adjusted_quantity = origin_quantity - quote['quantity'] # quantity is the delta quantity
                            quote['quantity']=adjusted_quantity
                            self.asks.update_order(quote)
                else:
                    #IGNORE cancel order
                    pass
            else:
                sys.exit('cancel_order() given neither "bid" nor "ask"')
                pass
                
            trades=[]
            order_in_book=quote
        elif type=='delete':
            self.cancel_order(quote) #this will work as-is, any issues with messages will be flagged. 
            trades=[]
            order_in_book=quote
        elif type=='skip':
            trades=[]
            order_in_book=quote
        else:
            sys.exit("Type is wrong")

        return trades,order_in_book

    def process_order_list(self, side, order_list, quantity_still_to_trade, quote, verbose):
        '''
        Takes an OrderList (stack of orders at one price) and an incoming order and matches
        appropriate trades given the order's quantity.
        '''
        trades = []
        quantity_to_trade = quantity_still_to_trade
        while len(order_list) > 0 and quantity_to_trade > 0:
            head_order = order_list.get_head_order()
            traded_price = head_order.price
            counter_party = head_order.trade_id
            new_book_quantity = None
            if quantity_to_trade < head_order.quantity:
                #Less to trade than size of order: match partially
                traded_quantity = quantity_to_trade
                # Do the transaction
                new_book_quantity = head_order.quantity - quantity_to_trade
                head_order.update_quantity(new_book_quantity, head_order.timestamp)
                quantity_to_trade = 0
            elif quantity_to_trade == head_order.quantity:
                traded_quantity = quantity_to_trade
                if side == 'bid':
                    self.bids.remove_order_by_obj(head_order)
                else:
                    self.asks.remove_order_by_obj(head_order)
                quantity_to_trade = 0
            else: # quantity to trade is larger than the head order
                traded_quantity = head_order.quantity
                if side == 'bid':
                    self.bids.remove_order_by_obj(head_order)
                else:
                    self.asks.remove_order_by_obj(head_order)
                quantity_to_trade -= traded_quantity
            if verbose:
                print(("TRADE: Time - {}, Price - {}, Quantity - {}, TradeID - {}, Matching TradeID - {}".format(self.time, traded_price, traded_quantity, counter_party, quote['trade_id'])))

            transaction_record = {
                    'timestamp': self.time,
                    'price': traded_price,
                    'quantity': traded_quantity,
                    'time': self.time
                    }

            if side == 'bid':
                transaction_record['party1'] = [counter_party, 'bid', head_order.order_id, new_book_quantity]
                transaction_record['party2'] = [quote['trade_id'], 'ask', None, None]
            else:
                transaction_record['party1'] = [counter_party, 'ask', head_order.order_id, new_book_quantity]
                transaction_record['party2'] = [quote['trade_id'], 'bid', None, None]

            self.tape.append(transaction_record)
            trades.append(transaction_record)
        return quantity_to_trade, trades
                    
    def process_market_order(self, quote, verbose):
        trades = []
        quantity_to_trade = quote['quantity']
        side = quote['side']
        if side == 'bid':
            while quantity_to_trade > 0 and self.asks:
                best_price_asks = self.asks.min_price_list()
                quantity_to_trade, new_trades = self.process_order_list('ask', best_price_asks, quantity_to_trade, quote, verbose)
                trades += new_trades
        elif side == 'ask':
            while quantity_to_trade > 0 and self.bids:
                best_price_bids = self.bids.max_price_list()
                quantity_to_trade, new_trades = self.process_order_list('bid', best_price_bids, quantity_to_trade, quote, verbose)
                trades += new_trades
        else:
            sys.exit('process_market_order() recieved neither "bid" nor "ask"')
        return trades

    def process_limit_order(self, quote, from_data, verbose):
        order_in_book = None
        trades = []
        quantity_to_trade = quote['quantity']
        side = quote['side']
        price = quote['price']
        if side == 'bid':
            while (self.asks and price >= self.asks.min_price() and quantity_to_trade > 0):
                best_price_asks = self.asks.min_price_list()
                quantity_to_trade, new_trades = self.process_order_list('ask', best_price_asks, quantity_to_trade, quote, verbose)
                trades += new_trades
            # If volume remains, need to update the book with new quantity
            if quantity_to_trade > 0:
                if not from_data:
                    quote['order_id'] = self.next_order_id
                quote['quantity'] = quantity_to_trade
                self.bids.insert_order(quote)
                order_in_book = quote
        elif side == 'ask':
            while (self.bids and price <= self.bids.max_price() and quantity_to_trade > 0):
                best_price_bids = self.bids.max_price_list()
                quantity_to_trade, new_trades = self.process_order_list('bid', best_price_bids, quantity_to_trade, quote, verbose)
                trades += new_trades
            # If volume remains, need to update the book with new quantity
            best_price_bids = self.bids.max_price_list()
            if quantity_to_trade > 0:
                if not from_data:
                    quote['order_id'] = self.next_order_id
                quote['quantity'] = quantity_to_trade
                self.asks.insert_order(quote)
                order_in_book = quote
        else:
            sys.exit('process_limit_order() given neither "bid" nor "ask"')
        return trades, order_in_book

    def cancel_order(self, order,time=None):
        if order['timestamp']:
            self.time = order['timestamp']
        else:
            self.update_time()

        if order['side'] == 'bid':
            if self.bids.order_exists(order['order_id']):
                print('should not be here')
                print(self.bids.order_map[order['order_id']])
                #Order ID is found, and order can be cancelled (deleted) by ID
                self.bids.remove_order_by_id(order['order_id'])
            elif self.bids.price_exists(order['price']):
                print('should be here')
                #Order ID is not found, but price is. Proceed to check if price contains an initial order. 
                orderlist=self.bids.get_price_list(order['price'])
                if orderlist.get_head_order().order_id>=INITID:
                    print('but should never get here as',orderlist.get_head_order().order_id,'is smalled than',INITID)
                    if orderlist.get_head_order().quantity<=order['quantity']:
                        self.bids.remove_order_by_id(orderlist.get_head_order().order_id)
                    else:
                        order['order_id']=orderlist.get_head_order().order_id
                        self.modify_order(order['order_id'],order,order['timestamp'])
            else:
                #No matching lim order found to cancel - ignore message. 
                pass
        elif order['side'] == 'ask':
            if self.asks.order_exists(order['order_id']):
                #Order ID is found, and order can be cancelled (deleted) by ID
                self.asks.remove_order_by_id(order['order_id'])
            elif self.asks.price_exists(order['price']):
                #Order ID is not found, but price is. Proceed to check if price contains an initial order. 
                orderlist=self.asks.get_price_list(order['price'])
                if orderlist.get_head_order().order_id>=INITID:
                    if orderlist.get_head_order().quantity<=order['quantity']:
                        self.asks.remove_order_by_id(orderlist.get_head_order().order_id)
                    else:
                        order['order_id']=orderlist.get_head_order().order_id
                        self.modify_order(order['order_id'],order,order['timestamp'])
            else:
                #No matching lim order found to cancel - ignore message. 
                pass
        else:
            raise NotImplementedError # tbd
            sys.exit('cancel_order() given neither "bid" nor "ask"')

    def modify_order(self, order_id, order_update, time=None):
        if time:
            self.time = time
        else:
            self.update_time()
        side = order_update['side']
        order_update['order_id'] = order_id
        order_update['timestamp'] = self.time
        if side == 'bid':
            if self.bids.order_exists(order_update['order_id']):
                self.bids.update_order(order_update)
        elif side == 'ask':
            if self.asks.order_exists(order_update['order_id']):
                self.asks.update_order(order_update)
        else:
            sys.exit('modify_order() given neither "bid" nor "ask"')

    def get_volume_at_price(self, side, price):
        price = int(price)
        # price = Decimal(price)
        if side == 'bid':
            volume = 0
            if self.bids.price_exists(price):
                volume = self.bids.get_price(price).volume
            return volume
        elif side == 'ask':
            volume = 0
            if self.asks.price_exists(price):
                volume = self.asks.get_price(price).volume
            return volume
        else:
            sys.exit('get_volume_at_price() given neither "bid" nor "ask"')

    def get_best_bid(self):
        return self.bids.max_price()

    def get_worst_bid(self):
        return self.bids.min_price()

    def get_best_ask(self):
        return self.asks.min_price()

    def get_worst_ask(self):
        return self.asks.max_price()

    def tape_dump(self, filename, filemode, tapemode):
        dumpfile = open(filename, filemode)
        for tapeitem in self.tape:
            dumpfile.write('Time: %s, Price: %s, Quantity: %s\n' % (tapeitem['time'],
                                                                    tapeitem['price'],
                                                                    tapeitem['quantity']))
        dumpfile.close()
        if tapemode == 'wipe':
            self.tape = []

    def __str__(self):
        tempfile = StringIO()
        
        string_len = 38
        tempfile.write("="*string_len + " Asks " + "="*string_len +"\n") 
        if self.asks != None and len(self.asks) > 0:
            for key, value in  reversed(self.asks.price_map.items()):
                tempfile.write('%s' % value)
        tempfile.write("\n\n") 
        # tempfile.write('\n'+"-"*string_len + "------" + "-"*string_len +"\n\n") 
        if self.bids != None and len(self.bids) > 0:
            for key, value in reversed(self.bids.price_map.items()):
                tempfile.write('%s' % value)
        tempfile.write("="*string_len + " Bids " + "="*string_len +"\n") 
                
        
        # tempfile.write("\n***Trades***\n")
        # if self.tape != None and len(self.tape) > 0:
        #     num = 0
        #     for entry in self.tape:
        #         if num < 10: # get last 5 entries
        #             tempfile.write(str(entry['quantity']) + " @ " + str(entry['price']) + " (" + str(entry['timestamp']) + ") " + str(entry['party1'][0]) + "/" + str(entry['party2'][0]) + "\n")
        #             num += 1
        #         else:
        #             break
    
        tempfile.write("\n")
        return tempfile.getvalue()

