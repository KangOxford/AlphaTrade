# -*- coding: utf-8 -*-

price_level = 10 # wont be changed during running
# =============================================================================
# 01 IMPORT PACKAGES
# =============================================================================
import numpy as np
import pandas as pd
from gym_trading.envs.data_orderbook_adapter import utils
from gym_trading.envs.data_orderbook_adapter.adjust_data_drift import DataAdjuster
from gym_trading.envs.orderbook import OrderBook
order_book = OrderBook()

# =============================================================================
# 02 READ DATA
# =============================================================================
df2 = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)

l1 = df2.iloc[0,:]
column_numbers=[i for i in range(price_level * 4) if i%4==2 or i%4==3]
l2 = l1.iloc[column_numbers]
l2 = l2.reset_index().drop(['index'],axis = 1)

# =============================================================================
column_numbers=[i for i in range(price_level * 4) if i%4==2 or i%4==3]
d2 = df2.iloc[:,column_numbers]  
# =============================================================================


df = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
df.columns = ["timestamp",'type','order_id','quantity','price','side','remark']
df["timestamp"] = df["timestamp"].astype(str)

# =============================================================================
# 03 CONFIGURARION OF ORDERBOOK
# =============================================================================

limit_orders = []
order_id_list = [15000000 + i for i in range(price_level)]
for i in range(price_level):
    trade_id = 90000
    # timestamp = datetime(34200.000000001)
    timestamp = str(34200.000000001)
    item = {'type' : 'limit', 
        'side' : 'bid', 
          'quantity' : l2.iloc[2 * i + 1,0], 
          'price' : l2.iloc[2 * i,0],
          'trade_id' : trade_id,
          'order_id' : order_id_list[i],
          "timestamp": timestamp}
    limit_orders.append(item)
# Add orders to order book

for order in limit_orders:
    # breakpoint()
    trades, order_id = order_book.process_order(order, True, False)   
# The current book may be viewed using a print
print(order_book)

# size = 50 # pass
# size = 75 # pass
# size = 85 # pass
# size = 86 # pass
# size = 88 # pass
# size = 100 # pass
# size = 134 # pass
# size = 200 # pass
# size = 400 # pass
# size = 604 # pass
# size = 654 # pass
# size = 658 # pass
# size = 1733 # pass
# size = 1830 # pass
# size = 1864 # pass
# size = 2189 # pass
# size = 2190 # pass
# size = 2199 # pass
# size = 2900 # pass type3 cancel
# size = 3019 # pass worsest bid partly cancelled outside price range
# size = 3225 # pass partly cancel the order oustside the price range
# size = 3444 
size = 4000 


data_adjuster = DataAdjuster(d2)

# =============================================================================
# 04 START ALGORITHM
# =============================================================================

for index in range(size):
    
    print("=="*10 + " " + str(index) + " "+ "=="*10)
    print("The order book used to be:")
    print(order_book)
    l1 = df.iloc[index,:]
    ttype = l1[1] 
    side = 'bid' if l1[5] ==1 else 'ask'
    quantity = l1[3]
    price = l1[4]
    trade_id = l1[2] # not sure, in the data it is order id
    order_id = trade_id
    timestamp = l1[0]
    message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
    print(l1)#tbd
    print("Message:")
    print(message)
    best_bid = order_book.get_best_bid()
    
    if side == 'bid':
        if ttype == 1:
            message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
        elif ttype == 2:
            # cancellation (partial deletion of a limit order)
            origin_quantity = order_book.bids.get_order(order_id).quantity # origin_quantity is the quantity in the order book
            adjusted_quantity = origin_quantity - quantity # quantity is the delta quantity
            message = {
                'type' : 'limit',
                'side' : 'bid',
                'quantity': adjusted_quantity,
                'price' : price,
                'order_id': order_id,
                'timestamp': timestamp # the new timestamp
                }
            order_book.bids.update_order(message)
            message = None
        elif ttype == 3:
            # if index == 3225: breakpoint()
            if price > best_bid:
                message = None
            else:
                # print(order_book)
                try: order_book.cancel_order(side, trade_id, time = timestamp)
                except: 

                    order_list = order_book.bids.get_price_list(price)
                    assert len(order_list) == 1
                    order = order_list.head_order
                    auto_generated_trade_id = order.order_id
                    order_book.cancel_order(side = 'bid', 
                                            order_id = order.order_id,
                                            time = order.timestamp, 
                                            )
                    
                # print(order_book)
                message  = None # !remember not to pass the message to be processed
                # breakpoint()
        elif ttype == 4 or ttype == 5: # not sure???
            if side == 'bid' and price <= best_bid:
                side = 'ask'
                message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
            else:
                message = None
        elif ttype == 6:
            message = None
        else:
            raise NotImplementedError
    else:
        message = None
        
    if message is not None:
        trades, order_in_book = order_book.process_order(message, True, False)
        print("Trade occurs as follows:")
        print(trades)
        print("The order book now is:")
        print(order_book)
        
    if index == 3444: breakpoint()
    order_book = data_adjuster.adjust_data_drift(order_book, timestamp, index)
    print("brief_order_book(order_book)")
    print(utils.brief_order_book(order_book))
    # order_book.asks = None # remove the ask side
    assert utils.is_right_answer(order_book, index, d2), "the orderbook if different from the data"
    print("=="*10 + "=" + "=====" + "="+ "=="*10+'\n')
    
   
breakpoint()    
# # tbd
# =============================================================================
