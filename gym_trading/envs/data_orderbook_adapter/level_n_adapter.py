# -*- coding: utf-8 -*-

# =============================================================================
import numpy as np
adjust_data_drift_id = 10000
price_level = 10 # wont be changed during running
t3_count = 0

# =============================================================================
# Price = 31155000
def cancel_by_price(order_book, Price):
    side = 'bid'
    order_list =  order_book.bids.get_price_list(Price)
    order = order_list.get_head_order()
    order_id = order.order_id
    trade_id = order.trade_id
    timestamp = order.timestamp
    order_book.cancel_order(side, trade_id, time = timestamp)
    return order_book


def get_two_list4compare(order_book, index):
    global price_level
    my_list = brief_order_book(order_book)[0:2*price_level]
    right_list = d2.iloc[index,:].reset_index().drop(['index'],axis= 1).iloc[:,0].to_list() 
    return my_list, right_list
    
def is_right_answer(order_book, index):
    my_list, right_list = get_two_list4compare(order_book, index)
    return len(list(set(right_list) - set(my_list))) == 0
    
    
def brief_order_book(order_book):
    my_list = []
    count = 0
    global  price_level
    for key, value in reversed(order_book.bids.price_map.items()):
        count +=1 
        quantity = value.volume
        price = value.head_order.price
        my_list.append(price)
        my_list.append(quantity)
        if count == price_level:
            break
    return my_list

def adjust_data_drift(order_book, timestamp, index):
        
    print(order_book)#tbd
    my_list, right_list = get_two_list4compare(order_book, index)
    my_array, right_array = np.array(my_list), np.array(right_list)
    
    right_order = list(set(right_list) - set(my_list))
    wrong_order = list(set(my_list) -set(right_list))
    if len(right_order) == 0 and len(wrong_order) == 0:
        print("no data drift: no incomming new limit order outside the 10 price levels")
        message = None
    else:
        global adjust_data_drift_id
        if len(right_order) == 1 and len(wrong_order) == 1:
            if my_list[-2] == right_list[-2] :
                price = right_list[-2]
                if right_order[0] == wrong_order[0]:
                    raise NotImplementedError
                elif right_order[0] > wrong_order[0]:
                    quantity = right_order[0] - wrong_order[0]
                    side = 'bid'
                elif right_order[0] < wrong_order[0]:
                    quantity = -right_order[0] + wrong_order[0]
                    side = 'ask' # cancel order here
                    return cancel_by_price(order_book, price)
            elif my_list[-1] == right_list[-1]:
                price = right_list[-2]
                quantity = right_list[-1]
                side = 'bid'
            else:
                raise NotImplementedError
            adjust_data_drift_id += 1
            trade_id = adjust_data_drift_id
            order_id = adjust_data_drift_id
            
            str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
            timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
            
            message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
            print('\n'+'-'*15)
            print(">>> ADJUSTED <<<")
            print('-'*15+'\n')   
        elif len(right_order) == 2 and len(wrong_order) == 2:
            right_order_price =  right_list[-2]
            wrong_order_price =  my_list[-2]
            if right_order_price > wrong_order_price: # just insert new order
                side = 'bid'
                price = right_order[0]
                quantity = right_order[1]
                adjust_data_drift_id += 1
                trade_id = adjust_data_drift_id
                order_id = adjust_data_drift_id
                
                str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
                timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
                
                message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
                print('\n'+'-'*15)
                print(">>> ADJUSTED <<<")
                print('-'*15+'\n')
            elif right_order_price < wrong_order_price:
                # wrong order been cancelled outside the order book 
                for price, order_list in reversed(order_book.bids.price_map.items()):
                    print(right_order_price, price , wrong_order_price)
                    if right_order_price < price  and price <= wrong_order_price:
                        for order in order_list:
                            order_book.cancel_order(side = 'bid', 
                                                    order_id = order.order_id,
                                                    time = order.timestamp, 
                                                    )
                message = None
            else: 
                raise NotImplementedError
        elif np.sum(my_array != right_array) == 2:
            side = 'bid'
            price = right_array[-2]
            quantity = right_array[-1]
            adjust_data_drift_id += 1
            trade_id = adjust_data_drift_id
            order_id = adjust_data_drift_id
            
            str_int_timestamp = str(int(timestamp[0:5]) * int(1e9) + (int(timestamp[6:15]) +1))
            timestamp = str(str_int_timestamp[0:5])+'.'+str(str_int_timestamp[5:15])
            
            message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id, "timestamp":timestamp, 'order_id':order_id}
            print('\n'+'-'*15)
            print(">>> ADJUSTED <<<")
            print('-'*15+'\n')
        elif np.sum(my_array != right_array) == 1:
            # =============================================================================
            # part of order_list at this price has been partly cancelled outside the order book
            # Quantity    20  |  Price 31155100  |  Trade_ID   15227277  |  Time 34200.290719105
            # Quantity     1  |  Price 31155100  |  Trade_ID      10003  |  Time 34204.721258569
            # ----------------------------------------------------------------------------
            # my_list
            # [31171400, 5, 31171000, 200, 31167100, 4, 31160000, 4, 31159800, 20, 
            #  31158100, 10, 31158000, 7, 31157700, 1, 31155500, 70, 31155100, 21]
            # ----------------------------------------------------------------------------
            # right_list
            # [31171400, 5, 31171000, 200, 31167100, 4, 31160000, 4, 31159800, 20, 
            #  31158100, 10, 31158000, 7, 31157700, 1, 31155500, 70, 31155100, 1]
            # =============================================================================
            price = right_array[-2]
            quantity = right_array[-1] # right quantity
            quantity_list = [] # wrong quantity list
            order_id_list = []
            timestamp_list = []
            for item in order_book.bids.get_price_list(price):
                quantity_list.append(item.quantity)
                order_id_list.append(item.order_id)
                timestamp_list.append(item.timestamp)
            try:quantity_index = quantity_list.index(quantity)
            except: raise NotImplementedError
            difference_index = np.array([i for i in range(len(quantity_list)) if i != quantity_index])
            if index == 4: breakpoint()
            returned_timestamp_array = np.array(timestamp_list)[difference_index]
            returned_order_id_array = np.array(order_id_list)[difference_index] # returned_order_id_array for cancel order
            assert len(returned_order_id_array) == 1, "NotImplemented, only implement the single order situation"
            order_book.cancel_order(side = 'bid', 
                                    order_id = returned_order_id_array[0],
                                    time = returned_timestamp_array[0], 
                                    )
            message = None
            print('\n'+'-'*15)
            print(">>> ADJUSTED <<<")
            print('-'*15+'\n')
        else:
            raise NotImplementedError
    if message is not None:
        trades, order_in_book = order_book.process_order(message, True, False)
    print(order_book) # tbd
    return order_book
    # print("Adjusted Order Book")
    # print(brief_order_book(order_book))
# my_list[~my_list.apply(tuple,1).isin(right_list.apply(tuple,1))]
# right_list[~right_list.apply(tuple,1).isin(my_list.apply(tuple,1))]
# my_list is right_list

# =============================================================================


# =============================================================================
# #tbd

import pandas as pd
df2 = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)

from gym_trading.envs.orderbook import OrderBook
order_book = OrderBook()

l1 = df2.iloc[0,:]
column_numbers=[i for i in range(price_level * 4) if i%4==2 or i%4==3]
l2 = l1.iloc[column_numbers]
l2 = l2.reset_index().drop(['index'],axis = 1)

# =============================================================================
column_numbers=[i for i in range(price_level * 4) if i%4==2 or i%4==3]
d2 = df2.iloc[:,column_numbers]  
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

# #tbd
# =============================================================================

# =============================================================================
# # tbd

import pandas as pd
df = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
df.columns = ["timestamp",'type','order_id','quantity','price','side','remark']
df["timestamp"] = df["timestamp"].astype(str)
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
# size = 2913 
size = 4000 

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
            if price > best_bid:
                message = None
            else:
                # print(order_book)
                order_book.cancel_order(side, trade_id, time = timestamp)
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
        
    # if index == 2199: breakpoint()
    order_book = adjust_data_drift(order_book, timestamp, index)
    print("brief_order_book(order_book)")
    print(brief_order_book(order_book))
    # order_book.asks = None # remove the ask side
    assert is_right_answer(order_book, index), "the orderbook if different from the data"
    print("=="*10 + "=" + "=====" + "="+ "=="*10+'\n')
    
   
breakpoint()    
# # tbd
# =============================================================================
