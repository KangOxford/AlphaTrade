# from gym_trading.envs.orderbook import OrderBook

# # Create an order book

# order_book = OrderBook()

# # Create some limit orders

# limit_orders = [{'type' : 'limit', 
#                    'side' : 'ask', 
#                     'quantity' : 5, 
#                     'price' : 101,
#                     'trade_id' : 100},
#                    {'type' : 'limit', 
#                     'side' : 'ask', 
#                     'quantity' : 5, 
#                     'price' : 103,
#                     'trade_id' : 101},
#                    {'type' : 'limit', 
#                     'side' : 'ask', 
#                     'quantity' : 5, 
#                     'price' : 101,
#                     'trade_id' : 102},
#                    {'type' : 'limit', 
#                     'side' : 'ask', 
#                     'quantity' : 5, 
#                     'price' : 101,
#                     'trade_id' : 103},
#                    {'type' : 'limit', 
#                     'side' : 'bid', 
#                     'quantity' : 5, 
#                     'price' : 99,
#                     'trade_id' : 100},
#                    {'type' : 'limit', 
#                     'side' : 'bid', 
#                     'quantity' : 5, 
#                     'price' : 98,
#                     'trade_id' : 101},
#                    {'type' : 'limit', 
#                     'side' : 'bid', 
#                     'quantity' : 5, 
#                     'price' : 99,
#                     'trade_id' : 102},
#                    {'type' : 'limit', 
#                     'side' : 'bid', 
#                     'quantity' : 5, 
#                     'price' : 97,
#                     'trade_id' : 103},
#                    ]


# =============================================================================
# #tbd

import pandas as pd
df2 = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv", header = None)

from gym_trading.envs.orderbook import OrderBook
order_book = OrderBook()

l1 = df2.iloc[0,:]
column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
l2 = l1.iloc[column_numbers]
l2 = l2.reset_index().drop(['index'],axis = 1)

# # convert to float
# new_column_numbers = [i for i in range(20) if i%2 == 0]
# l2[new_column_numbers] = l2[new_column_numbers].apply(lambda x:x/10000)
# l2

limit_orders = []
order_id_list = [15000000 + i for i in range(10)]
for i in range(10):
    trade_id = 10086
    timestamp = 34200.000000001
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
    breakpoint()
    trades, order_id = order_book.process_order(order, True, False)   
# The current book may be viewed using a print
print(order_book)

# #tbd
# =============================================================================

# =============================================================================
# # tbd

import pandas as pd
df = pd.read_csv("/Users/kang/Data/AMZN_2021-04-01_34200000_57600000_message_10.csv", header=None)
# index = 0 
size = 100
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
    message = {'type': 'limit','side': side,'quantity': quantity,'price': price,'trade_id': trade_id}
    print("Message:")
    print(message)
    best_bid = order_book.get_best_bid()
    if ttype == 1:
        pass
    elif ttype == 4 or ttype == 5:
        if price > best_bid:
            message = None
        else:
            pass
    elif ttype == 3:
        if price > best_bid:
            message = None
        else:
            breakpoint()
            print(order_book)
            order_book.cancel_order(side, trade_id)
            print(order_book)
    else:
        raise NotImplementedError
    
    if message is not None:
        breakpoint()
        trades, order_in_book = order_book.process_order(message, True, False)
        print("Trade occurs as follows:")
        print(trades)
        print("The order book now is:")
        print(order_book)
        print("=="*10 + "=" + "=====" + "="+ "=="*10)
    
# # tbd
# =============================================================================



# Add orders to order book
for order in limit_orders:
    trades, order_id = order_book.process_order(order, False, False)

# The current book may be viewed using a print
print(order_book)

# Submitting a limit order that crosses the opposing best price will result in a trade
crossing_limit_order = {'type': 'limit',
                        'side': 'bid',
                        'quantity': 2,
                        'price': 102,
                        'trade_id': 109}

print(crossing_limit_order)
trades, order_in_book = order_book.process_order(crossing_limit_order, False, False)
print("Trade occurs as incoming bid limit crosses best ask")
print(trades)
print(order_book)

# If a limit crosses but is only partially matched, the remaning volume will
# be placed in the book as an outstanding order
big_crossing_limit_order = {'type': 'limit',
                            'side': 'bid',
                            'quantity': 50,
                            'price': 102,
                            'trade_id': 110}
print(big_crossing_limit_order)
trades, order_in_book = order_book.process_order(big_crossing_limit_order, False, False)
print("Large incoming bid limit crosses best ask. Remaining volume is placed in book.")
print(trades)
print(order_book)


# Market Orders

# Market orders only require that a user specifies a side (bid or ask), a quantity, and their unique trade id
market_order = {'type': 'market',
                'side': 'ask',
                'quantity': 40,
                'trade_id': 111}
trades, order_id = order_book.process_order(market_order, False, False)
print("A market order takes the specified volume from the inside of the book, regardless of price")
print("A market ask for 40 results in:")
print(order_book)
