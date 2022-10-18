# -*- coding: utf-8 -*-
price_level = 10

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
    global price_level
    my_list = brief_order_book(order_book)[0:2*price_level]
    right_list = d2.iloc[index,:].reset_index().drop(['index'],axis= 1).iloc[:,0].to_list() 
    return my_list, right_list
    
def is_right_answer(order_book, index, d2):
    my_list, right_list = get_two_list4compare(order_book, index, d2)
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
