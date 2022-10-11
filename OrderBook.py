#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 21:11:38 2022

@author: kang
"""
from gym_trading.envs.orderbook import OrderBook

# =============================================================================
from decimal import Decimal
import requests
from order_book import OrderBook

ob = OrderBook()

# get some orderbook data
data = requests.get("https://api.pro.coinbase.com/products/BTC-USD/book?level=2").json()

ob.bids = {Decimal(price): size for price, size, _ in data['bids']}
ob.asks = {Decimal(price): size for price, size, _ in data['asks']}

# OR

for side in data:
    # there is additional data we need to ignore
    if side in {'bids', 'asks'}:
        ob[side] = {Decimal(price): size for price, size, _ in data[side]}


# Data is accessible by .index(), which returns a tuple of (price, size) at that level in the book
price, size = ob.bids.index(0)
print(f"Best bid price: {price} size: {size}")

price, size = ob.asks.index(0)
print(f"Best ask price: {price} size: {size}")

print(f"The spread is {ob.asks.index(0)[0] - ob.bids.index(0)[0]}\n\n")

# Data is accessible via iteration
# Note: bids/asks are iterators

print("Bids")
for price in ob.bids:
    print(f"Price: {price} Size: {ob.bids[price]}")


print("\n\nAsks")
for price in ob.asks:
    print(f"Price: {price} Size: {ob.asks[price]}")


# Data can be exported to a sorted dictionary
# In Python3.7+ dictionaries remain in insertion ordering. The
# dict returned by .to_dict() has had its keys inserted in sorted order
print("\n\nRaw asks dictionary")
print(ob.asks.to_dict())
# =============================================================================
