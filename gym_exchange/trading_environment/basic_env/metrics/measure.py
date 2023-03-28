import numpy as np
import pandas as pd
from gym_exchange import Config
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.data_orderbook_adapter.utils import brief_order_book

def distance(generated_orderbook, data_orderbook):
    result = 0
    d_prices = data_orderbook.flatten()[::2]
    # for d_pair in data_orderbook:
    for g_pair in generated_orderbook:
        if g_pair[0] in d_prices:
            d_quantity = data_orderbook[np.where(data_orderbook[:, 0] == g_pair[0])[0][0], 1]
            result += 1 * (d_quantity != g_pair[1])
        else:
            result += 2  # 1+1: price and quantity
    return result

class OrderbookDistance:
    def __init__(self):
        self.data_orderbook = DataPipeline()()['historical_data']

    def get_distance(self, Self):
        index = Self.exchange.index
        data_orderbook = np.array(self.data_orderbook.iloc[index,:]).reshape(-1,2)
        generated_orderbook_ask, generated_orderbook_bid = brief_order_book(Self.exchange.order_book, 'ask'), brief_order_book(Self.exchange.order_book, 'bid')
        generated_orderbook_ask= [[generated_orderbook_ask[2 * i], generated_orderbook_ask[2 * i + 1]] for i in range(len(generated_orderbook_ask) // 2)]
        generated_orderbook_bid= [[generated_orderbook_bid[2 * i], generated_orderbook_bid[2 * i + 1]] for i in range(len(generated_orderbook_bid) // 2)]
        generated_orderbook = np.array([generated_orderbook_ask, generated_orderbook_bid])
        generated_orderbook = generated_orderbook.flatten().reshape(-1,2)  # sequence/order doesn't matter
        result = distance(generated_orderbook, data_orderbook)
        return result

