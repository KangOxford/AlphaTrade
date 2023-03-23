import numpy as np
from gym_exchange.data_orderbook_adapter.utils import brief_order_book

def get_state_memo(order_book):
    state_memo = tuple(map(lambda side: brief_order_book(order_book, side), ('ask', 'bid')))
    state_memo_tuple = state_memo
    sm = np.array([[[list[2 * i], list[2 * i + 1]] for i in range(len(list) // 2)] for list in state_memo_tuple])
    sm[0] = sm[0][np.argsort(sm[0][:, 0])[::-1]]
    return sm
