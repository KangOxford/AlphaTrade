from gym_exchange import Config
from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange

'''
function:
01 only step after a timewindow
02 remember all the (level 10) brief order_book data during the past timewindow
'''

import numpy as np
from gym_exchange.data_orderbook_adapter.utils import brief_order_book
from gym_exchange.environment.base_env.utils import broadcast_lists
def get_state_memo(order_book):
    state_memo_tuple = tuple(map(lambda side: brief_order_book(order_book, side), ('ask', 'bid')))
    try:
        assert len(state_memo_tuple[0]) == len(state_memo_tuple[1])
    except:
        array =  broadcast_lists(*state_memo_tuple)
        state_memo_tuple = (array[0], array[1])
    sm = np.array([[[list[2 * i], list[2 * i + 1]] for i in range(len(list) // 2)] for list in state_memo_tuple])
    sm[0] = sm[0][np.argsort(sm[0][:, 0])[::-1]]
    return sm

# ========================= 01 =========================
class TimewindowExchange(Exchange):
    def __init__(self):
        super().__init__()
        # self.out_executed_pairs_recoder =

    # -------------------------- 01.01 ----------------------------
    def step(self, action=None):  # action : Action(for the definition of type)
        self.state_memos = [] # # init update_state_memos
        # ···················· 01.01.01 ····················
        for i in range(Config.window_size-1):
            # print(f"innerloop step {i}") #$
            # if i == 2:
            #     print() #$
            super(TimewindowExchange, self).step()
            self.state_memos.append(get_state_memo(self.order_book)) # update_state_memos
        # ···················· 01.01.02 ····················
        super().step(action)
        self.state_memos.append(get_state_memo(self.order_book)) # update_state_memos
        return self.order_book

    # -------------------------- 01.02 ----------------------------
    def reset(self):
        super(TimewindowExchange, self).reset()
        self.state_memos = []

if __name__ == "__main__":
    exchange = TimewindowExchange()
    exchange.reset()
    for i in range(2048):
        # print(f">>> outerloop step {i}")
        # if i == 2:
        #     print() #$
        exchange.step()
















































