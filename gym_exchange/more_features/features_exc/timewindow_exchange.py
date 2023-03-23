from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange
from gym_exchange.more_features.features_exc.utils import get_state_memo

time_window = 300
'''
function:
01 only step after a timewindow
02 remember all the (level 10) brief order_book data during the past timewindow
'''

# ========================= 01 =========================
class TimewindowExchange(Exchange):
    def __init__(self):
        super().__init__()

    # -------------------------- 01.01 ----------------------------
    def step(self, action=None):  # action : Action(for the definition of type)
        self.state_memos.append([]) # # init update_state_memos
        # ···················· 01.01.01 ····················
        for i in range(time_window-1):
            # print(f"innerloop step {i}") #$
            # if i == 2:
            #     print() #$
            super(TimewindowExchange, self).step()
            self.state_memos[-1].append(get_state_memo(self.order_book)) # update_state_memos
        super().step(action)
        # ···················· 01.01.02 ····················
        self.state_memos[-1].append(get_state_memo(self.order_book)) # update_state_memos
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
















































