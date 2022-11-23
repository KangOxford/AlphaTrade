# ========================= 01 =========================
# import numpy as np
# from gym_exchange.data_orderbook_adapter.utils import get_two_list4compare
# from gym_exchange.exchange.utils import latest_timestamp, timestamp_increase
# from gym_exchange.exchange.utils.executed_pairs import ExecutedPairsRecorder
# from gym_exchange.exchange.order_flow import OrderFlow


from gym_exchange import Config
from gym_exchange.exchange import Debugger
from gym_exchange.data_orderbook_adapter import utils
from gym_exchange.exchange.base_exchange import BaseExchange


# ========================= 03 =========================
class DebugBase(BaseExchange):
    def __init__(self):
        super().__init__()
        if Debugger.on: print(">>> BaseExchange Initialized") #$
        
    # -------------------------- 03.01 ----------------------------
    def reset(self):
        super().reset()
        if Debugger.on == True:
            from gym_exchange import Config
            from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
            historical_data = (DataPipeline()())['historical_data']
            column_numbers_bid = [i for i in range(Config.price_level * 4) if i%4==2 or i%4==3]
            column_numbers_ask = [i for i in range(Config.price_level * 4) if i%4==0 or i%4==1]
            bid_sid_historical_data = historical_data.iloc[:,column_numbers_bid]
            ask_sid_historical_data = historical_data.iloc[:,column_numbers_ask]
            self.d2 = bid_sid_historical_data; self.l2 = ask_sid_historical_data
            
    # -------------------------- 03.02 ----------------------------
    
    # ···················· 03.02.01 ···················· 
    def order_book_data_consistency_check(self):
        if Debugger.on == True:
            # pass # TODO to be tested
            # ................ 03.02.01.01 ................
            self.column_numbers_bid = [i for i in range(Config.price_level * 4) if i%4==2 or i%4==3]
            self.column_numbers_ask = [i for i in range(Config.price_level * 4) if i%4==0 or i%4==1]
            self.bid_sid_historical_data = self.historical_data.iloc[:,self.column_numbers_bid]
            self.ask_sid_historical_data = self.historical_data.iloc[:,self.column_numbers_ask]
            # ................ 03.02.01.02 ................
            if self.order_book.bids.depth != 0:
                single_side_historical_data = self.bid_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data, side = 'bid'), "the orderbook if different from the data"
            if self.order_book.asks.depth != 0:
                single_side_historical_data = self.ask_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data, side = 'ask'), "the orderbook if different from the data"
            print(">>> Right_order_book"); print(utils.get_right_answer(self.index, single_side_historical_data))
            # ................ 03.02.01.03 ................
            print(">>> Brief_self.order_book(self.order_book)")
            side = 'bid' if self.historical_message[5] == 1 else 'ask'
            print(utils.brief_order_book(self.order_book, side))
            print("The orderbook is right!\n")
    
    # ···················· 03.02.02 ···················· 
    def type1_handler(self, message, index):
        if Debugger.on: print(f"message:{message}") #$
        # print(f"---before trading {utils.brief_order_book(self.order_book,message['side'])}")
        if Debugger.on: print(f"---before trading\n {(self.order_book)}")
        trades, order_in_book = self.order_book.process_order(message, True, False)
        if Debugger.on: print(f"---after trading\n {(self.order_book)}")
        # print(f"---after trading {utils.brief_order_book(self.order_book,message['side'])}")
        # if len(trades) != 0:
        #     breakpoint()
        #     print() #$
        if Debugger.on: self.order_book_data_consistency_check()
        self.executed_pairs_recoder.step(trades, 'agent' if index == 0 else 'market', self.index) # 2nd para: kind
        
    # ···················· 03.02.03 ····················     
    def process_tasks(self): # para: self.task_list; return: self.order_book
        if Debugger.on: print(f">>>>>>>> self.index : {self.index}") #$
        super().process_tasks()
        
    # ···················· 03.02.04 ····················                     
    def step(self, action = None): # action : Action(for the definition of type)
        self.order_book = super().step(action)
        if action == None and Debugger.DebugBase.on:
            pass
            # for side in ['bid', 'ask']:
            #     history_data = self.d2 if side == 'bid' else self.l2
            #     my_list, right_list = get_two_list4compare(self.order_book, self.index, history_data, side)
            #     my_list = np.array(my_list); right_list = np.array(right_list)
            #     difference = my_list - right_list
            #     is_the_same = (not any(difference))
            #     if Debugger.BaseExchange.on == True:
            #         print(f"self.index: {self.index}")#$
            #         print(my_list) #$
            #         print(right_list) #$
            #         print(f"is_the_same:{is_the_same}") #$
            #         print() #$ #TODO
        return self.order_book 
        
    
if __name__ == "__main__":
    exchange = DebugBase()
    exchange.reset()
    for _ in range(2048):
        exchange.step()
        















































