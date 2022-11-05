# ========================= 01 =========================
import numpy as np
from gym_exchange.data_orderbook_adapter.utils import get_two_list4compare
from gym_exchange.exchange import Debugger
from gym_exchange.exchange.exchange_interface import Exchange_Interface
from gym_exchange.exchange.utils import latest_timestamp, timestamp_increase
from gym_exchange.exchange.utils.executed_pairs import ExecutedPairs
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.data_orderbook_adapter import utils


# ========================= 03 =========================
class BaseExchange(Exchange_Interface):
    def __init__(self):
        super().__init__()
        
    # -------------------------- 03.01 ----------------------------
    def reset(self):
        super().reset()
        self.executed_pairs = ExecutedPairs()
        if Debugger.BaseExchange.on == True:
            from gym_exchange import Config
            from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
            historical_data = (DataPipeline()())['historical_data']
            column_numbers_bid = [i for i in range(Config.price_level * 4) if i%4==2 or i%4==3]
            column_numbers_ask = [i for i in range(Config.price_level * 4) if i%4==0 or i%4==1]
            bid_sid_historical_data = historical_data.iloc[:,column_numbers_bid]
            ask_sid_historical_data = historical_data.iloc[:,column_numbers_ask]
            self.d2 = bid_sid_historical_data; self.l2 = ask_sid_historical_data
        
        
    def process_tasks(self): # para: self.task_list; return: self.order_book
        for index, item in enumerate(self.task_list): # advantange for ask limit order (in liquidation problem)
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    trades, order_in_book = self.order_book.process_order(message, True, False)
                    self.executed_pairs.step(trades, 'agent' if index == 0 else 'market') # 2nd para: kind
                elif item.type == 2:
                    tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
                    in_book_quantity = tree.get_order(message['order_id']).quantity
                    message['quantity'] = min(message['quantity'], in_book_quantity)# adjuested_message 
                    (self.order_book.bids if message['side'] == 'bid' else self.order_book.asks).update_order(message)
                elif item.type == 3:
                    done = False
                    right_tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
                    if right_tree.order_exists(message['order_id']) == False:
                        try: # message['price'] in the order_book
                            right_price_list = right_tree.get_price_list(message['price']) # my_price
                            for order in right_price_list:
                                if 90000000 <= order.order_id and order.order_id < 100000000: # if my_order_id in initial_orderbook_ids
                                    '''Initial orderbook id is created via the exchange
                                    cannot be the same with the message['order_id'].
                                    Solution: total delete the first (90000000) order in the orderlist
                                    at the price we want to totally delete.
                                    message['order_id'] not in the order_book.
                                    message['timestamp'] not in the order_book.
                                    Only tackle with single order. If found, break.
                                    Solution code: 31'''
                                    self.order_book.cancel_order(
                                        side = message['side'], 
                                        order_id = order.order_id,
                                        time = order.timestamp, 
                                    )
                                    self.cancelled_quantity = order.quantity
                                    done = True; break
                            if not done:
                                raise NotImplementedError
                        except: # message['price'] not in the order_book
                            pass
                            # print()#$
                            # raise NotImplementedError #TODO
                    else: #right_tree.order_exists(message['order_id']) == True
                        self.order_book.cancel_order(
                            side = message['side'], 
                            order_id = message['order_id'],
                            time = message['timestamp'], 
                        )
                        self.cancelled_quantity =  message['quantity']
                        
    def step(self, action = None): # action : Action(for the definition of type)
        self.order_book = super().step(action)
        if action == None and Debugger.BaseExchange.on == True:
            for side in ['bid', 'ask']:
                print(f"self.index: {self.index}")#$
                history_data = self.d2 if side == 'bid' else self.l2
                my_list, right_list = get_two_list4compare(self.order_book, self.index, history_data, side)
                my_list = np.array(my_list); right_list = np.array(right_list)
                difference = my_list - right_list
                is_the_same = (not any(difference))
                print(my_list) #$
                print(right_list) #$
                print(f"is_the_same:{is_the_same}") #$
                print() #$ #TODO
        return self.order_book 
        
    
if __name__ == "__main__":
    exchange = BaseExchange()
    exchange.reset()
    for _ in range(2048):
        exchange.step()
        















































