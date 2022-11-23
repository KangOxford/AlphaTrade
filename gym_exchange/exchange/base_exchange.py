# ========================= 01 =========================
import numpy as np
from gym_exchange.data_orderbook_adapter.utils import get_two_list4compare
from gym_exchange.exchange import Debugger
from gym_exchange.exchange.exchange_interface import Exchange_Interface
from gym_exchange.exchange.utils import latest_timestamp, timestamp_increase
from gym_exchange.exchange.utils.executed_pairs import ExecutedPairsRecorder
from gym_exchange.exchange.order_flow import OrderFlow
from gym_exchange.data_orderbook_adapter import utils


# ========================= 03 =========================
class BaseExchange(Exchange_Interface):
    def __init__(self):
        super().__init__()
        
    # -------------------------- 03.01 ----------------------------
    def reset(self):
        super().reset()
        self.executed_pairs_recoder = ExecutedPairsRecorder()

        
    # -------------------------- 03.02 ----------------------------
    def type1_handler(self, message, index):
        trades, order_in_book = self.order_book.process_order(message, True, False)
        self.executed_pairs_recoder.step(trades, 'agent' if index == 0 else 'market', self.index) # 2nd para: kind
    
    def type2_handler(self, message):
        tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
        in_book_quantity = tree.get_order(message['order_id']).quantity
        message['quantity'] = min(message['quantity'], in_book_quantity)# adjuested_message 
        (self.order_book.bids if message['side'] == 'bid' else self.order_book.asks).update_order(message)
        
    def type3_handler(self, message):
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
        
    def process_tasks(self): # para: self.task_list; return: self.order_book
        for index, item in enumerate(self.task_list): # advantange for ask limit order (in liquidation problem)
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    self.type1_handler(message, index)
                elif item.type == 2:
                    self.type2_handler(message)
                elif item.type == 3:
                    self.type3_handler(message)
                        
    def step(self, action = None): # action : Action(for the definition of type)
        self.order_book = super().step(action)
        return self.order_book 
        
    
if __name__ == "__main__":
    exchange = BaseExchange()
    exchange.reset()
    for _ in range(2048):
        exchange.step()
        















































