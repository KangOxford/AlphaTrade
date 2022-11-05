# ========================= 01 =========================
# import numpy as np
from gym_exchange.exchange import Debugger
from gym_exchange.exchange.exchange_interface import Exchange_Interface
from gym_exchange.exchange.utils import latest_timestamp, timestamp_increase
# from exchange.utils.deletion_handler import PartDeletion, TotalDeletion
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
        
    
    def update_task_list(self, action = None):# action : Action(for the definition of type)
        flow = next(self.flow_generator)#used for historical data
        self.task_list = [action, flow] 
        
    def process_tasks(self): # para: self.task_list; return: self.order_book
        for index, item in enumerate(self.task_list): # advantange for ask limit order (in liquidation problem)
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    trades, order_in_book = self.order_book.process_order(message, True, False)
                    if len(trades) != 0:
                         breakpoint()#$
                         self.executed_pairs.step(trades, 'agent' if index == 0 else 'market') # 2nd para: kind
                    # self.executed_pairs.step(trades, 'agent' if index == 0 else 'market') # 2nd para: kind
                elif item.type == 2:
                    tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
                    in_book_quantity = tree.get_order(message['order_id']).quantity
                    message['quantity'] = min(message['quantity'], in_book_quantity)# adjuested_message 
                    (self.order_book.bids if message['side'] == 'bid' else self.order_book.asks).update_order(message)
                elif item.type == 3:
                    done = False
                    right_tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
                    if right_tree.order_exists(message['order_id']) == False:
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
                    else: #right_tree.order_exists(message['order_id']) == True
                        self.order_book.cancel_order(
                            side = message['side'], 
                            order_id = message['order_id'],
                            time = message['timestamp'], 
                        )
                        self.cancelled_quantity =  message['quantity']
        
    # -------------------------- 03.02 ----------------------------
    def step(self, action = None): # action : Action(for the definition of type)
        self.update_task_list(action)
        self.process_tasks()
        return self.order_book
    
    # ···················· 03.02.01 ···················· 
    def time_wrapper(self, order_flow: OrderFlow) -> OrderFlow:
        timestamp = latest_timestamp(self.order_book)
        return timestamp_increase(timestamp, order_flow) 
    
if __name__ == "__main__":
    exchange = BaseExchange()
    exchange.reset()
    for _ in range(1000):
        exchange.step()
        















































