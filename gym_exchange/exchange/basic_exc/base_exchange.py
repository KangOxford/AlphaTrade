# ========================= 01 =========================
from gym_exchange import Config
from gym_exchange.orderbook import OrderBook

from gym_exchange.exchange.basic_exc.assets.executed_pairs import ExecutedPairsRecorder


# ========================= 03 =========================
class BaseExchange():
    # -------------------------- 03.01 ----------------------------
    def __init__(self, flow_lists_initialized):
        self.flow_lists = flow_lists_initialized

    # -------------------------- 03.02 ----------------------------
    def reset(self):
        self.index = -1 # if initialized,index0th; else, index-1th.
        self.flow_generator = (flow_list for flow_list in self.flow_lists)
        self.order_book = OrderBook()
        self.initialize_orderbook()
        self.executed_pairs_recoder = ExecutedPairsRecorder()
        self.mid_prices = [(self.order_book.get_best_ask() + self.order_book.get_best_bid())/2]
        self.best_bids = [self.order_book.get_best_bid()]
        self.best_asks = [self.order_book.get_best_ask()]
        self.latest_timestamp = Config.init_latest_timestamp
        # self.latest_timestamp = "34200.000000002" # TODO

    def initialize_orderbook(self):
        '''only take the index0, the first one to init the lob'''
        flow_list = next(self.flow_generator)
        for flow in flow_list:
            self.order_book.process_order(flow.to_message, True, False)
        self.index += 1
        '''for this step is index0, for next step is index1'''

    # -------------------------- 03.03 ----------------------------
    def step(self, action=None):  # action : Action(for the definition of type)
        self.update_task_list(action)
        # if self.index ==16:
        #     print(self.order_book)
        #     print()#$
        try:
            self.process_tasks()
        except:
            breakpoint()
        self.accumulating()
        return self.order_book

    # ························ 03.03.01 ·························
    # ··················· component of the step ·················
    def update_task_list(self, action=None):  # action : Action(for the definition of type)
        flow_list = next(self.flow_generator)  # used for historical data
        try:self.task_list += [action] + [flow for flow in flow_list]
        except:self.task_list = [action] + [flow for flow in flow_list]
    def process_tasks(self):  # para: self.task_list; return: self.order_book
        # if self.index ==16:
        #     print()#$
        for index, item in enumerate(self.task_list):  # advantange for ask limit order (in liquidation problem)
            if not (item is None or item.quantity == 0):
                message = item.to_message
                if item.type == 1 or item.type == 0:
                    try:
                        self.type1_handler(message, index)
                    except:
                        # breakpoint()
                        pass
                        print(f"skip: {message}")
                    # print("1")
                    # print(item.type,item.timestamp,self.latest_timestamp) #$
                elif item.type == 2:
                    try:
                        self.type2_handler(message)
                    except:
                        breakpoint()
                    # print(item.type,item.timestamp,self.latest_timestamp) #$
                    # print("2")
                elif item.type == 3:
                    try:
                        self.type3_handler(message)
                    except:
                        breakpoint()
                    # print("3")
                    # print(item.type,item.timestamp,self.latest_timestamp) #$
                # assert item.timestamp >= self.latest_timestamp, 'The timestamp of a new order should be later than the timestamp of the auto_cancel action in the previous step' #$
                self.latest_timestamp = item.timestamp

    def accumulating(self):
        try: #$
            self.mid_prices.append((self.order_book.get_best_ask() + self.order_book.get_best_bid())/2)
            self.best_bids.append(self.order_book.get_best_bid())
            self.best_asks.append(self.order_book.get_best_ask())
            # print(self.order_book) #$
        except:
            # print() #$
            pass
        self.index += 1


    # ························ 03.03.02 ·························
    # ··········· component of the process_tasks ················
    def type1_handler(self, message, index):
        trades, order_in_book = self.order_book.process_order(message, True, False)
        try:
            self.executed_pairs_recoder.step(trades, self.index) # 2nd para: kind
        except:
            breakpoint()
            print("type1_handler")

    def type2_handler(self, message):
        ''' Cancellation (Partial deletion of a limit order)'''
        tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
        try:
            in_book_quantity = tree.get_order(message['order_id']).quantity
            message['quantity'] = min(message['quantity'], in_book_quantity)# adjuested_message
            (self.order_book.bids if message['side'] == 'bid' else self.order_book.asks).update_order(message)
        except:
            '''EXAMPLE: in get_order return self.order_map[order_id] KeyError: 17142637'''
            pass #TODO

    def type3_handler(self, message):
        '''Deletion (Total deletion of a limit order)'''
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

                    


        
    
if __name__ == "__main__":
    exchange = BaseExchange()
    exchange.reset()
    for _ in range(2048):
        exchange.step()

"""
=========================================================
ORDERBOOK:(Initialized, AMZN 2021.04.01)
31240000 4
31237900 1
31230000 24
31229800 100
31220000 4
31214000 2
31210000 3
31200000 18
31190000 2
31180100 48

31161600 3
31160000 4
31152200 16
31151000 2
31150100 2
31150000 506
31140000 4
31130000 2
31120300 35
31120200 35
*********************************************************
CODES:
lst = [$data]
new = [[lst[2*i],lst[2*i+1]] for i in range(len(lst)//2)]
s = sorted(new, key=lambda pair: pair[0], reverse = True)
lst = [elem for pair in s for elem in pair]
[print(lst[2*i], lst[2*i+1]) for i in range(len(lst))]
=========================================================
"""











































