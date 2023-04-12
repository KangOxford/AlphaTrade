# ========================= 01 =========================
from gym_exchange import Config
from gym_exchange.orderbook import OrderBook
from gym_exchange.data_orderbook_adapter.raw_encoder import RawDecoder, RawEncoder
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.encoder import Encoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.exchange.basic_exc.assets.executed_pairs import ExecutedPairsRecorder
import orderbook


# ========================= 03 =========================
class BaseExchange():
    # -------------------------- 03.01 ----------------------------
    def __init__(self):
        self.flow_lists = self.flow_lists_initialization()

    def flow_lists_initialization(self):
        if Config.exchange_data_source == "encoder":
            decoder = Decoder(**DataPipeline()())
            encoder = Encoder(decoder)
        elif Config.exchange_data_source == "raw_encoder":
            decoder = RawDecoder(**DataPipeline()())
            encoder = RawEncoder(decoder)
        else:
            raise NotImplementedError
        flow_lists = encoder()
        flow_lists = self.to_order_flow_lists(flow_lists)
        return flow_lists

    def to_order_flow_lists(self, flow_lists):
        '''change side format from bid/ask to 1/-1
        side = -1 if item.side == 'ask' else 1'''
        for flow_list in flow_lists:
            for item in flow_list:
                side = -1 if item.side == 'ask' else 1
                item.side = side
        return flow_lists

    # -------------------------- 03.02 ----------------------------
    def reset(self):
        self.index = -1  # if initialized,index0th; else, index-1th.
        self.flow_generator = (flow_list for flow_list in self.flow_lists)
        self.order_book = OrderBook()
        self.initialize_orderbook()
        self.executed_pairs_recoder = ExecutedPairsRecorder()
        self.mid_prices = [(self.order_book.get_best_ask() + self.order_book.get_best_bid()) / 2]
        self.best_bids = [self.order_book.get_best_bid()]
        self.best_asks = [self.order_book.get_best_ask()]

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
        self.process_tasks()
        self.accumulating()
        return self.order_book

    # ························ 03.03.01 ·························
    # ··················· component of the step ·················
    def update_task_list(self, action=None):  # action : Action(for the definition of type)
        flow_list = next(self.flow_generator)  # used for historical data
        self.task_list = [action] + [flow for flow in flow_list]

    def process_tasks(self):  # para: self.task_list; return: self.order_book
        # if self.index ==125:
        #     print()#$
        for index, item in enumerate(self.task_list):  # advantange for ask limit order (in liquidation problem)
            if not (item is None or item.quantity == 0):
                message = item.to_message
                if item.type == 1:
                    self.type1_handler(message, index)
                elif item.type == 2:
                    self.type2_handler(message)
                elif item.type == 3:
                    self.type3_handler(message)

    def accumulating(self):
        try:  # $
            self.mid_prices.append((self.order_book.get_best_ask() + self.order_book.get_best_bid()) / 2)
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
        self.executed_pairs_recoder.step(trades, self.index)  # 2nd para: kind

    def type2_handler(self, message):
        ''' Cancellation (Partial deletion of a limit order)'''
        tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
        try:
            in_book_quantity = tree.get_order(message['order_id']).quantity
            message['quantity'] = min(message['quantity'], in_book_quantity)  # adjuested_message
            (self.order_book.bids if message['side'] == 'bid' else self.order_book.asks).update_order(message)
        except:
            '''EXAMPLE: in get_order return self.order_map[order_id] KeyError: 17142637'''
            pass  # TODO

    def type3_handler(self, message):
        '''Deletion (Total deletion of a limit order)'''
        done = False
        right_tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
        if right_tree.order_exists(message['order_id']) == False:
            try:  # message['price'] in the order_book
                right_price_list = right_tree.get_price_list(message['price'])  # my_price
                for order in right_price_list:
                    if 90000000 <= order.order_id and order.order_id < 100000000:  # if my_order_id in initial_orderbook_ids
                        '''Initial orderbook id is created via the exchange
                        cannot be the same with the message['order_id'].
                        Solution: total delete the first (90000000) order in the orderlist
                        at the price we want to totally delete.
                        message['order_id'] not in the order_book.
                        message['timestamp'] not in the order_book.
                        Only tackle with single order. If found, break.
                        Solution code: 31'''
                        self.order_book.cancel_order(
                            side=message['side'],
                            order_id=order.order_id,
                            time=order.timestamp,
                        )
                        self.cancelled_quantity = order.quantity
                        done = True;
                        break
                if not done:
                    raise NotImplementedError
            except:  # message['price'] not in the order_book
                pass
                # print()#$
                # raise NotImplementedError #TODO
        else:  # right_tree.order_exists(message['order_id']) == True
            self.order_book.cancel_order(
                side=message['side'],
                order_id=message['order_id'],
                time=message['timestamp'],
            )
            self.cancelled_quantity = message['quantity']


if __name__ == "__main__":
    exchange = BaseExchange()
    """
    for i in range(100):
        exchange.reset()
        for _ in range(2048):
            exchange.step()"""
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
