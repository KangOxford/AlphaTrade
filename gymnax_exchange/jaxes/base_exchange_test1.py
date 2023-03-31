from gym_exchange.exchange.basic_exc.assets.executed_pairs import ExecutedPairsRecorder
from gymnax_exchange.jaxob.jorderbook import OrderBook


class BaseExchange():
    def __init__(self):
        super().__init__()

    def reset(self):
        self.index = 0
        self.flow_generator = (flow for flow in self.flow_list)
        self.order_book = OrderBook()
        self.initialize_orderbook()
        self.executed_pairs_recoder = ExecutedPairsRecorder()

    def type1_handler(self, message, index):
        trades, order_in_book = self.order_book.process_order(message, True, False)
        self.executed_pairs_recoder.step(trades, self.index)

    def type2_handler(self, message):
        tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
        in_book_quantity = tree.get_order(message['order_id']).quantity
        message['quantity'] = min(message['quantity'], in_book_quantity)
        (self.order_book.bids if message['side'] == 'bid' else self.order_book.asks).update_order(message)

    def type3_handler(self, message):
        done = False
        right_tree = self.order_book.bids if message['side'] == 'bid' else self.order_book.asks
        if right_tree.order_exists(message['order_id']) == False:
            try:
                right_price_list = right_tree.get_price_list(message['price'])
                for order in right_price_list:
                    if 90000000 <= order.order_id and order.order_id < 100000000:
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
            except:
                pass
        else:
            self.order_book.cancel_order(
                side=message['side'],
                order_id=message['order_id'],
                time=message['timestamp'],
            )
            self.cancelled_quantity = message['quantity']

    # @jit
    def process_tasks(self):
        for index, item in enumerate(self.task_list):
            if item is not None:
                message = item.to_message
                if item.type == 1:
                    self.type1_handler(message, index)
                elif item.type == 2:
                    self.type2_handler(message)
                elif item.type == 3:
                    self.type3_handler(message)

    def update_task_list(self, action=None):
        flow_list = next(self.flow_generator)
        self.task_list = [action] + [flow for flow in flow_list]

    def accumulating(self):
        self.index += 1

    def step(self, action=None):
        self.update_task_list(action)
        self.process_tasks()
        self.accumulating()
        return self.order_book


if __name__ == "__main__":
    exchange = BaseExchange()
    exchange.reset()
    for _ in range(2048):
        exchange.step()
