import pandas as pd
from gym_exchange.data_orderbook_adapter import Configuration, Debugger
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.exchange.basic_exc.assets.order_flow import OrderFlow
from gym_exchange.exchange.basic_exc.assets.order_flow_list import FlowList
from gym_exchange.data_orderbook_adapter import Debugger, Configuration
from gym_exchange.data_orderbook_adapter.data_adjuster import DataAdjuster
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.orderbook import OrderBook


class RawDecoder():
    def __init__(self, price_level, horizon, historical_data, data_loader):
        self.historical_data = historical_data
        self.price_level = price_level
        self.horizon = horizon
        self.data_loader_iterrows = data_loader.iterrows()
        self.index = 0
        # --------------- NEED ACTIONS --------------------
        self.column_numbers_bid = [i for i in range(price_level * 4) if i % 4 == 2 or i % 4 == 3]
        self.column_numbers_ask = [i for i in range(price_level * 4) if i % 4 == 0 or i % 4 == 1]
        self.order_book = OrderBook()
        self.initialize_orderbook()
        self.length = (self.order_book.bids.depth != 0) + (self.order_book.asks.depth != 0)

    # -------------------------- 01 ----------------------------
    def initiaze_orderbook_message(self, side):
        columns = self.column_numbers_bid if side == 'bid' else self.column_numbers_ask
        l2 = self.historical_data.iloc[0, :].iloc[columns].reset_index().drop(['index'], axis=1)
        limit_orders = []
        order_id_list = [90000000 + 100000 * (side == 'bid') + i for i in range(self.price_level)]
        for i in range(self.price_level):
            trade_id = 90000000 + 100000 * (side == 'bid')
            timestamp = str(34200.000000002) if (side == 'bid') else str(34200.000000001)
            item = {'type': 'limit',
                    'side': side,
                    'quantity': l2.iloc[2 * i + 1, 0],
                    'price': l2.iloc[2 * i, 0],
                    'trade_id': trade_id,
                    'order_id': order_id_list[i],
                    "timestamp": timestamp}
            limit_orders.append(item)
        return limit_orders

    def initialize_orderbook_with_side(self, side):  # Add orders to order book
        limit_orders = self.initiaze_orderbook_message(side)
        for order in limit_orders:  _trades, _order_id = self.order_book.process_order(order, True, False)

    def initialize_orderbook(self):
        self.initialize_orderbook_with_side('bid')
        self.initialize_orderbook_with_side('ask')

    # -------------------------- 02 ----------------------------
    def step(self):
        message = next(self.data_loader_iterrows)[1]
        if message['type'] in (1,2,3):
            order_flow = OrderFlow(
                Type = message['type'],
                direction='bid' if message['side'] == 1 else 'ask' if message['side'] == -1 else 'Error',
                # direction = 'ask' if message['side'] == 1 else 'bid',
                size = message['quantity'],
                price = message['price'],
                trade_id = message['order_id'],
                order_id = message['order_id'],
                time = message['timestamp']
            )
        elif message['type'] in (4,):
            order_flow = OrderFlow(
                Type = 1,
                direction = 'bid' if (-1 * message['side']) == 1 else 'ask' if (-1 * message['side']) == -1 else 'Error',
                # direction = 'ask' if (-1 * message['side']) == 1 else 'bid',
                size = message['quantity'],
                price = message['price'],
                trade_id = message['order_id'],
                order_id = message['order_id'],
                time = message['timestamp']
            )
        else:
            order_flow = None # Not sure !TODO
        # assert message['type'] in (1,2,3), "the exchange can only handle this three situations"
        return order_flow


class RawEncoder():
    def __init__(self, decoder):
        self.decoder = decoder
        self.flow_lists = []  # [flow_list, flow_list, ... ], for each step we get a flow_list

    # -------------------------- 01 ----------------------------
    def initialize_order_flows(self):
        flow_list = FlowList()
        for side in ['ask', 'bid']:
            List = self.decoder.initiaze_orderbook_message(side)
            for Dict in List:
                order_flow = OrderFlow(
                    time=Dict['timestamp'],
                    Type=1,
                    order_id=Dict['order_id'],
                    size=Dict['quantity'],
                    price=Dict['price'],
                    direction=Dict['side'],
                    trade_id=Dict['trade_id']
                )
                flow_list.append(order_flow)
        self.flow_lists.append(flow_list)
        return self.flow_lists

    def get_all_running_order_flows(self):
        [self.step() for _ in range(Configuration.horizon)]
        return self.flow_lists

    def step(self):  # get_single_running_order_flows
        order_flow = self.decoder.step()  # the decoder return single data in step()
        # ···················· 02.01 ····················
        flow_list = FlowList()
        if order_flow is not None:
            flow_list.append(order_flow)
        # ···················· 02.02 ····················
        self.flow_lists.append(flow_list)

        return flow_list

    # -------------------------- 03 ----------------------------
    def __call__(self):
        self.initialize_order_flows()
        self.get_all_running_order_flows()
        return self.flow_lists




if __name__ == "__main__":
    decoder = RawDecoder(**DataPipeline()())
    encoder = RawEncoder(decoder)
    Ofs = encoder()

    with open("/Users/kang/AlphaTrade/gym_exchange/outputs/raw_encoder_ofs.log", "w+") as f:
        for i in range(len(Ofs)):
            f.write(f"------ {i} ------\n")
            f.write(Ofs[i].__str__())



