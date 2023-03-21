import pandas as pd
from gym_exchange.data_orderbook_adapter import Configuration, Debugger
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.exchange.basic_exc.assets.order_flow import OrderFlow
from gym_exchange.exchange.basic_exc.assets.order_flow_list import FlowList
from gym_exchange.data_orderbook_adapter import Debugger, Configuration
from gym_exchange.data_orderbook_adapter import utils
from gym_exchange.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_exchange.data_orderbook_adapter.utils.InsideSignalEncoder import InsideSignalEncoder
from gym_exchange.data_orderbook_adapter.data_adjuster import DataAdjuster
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.orderbook import OrderBook


class RawDecoder():
    def __init__(self, price_level, horizon, historical_data, data_loader):
        self.historical_data = historical_data
        self.price_level = price_level
        self.horizon = horizon
        self.data_loader = data_loader
        self.index = 0
        # --------------- NEED ACTIONS --------------------
        self.column_numbers_bid = [i for i in range(price_level * 4) if i % 4 == 2 or i % 4 == 3]
        self.column_numbers_ask = [i for i in range(price_level * 4) if i % 4 == 0 or i % 4 == 1]
        self.bid_sid_historical_data = historical_data.iloc[:, self.column_numbers_bid]
        self.ask_sid_historical_data = historical_data.iloc[:, self.column_numbers_ask]
        self.order_book = OrderBook()
        self.initialize_orderbook()
        self.length = (self.order_book.bids.depth != 0) + (self.order_book.asks.depth != 0)
        self.data_adjuster = DataAdjuster(d2=self.bid_sid_historical_data, l2=self.ask_sid_historical_data)

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
        for order in limit_orders:  trades, order_id = self.order_book.process_order(order, True,
                                                                                     False)  # The current book may be viewed using a print
        if Debugger.on: print(self.order_book)

    def initialize_orderbook(self):
        self.initialize_orderbook_with_side('bid')
        self.initialize_orderbook_with_side('ask')

    def step(self):
        # -------------------------- 01 ----------------------------
        if Debugger.on:
            print("##" * 25 + '###' + "##" * 25);
            print("==" * 25 + " " + str(self.index) + " " + "==" * 25)
            print("##" * 25 + '###' + "##" * 25 + '\n');
            print("The order book used to be:");
            print(self.order_book)
        self.historical_message = self.data_loader.iloc[self.index, :]
        historical_message = list(self.historical_message)  # tbd
        inside_signal = InsideSignalEncoder(self.order_book, self.historical_message)()
        self.order_book = SignalProcessor(self.order_book)(inside_signal)

        if self.order_book.bids.depth != 0:
            outside_signal_bid, self.order_book = self.data_adjuster.adjust_data_drift(self.order_book,
                                                                                       self.historical_message[0],
                                                                                       self.index,
                                                                                       side='bid')  # adjust only happens when the side of lob is existed(initialised)
        if self.order_book.asks.depth != 0:
            outside_signal_ask, self.order_book = self.data_adjuster.adjust_data_drift(self.order_book,
                                                                                       self.historical_message[0],
                                                                                       self.index,
                                                                                       side='ask')  # adjust only happens when the side of lob is existed(initialised)

        if Debugger.on:
            # -------------------------- 04.01 ----------------------------
            if self.order_book.bids.depth != 0:
                single_side_historical_data = self.bid_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data,
                                             side='bid'), "the orderbook if different from the data"
            if self.order_book.asks.depth != 0:
                single_side_historical_data = self.ask_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data,
                                             side='ask'), "the orderbook if different from the data"
            print("********** Print orderbook for comparison **********");
            print(">>> Right_order_book");
            print(utils.get_right_answer(self.index, self.ask_sid_historical_data))
            print(">>> Right_order_book");
            print(utils.get_right_answer(self.index, self.bid_sid_historical_data))
            # -------------------------- 04.02 ----------------------------
            print(">>> Brief_self.order_book(self.order_book)")
            print(utils.brief_order_book(self.order_book, 'ask'))
            print(utils.brief_order_book(self.order_book, 'bid'))
            print("The orderbook is right!\n")
        try:
            outside_signals = [outside_signal_bid, outside_signal_ask]
        except:
            try:
                outside_signals = [outside_signal_bid]
            except:
                try:
                    outside_signals = [outside_signal_bid]
                except:
                    outside_signals = []
        self.index += 1
        return inside_signal, outside_signals

    def process(self):
        signals_list = []
        for index in range(self.horizon):  # size : self.horizon
            signals = self.step()
            signals_list.append(signals)
        return signals_list

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

    # -------------------------- 02 ----------------------------
    def inside_signal_encoding(self, inside_signal):
        if inside_signal['sign'] in (1, 2, 3,):
            order_flow = OrderFlow(
                time=inside_signal['timestamp'],
                Type=inside_signal['sign'],
                order_id=inside_signal['order_id'],
                size=inside_signal['quantity'],
                price=inside_signal['price'],
                direction=inside_signal['side'],
                trade_id=inside_signal['trade_id']
            )
        elif inside_signal['sign'] in (4,):
            order_flow = OrderFlow(
                time=inside_signal['timestamp'],
                Type=1,
                order_id=inside_signal['order_id'],
                size=inside_signal['quantity'],
                price=inside_signal['price'],
                direction=inside_signal['side'],
                trade_id=inside_signal['trade_id']
            )
        elif inside_signal['sign'] in (5, 6,):
            order_flow = None
        else:
            raise NotImplementedError
        return order_flow

    def outside_signal_encoding(self, signal):
        if signal['sign'] in (10, 11):
            order_flow = OrderFlow(
                time=signal['timestamp'],
                Type=signal['sign'] // 10,
                order_id=signal['order_id'],
                size=signal['quantity'],
                price=signal['price'],
                direction=signal['side'],
                trade_id=signal['trade_id']
            )
        # elif signal['sign'] in (20,): #TODO
        #     '''signal
        #     {'sign': 20, 'right_order_price': 31210000, 'wrong_order_price': 31209000, 'side': 'ask'}'''
        elif signal['sign'] in (60,):
            order_flow = None
        else:
            order_flow = None  # !not implemented yet
        return order_flow

    def get_all_running_order_flows(self):
        for index in range(Configuration.horizon):
            _ = self.step(index)
        return self.flow_lists

    def step(self, index=None):  # get_single_running_order_flows
        inside_signal, outside_signals = self.decoder.step()  # the decoder return single data in step()
        inside_order_flow = self.inside_signal_encoding(inside_signal)
        # ···················· 02.01 ····················
        flow_list = FlowList()
        if inside_order_flow is not None:
            flow_list.append(inside_order_flow)
        for signal in outside_signals:
            if type(signal) is list:
                for s in signal:
                    outside_order_flow = self.outside_signal_encoding(s)
                    if outside_order_flow is not None:
                        flow_list.append(outside_order_flow)
            else:
                outside_order_flow = self.outside_signal_encoding(signal)
                if outside_order_flow is not None:
                    flow_list.append(outside_order_flow)
        # ···················· 02.02 ····················
        self.flow_lists.append(flow_list)
        # ···················· 02.03 ····················
        if Debugger.Encoder.on:
            try:
                print("=" * 10 + ' ' + str(index) + " " + "=" * 10)
                print(">>> inside_signal");
                print(inside_signal)
                print(">>> outside_signal");
                [print(signal) for signal in outside_signals]
                print("-" * 23)
            except:
                pass
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

    with open("/Users/kang/AlphaTrade/gym_exchange/outputs/log_encoder_ofs.txt", "w+") as f:
        for i in range(len(Ofs)):
            f.write(f"------ {i} ------\n")
            f.write(Ofs[i].__str__())



