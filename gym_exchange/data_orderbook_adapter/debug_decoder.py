# =============================================================================
# 01 IMPORT PACKAGES
# =============================================================================
import pandas as pd
from gym_exchange.data_orderbook_adapter import Debugger, Configuration 
from gym_exchange.data_orderbook_adapter.decoder import Decoder
from gym_exchange.data_orderbook_adapter import utils
from gym_exchange.data_orderbook_adapter.utils.SignalProcessor import SignalProcessor
from gym_exchange.data_orderbook_adapter.utils.InsideSignalEncoder import InsideSignalEncoder
from gym_exchange.data_orderbook_adapter.data_adjuster import DataAdjuster
from gym_exchange.data_orderbook_adapter.data_pipeline import DataPipeline
from gym_exchange.orderbook import OrderBook

class DebugDecoder(Decoder):
    def __init__(self, price_level, horizon, historical_data, data_loader): 
        super().__init__(price_level, horizon, historical_data, data_loader)
        
    
    def step(self):
        # -------------------------- 01 ----------------------------
        if Debugger.on: 
            print("##"*25 + '###' + "##"*25);print("=="*25 + " " + str(self.index) + " "+ "=="*25)
            print("##"*25 + '###' + "##"*25+'\n');print("The order book used to be:"); print(self.order_book)
        self.historical_message = self.data_loader.iloc[self.index,:]
        historical_message = list(self.historical_message) # tbd 
        # if self.index == 237:breakpoint();
        # if self.index == 299:breakpoint();
        # print(self.order_book)#tbd
        inside_signal = InsideSignalEncoder(self.order_book, self.historical_message)()
        # print(inside_signal)#tbd
        # print(self.order_book)#tbd
        self.order_book = SignalProcessor(self.order_book)(inside_signal)
        # print(self.order_book)#tbd
        
        
        if self.order_book.bids.depth != 0:
            outside_signal_bid, self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, self.historical_message[0], self.index, side = 'bid') # adjust only happens when the side of lob is existed(initialised)
        if self.order_book.asks.depth != 0:
            # if self.index == 238:breakpoint()
            outside_signal_ask, self.order_book = self.data_adjuster.adjust_data_drift(self.order_book, self.historical_message[0], self.index, side = 'ask') # adjust only happens when the side of lob is existed(initialised)
            
        
        if Debugger.on: 
            # -------------------------- 04.01 ----------------------------
            if self.order_book.bids.depth != 0:
                single_side_historical_data = self.bid_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data, side = 'bid'), "the orderbook if different from the data"
            if self.order_book.asks.depth != 0:
                single_side_historical_data = self.ask_sid_historical_data
                assert utils.is_right_answer(self.order_book, self.index, single_side_historical_data, side = 'ask'), "the orderbook if different from the data"
            print("********** Print orderbook for comparison **********");
            print(">>> Right_order_book"); print(utils.get_right_answer(self.index, self.ask_sid_historical_data))
            print(">>> Right_order_book"); print(utils.get_right_answer(self.index, self.bid_sid_historical_data))
            # -------------------------- 04.02 ----------------------------
            print(">>> Brief_self.order_book(self.order_book)")
            print(utils.brief_order_book(self.order_book, 'ask'))
            print(utils.brief_order_book(self.order_book, 'bid'))
            print("The orderbook is right!\n")
        try:outside_signals = [outside_signal_bid, outside_signal_ask]
        except: 
            try:outside_signals = [outside_signal_bid]
            except: 
                try:outside_signals = [outside_signal_bid]
                except: outside_signals = []
        self.index += 1
        return inside_signal, outside_signals
        
    def process(self):
        signals_list = []
        for index in range(self.horizon): # size : self.horizon
            # if index == 124: breakpoint()#$
            signals = self.step()
            # print()
            # _, _ = self.step()
            signals_list.append(signals)
        return signals_list
                    
if __name__ == "__main__":
    # =============================================================================
    # 02 REVISING OF ORDERBOOK
    # =============================================================================
    decoder = Decoder(**DataPipeline()())
    signals_list = decoder.process()
    # breakpoint() # tbd
    
        
    with open("/Users/kang/AlphaTrade/gym_exchange/outputs/log_decoder_ofs.txt","w+") as f:
        for i in range(len(signals_list)):
            f.write(f"\n------ {i} ------\n")
            f.write(signals_list[i].__str__())
