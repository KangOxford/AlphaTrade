"""Docstring TBD"""
from os import listdir
from os.path import isfile, join
import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning


from jax import numpy as jnp
import jax
from jax import lax

            





class LoadLOBSTER():
    """Docstring
    """
    def __init__(self,alphatradepath,n_Levels):
        self.atpath=alphatradepath
        self.messagePath = alphatradepath+"/data/Flow_"+str(n_Levels)+"/"
        self.orderbookPath = alphatradepath+"/data/Book_"+str(n_Levels)+"/"

    def run_loading(self):
        messages, orderbooks = self.load_files()
        pairs = [self.pre_process_msg_ob(msg,ob) for msg,ob in zip(messages,orderbooks)]
        messages, orderbooks = zip(*pairs)


    def load_files(self):
        """Loads the csvs as pandas arrays

        Could potentially be optimised to work around pandas, very slow.         
        """
        readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
        messageFiles, orderbookFiles = readFromPath(self.messagePath), readFromPath(self.orderbookPath)
        dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
        messageCSVs = [pd.read_csv(self.messagePath + file, usecols=range(6), dtype=dtype, header=None) for file in messageFiles if file[-3:] == "csv"]
        orderbookCSVs = [pd.read_csv(self.orderbookPath + file, header=None) for file in orderbookFiles if file[-3:] == "csv"]
        return messageCSVs, orderbookCSVs
    
    def pre_process_msg_ob(self,message_day,orderbook_day):
        """Adjust message_day data and orderbook_day data. 
        Splits time into two fields, drops unused message_day types,
        transforms executions into limit orders and delete into cancel
        orders, and adds the traderID field. 
        """
        #split the time into two integer fields.
        message_day[6] = message_day[0].apply(lambda x: int(x))
        message_day[7] = ((message_day[0] - message_day[6]) * int(1e9)).astype(int)
        message_day.columns = ['time','type','order_id','qty','price','direction','time_s','time_ns']
        #Drop all message_days of type 5,6,7 (Hidden orders, Auction, Trading Halt)
        message_day = message_day[message_day.type.isin([1,2,3,4])]
        valid_index = message_day.index.to_numpy()
        message_day.reset_index(inplace=True,drop=True)
        # Turn executions into limit orders on the opposite book side
        message_day.loc[message_day['type'] == 4, 'direction'] *= -1
        message_day.loc[message_day['type'] == 4, 'type'] = 1
        #Turn delete into cancel orders
        message_day.loc[message_day['type'] == 3, 'type'] = 2
        #Add trader_id field (copy of order_id)
        warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        message_day['trader_id'] = message_day['order_id']
        orderbook_day.iloc[valid_index,:].reset_index(inplace=True, drop=True)
        return message_day,orderbook_day

    def slice_day_no_overlap(self, message_day, orderbook_day):
        def index_of_sliceWithoutOverlap_by_lines(start_time, end_time, interval):
            indices = list(range(start_time, end_time, interval))
            return indices
        indices = index_of_sliceWithoutOverlap_by_lines(0, message_day.shape[0]//100*100, 100*100)
        # def index_of_sliceWithoutOverlap_by_time(start_time, end_time, interval):
        #     indices = list(range(start_time, end_time, interval))
        #     return indices
        # indices = index_of_sliceWithoutOverlap_by_time(start_time, end_time, sliceTimeWindow)
        
        def splitMessage(message_day, orderbook_day):
            sliced_parts = []
            init_OBs = []
            for i in range(len(indices) - 1):
                start_index = indices[i]
                end_index = indices[i + 1]
                sliced_part = message_day[(message_day.index > start_index) & (message_day.index <= end_index)]
                sliced_parts.append(sliced_part)
                init_OBs.append(orderbook_day.iloc[start_index,:])
            # # Last sliced part from last index to end_time
            # start_index = indices[i]
            # end_index = indices[i + 1]
            # index_s, index_e = message_day[(message_day.index >= start_index) & (message_day.index < end_index)].index[[0, -1]].tolist()
            # # index_s, index_e = message_day[(message_day['time'] >= start_index) & (message_day['time'] < end_index)].index[[0, -1]].tolist()
            # index_s = (index_s // stepLines - 10) * stepLines + index_e % stepLines
            # assert (index_e - index_s) % stepLines == 0, 'wrong code 32'
            # last_sliced_part = message_day.loc[np.arange(index_s, index_e)]
            # sliced_parts.append(last_sliced_part)
            # init_OBs.append(orderbook_day.iloc[index_s, :])
            
            for part in sliced_parts:
                # print("start")
                # assert part.time_s.iloc[-1] - part.time_s.iloc[0] >= sliceTimeWindow, f'wrong code 33, {part.time_s.iloc[-1] - part.time_s.iloc[0]}, {sliceTimeWindow}'
                assert len(sliced_parts) == len(indices)-1, 'wrong code 33'
                assert part.shape[0] % stepLines == 0, 'wrong code 34'
            return sliced_parts, init_OBs
        sliced_parts, init_OBs = splitMessage(message_day, orderbook_day)
        def sliced2cude(sliced):
            columns = ['type','direction','qty','price','trader_id','order_id','time_s','time_ns']
            cube = sliced[columns].to_numpy()
            cube = cube.reshape((-1, stepLines, 8))
            return cube
        # def initialOrderbook():
        slicedCubes = [sliced2cude(sliced) for sliced in sliced_parts]
        # Cube: dynamic_horizon * stepLines * 8
        slicedCubes_withOB = zip(slicedCubes, init_OBs)
        return slicedCubes_withOB
    
    def daily_slice_indeces(self,type,overlap,start_time, end_time, interval):
        if type=="Time":
            indices = list(range(start_time, end_time, interval))
        elif type=="Line":
            indices = list(range(start_time, end_time, interval))

