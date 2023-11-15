"""Docstring TBD"""
from os import listdir
from os.path import isfile, join
import warnings

import itertools
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np

from jax import numpy as jnp
import jax
from jax import lax

            





class LoadLOBSTER():
    """Docstring
    """
    def __init__(self,
                 alphatradepath,
                 n_Levels=10,
                 type_="fixed_time",
                 window_length=1800,
                 n_msg_per_step=100):
        self.atpath=alphatradepath
        self.messagePath = alphatradepath+"/data/Flow_"+str(n_Levels)+"/"
        self.orderbookPath = alphatradepath+"/data/Book_"+str(n_Levels)+"/"
        self.window_type=type_
        self.window_length=window_length
        self.n_messages=n_msg_per_step
        self.day_start=34200
        self.day_end=57600

    def run_loading(self):
        message_days, orderbook_days = self.load_files()
        pairs = [self.pre_process_msg_ob(msg,ob) 
                 for msg,ob 
                 in zip(message_days,orderbook_days)]
        message_days, orderbook_days = zip(*pairs)
        slicedCubes_withOB_list = [self.slice_day_no_overlap(msg_day,ob_day) 
                                   for msg_day,ob_day 
                                   in zip(message_days,orderbook_days)]
        cubes_withOB = list(itertools.chain \
                            .from_iterable(slicedCubes_withOB_list))
        max_steps_in_windows_arr = jnp.array([m.shape[0] 
                                              for m,o 
                                              in cubes_withOB],jnp.int32)
        cubes_withOB=self.pad_window_cubes(cubes_withOB)
        loaded_msg_windows,loaded_book_windows=map(jnp.array,
                                                    zip(*cubes_withOB))
        n_windows=len(loaded_book_windows)
        return (loaded_msg_windows,
                loaded_book_windows,
                max_steps_in_windows_arr,
                n_windows)

    def pad_window_cubes(self,cubes_withOB):
        #Get length of longest window
        max_win = max(w.shape[0] for w, o in cubes_withOB)
        new_cubes_withOB = []
        for cube, OB in cubes_withOB:
            cube = self.pad_cube(cube, max_win)
            new_cubes_withOB.append((cube, OB))
        return new_cubes_withOB
    
    def pad_cube(self, cube, target_shape):
                # Calculate the amount of padding required
                padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
                padded_cube = np.pad(cube, padding, mode='constant', constant_values=0)
                return padded_cube

    def slice_day_no_overlap(self, message_day, orderbook_day):
        sliced_parts, init_OBs = self.split_day_to_windows(message_day, orderbook_day)
        slicedCubes = [self._slice_to_cube(slice_) for slice_ in sliced_parts]
        slicedCubes_withOB = zip(slicedCubes, init_OBs)
        return slicedCubes_withOB

    def load_files(self):
        """Loads the csvs as pandas arrays. Files are seperated by days

        Could potentially be optimised to work around pandas, very slow.         
        """
        from os import listdir; from os.path import isfile, join; import pandas as pd
        readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
        messageFiles, orderbookFiles = readFromPath(messagePath), readFromPath(orderbookPath)
        dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
        messageCSVs = [pd.read_csv(messagePath + file, usecols=range(6), dtype=dtype, header=None) for file in messageFiles if file[-3:] == "csv"]
        orderbookCSVs = [pd.read_csv(orderbookPath + file, header=None) for file in orderbookFiles if file[-3:] == "csv"]
        return messageCSVs, orderbookCSVs
    messages, orderbooks = load_files()
    def preProcessingMassegeOB(message, orderbook):
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
    
    def daily_slice_indeces(self,type,start, end, interval, msgs_length=100000):
        """Returns a list of times of indices at which to cut the daily
        message data into data windows.
            Parameters:
                type (str): "fixed_steps" or "fixed_time" mode
                start (int): start time of the day or index of
                                first message to consider
                end (int): end time or last index to consider.
                            If = -1 and using fixe steps then end
                            index is the full array.
                interval (int): length of an episode window in
                                terms of time (s) or number of 
                                steps.
                msgs_length (int): Only used in the event of 
                                    end_time=-1. Total number
                                    of messages in data. 
            Returns:
                    indices (List): Either times or indices at which
                    to slice the data array. 
        """
        if type == "fixed_steps":
            end_index = ((msgs_length-start)
                            //self.n_messages*self.n_messages+start
                          if end==-1 else (end-start)
                            //self.n_messages*self.n_messages+start)
            indices = list(range(start, end_index, self.n_messages*interval))
        elif type == "fixed_time":
            if end==-1:
                raise IndexError('Cannot use -1 as an index if fixed_time')
            indices = list(range(start, end, interval))
            print(indices)
        else: raise NotImplementedError('Use either "fixed_time" or' 
                                        + ' "fixed_steps"')
        if len(indices)<2:
            raise ValueError("Not enough range to get a slice")
        return indices

    def split_day_to_windows(self,message,orderbook):
        """Splits a day of messages into given windows.
        The windows are either defined by a fixed time interval or a 
        fixed number of steps, whereby each step is a fixed number of 
        messages.
        For each window, the initial orderbook state is taken from the
        orderbook data. 
            Parameters:
                message (Array): Array of all messages in a day
                orderbook (Array): All order book states in a day.
                                    1st dim of equal size as message. 
           Returns:
                sliced_parts (List): List of arrays each repr. a window.  
                init_OBs (List): List of arrays repr. init. orderbook
                                    data for each window. 
        """
        indices=self.daily_slice_indeces(self.window_type,
                                         self.day_start,
                                         self.day_end,
                                         self.window_length,
                                         message.shape[0])
        sliced_parts = []
        init_OBs = []
        for i in range(len(indices) - 1):
            start_index = indices[i]
            end_index = indices[i + 1]
            if self.window_type == "fixed_steps":
                sliced_part = message[(message.index > start_index) &
                                             (message.index <= end_index)]
            elif self.window_type == "fixed_time":
                index_s, index_e = message[(message['time'] >= start_index) &
                                            (message['time'] < end_index)].index[[0, -1]].tolist()
                
                index_e = ((index_e // self.n_messages + 10) * self.n_messages
                            + index_s % self.n_messages)
                assert ((index_e - index_s) 
                        % self.n_messages == 0), 'wrong code 31'
                sliced_part = message.loc[np.arange(index_s, index_e)]
            sliced_parts.append(sliced_part)
            init_OBs.append(orderbook.iloc[start_index,:])
        
            if self.window_type == "fixed_steps":
                assert len(sliced_parts) == len(indices)-1, 'wrong code 33'
                for part in sliced_parts:
                    assert part.shape[0] % self.n_messages == 0, 'wrong code 34'
            elif self.window_type == "fixed_time":
                for part in sliced_parts:
                    assert (part.time_s.iloc[-1] 
                            - part.time_s.iloc[0] 
                            >= self.window_length), \
                            f"wrong code 33, {part.time_s.iloc[-1] - part.time_s.iloc[0]}, {self.window_length}"
                    assert part.shape[0] % self.n_messages == 0, 'wrong code 34'
        return sliced_parts, init_OBs
    
    def _slice_to_cube(self,sliced):
        """Turn a 2D pandas table of messages into a 3D numpy array
        whereby the additional dimension is due to the message
        stream being split into fixed-size blocks"""
        columns = ['type','direction','qty','price',
                   'trader_id','order_id','time_s','time_ns']
        cube = sliced[columns].to_numpy()
        cube = cube.reshape((-1, self.n_messages, 8))
        return cube


if __name__ == "__main__":
    loader=LoadLOBSTER(".",10)
    print(loader.daily_slice_indeces(loader.window_type,loader.day_start,loader.day_end,loader.window_length))
    msgs,books,window_lengths,n_windows=loader.run_loading()
    print(msgs.shape,books.shape,window_lengths,n_windows)
    print(list(range(0,10+1,2)))


