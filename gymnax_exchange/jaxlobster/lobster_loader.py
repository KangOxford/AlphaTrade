"""
load_LOBSTER 

University of Oxford
Corresponding Author: 
Kang Li     (kang.li@keble.ox.ac.uk)
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
Peer Nagy   (peer.nagy@reuben.ox.ac.uk)
V1.0

Module Description
This module loads data from load_LOBSTER, initializes orders data in cubes from
 messages for downstream tasks, and creates auxiliary arrays with 
 essential information for these tasks.

Key Components 
The return of the funcion is:
    Cubes_withOB: a list of (cube, OB), where cube is of of three dimension
                  0-aixs is index of data windows 
                  1-axis is index of steps inside the data window
                  2-axis is index of lines of data inside the steps
    max_steps_in_episode_arr: horizon(max steps) of one data window
    taskSize_array: the amount of share for n%(default as 1) of 
                    the traded volume in the window

Functionality Overview
load_files:             loads the csvs as pandas arrays
preProcessingMassegeOB: adjust message_day data and orderbook_day data. 
sliceWithoutOverlap:    split the message of one day into multiple cubes.
Cubes_withOB_padding:   pad the cubes to have the shape shape
                        to be stored in single array.
"""

from re import L
import jax.numpy as jnp
import numpy as np
import itertools


def load_LOBSTER(sliceTimeWindow, stepLines, messagePath, orderbookPath, start_time, end_time, tradeVolumePercentage=0.01):
    """Docstring
    """
    def preProcessingData_csv2pkl():
        return 0
    def load_files():
        """Loads the csvs as pandas arrays
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
        def splitTimeStamp(m):
            m[6] = m[0].apply(lambda x: int(x))
            m[7] = ((m[0] - m[6]) * int(1e9)).astype(int)
            m[8] = ((m[1]==4) | (m[1]==5)).astype(int)
            m.columns = ['time','type','order_id','qty','price','direction','time_s','time_ns','ifTraded']
            return m
        message = splitTimeStamp(message)
        def selectTradingInterval(m, o):
            m = m[(m.time_s>=34200) & (m.time_s<=57600)]
            o = o.iloc[m.index.to_numpy(),:] # valid_index 
            return m.reset_index(drop=True), o.reset_index(drop=True)
        message, orderbook = selectTradingInterval(message, orderbook)
        def filterValid(message):
            message = message[message.type.isin([1,2,3,4])]
            valid_index = message.index.to_numpy()
            message.reset_index(inplace=True,drop=True)
            return message, valid_index
        message, valid_index = filterValid(message)
        def adjustExecutions(message):
            message.loc[message['type'] == 4, 'direction'] *= -1
            message.loc[message['type'] == 4, 'type'] = 1
            return message
        message = adjustExecutions(message)
        def removeDeletes(message):
            message.loc[message['type'] == 3, 'type'] = 2
            return message
        message = removeDeletes(message)
        def addTraderId(message):
            import warnings
            from pandas.errors import SettingWithCopyWarning
            warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
            message['trader_id'] = message['order_id']
            return message

        message = addTraderId(message)
        orderbook.iloc[valid_index,:].reset_index(inplace=True, drop=True)
        return message,orderbook
    pairs = [preProcessingMassegeOB(message, orderbook) for message,orderbook in zip(messages,orderbooks)]
    messages, orderbooks = zip(*pairs)

    def sliceWithoutOverlap(message, orderbook):
        """split the message of one day into multiple cubes

        Args:
            message (_type_): _description_
            orderbook (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_horizon_of_message = message.shape[0]//100*100
        sliced_parts, init_OBs = [message.iloc[:max_horizon_of_message,:]], [orderbook.iloc[0,:]]
        def sliced2cude(sliced):
            columns = ['type','direction','qty','price','trader_id','order_id','time_s','time_ns','ifTraded']
            cube = sliced[columns].to_numpy()
            cube = cube.reshape((-1, stepLines, 9))
            return cube
        # def initialOrderbook():
        slicedCubes = [sliced2cude(sliced) for sliced in sliced_parts]
        # Cube: dynamic_horizon * stepLines * 9
        slicedCubes_withOB = zip(slicedCubes, init_OBs)
        return slicedCubes_withOB
    slicedCubes_withOB_list = [sliceWithoutOverlap(message, orderbook) for message,orderbook in zip(messages,orderbooks)]
    # i = 6 ; message,orderbook = messages[i],orderbooks[i]
    # slicedCubes_list(nested list), outer_layer : day, inter_later : time of the day
    def nestlist2flattenlist(nested_list):
        flattened_list = list(itertools.chain.from_iterable(nested_list))
        return flattened_list
    Cubes_withOB = nestlist2flattenlist(slicedCubes_withOB_list)
    
    taskSize_array = np.array([int((m[:,:,2]*m[:,:,8]).sum()*tradeVolumePercentage) for m,o in Cubes_withOB])
    max_steps_in_episode_arr = jnp.array([m.shape[0] for m,o in Cubes_withOB],jnp.int32)
    
    def get_start_idx_array_list():
        def get_start_idx_array(idx):
            cube=Cubes_withOB[idx][0]
            print(f"cube{idx} shape:{cube.shape}")
            start_time_array = cube[:,0,[6,7]]
            start_time_stamp_array = np.arange(34200,57600,900)
            start_idx_list = [(0, 34200, 34200, 34200)]
            for start_time_stamp in start_time_stamp_array:
                for i in range(1, start_time_array.shape[0]):
                    timestamp = lambda i: float(str( start_time_array[i][0])+"."+str( start_time_array[i][1]))
                    timestamp_before = timestamp(i-1)
                    timestamp_current = timestamp(i)
                    if timestamp_before< start_time_stamp <timestamp_current:
                        start_idx_list.append((i,timestamp_before,start_time_stamp,timestamp_current))
            start_idx_array = np.array(start_idx_list)
            start_idx_array[:,0] = np.array(start_idx_array[1:,0].tolist()+[max_steps_in_episode_arr[idx]])
            return start_idx_array
        return [get_start_idx_array(idx) for idx in range(len(Cubes_withOB))] # start_idx_array_list
    start_idx_array_list = get_start_idx_array_list()
    
    Cubes_withOB = [(cube[:,:,:-1], OB) for cube, OB in Cubes_withOB] # remove_ifTraded
    
    def Cubes_withOB_padding(Cubes_withOB):
        """pad the cubes to have the shape shape to be stored in single array

        Args:
            Cubes_withOB (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_m = max(m.shape[0] for m, o in Cubes_withOB)
        new_Cubes_withOB = []
        for cube, OB in Cubes_withOB:
            def padding(cube, target_shape):
                pad_width = np.zeros((100, 8))
                # Calculate the amount of padding required
                padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
                padded_cube = np.pad(cube, padding, mode='constant', constant_values=0)
                return padded_cube
            cube = padding(cube, max_m)
            new_Cubes_withOB.append((cube, OB))
        return new_Cubes_withOB 
    Cubes_withOB = Cubes_withOB_padding(Cubes_withOB)

    return Cubes_withOB, max_steps_in_episode_arr, start_idx_array_list, taskSize_array


# if __name__ == "__main__":
#     Cubes_withOB, max_steps_in_episode_arr, start_idx_array_list, taskSize_array = load_LOBSTER(
#         sliceTimeWindow,
#         stepLines,
#         messagePath,
#         orderbookPath,
#         start_time,
#         end_time,
#         tradeVolumePercentage,
#     )