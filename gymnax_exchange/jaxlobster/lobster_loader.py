"""
#TODO: Update to fit reality. 

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
from os import listdir
from os.path import isfile, join
import warnings
import os

import jax
import itertools
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np

# from jax import numpy as jnp
# import jax
# from jax import lax
from glob import glob
from functools import partial

class LoadLOBSTER():
    """
    Class which completes all of the loading from the lobster data
    set files. 

    ...

    Attributes
    ----------
    atpath : str
        Path to the "AlphaTrade" repository folder
    messagePath : str
        Path to folder containing all message data
    orderbookPath : str
        Path to folder containing all orderbook state data
    window_type : str
        "fixed_time" or "fixed_steps" defines whether episode windows
        are defined by time (e.g. 30mins) or n_steps (e.g. 150)
    window_length : int
        Length of an episode window. In seconds or steps (see above)
    n_data_msg_per_step : int
        number of messages to process from data per step

    Methods
    -------
    run_loading():
        Returns jax.numpy arrays with messages sliced into fixed-size
        windows. Dimensions: (Window, Step, Message, Features)
        Additionally returns initial state data for each window, and 
        the lengths (horizons) of each window. 
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
        self.n_data_msg_per_step=n_msg_per_step


    def run_loading(self):
        """Returns jax.numpy arrays with messages sliced into fixed-size
        windows. Dimensions: (Window, Step, Message, Features)
        
            Parameters:
                NA   
            Returns:
                loaded_msg_windows (Array): messages sliced into fixed-
                                            size windows.
                                            Dimensions: 
                                            (Window, Step, Message, Features)
        """
        message_days, orderbook_days = self._load_files()
        pairs = [self._pre_process_msg_ob(msg,ob) 
                 for msg,ob 
                 in zip(message_days,orderbook_days)]
        message_days, orderbook_days = zip(*pairs)
        slicedCubes_withOB_list = [self._slice_day_no_overlap(msg_day,ob_day) 
                                   for msg_day,ob_day 
                                   in zip(message_days,orderbook_days)]
        cubes_withOB = list(itertools.chain \
                            .from_iterable(slicedCubes_withOB_list))
        max_steps_in_windows_arr = np.array([m.shape[0] 
                                              for m,o 
                                              in cubes_withOB],np.int32)
        cubes_withOB=self._pad_window_cubes(cubes_withOB)
        loaded_msg_windows,loaded_book_windows=map(np.array,
                                                    zip(*cubes_withOB))
        n_windows=len(loaded_book_windows)
        return (loaded_msg_windows,
                loaded_book_windows,
                max_steps_in_windows_arr,
                n_windows)

    def _pad_window_cubes(self,cubes_withOB):
        #Get length of longest window
        max_win = max(w.shape[0] for w, o in cubes_withOB)
        new_cubes_withOB = []
        for cube, OB in cubes_withOB:
            cube = self._pad_cube(cube, max_win)
            new_cubes_withOB.append((cube, OB))
        return new_cubes_withOB
    
    def _pad_cube(self, cube, target_shape):
        """Given a 'cube' of data, representing one episode window, pad
        it with extra entries of 0 to reach a target number of steps.
        """
        # Calculate the amount of padding required
        padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
        padded_cube = np.pad(cube, padding, mode='constant', constant_values=0)
        return padded_cube

    def _slice_day_no_overlap(self, message_day, orderbook_day):
        """ Given a day of message and orderbook data, slice it into
        'cubes' of episode windows. 
        """
        sliced_parts, init_OBs = self._split_day_to_windows(message_day, orderbook_day)
        slicedCubes = [self._slice_to_cube(slice_) for slice_ in sliced_parts]
        slicedCubes_withOB = zip(slicedCubes, init_OBs)
        return slicedCubes_withOB

    def _load_files(self):
        """Loads the csvs as pandas arrays. Files are seperated by days
        Could potentially be optimised to work around pandas, very slow.         
        """
        readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
        messageFiles, orderbookFiles = readFromPath(self.messagePath), readFromPath(self.orderbookPath)
        dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
        messageCSVs = [pd.read_csv(self.messagePath + file, usecols=range(6), dtype=dtype, header=None) for file in messageFiles if file[-3:] == "csv"]
        orderbookCSVs = [pd.read_csv(self.orderbookPath + file, header=None) for file in orderbookFiles if file[-3:] == "csv"]
        return messageCSVs, orderbookCSVs
    
    def _pre_process_msg_ob(self,message_day,orderbook_day):
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
    
    def _daily_slice_indeces(self,type,start, end, interval):
        """Returns a list of times of indices at which to cut the daily
        message data into data windows.
            Parameters:
                type (str): "fixed_steps" or "fixed_time" mode
                start (int): start time of the day or index of
                                first message to consider
                end (int): end time or last index to consider.
                            
                interval (int): length of an episode window in
                                terms of time (s) or number of 
                                steps.
            Returns:
                    indices (List): Either times or indices at which
                    to slice the data array. 
        """
        if type == "fixed_steps":
            end_index = ((end-start)
                         // self.n_data_msg_per_step*self.n_data_msg_per_step+start+1)
            indices = list(range(start, end_index, self.n_data_msg_per_step*interval))
        elif type == "fixed_time":
            indices = list(range(start, end+1, interval))
        else: raise NotImplementedError('Use either "fixed_time" or' 
                                        + ' "fixed_steps"')
        if len(indices)<2:
            raise ValueError("Not enough range to get a slice")
        return indices

    def _split_day_to_windows(self,message_day,orderbook_day):
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
        d_end = (message_day['time_s'].max()+1 
                 if self.window_type=="fixed_time"  
                 else message_day.shape[0])
        d_start = (message_day['time_s'].min() 
                 if self.window_type=="fixed_time"  
                 else 0)
        indices=self._daily_slice_indeces(self.window_type,
                                         d_start,
                                         d_end,
                                         self.window_length)
        sliced_parts = []
        init_OBs = []
        for i in range(len(indices) - 1):
            start_index = indices[i]
            end_index = indices[i + 1]
            if self.window_type == "fixed_steps":
                sliced_part = message_day[(message_day.index > start_index) &
                                             (message_day.index <= end_index)]
            elif self.window_type == "fixed_time":
                index_s, index_e = message_day[(message_day['time'] >= start_index) &
                                            (message_day['time'] < end_index)].index[[0, -1]].tolist()
                index_e = ((index_e // self.n_data_msg_per_step - 1) * self.n_data_msg_per_step
                            + index_s % self.n_data_msg_per_step)
                assert ((index_e - index_s) 
                        % self.n_data_msg_per_step == 0), 'wrong code 31'
                sliced_part = message_day.loc[np.arange(index_s, index_e)]
            sliced_parts.append(sliced_part)
            init_OBs.append(orderbook_day.iloc[start_index,:])
        
        if self.window_type == "fixed_steps":
            #print(indices)
           # print(len(sliced_parts))
            assert len(sliced_parts) == len(indices)-1, 'wrong code 33'
            for part in sliced_parts:
                assert part.shape[0] % self.n_data_msg_per_step == 0, 'wrong code 34'
        elif self.window_type == "fixed_time":
            for part in sliced_parts:
                assert part.shape[0] % self.n_data_msg_per_step == 0, 'wrong code 34'
        return sliced_parts, init_OBs
    
    def _slice_to_cube(self,sliced):
        """Turn a 2D pandas table of messages into a 3D numpy array
        whereby the additional dimension is due to the message
        stream being split into fixed-size blocks"""
        columns = ['type','direction','qty','price',
                   'trader_id','order_id','time_s','time_ns']
        cube = sliced[columns].to_numpy()
        cube = cube.reshape((-1, self.n_data_msg_per_step, 8))
        return cube


class LoadLOBSTER_resample():
    """
    Class which completes all of the loading from the lobster data
    set files as a single array of all messages of interest.
    
    Assumes that the split into padded chunks happens 'live'.
    Provides jittable methods to make this happen.

    ...

    Attributes
    ----------
    atpath : str
        Path to the "AlphaTrade" repository folder
    messagePath : str
        Path to folder containing all message data
    orderbookPath : str
        Path to folder containing all orderbook state data
    window_type : str
        "fixed_time" or "fixed_steps" defines whether episode windows
        are defined by time (e.g. 30mins) or n_steps (e.g. 150)
    window_length : int
        Length of an episode window. In seconds or steps (see above)
    window_resolution : int 
        Places at which a window may start. Every minute, 
            or N-thousand step based on window_type. 
    n_data_msg_per_step : int
        number of messages to process from data per step (omits option
        to consider a fixed time per step)
    
    

    Methods
    -------
    run_loading():
        Returns jax.numpy arrays with messages sliced into fixed-size
        windows. Dimensions: (Window, Step, Message, Features)
        Additionally returns initial state data for each window, and 
        the lengths (horizons) of each window. 
    """
    def __init__(self,
                 datapath,
                 atpath,
                 n_Levels=10,
                 type_="fixed_time",
                 window_length=1800,
                 window_resolution=60,
                 n_data_msg_per_step=100, #TODO rename this to n_data_msg_per_step?
                 day_start=34200,  
                 day_end=57600,
                 stock="AMZN",
                 time_period="2017Jan_oneday"):
        self.datapath=datapath+f"/rawLOBSTER/{stock}/{time_period}/"
        self.window_type=type_
        self.window_length=window_length
        self.window_resolution=window_resolution
        self.n_data_msg_per_step=n_data_msg_per_step 
        self.index_offest=0
        self.day_start=day_start
        self.day_end=day_end
        self.stock=stock
        self.time_period=time_period
        self.n_Levels=n_Levels
        self.alphatrade_path=atpath


        print("self.datapath",self.datapath)
        self.message_files = sorted(glob(self.datapath + '*message*.csv'))
        self.book_files = sorted(glob(self.datapath + '*orderbook*.csv'))
        #self.message_files = sorted([f for f in glob(self.datapath + '*message*.csv') if os.path.getsize(f) > 0])
        #self.book_files = sorted([f for f in glob(self.datapath + '*orderbook*.csv') if os.path.getsize(f) > 0])

        print('found', len(self.message_files), 'message files')
        print('found', len(self.book_files), 'book files')
   
        



    def run_loading(self):
        """Returns jax.numpy array with all messages aligned in series. 
        
            Parameters:
                NA   
            Returns:
                loaded_msgs (Array): messages 
                                            Dimensions: 
                                            (Messages, Features)
                max_window_size (Int)
        """
        # jax.profiler.start_trace("/tmp/profile-data")

        save_path = self._get_save_filename()

        if os.path.exists(save_path):
            print(f"Loading cached arrays from {save_path}")
            data = np.load(save_path, allow_pickle=True)
            msgs = data['msgs']
            starts = data['starts']
            ends = data['ends']
            obs = data['obs']
            max_msgs_in_windows_arr = data['max_msgs_in_windows_arr']
        else:

            msgs,starts,ends,obs = self._load_files()

            # jax.profiler.stop_trace()

            # jax.profiler.start_trace("/tmp/profile-data")

            
            #Concatenate the data from all the days.
            msgs=np.concatenate(msgs,0)
            starts=np.concatenate(starts,0)
            ends=np.concatenate(ends,0)
            obs=np.concatenate(obs,0)
            max_msgs_in_windows_arr=ends - starts

            if self.n_data_msg_per_step !=0:
                (msgs,
                max_msgs_in_windows_arr)=self._pad_last_ep(msgs,
                                                            max_msgs_in_windows_arr)
            

            print(f"Saving arrays to {save_path}")
            np.savez_compressed(
                save_path,
                msgs=msgs,
                starts=starts,
                ends=ends,
                obs=obs,
                max_msgs_in_windows_arr=max_msgs_in_windows_arr
            )

        return msgs,starts,ends,obs,max_msgs_in_windows_arr
    
    def _get_save_filename(self):
        # Create a unique filename based on config parameters
        params = [
            str(self.stock),
            str(self.time_period),
            str(self.n_Levels),
            str(self.window_type),
            str(self.window_length),
            str(self.window_resolution),
            str(self.n_data_msg_per_step),
            str(self.day_start),
            str(self.day_end),
        ]
        base = "_".join(params)
        # Use a hash to avoid overly long filenames
        # hash_str = hashlib.md5(base.encode()).hexdigest()
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(self.alphatrade_path, "saved_npz"), exist_ok=True)
        fname = f"saved_npz/lobster_{base}.npz"
        return os.path.join(self.alphatrade_path, fname)

    def _pad_last_ep(self,messages,max_msgs_in_windows_arr):
        length_last_ep=max_msgs_in_windows_arr[-1]
        new_length=(length_last_ep//self.n_data_msg_per_step+1)*self.n_data_msg_per_step
        pad=np.zeros((new_length-length_last_ep,messages.shape[1]),dtype=np.int32)
        last_time=np.array([messages[-1,-2:][0]+1,0])
        pad[:,-2:]=last_time
        messages=np.concatenate((messages,pad))
        max_msgs_in_windows_arr[-1]=new_length
        return messages,max_msgs_in_windows_arr
    

    
    # def _load_files(self):
    #         """Loads the csvs as pandas arrays. Files are seperated by days
    #         Could potentially be optimised to work around pandas, very slow.         
    #         """
    #         dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
    #         print("self.message_files",self.message_files)
    #         messageCSVs = [pd.read_csv(file, usecols=range(6), dtype=dtype, header=None) for file in self.message_files if file[-3:] == "csv"]
    #         orderbookCSVs = [pd.read_csv(file, header=None) for file in self.book_files if file[-3:] == "csv"]
    #         return messageCSVs, orderbookCSVs

    def _load_files(self):
        """Loads the csvs as pandas arrays. Files are seperated by days
        Could potentially be optimised to work around pandas, very slow.         
        """
        import concurrent.futures
        import multiprocessing as mp
        import os
        import time
        from threading import Semaphore
        import hashlib
        
        dtypes = {0: float, 1: int, 2: int, 3: int, 4: int, 5: int}
        print("self.message_files",self.message_files)
        
        # Adaptive worker count based on file size and system resources
        total_files = len(self.message_files)
        
        # Start with fewer workers and scale based on system performance
        # I/O bound tasks benefit from more workers, but too many cause contention
        base_workers = min(mp.cpu_count() // 4, 8)  # Conservative start
        n_workers = min(base_workers, total_files, 16)  # Cap at 16 to avoid thrashing
        
        print(f"Using {n_workers} workers for parallel loading ({total_files} files)")
        
        # Semaphore to limit concurrent file operations (prevent disk thrashing)
        file_semaphore = Semaphore(n_workers * 2)  # Allow some buffering

        def read_pair(files):
            message_file, book_file = files
            if message_file[-3:] == "csv" and book_file[-3:] == "csv":
                with file_semaphore:  # Limit concurrent disk access
                    try:
                        start_time = time.time()
                        
                        # Read files more efficiently with chunking for large files
                        df_message = pd.read_csv(
                            message_file, 
                            usecols=range(6), 
                            dtype=dtypes, 
                            header=None, 
                            engine='c',
                            low_memory=True,  # Changed to True for better memory management
                            chunksize=None,   # No chunking for now, but option for future
                            na_filter=False,  # Skip NA detection for speed
                            skip_blank_lines=True
                        )
                        
                        df_book = pd.read_csv(
                            book_file, 
                            header=None, 
                            engine='c',
                            low_memory=True,
                            na_filter=False,
                            skip_blank_lines=True
                        )
                        
                        read_time = time.time() - start_time
                        
                        if not df_message.empty and not df_book.empty:
                            process_start = time.time()
                            
                            # Optimize pandas operations with copy=False where safe
                            msg, book = self._pre_process_msg_ob(df_message, df_book)
                            message_day, index_s, index_e, init_OBs = self._get_inits_day(msg, book)
                            
                            process_time = time.time() - process_start
                            total_time = time.time() - start_time
                            
                            file_size_mb = (os.path.getsize(message_file) + os.path.getsize(book_file)) / 1024 / 1024
                            throughput = file_size_mb / total_time if total_time > 0 else 0
                            
                            print(f"✓ {os.path.basename(message_file)} "
                                  f"({file_size_mb:.1f}MB, {throughput:.1f}MB/s) "
                                  f"read:{read_time:.2f}s proc:{process_time:.2f}s total:{total_time:.2f}s")
                            
                            return (message_day, index_s, index_e, init_OBs)
                        else:
                            if df_message.empty:
                                print(f"⚠ Empty message file: {os.path.basename(message_file)}")
                            if df_book.empty:
                                print(f"⚠ Empty orderbook file: {os.path.basename(book_file)}")
                    
                    except pd.errors.EmptyDataError:
                        print(f"⚠ Truly empty file: {os.path.basename(message_file)}")
                    except Exception as e:
                        print(f"✗ Error processing {os.path.basename(message_file)}: {e}")
            return None

        pairs = list(zip(self.message_files, self.book_files))
        messageDays = []
        startIndeces = []
        endIndeces = []
        initOrderboks = []

        # Process files with better resource management
        start_total = time.time()
        completed_files = 0
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="FileLoader"
        ) as executor:
            # Submit tasks in batches to avoid memory buildup
            batch_size = max(1, total_files // 4)  # Process in 4 batches
            
            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch_pairs = pairs[batch_start:batch_end]
                batch_number = batch_start // batch_size + 1
                total_batches = (total_files - 1) // batch_size + 1
                
                print(f"\n📦 Processing batch {batch_number}/{total_batches} "
                      f"({len(batch_pairs)} files)...")
                
                # Submit batch and process results as they complete
                future_to_pair = {
                    executor.submit(read_pair, pair): pair 
                    for pair in batch_pairs
                }
                
                batch_start_time = time.time()
                
                for future in concurrent.futures.as_completed(future_to_pair):
                    try:
                        result = future.result()
                        if result is not None:
                            message_day, index_s, index_e, init_OBs = result
                            messageDays.append(message_day)
                            startIndeces.append(index_s)
                            endIndeces.append(index_e)
                            initOrderboks.append(init_OBs)
                            completed_files += 1
                    except Exception as exc:
                        pair = future_to_pair[future]
                        print(f'✗ Batch task {pair} failed: {exc}')
                
                batch_time = time.time() - batch_start_time
                avg_time_per_file = batch_time / len(batch_pairs) if batch_pairs else 0
                print(f"   Batch {batch_number} completed in {batch_time:.2f}s "
                      f"(avg: {avg_time_per_file:.2f}s/file)")

        total_time = time.time() - start_total
        avg_time_per_file = total_time / completed_files if completed_files > 0 else 0
        
        print(f"\n🎉 Parallel loading completed!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Files processed: {completed_files}/{total_files}")
        print(f"   Average time per file: {avg_time_per_file:.2f}s")
        print(f"   Effective throughput: {completed_files / total_time:.2f} files/s")
        
        return messageDays, startIndeces, endIndeces, initOrderboks
    
    def _pre_process_msg_ob(self,message_day,orderbook_day):
        """Adjust message_day data and orderbook_day data. 
        Splits time into two fields, drops unused message_day types,
        transforms executions into limit orders and delete into cancel
        orders, and adds the traderID field. 
        """
        # Optimize pandas operations by avoiding unnecessary copies
        # and using vectorized operations where possible
        
        # Split the time into two integer fields (vectorized)
        time_int = message_day[0].astype(np.int64)  # More efficient than apply
        message_day[6] = time_int
        message_day[7] = ((message_day[0] - time_int) * 1_000_000_000).astype(np.int64)
        
        
        # Filter messages outside of trading hours (before day_start and after day_end)
        time_mask = (message_day[6] >= self.day_start) & (message_day[6] <= self.day_end)
        dropped_count = (~time_mask).sum()
        if dropped_count > 0:
            print(f"Dropped {dropped_count} messages outside trading hours ({self.day_start}-{self.day_end}s)")
        
        message_day = message_day[time_mask]
        
        message_day.columns = ['time','type','order_id','qty','price','direction','time_s','time_ns']
        
        # Filter message types more efficiently
        type_mask = message_day['type'].isin([1,2,3,4])
        message_day = message_day[type_mask].copy()  # Explicit copy to avoid warnings
        valid_index = message_day.index.to_numpy()
        message_day.reset_index(inplace=True, drop=True)

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
        
        # Vectorized transformations (faster than loc operations)
        execution_mask = message_day['type'] == 4
        delete_mask = message_day['type'] == 3
        
        # Turn executions into limit orders on the opposite book side
        message_day.loc[execution_mask, 'direction'] *= -1
        message_day.loc[execution_mask, 'type'] = 1
        
        # Turn delete into cancel orders
        message_day.loc[delete_mask, 'type'] = 2
        
        # Add trader_id field (copy of order_id) - faster assignment
        message_day['trader_id'] = message_day['order_id'].values  # Use .values for speed
        
        # Filter orderbook efficiently
        orderbook_day = orderbook_day.iloc[valid_index].copy()
        orderbook_day.reset_index(inplace=True, drop=True)
        
        # Suppress pandas warnings
        warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        
        return message_day, orderbook_day

    
    def _daily_slice_indeces(self,type,start, end, interval):
        """Returns a list of times of indices at which an episode
        window may start.
            Parameters:
                type (str): "fixed_steps" or "fixed_time" mode
                start (int): start time of the day or index of
                                first message to consider
                end (int): end time or last index to consider.
                            
                interval (int): length between starts in
                                terms of time (s) or number of 
                                steps.
            Returns:
                    indices (List): Either times or indices at which
                    to slice the data array. 
        """
        if type == "fixed_steps":
            if self.n_data_msg_per_step == 0:
                raise ValueError("n_data_msg_per_step cannot be 0 if using 'fixed_steps' as an episode end condition.")
            elif self.n_data_msg_per_step <0:
                raise ValueError("Negative messages per step makes no sense...")
            else:
                end_index = ((end-start)
                            // self.n_data_msg_per_step*self.n_data_msg_per_step+start+1)
                indices = list(range(start, end_index, self.n_data_msg_per_step*interval))
        elif type == "fixed_time":
            indices = list(range(start, end+1, interval))
        else: raise NotImplementedError('Use either "fixed_time" or' 
                                        + ' "fixed_steps"')
        if len(indices)<2:
            raise ValueError("Not enough range to get a slice")
        return indices

    def _get_inits_day(self,message_day,orderbook_day):
        """Obtains the starting indeces for each of the possible
        message windows and a list of the initial book states at
        these times. Doesn't return any message data due to this
        not being sliced at this time (expected on reset).

            Parameters:
                message (Array): Array of all messages in a day
                orderbook (Array): All order book states in a day.
                                    1st dim of equal size as message. 
           Returns:
                indices (Array): Array of indices of starting points.
                init_OBs (List): List of arrays repr. init. orderbook
                                    data for each starting point.
        """
        d_end = (#message_day['time_s'].max()+1-self.window_length+self.window_resolution
                 self.day_end
                 if self.window_type=="fixed_time"  
                 else message_day.shape[0]-
                    self.window_length*self.n_data_msg_per_step)
        d_start = (#message_day['time_s'].min() 
                    self.day_start
                 if self.window_type=="fixed_time"  
                 else 0)
        #Note indices may be either time or index. Confusing. 
        indices=self._daily_slice_indeces(self.window_type,
                                         d_start,
                                         d_end,
                                         self.window_resolution)
        index_s = []
        index_e = []
        if self.window_type == "fixed_steps":
                index_s=np.array(indices)
                index_e=np.array(indices)+np.ones_like(index_s)*self.n_data_msg_per_step*self.window_length
                max_msgs=self.n_data_msg_per_step*self.window_length

        elif self.window_type == "fixed_time":
            for i in range(len(indices) - 1):
                window_start = indices[i]
                window_end = indices[i] + self.window_length
                # Filter the messages for the current window
                filtered = message_day[
                    (message_day['time'] >= window_start) &
                    (message_day['time'] < window_end)
                ]
                # Debug: print out information about the candidate window
                #print(f"Window {i}: start time {window_start}, end time {window_end}, "
                #    f"found {len(filtered)} rows in the message data.")
                if not filtered.empty:
                    # Get the first and last row indices of this window
                    (i_s, i_e) = filtered.index[[0, -1]]
                    #print(f"  Window {i} indices: first index = {i_s}, last index = {i_e}")
                    index_s.append(i_s)
                    index_e.append(i_e)
                else:
                    # If no data is found, print a warning (seems to happen quite often for smaller window sizes)
                    print(f"  Warning: Window {i} has no data!")

        init_OBs=np.array(orderbook_day.iloc[np.array(index_s),:])
        index_s=np.array(index_s)+np.ones_like(np.array(index_s))*self.index_offest
        index_e=np.array(index_e)+np.ones_like(np.array(index_e))*self.index_offest
        self.index_offest=self.index_offest+message_day.shape[0]
        columns = ['type','direction','qty','price',
                   'trader_id','order_id','time_s','time_ns']
        message_day=message_day[columns].to_numpy()
        return message_day,index_s,index_e,init_OBs
    



if __name__ == "__main__":
    #Load data from 50 Levels, fixing each episode to 150 steps
    #containing 100 messages each. 
    loader=LoadLOBSTER_resample("/AlphaTrade/training_oneDay",10,"fixed_time",window_length=1800,n_data_msg_per_step=100,window_resolution=60)
    msgs,starts,ends,obs,max_msgs=loader.run_loading()
    print(msgs.shape)
    print(starts.shape)
    print(ends.shape)
    print(obs.shape)
    print(max_msgs)

    print(starts[720:722])

    print(msgs[starts[720:722]])


    print(msgs[-100:])


    """
    #Load data from 50 Levels, fixing each episode to 30 minutes
    #(1800 seconds) containing a varied number of steps of 100
    # messages each.
    loader=LoadLOBSTER_resample("./AlphaTrade",10,"fixed_steps",window_length=300,n_msg_per_step=100,window_resolution=5)
    msgs,starts,ends,obs,max_msgs=loader.run_loading()
    print(msgs.shape)
    print(starts.shape)
    print(ends.shape)
    print(obs.shape)
    print(max_msgs)


    loader=LoadLOBSTER("./AlphaTrade",10,"fixed_time",window_length=1800,n_msg_per_step=100)
    msgs,obs,max_steps,n_windows=loader.run_loading()
    print(msgs.shape)
    print(msgs[0,0,0,:])
    """
