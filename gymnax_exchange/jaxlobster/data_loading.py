from pathlib import Path
from typing import Generator, Optional
import pandas as pd
from itertools import groupby
import re
from tqdm import tqdm
import math
import numpy as np
import torch.distributed as dist


from constants import MESSAGE_TOKEN_DTYPE_MAP, MESSAGE_TOKEN_TYPES, TIME_COL, get_orderbook_token_types


def _batch(iterable: list[str], n=1) -> Generator[list[str], None, None]:
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def _df_to_str(df: pd.DataFrame, n_msgs: int = 1) -> list[str]:
    columns = df.columns.tolist()
    values = df.to_numpy()
    row_strings = [','.join([f"{col},{val}" for col, val in zip(columns, row) if not pd.isna(val)]) for row in values]
    concatted_strings = ['\n'.join(rows) for rows in _batch(row_strings, n_msgs)]
    # if len(concatted_strings[-1].split("\n")) % n_msgs:
        # TODO: Currently the last incomplete batch is thrown away.
        # This can be improved by filling up the batch with the next file batch, but it's quiet tedious
        # concatted_strings = concatted_strings[:-1]
    return concatted_strings

def load_message_df(file: str, cast_dtypes: bool = False, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=object, header=None, nrows=nrows)
    removed_indices = []
    if df.iloc[:, -1].isna().sum():
        mask = df.iloc[:, -1].isna()
        removed_indices = df[~mask].index.to_list()
        df = df[mask]
    df = df.dropna(axis=1).reset_index(drop=True)
    try:
        df.columns = MESSAGE_TOKEN_TYPES
    except ValueError as e:
        print(f"Warning: message file {file} was expected to have {len(MESSAGE_TOKEN_TYPES)} columns" + 
                f" but got {len(df.columns)}). This file will be skipped.")
        return pd.DataFrame()
    if cast_dtypes:
        df = df.astype(MESSAGE_TOKEN_DTYPE_MAP)
    return df, removed_indices

def load_orderbook_df(file: str, cast_dtypes: bool = False) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=object, header=None)
    levels = len(df.columns) // 4
    columns = get_orderbook_token_types(levels)
    try:
        df.columns = columns
    except ValueError as e:
        print(f"Warning: orderbook file {file} was expected to have {len(columns)} columns," + 
                f" but got {len(df.columns)}). This file will be skipped.")
        return pd.DataFrame()
    if cast_dtypes:
        df = df.astype(int)
    return df

def extract_date(path: Path) -> Optional[str]:
    # Use a regex to find the date in the format YYYY-MM-DD
    match = re.search(r'\d{4}-\d{2}-\d{2}', path.name)
    return match.group(0) if match else None

def convert_to_nanoseconds(s: pd.Series):
    split_times = s.str.split('.', expand=True)
    split_times.columns = ['seconds', 'fractional']
    seconds = split_times['seconds'].astype(int)
    fractional = split_times['fractional'].fillna('0').str.ljust(9, '0').str[:9].astype(int)
    return seconds * 1_000_000_000 + fractional

def merge_dfm_dfo(dfm: pd.DataFrame, dfo: pd.DataFrame, n_msgs: int = 50):
    # reduce n_msgs by 1 due to the prepended orderbook row so that in total one batch has again n_msgs elements
    n_msgs -= 1
    num_blocks = math.ceil(len(dfm) / n_msgs)
    merged_len = len(dfm) + num_blocks
    merged_df = pd.DataFrame(index=range(merged_len), columns=[*dfm.columns, *dfo.columns])
    b_indices = np.arange(num_blocks) * (n_msgs + 1)
    b_vals = np.arange(num_blocks) * n_msgs
    merged_df.loc[b_indices, dfo.columns] = dfo.iloc[b_vals].values
    a_indices = np.setdiff1d(np.arange(merged_len), b_indices)
    merged_df.loc[a_indices, dfm.columns] = dfm.values
    return merged_df

def compute_df_from_file_group(group: list[str], n_msgs: int = 50, only_use_message_orderbook_matches: bool = True, differentiate_time: bool = True):
    df, dfm, dfo = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if only_use_message_orderbook_matches and len(group) != 2:
        return pd.DataFrame()
    for file in list(group):
        file = str(file)
        if "message" in file:
            dfm, removed_indices = load_message_df(file)
            if dfm.empty:
                continue
            if differentiate_time:
                dfm[TIME_COL] = convert_to_nanoseconds(dfm[TIME_COL])
                dfm[TIME_COL] = dfm[TIME_COL].diff()
                dfm = dfm.dropna(subset=TIME_COL).reset_index(drop=True)
                dfm[TIME_COL] = dfm[TIME_COL].astype(int)
            df = dfm.copy()
        elif "orderbook" in file:
            dfo = load_orderbook_df(file)
            if dfo.empty:
                continue
            df = dfo.copy()
        else:
            print(f"File {file} not known. Expected 'orderbook' or 'message' in file")
    if not dfm.empty and not dfo.empty:
        if removed_indices:
            dfo = dfo.drop(index=removed_indices)
        if differentiate_time:
            dfo = dfo[1:] # adjust to fit to message dataframe
        dfo = dfo.reset_index(drop=True)
        df = merge_dfm_dfo(dfm, dfo, n_msgs)
    return df
    
def get_data_stream_generator(data_dir: str, filter_str: str = "", n_msgs: int = 50, only_use_message_orderbook_matches: bool = True, 
                              differentiate_time: bool = True, world_size: int = 1, rank: int = 0, n_files: int = -1) -> Generator[list[str], None, None]:
    """Generate batches of message files and orderbook files in concatted format (depending on the filter_str) and yield them

    :param data_dir: directory of csv files. Expects files to be of format GOOG_2022-01-11_34200000_57600000_message_10.csv
    :param batch_size: batch size that is yielded. If None, return single elements, otherwise return 
        list of length batch size where each element contains n_msgs rows, defaults to None
    :param filter_str: optional filter of files, if e.g. only message files should be processed, defaults to ""
    :param n_msgs: number of messages in one batch element. One message/ row will look like this:

        '
        \<time\>,123345567768,\<event_type\>,1,\<order_id\>,123456,\<size\>,100,\<price\>,200,\<direction\>,1,
        \<ask_price_1\>,200,\<ask_size_1\>,20,\<bid_price_1\>,180,\<bid_size_1\>30,\<ask_price_N\>,210,\<ask_size_N\>,20,\<bid_price_N\>,170,\<bid_size_N\>30,
        '
        
        defaults to 20
    :param only_use_message_orderbook_matches: Whether to only use days where message data and orderbook data is available, defaults to True
    :yield: batch or single element of n_msgs
    """
    if filter_str and only_use_message_orderbook_matches:
        print("Warning: Data files are filtered but 'only_use_message_orderbook_matches' is True," +
              " meaning orderbook files and message files are expected. If you want to train purely on one of the two," + 
              " set 'only_use_message_orderbook_matches' to False")
    filter_str = f"*{filter_str}*.csv" if filter_str else "*.csv"

    # TODO: There is probably a more efficient way to do this, since extract_date is called two times.
    # But the array is so small that is basically doesn't matter
    msg_files = sorted(list(Path(data_dir).glob(filter_str)), key=extract_date)
    file_groups = [(key, list(group)) for key, group in groupby(msg_files, key=extract_date)]
    # taken_over_batch = None
    # if "test" not in data_dir:
    #     file_groups = file_groups[26:]
    print(f"Unique days: {len(file_groups)}")
    if n_files > 0 and n_files < len(file_groups):
        print(f"Using first {n_files} of {len(file_groups)}")
        file_groups = file_groups[:n_files]
    for _, group in tqdm(file_groups):
        print("Loading file group " + str([str(g) for g in group]))
        df = compute_df_from_file_group(group, n_msgs, only_use_message_orderbook_matches, differentiate_time)
        if df.empty:
            continue
        dfs = _df_to_str(df, n_msgs)
        # if batch_size is None:
        if "test" in data_dir:  # TODO: FIXME: Remove me possibly
            dfs = dfs[:50]  # take only first 50 elements of n_msgs just so validation doesn't take hours
        for i, el in enumerate(dfs):
            if i % world_size == rank:
                yield el
        # else:
        #     if taken_over_batch:
        #         remainder = batch_size - len(batch)
        #         batch = taken_over_batch + dfs[:remainder]
        #         dfs = dfs[remainder:]
        #         taken_over_batch = None
        #         yield batch
        #     for batch in _batch(dfs, batch_size):
        #         if len(batch) == batch_size:
        #             yield batch
        #         else:
        #             taken_over_batch = batch


def get_data_stream(data_dir: str, filter_str: str = "", n_msgs: int = 50, only_use_message_orderbook_matches: bool = True, 
                    differentiate_time: bool = True, n_files: int = -1) -> list[str]:
    """Generate batches of message files and orderbook files in concatted format (depending on the filter_str) and yield them

    :param data_dir: directory of csv files. Expects files to be of format GOOG_2022-01-11_34200000_57600000_message_10.csv
    :param batch_size: batch size that is yielded. If None, return single elements, otherwise return 
        list of length batch size where each element contains n_msgs rows, defaults to None
    :param filter_str: optional filter of files, if e.g. only message files should be processed, defaults to ""
    :param n_msgs: number of messages in one batch element. One message/ row will look like this:

        '
        \<time\>,123345567768,\<event_type\>,1,\<order_id\>,123456,\<size\>,100,\<price\>,200,\<direction\>,1,
        \<ask_price_1\>,200,\<ask_size_1\>,20,\<bid_price_1\>,180,\<bid_size_1\>30,\<ask_price_N\>,210,\<ask_size_N\>,20,\<bid_price_N\>,170,\<bid_size_N\>30,
        '
        
        defaults to 20
    :param only_use_message_orderbook_matches: Whether to only use days where message data and orderbook data is available, defaults to True
    :yield: batch or single element of n_msgs
    """
    if filter_str and only_use_message_orderbook_matches:
        print("Warning: Data files are filtered but 'only_use_message_orderbook_matches' is True," +
              " meaning orderbook files and message files are expected. If you want to train purely on one of the two," + 
              " set 'only_use_message_orderbook_matches' to False")
    filter_str = f"*{filter_str}*.csv" if filter_str else "*.csv"

    # TODO: There is probably a more efficient way to do this, since extract_date is called two times.
    # But the array is so small that is basically doesn't matter
    msg_files = sorted(list(Path(data_dir).glob(filter_str)), key=extract_date)
    file_groups = [(key, list(group)) for key, group in groupby(msg_files, key=extract_date)]
    dfs = []
    print(f"Unique days: {len(file_groups)}")
    if n_files > 0 and n_files < len(file_groups):
        print(f"Using first {n_files} file_groups out of {len(file_groups)}")
        file_groups = file_groups[:n_files]
    for _, group in tqdm(file_groups):
        print("Loading file group " + str([str(g) for g in group]))
        df = compute_df_from_file_group(group, n_msgs, only_use_message_orderbook_matches, differentiate_time)
        if df.empty:
            continue
        dfs.extend(_df_to_str(df, n_msgs))
    return dfs