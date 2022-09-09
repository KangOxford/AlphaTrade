import numpy as np
import pandas as pd
from gym_trading.envs.broker import Flag

def get_price_list(flow):
    price_list = []
    column_index = [i*2 for i in range(0,flow.shape[1]//2)]
    for i in range(flow.shape[0]):
        price_list.extend(flow.iloc[i,column_index].to_list())
    price_set = set(price_list)
    price_list = sorted(list(price_set), reverse = True)
    return price_list

def get_adjusted_obs(stream, price_list):
    result = [0 for _ in range(len(price_list))]
    for i in range(len(price_list)):
        for j in range(stream.shape[0]//2):
            if price_list[i] == stream.iloc[j*2]:
                result[i] = stream.iloc[j*2+1]
    return result 

def get_max_quantity(flow):
    price_list = []
    column_index = [i*2 + 1 for i in range(0,flow.shape[1]//2)]
    for i in range(flow.shape[0]):
        price_list.extend(flow.iloc[i,column_index].to_list())
    price_set = max(price_list)
    return price_set

def get_quantity_from_stream(stream):
    column_index = [i*2 + 1 for i in range(0,stream.shape[0]//2)]
    return stream.iloc[column_index].to_list()

def get_price_from_stream(stream):
    column_index = [i*2 for i in range(0,stream.shape[0]//2)]
    return stream.iloc[column_index].to_list()

def from_pairs2lst_pairs(pairs):
    lst = [[],[]]
    for pair in pairs:
        # if type(pair) == int: pair = np.expand_dims(pair,axis=0)
        try:
            lst[0].append(pair[0])
            lst[1].append(pair[1])
        except: pass
    return lst
def from_series2obs(series):
    price_level = Flag.price_level
    min_price = Flag.min_price 
    dictionary = {
        'price':np.array([min_price for _ in range(price_level)]).astype(np.int32),
        'quantity':np.array([0 for _ in range(price_level)]).astype(np.int32)
        }
    for i in range(min(len(series)//2, price_level)):
        # as prices could be 11 or 12 which is more than current price level
        dictionary['price'][i] = series.iloc[2*i]
        dictionary['quantity'][i] = series.iloc[2*i+1]
        # print("series.iloc[2*i+1] : ",series.iloc[2*i+1])
    return dictionary



def timing(f):
    from functools import wraps
    from time import time
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' %(f.__name__, args, kw, te-ts))
        return result
    return wrap


def exit_after(fn):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    # from __future__ import print_function
    s = 60
    import sys
    import threading
    from time import sleep
    try:
        import thread
    except ImportError:
        import _thread as thread
    def quit_function(fn_name):
        # print to stderr, unbuffered in Python 2.
        print('{0} took too long'.format(fn_name), file=sys.stderr)
        sys.stderr.flush() # Python 3 stderr is likely buffered.
        thread.interrupt_main() # raises KeyboardInterrupt
    def inner(*args, **kwargs):
        timer = threading.Timer(s, quit_function, args=[fn.__name__])
        timer.start()
        try:
            result = fn(*args, **kwargs)
        finally:
            timer.cancel()
        return result
    return inner


def dict_to_nparray(observation: dict) -> np.ndarray:
    price, quantity = observation['price'], observation['quantity']
    return np.array([price,quantity])

class Utils():
    def from_series2pair(stream):
        num = stream.shape[0]
        previous_list = list(stream)
        previous_flow = []
        for i in range(num):
            if i%2==0:
                previous_flow.append(
                    [previous_list[i],previous_list[i+1]]
                    ) 
        return previous_flow

    def from_pair2series(stream):
        def namelist():
            name_lst = []
            for i in range(len(stream)):
                name_lst.append("bid"+str(i+1))
                name_lst.append("bid"+str(i+1)+"_quantity")
            return name_lst
        name_lst = namelist()
        stream = sorted(stream,reverse=True)
        result = []
        for item in stream:
            result.append(item[0])
            result.append(item[1])
        return pd.Series(data=result, index = name_lst)
        # TODO deal with the empty data, which is object not float

    def remove_replicate(diff_list):
        # remove_replicate
        # diff_list = sorted(diff_list) ## !TODO not sure
        diff_list_keys = []
        for item in diff_list:
            diff_list_keys.append(item[0])
        set_diff_list_keys = sorted(set(diff_list_keys))
        
        index_list = []
        for item in set_diff_list_keys:
            index_list.append(diff_list_keys.index(item))
        index_list = sorted(index_list)
        
        present_flow = []
        for i in range(len(index_list)-1):
            index = index_list[i]
            if diff_list[index][0] == diff_list[index+1][0] :
                present_flow.append([
                    diff_list[index][0], 
                    diff_list[index][1]+diff_list[index+1][1]
                    ]) 
            elif diff_list[index][0] != diff_list[index+1][0] :
                present_flow.append([
                    diff_list[index][0],
                    diff_list[index][1]
                    ])
        if index_list[-1] == len(diff_list)-1:
            present_flow.append([
                diff_list[-1][0],
                diff_list[-1][1]
                ])
        elif index_list[-1] != len(diff_list)-1:
            present_flow.append([
                diff_list[-2][0], 
                diff_list[-2][1]+diff_list[-1][1]
                ])     
        result = present_flow.copy()
        for j in range(len(present_flow)):
            if present_flow[j][1] == 0:
                result.remove(present_flow[j])
        return result

def get_avarage_price(pairs):
    sum_product, sum_quantity = 0, 0 
    for item in pairs:
        for value in item:
            sum_product += value[0] * value[1]
            sum_quantity+= value[1]
    avarage_price = sum_product / sum_quantity
    return avarage_price



    
if __name__=="__main__":
    pairs = [[123,1],[133324,1],[132312,3]]##
    # series = observation[0]
    

    

    