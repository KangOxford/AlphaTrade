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
    return stream[column_index]
    # return stream.iloc[column_index].to_list()

def get_price_from_stream(stream):
    column_index = [i*2 for i in range(0,stream.shape[0]//2)]
    return stream[column_index]
    # return stream.iloc[column_index].to_list()

# def from_pairs2lst_pairs(pairs):
#     lst = [[],[]]
#     for pair in pairs:
#         # if type(pair) == int: pair = np.expand_dims(pair,axis=0)
#         try:
#             lst[0].append(pair[0])
#             lst[1].append(pair[1])
#         except: pass
#     return lst

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
    # remove_replicate and remove zero quantity
    
    diff_list = arg_sort(diff_list)
    
    diff_list_keys = []
    for item in diff_list[0,:]:
        diff_list_keys.append(item)
    set_diff_list_keys = sorted(set(diff_list_keys), reverse = True)
    index_list = []
    for item in set_diff_list_keys:
        index_list.append(diff_list_keys.index(item))
    index_list = sorted(index_list)
    
    present_flow = []
    for i in range(len(index_list)-1):
        index = index_list[i]
        if diff_list[0][index] == diff_list[0][index+1] :
            present_flow.append([
                diff_list[0][index], 
                diff_list[1][index]+diff_list[1][index+1]
                ]) 
        elif diff_list[0][index] != diff_list[0][index+1] :
            present_flow.append([
                diff_list[0][index],
                diff_list[1][index]
                ])
            
    if index_list[-1] == diff_list.shape[1]-1:
        present_flow.append([
            diff_list[0][-1],
            diff_list[1][-1]
            ])
    else:
        present_flow.append([
            diff_list[0][-2], 
            diff_list[1][-2]+diff_list[1][-1]
            ])     
        
    result = present_flow.copy()
    for j in range(len(present_flow)):
        if present_flow[j][1] == 0:
            result.remove(present_flow[j])
            # remove zero quantity
    # -----------
    result = np.array(result).T # change to satisfy the gym dimension requirements
    return result

def get_avarage_price(pairs):
    sum_product, sum_quantity = 0, 0 
    for item in pairs:
        for value in item:
            sum_product += value[0] * value[1]
            sum_quantity+= value[1]
    avarage_price = sum_product / sum_quantity
    return avarage_price

def change_to_gym_state(stream): 
    if stream.shape == (20,):
        return np.array([[stream[2*i] for i in range(len(stream)//2)], [stream[2*i+1] for i in range(len(stream)//2)]])
    else:
        raise NotImplementedError
    

def remove_zero_quantity(x):
    for i in range(x.shape[1]):
        if x[1][i] == 0:
            return x[:,:i]
    return x
    '''e.g. 1
    x == np.array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
            31140000, 31130000, 31120300, 31120200],
           [       3,        4,       16,        2,        2,      506,
                   4,        2,        0,        0]])
    return array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
            31140000, 31130000],
           [       3,        4,       16,        2,        2,      506,
                   4,        2]])
    e.g. 2
    x == np.array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
            31140000, 31130000, 31120300, 31120200],
           [       3,        4,       16,        2,        2,      506,
                   4,        2,        1,        0]])
    return array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
            31140000, 31130000, 31120300],
           [       3,        4,       16,        2,        2,      506,
                   4,        2,       1]])
    '''
    
def check_positive_and_remove_zero(updated_state):
    if updated_state.shape == (0,): return np.array([])
    index_list = []
    try: updated_state.shape[1]
    except:breakpoint()
    for i in range(updated_state.shape[1]):
        if updated_state[1,i]>0: # todo check here if it should be negative # checking? here it should be negative
            index_list.append(i)
            '''e.g. if item == [31120200, -35] and item[1] == -35 < 0, 
            it means the order has been withdrawned, and we can directly
            remove it from the order book.
            '''
    return updated_state[:,index_list]
 

def keep_dimension(updated_state,size):
    def extend_dimension(updated_state, to_be_extended_size):
        # updated_state.extend([[Flag.min_price,0] for i in range(size - len(updated_state))])
        to_be_extended = np.array([[Flag.min_price,0] for i in range(to_be_extended_size)]).T
        if updated_state.size != 0: updated_state = np.hstack((updated_state, to_be_extended))
        else: updated_state = to_be_extended
        return updated_state
        # todo not sure
    if updated_state.shape == (0,):     updated_state = extend_dimension(updated_state, Flag.price_level)
    elif updated_state.shape[1] > size: updated_state = updated_state[:size]
    elif updated_state.shape[1] < size: updated_state = extend_dimension(updated_state, size - updated_state.shape[1])
    else: raise NotADirectoryError
    return updated_state
           

def list_to_gym_state(diff_list):
    container = np.zeros((len(diff_list[0]), len(diff_list)), dtype = np.int64)
    for i in range(container.shape[1]):
        container[:,i] = diff_list[i]
    return container

def check_if_sorted(obs):
    print("check_if_sorted called") #tbd
    # if not (obs == arg_sort(obs))[0].all(): return arg_sort(obs)
    # else: return obs
    
    assert (obs == arg_sort(obs))[0].all(), "price in obs is ont in the ascending order" 
    
    # only check whether the price is in ascending order
    # assert (obs == -np.sort(-obs)).all(), "price in obs is ont in the ascending order" 
    # todo check it 
    
def set_sorted(obs): return arg_sort(obs) # todo whether need to delete the check_if_sorted

def check_get_difference_and_update_0(skip, action, right_answer, my_answer): 
    ''''If there is a new order in the data, but the action does not produce 
    a new order instruction, then the new state should be consistent 
    with the next state in the data
    '''
    if action == 0 and skip == 1:
        right_answer = change_to_gym_state(right_answer) # it should be this one
        assert (my_answer == right_answer).all(), "error in the update in match_engine.core"
    else: pass
    
def check_get_difference_and_update_1(skip, action, index, my_answer):
    state1 = np.array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
                        31140000, 31130000, 31120300, 31120200],
                       [       2,        4,       16,        2,        2,      506,
                               4,        2,       35,       35]])
    state2 = np.array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
                        31140000, 31130000, 31120300, 31120200],
                       [       1,        4,       16,        2,        2,      506,
                               4,        2,       35,       35]])
    state3 = np.array([[31160000, 31152200, 31151000, 31150100, 31150000, 31140000,
                        31130000, 31120300, 31120200, 30000000],
                       [       4,       16,        2,        2,      506,        4,
                               2,       35,       35,        0]])
    state4 = np.array([[31160000, 31155000, 31152200, 31151000, 31150100, 31150000,
                        31140000, 31130000, 31120300, 30000000],
                       [       3,       28,       16,        2,        2,      506,
                               4,        2,       35,        0]])
    state_list = [state1, state2, state3, state4]
    if action == 1 and index<=3 and skip == 1:
        (state_list[index] == my_answer).all()
    else: pass

def arg_sort(x):
    return x[:,x[0, :].argsort()[::-1]]
                    
if __name__=="__main__":
    pairs = [[123,1],[133324,1],[132312,3]]##
    # series = observation[0]
    

    

    