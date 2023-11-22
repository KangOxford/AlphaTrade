# from jax import config
# config.update("jax_enable_x64",True)

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

import chex
import flax
import flax.linen as nn
import gymnax
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Any, Dict, NamedTuple, Sequence
import distrax
from gymnax.environments import spaces

sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
#Code snippet to disable all jitting.
from jax import config

from gymnax_exchange.jaxen.exec_env import ExecutionEnv

config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.
np.set_printoptions(suppress=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  

@jax.jit
def hamilton_apportionment_permuted_jax(votes, seats, key):
    init_seats, remainders = jnp.divmod(votes, jnp.sum(votes) / seats) # std_divisor = jnp.sum(votes) / seats
    remaining_seats = jnp.array(seats - init_seats.sum(), dtype=jnp.int32) # in {0,1,2,3}
    def f(carry,x):
        key,init_seats,remainders=carry
        key, subkey = jax.random.split(key)
        chosen_index = jax.random.choice(subkey, remainders.size, p=(remainders == remainders.max())/(remainders == remainders.max()).sum())
        return (key,init_seats.at[chosen_index].add(jnp.where(x < remaining_seats,1,0)),remainders.at[chosen_index].set(0)),x
    (key,init_seats,remainders), x = jax.lax.scan(f,(key,init_seats,remainders),xs=jnp.arange(votes.shape[0]))
    return init_seats.astype(jnp.int32)

def TWAP_Scheduling(state, env, key):
    allocation_array_final_lst = []
    for idx in range(env.taskSize_array.shape[0]): # TODO not sure
        print(f"TWAP_Scheduling idx {idx}")
        allocation_array_final = hamilton_apportionment_permuted_jax(
                                 jnp.ones(env.max_steps_in_episode_arr[idx]), 
                                 env.taskSize_array[idx], key)
        allocation_array_final_lst.append(allocation_array_final)
    return allocation_array_final_lst

def RM_Scheduling(state, env, forcasted_volume_, key):
    """slice the order by rolling mean of the past one week

    Args:
        env (_type_): _description_
        forcasted_volume_: forecasted trading volume by rolling mean 
                           with window length to be 21 days.

    Returns:
        _type_: _description_
    """
    allocation_array_final_lst = []
    for idx in range(forcasted_volume_.shape[0]):
        forcasted_volume = forcasted_volume_[idx]
        print(f"RM_Scheduling idx {idx}")
        start_idx_array = env.start_idx_array_list[idx]
        forcasted_volume = hamilton_apportionment_permuted_jax(forcasted_volume, env.taskSize_array[idx], key)
        assert forcasted_volume.sum() == env.taskSize_array[idx], f"Error code RM10"
        
        allocation_array_full = jnp.concatenate([start_idx_array,forcasted_volume.reshape(-1,1)],axis=1)
        allocation_array_breif = allocation_array_full[:,[0,-1]].astype(jnp.int32)
        allocation_array_breif = jnp.concatenate([allocation_array_breif, np.insert(np.diff(allocation_array_breif[:, 0]),0,allocation_array_breif[0, 0]).reshape(-1,1)],axis=1)
        
        lst = []
        key = jax.random.PRNGKey(100)
        for i in range(allocation_array_breif.shape[0]):
            print(i)
            key, subkey = jax.random.split(key)
            lst.append(hamilton_apportionment_permuted_jax(jnp.ones(allocation_array_breif[i,2]), allocation_array_breif[i,1], key))
        allocation_array_final = np.concatenate(lst)
        allocation_array_final_lst.append(allocation_array_final)
    return allocation_array_final_lst

def VWAP_Scheduling(state, env, forcasted_volume_, key):
    allocation_array_final_lst = []
    for idx in range(forcasted_volume_.shape[0]):
        forcasted_volume = forcasted_volume_[idx,:]
        print(f"VWAP_Scheduling idx {idx}")
        start_idx_array = env.start_idx_array_list[idx]
        forcasted_volume = hamilton_apportionment_permuted_jax(forcasted_volume, env.taskSize_array[idx], key)
        assert forcasted_volume.sum() == env.taskSize_array[idx], f"Error code V10"
        
        allocation_array_full = jnp.concatenate([start_idx_array,forcasted_volume.reshape(-1,1)],axis=1)
        allocation_array_breif = allocation_array_full[:,[0,-1]].astype(jnp.int32)
        allocation_array_breif = jnp.concatenate([allocation_array_breif, np.insert(np.diff(allocation_array_breif[:, 0]),0,allocation_array_breif[0, 0]).reshape(-1,1)],axis=1)
        
        lst = []
        key = jax.random.PRNGKey(100)
        for i in range(allocation_array_breif.shape[0]):
            print(i)
            key, subkey = jax.random.split(key)
            lst.append(hamilton_apportionment_permuted_jax(jnp.ones(allocation_array_breif[i,2]), allocation_array_breif[i,1], key))
        allocation_array_final = np.concatenate(lst)
        allocation_array_final_lst.append(allocation_array_final)
    # result = np.array(allocation_array_final_lst)
    # breakpoint()
    return allocation_array_final_lst

def load_forecasted_and_original_volume_vwap():
    import pandas as pd
    import numpy as np
    dir = '/homes/80/kang/cmem/output/0900_r_output_with_features_csv_fractional_shares_clipped_vwap/'
    name = 'AAP.csv'
    df = pd.read_csv(dir+name,index_col=0)
    g = df.groupby('date')
    lst0,lst1,lst2=[],[],[]
    for idx,itm in g:
        lst0.append(idx)
        lst1.append(itm.x.to_numpy())
        lst2.append(itm.qty.to_numpy())
    dates = np.array(lst0)
    x = np.array(lst1)
    qty = np.array(lst2)
    return dates, x, qty

def load_forecasted_and_original_volume_rolling_mean():
    import pandas as pd
    import numpy as np
    dir = '/homes/80/kang/cmem/data/01_raw_rolling_mean_15min_bin/'
    name = 'AAP.csv'
    df = pd.read_csv(dir+name,index_col=0)
    g = df.groupby('date')
    lst0,lst1=[],[]
    for idx,itm in g:
        lst0.append(idx)
        lst1.append(itm.qty.to_numpy()) # Actually it is rolling mean, it is a forecasted value
    dates = np.array(lst0)
    dates = np.array([f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}" for d in dates])
    x = np.array(lst1)
    # x = np.array(lst1,dtype=np.float32)
    qty = None
    return dates, x, qty

def load_files(alphatradePath):
    messagePath = alphatradePath+"/data/Flow_10/"
    orderbookPath = alphatradePath+"/data/Book_10/"
    from os import listdir; from os.path import isfile, join; import pandas as pd
    readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    messageFiles, orderbookFiles = readFromPath(messagePath), readFromPath(orderbookPath)
    message_dates = np.array([m[4:14] for m in messageFiles])
    return message_dates


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        ATFolder = "/homes/80/kang/aap2017half"
        # ATFolder = "/homes/80/kang/aap2017_07_20"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
        
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell",
        "TASK_SIZE": 1000, # 8000, #100, # 500,
        "WINDOW_INDEX": -1,
        "ACTION_TYPE": "delta", # "pure",
        "REWARD_LAMBDA": 1.0,
        # "FORECASTED_VOLUME": qty,
        # "FORECASTED_VOLUME": load_forecasted_and_original_volume('2017-07-20').qty.to_numpy(),
        # "FORECASTED_VOLUME": jax.random.permutation(jax.random.PRNGKey(0),jnp.arange(1,27)),
        }

    def data_alignment():
        message_dates = load_files(config['ATFOLDER']) 
        dates, x, qty = load_forecasted_and_original_volume_vwap()  
        # difference = np.setdiff1d(message_dates, dates)
        # assert difference == None, f"load data error Wrong Code: D10"
        # TODO check the dates and the dates from the base_env 
        dates_rm, x_rm, qty_rm = load_forecasted_and_original_volume_rolling_mean()
        # difference0 = np.setdiff1d(message_dates, dates_rm) 
        difference1 = np.setdiff1d(dates_rm, message_dates) 
        index1 = np.array([np.where(dates_rm ==  difference1[i])[0] 
                        for i in range(len(difference1))]).reshape(-1)
        remaining_dates1 = np.delete(dates_rm, index1)
        remaining_x_rm1 = np.delete(x_rm, index1)
        x_rm = remaining_x_rm1
        return x, qty, x_rm
    x, qty, x_rm = data_alignment()
    


    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],
                      config["WINDOW_INDEX"],config["ACTION_TYPE"],
                      config["TASK_SIZE"],config["REWARD_LAMBDA"])
    env_params=env.default_params
    obs,state=env.reset(key_reset,env_params)
    allocation_array_final_rm = RM_Scheduling(state, env, x_rm, key_reset)
    allocation_array_final_vwap = VWAP_Scheduling(state, env, qty, key_reset)
    # allocation_array_final_twap = TWAP_Scheduling(state, env, key_reset)
    # assert len(allocation_array_final_vwap) == len(allocation_array_final_twap)
    assert len(allocation_array_final_vwap) == len(allocation_array_final_rm)
    print("allocation_array_final_vwap, num of arrays: ", len(allocation_array_final_vwap))

    vwap_info_lst = []
    # twap_info_lst = []
    rm_info_lst = []
    for reset_window_index in tqdm(range(len(allocation_array_final_vwap))):
        print(f"+++ reset_window_index idx {reset_window_index}")
        def get_final_info(strategy_type, reset_window_index, rng):
            rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
            obs,state=env.reset_env(key_reset,env_params,reset_window_index)
            for i in range(1,100000):
                # ==================== ACTION ====================
                # ---------- acion from random sampling ----------
                print("-"*20)
                key_policy, _ =  jax.random.split(key_policy, 2)
                key_step, _ =  jax.random.split(key_step, 2)
                print("window_index: ",state.window_index)
                if   strategy_type == "vwap":
                    test_action=allocation_array_final_vwap[state.window_index][state.step_counter-1] 
                elif strategy_type == "rm":
                    test_action=allocation_array_final_rm[state.window_index][state.step_counter-1]
                # elif strategy_type == "twap":
                #     test_action=allocation_array_final_twap[state.window_index][state.step_counter-1]
                print(state.task_to_execute)
                print(f"Sampled {i}th actions are: ",test_action)
                obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
                print("state.task_to_execute",state.task_to_execute)
                for key, value in info.items():
                    print(key, value)
                if done:
                    print("==="*20)
                    break
            return info
        vwap_info = get_final_info("vwap", reset_window_index, rng= jax.random.PRNGKey(0))
        vwap_info_lst.append((vwap_info['window_index'],vwap_info['average_price']))
        rm_info = get_final_info("rm", reset_window_index, rng= jax.random.PRNGKey(0))
        rm_info_lst.append((rm_info['window_index'],rm_info['average_price']))
        print(">>> vwap_info:",vwap_info)
        print(">>> rm_info:",rm_info)
        # twap_info = get_final_info("twap", reset_window_index, rng= jax.random.PRNGKey(0))
        # twap_info_lst.append((twap_info['window_index'],twap_info['average_price']))
    r = jnp.array(rm_info_lst)
    # t = jnp.array(twap_info_lst)
    v = jnp.array(vwap_info_lst)
    # tv=jnp.concatenate([t,v[:,1].reshape(-1,1)],axis=1)
    rv=jnp.concatenate([r,v[:,1].reshape(-1,1)],axis=1)
    df = pd.DataFrame(rv,columns = ['t_idx','rolling_mean','vwap'])
    # df = pd.DataFrame(tv,columns = ['t_idx','twap','vwap'])
    df['advantage_VoverR']=(df.vwap-df.rolling_mean)/df.rolling_mean*10000
    # df['advantage_VoverT']=(df.vwap-df.twap)/df.twap*10000
    print("summary: \n",df)
    print("summary: \n",df.mean())
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    df.to_csv(f"VWAP_Scheduling_{timestamp}.csv")
