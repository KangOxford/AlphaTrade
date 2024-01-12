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
from joblib import Parallel, delayed
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

# def TWAP_Scheduling(state, env, key):
#     allocation_array_final_lst = []
#     for idx in range(env.taskSize_array.shape[0]): # TODO not sure
#         print(f"TWAP_Scheduling idx {idx}")
#         allocation_array_final = hamilton_apportionment_permuted_jax(
#                                  jnp.ones(env.max_steps_in_episode_arr[idx]), 
#                                  env.taskSize_array[idx], key)
#         allocation_array_final_lst.append(allocation_array_final)
#     return allocation_array_final_lst




def VWAP_Scheduling(state, env, forcasted_volume_, key):
    allocation_array_final_lst = []
    print(f"VWAP_Scheduling")
    for idx in tqdm(range(forcasted_volume_.shape[0])):
        forcasted_volume = forcasted_volume_.iloc[idx,:].to_numpy()
        start_idx_array = env.start_idx_array_list[idx]
        forcasted_volume = hamilton_apportionment_permuted_jax(forcasted_volume, env.taskSize_array[idx], key)
        assert forcasted_volume.sum() == env.taskSize_array[idx], f"Error code V10"
        
        allocation_array_full = jnp.concatenate([start_idx_array,forcasted_volume.reshape(-1,1)],axis=1)
        allocation_array_breif = allocation_array_full[:,[0,-1]].astype(jnp.int32)
        allocation_array_breif = jnp.concatenate([allocation_array_breif, np.insert(np.diff(allocation_array_breif[:, 0]),0,allocation_array_breif[0, 0]).reshape(-1,1)],axis=1)
        
        lst = []
        key = jax.random.PRNGKey(100)
        for i in range(allocation_array_breif.shape[0]):
            key, subkey = jax.random.split(key)
            lst.append(hamilton_apportionment_permuted_jax(jnp.ones(allocation_array_breif[i,2]), allocation_array_breif[i,1], key))
        allocation_array_final = np.concatenate(lst)
        allocation_array_final_lst.append(allocation_array_final)
    # result = np.array(allocation_array_final_lst)
    # breakpoint()
    return allocation_array_final_lst

def ORACLE_Scheduling(state, env, oracles, key):
    return VWAP_Scheduling(state, env, oracles, key)

def TWAP_Scheduling(state, env, twaps, key):
    return VWAP_Scheduling(state, env, twaps, key)

def RM_Scheduling(state, env, rms, key):
    """slice the order by rolling mean of the past one week

    Args:
        env (_type_): _description_
        forcasted_volume_: forecasted trading volume by rolling mean 
                           with window length to be 5 days.

    Returns:
        _type_: _description_
    """
    return VWAP_Scheduling(state, env, rms, key)

def data_alignment(ATFolder):
    '''
    The logic here is to load the three dir: messages data, rolling mean and vwap
    Then find the common_dates of all these three,
    Then pass into the init of the base env, 
    To only load the data window in the common dates.
    But we should also align the symbols, it has been done, as we will pass symbols 
    into the load_forecasted_and_original_volume_VWAP and load_forecasted_volume_RM
    '''
    def load_forecasted_and_original_volume_VWAP(symbol):
        import pandas as pd
        import numpy as np
        dir = '/homes/80/kang/cmem/output/0900_r_output_with_features_csv_fractional_shares_clipped_vwap/'
        df = pd.read_csv(dir+f'{symbol}.csv',index_col=0)
        df['symbol'] = symbol
        from datetime import datetime, timedelta
        timeHMs = np.array([int((datetime(2023, 1, 1, 9, 30) + i * timedelta(minutes=15)).strftime('%H%M')) for i in range(26)])
        timeHMs = np.tile(timeHMs, (1, int(df.shape[0]/timeHMs.shape[0]))).squeeze()
        df['timeHMs'] = timeHMs
        df = df[['date', 'timeHMs', 'x', 'qty', 'symbol']]
        d = df[['date', 'timeHMs', 'x']].pivot(index = 'date', columns = 'timeHMs')
        f = df[['date', 'timeHMs', 'qty']].pivot(index = 'date', columns = 'timeHMs')
        return d, f # x, qty

    def load_forecasted_volume_RM(symbol):
        import pandas as pd
        import numpy as np
        dir = '/homes/80/kang/cmem/data/01_raw_rolling_mean_15min_bin/'
        df = pd.read_csv(dir+f'{symbol}.csv',index_col=0)
        df.columns = ['date', 'timeHMs', 'timeHMe', 'x_rm', 'symbol']
        df.date = df.date.apply(lambda d: str(d)[:4]+'-'+str(d)[4:6]+'-'+str(d)[6:8]) if len(str(df.date[0])) == 8 else df.date
        d = df[['date', 'timeHMs', 'x_rm']].pivot(index = 'date', columns = 'timeHMs')
        d = d.fillna(method = "ffill")
        return d # x

    def load_files(symbol):
        alphatradePath = ATFolder
        # alphatradePath = f"{ATFolder}/{symbol}_data"
        messagePath = alphatradePath+"/Flow_10/"
        orderbookPath = alphatradePath+"/Book_10/"
        from os import listdir; from os.path import isfile, join; import pandas as pd
        readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
        messageFiles, orderbookFiles = readFromPath(messagePath), readFromPath(orderbookPath)
        message_dates = np.array([m[4:14] for m in messageFiles])
        message_dates = np.array([m.split("_")[1] for m in messageFiles])
        return message_dates
    
    def get_symbols():
        from os.path import isdir
        from os import listdir
        path_dir1 = '/homes/80/kang/cmem/output/0900_r_output_with_features_csv_fractional_shares_clipped_vwap/'
        path_dir2 = ATFolder
        symbols1 = np.array([dirname.split(".")[0] for dirname in listdir(path_dir1)])
        symbols2 = np.array([path_dir2.split("_")[0].split("/")[-1]])
        # symbols2 = np.array([dirname.split("_")[0] for dirname in listdir(path_dir2) 
        #             if isdir(os.path.join(path_dir2, dirname))])
        intersection = np.intersect1d(symbols1, symbols2)
        return intersection
    common_stocks = get_symbols()
    assert len(common_stocks) >= 1
    
    def get_raw_VWAPs_ORACLEs_RMs():
        ds = [(symbol, *load_forecasted_and_original_volume_VWAP(symbol)) for symbol in common_stocks]
        VWAPs = [(d[0], d[1]) for d in ds]
        ORACLEs = [(d[0], d[2]) for d in ds]
        RMs = [(symbol, load_forecasted_volume_RM(symbol)) for symbol in common_stocks]
        return VWAPs, ORACLEs, RMs
    VWAPs, ORACLEs, RMs = get_raw_VWAPs_ORACLEs_RMs()
    
    def get_common_dates():
        dates_vwap =  VWAPs[-1][-1].index.to_numpy()
        dates_rm =  RMs[-1][-1].index.to_numpy()
        message_dates = load_files(common_stocks[-1]) 
        common_dates = np.intersect1d(np.intersect1d(dates_vwap, message_dates),dates_rm)
        return common_dates
    common_dates = get_common_dates()
    
    def align_dates(Xs):
        Xs = {d[0]: d[1].loc[common_dates] for d in Xs}
        return Xs
    VWAPs, ORACLEs, RMs = list(map(align_dates, [VWAPs, ORACLEs, RMs]))

    def generate_TWAPs(RMs):
        def f(df):
            dff = df.copy()
            dff.loc[:, :] = 1
            return dff
        return {symbol: f(df) for symbol, df in RMs.items()}
    TWAPs = generate_TWAPs(RMs)
            
    return common_dates, common_stocks, VWAPs, ORACLEs, RMs, TWAPs


def main1(symbol):
# if __name__ == "__main__":
        
    ATFolder = f"/homes/80/kang/SP500/{symbol}_data"
    # ATFolder = f"/homes/80/kang/SP500/ABC_data"
    # ATFolder = f"/homes/80/kang/SP500/ACN_data"
    # ATFolder = f"/homes/80/kang/SP500/AAP_data"
    common_dates, common_stocks, VWAPs, ORACLEs, RMs, TWAPs = data_alignment(ATFolder)

    # pre setup the env
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell",
        # "TASK_SIZE": 0.05, #1000, # 8000, #100, # 500,
        "TASK_SIZE": 0.01, #1000, # 8000, #100, # 500,
        "WINDOW_INDEX": -1,
        "REWARD_LAMBDA": 1.0,
        "DATES":common_dates,
        }
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],
                      config["WINDOW_INDEX"],config["DATES"],
                      config["TASK_SIZE"],config["REWARD_LAMBDA"],
                      )
    env_params=env.default_params
    obs,state=env.reset(key_reset,env_params)
    print("reset_finished")
    

    for symbol in common_stocks:
        print("+++ symbol : ", symbol)
        allocation_array_final_rm = RM_Scheduling(state, env, RMs[symbol], key_reset)
        print("allocation_array_final_rm finished")
        print(f"Size of the allocation_array_final_rm in MB: {sys.getsizeof(allocation_array_final_rm) / (1024 ** 2)}")
        allocation_array_final_oracle = ORACLE_Scheduling(state, env, ORACLEs[symbol], key_reset)
        print("allocation_array_final_oracle finished")
        print(f"Size of the allocation_array_final_oracle in MB: {sys.getsizeof(allocation_array_final_oracle) / (1024 ** 2)}")
        allocation_array_final_vwap = VWAP_Scheduling(state, env, VWAPs[symbol], key_reset)
        print("allocation_array_final_vwap finished")
        print(f"Size of the allocation_array_final_vwap in MB: {sys.getsizeof(allocation_array_final_vwap) / (1024 ** 2)}")
        allocation_array_final_twap = TWAP_Scheduling(state, env, TWAPs[symbol], key_reset)
        print("allocation_array_final_twap finished")
        print(f"Size of the allocation_array_final_twap in MB: {sys.getsizeof(allocation_array_final_twap) / (1024 ** 2)}")
        
        assert len(allocation_array_final_oracle) == len(allocation_array_final_rm)
        print("allocation_array_final_oracle, num of arrays: ", len(allocation_array_final_oracle))
    
        vwap_info_lst = []
        rm_info_lst = []
        oracle_info_lst = []
        twap_info_lst = []
        # for reset_window_index in tqdm(range(2)):
        for reset_window_index in tqdm(range(len(allocation_array_final_oracle))):
            # print(f"+++ reset_window_index idx {reset_window_index}")
            def get_final_info(strategy_type, reset_window_index, rng):
                rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
                obs,state=env.reset_env(key_reset,env_params,reset_window_index)
                for i in range(1,100000):
                    # ==================== ACTION ====================
                    # ---------- acion from random sampling ----------
                    # print("-"*20)
                    key_policy, _ =  jax.random.split(key_policy, 2)
                    key_step, _ =  jax.random.split(key_step, 2)
                    # print("window_index: ",state.window_index)
                    if  strategy_type == "vwap":
                        test_action=allocation_array_final_vwap[state.window_index][state.step_counter-1] 
                    elif strategy_type == "rm":
                        test_action=allocation_array_final_rm[state.window_index][state.step_counter-1]
                    elif strategy_type == "oracle":
                        test_action=allocation_array_final_oracle[state.window_index][state.step_counter-1]
                    elif strategy_type == "twap":
                        test_action=allocation_array_final_twap[state.window_index][state.step_counter-1]
                    else: raise NotImplementedError
                    # print(state.task_to_execute)
                    # print(f"Sampled {i}th actions are: ",test_action)
                    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
                    # print("state.task_to_execute",state.task_to_execute)
                    # for key, value in info.items():
                    #     print(key, value)
                    if done:
                        print("==="*20)
                        break
                return info
            
            
            vwap_info = get_final_info("vwap", reset_window_index, rng= jax.random.PRNGKey(0))
            vwap_info_lst.append((vwap_info['window_index'],vwap_info['average_price']))
            print(">>> vwap_info:",vwap_info)
            rm_info = get_final_info("rm", reset_window_index, rng= jax.random.PRNGKey(0))
            rm_info_lst.append((rm_info['window_index'],rm_info['average_price']))
            print(">>> rm_info:",rm_info)
            oracle_info = get_final_info("oracle", reset_window_index, rng= jax.random.PRNGKey(0))
            oracle_info_lst.append((oracle_info['window_index'],oracle_info['average_price']))
            print(">>> oracle_info:",oracle_info)
            twap_info = get_final_info("twap", reset_window_index, rng= jax.random.PRNGKey(0))
            twap_info_lst.append((twap_info['window_index'],twap_info['average_price']))
            print(">>> twap_info:",twap_info)

            

            # def process_info(method, reset_window_index, rng_key_num):
            #     info = get_final_info(method, reset_window_index, rng=jax.random.PRNGKey(rng_key_num))
            #     return (info['window_index'], info['average_price'])
            # results = Parallel(n_jobs=3)(delayed(process_info)(method, reset_window_index, rng_key_num = 0) for method in ["vwap", "rm", "oracle"])
            # vwap_info_lst, rm_info_lst, oracle_info_lst = results
            
        r = pd.DataFrame(jnp.array(rm_info_lst), columns = ['window_index', 'average_price_RM'])
        v = pd.DataFrame(jnp.array(vwap_info_lst), columns = ['window_index', 'average_price_VWAP'])
        o = pd.DataFrame(jnp.array(oracle_info_lst), columns = ['window_index', 'average_price_ORACLE'])
        t = pd.DataFrame(jnp.array(twap_info_lst), columns = ['window_index', 'average_price_TWAP'])
        merged_df = pd.merge(r, v, on='window_index', how='inner')
        merged_df = pd.merge(merged_df, t, on='window_index', how='inner')
        df = pd.merge(merged_df, o, on='window_index', how='inner')
        timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
        df.to_csv(f"VWAP_Scheduling_{symbol}_{timestamp}.csv")
        

        
        # # tv=jnp.concatenate([t,v[:,1].reshape(-1,1)],axis=1)
        # rv=jnp.concatenate([r,v[:,1].reshape(-1,1)],axis=1)
        # df = pd.DataFrame(rv,columns = ['t_idx','rolling_mean','vwap'])
        # # df = pd.DataFrame(tv,columns = ['t_idx','twap','vwap'])
        # df['advantage_VoverR']=(df.vwap-df.rolling_mean)/df.rolling_mean*10000
        # # df['advantage_VoverT']=(df.vwap-df.twap)/df.twap*10000
        # print("summary: \n",df)
        # print("summary: \n",df.mean())
        # timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
        # df.to_csv(f"VWAP_Scheduling_{symbol}_{timestamp}.csv")
        
    

if __name__ == "__main__":
    # symbols = ["ABC"]
    # symbols = ["ACN"]
    # symbols = ["AAP"]
    symbols = ["AAL", "ABT", "ADBE"  ,"ABBV" , "ABMD"]
    for symbol in symbols:
        try:
            main1(symbol)
        except:
            print(f"*** No {symbol} ***")