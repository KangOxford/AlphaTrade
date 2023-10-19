import os
import sys
import time
import csv
import datetime
from os import listdir
from os.path import isfile,join

import jax
import flax
import numpy as np
import jax.numpy as jnp

sys.path.append('/homes/80/kang/AlphaTrade')
from gymnax_exchange.jaxrl.ppoS5ExecCont import ActorCriticS5
from gymnax_exchange.jaxrl.ppoS5ExecCont import StackedEncoderModel, ssm_size, n_layers
from purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from gymnax_exchange.jaxen.exec_env import *

ppo_config = {    
    "ENV_NAME": "alphatradeExec-v0",
    "DEBUG": True,
    "TASKSIDE":'sell',

    "REWARD_LAMBDA":0,
    "ACTION_TYPE":"pure",
    # "ACTION_TYPE":"delta",
    "TASK_SIZE":500,
    
    "WINDOW_INDEX": -1,
    "ATFOLDER": "/homes/80/kang/AlphaTrade/testing_oneDay",
    "RESULTS_FILE":"/homes/80/kang/AlphaTrade/results_file_"+f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
    
    
    "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-19_06-27/", # N.O. 23, pure quant
    "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-19_06-27/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-15_00-33/", # N.O. 11, pure quant
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-15_00-33/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-15_10-03/", # N.O. 11, pure quant
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-15_10-03/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-14_10-16/", # N.O. 10, pure quant
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-14_10-16/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-11_04-22/", # N.O. 3, pure quant
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-11_04-22/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/ckpt/",
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/ckpt/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-07_09-09/",
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-07_09-09/csv/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-06_12-57/",
    # "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-06_12-57/csv/",
}

dir = ppo_config['CHECKPOINT_DIR']
csv_dir = ppo_config["CHECKPOINT_CSV_DIR"]


def make_evaluation(network_config):    
    network,trainstate_params,checkpoint,env,env_params,key_step = network_config
    def step(rng, obs, done, hstate,state):
        
        ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
        assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
        assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
        hstate, pi, value = network.apply(trainstate_params, hstate, ac_in) 
        raw_action = pi.sample(seed=rng)
        # action = raw_action.round().astype(jnp.int32)[0,0,:].clip(0, None)
        action = raw_action[0,0,:]
        obs,state,reward,done,info=env.step(key_step, state,action, env_params)
        row_data = [
            checkpoint, info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],raw_action[0,0,0],raw_action[0,0,1],
            info['done'], info['slippage'], info['price_drift'], info['advantage_reward'], 
            info['drift_reward'], info['quant_executed'], 
            info['task_to_execute'], info['total_revenue']
        ]
        return row_data,(obs, done, hstate,state,reward,info)
    return step


    
    
def evaluate_savefile(paramsFile,window_idx):
    # env=ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx)
    env= ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx,ppo_config["ACTION_TYPE"],ppo_config["TASK_SIZE"],ppo_config["REWARD_LAMBDA"])
    env_params=env.default_params
    assert env.task_size == 500
    # Automatically create the directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    with open(csv_dir+paramsFile.split(".")[0]+f'_wdw_idx_{window_idx}.csv', 'w', newline='') as csvfile:
        print(paramsFile)
        csvwriter = csv.writer(csvfile)
        # Add a header row if needed
        row_title = [
            'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','raw_delta_aggressive','raw_delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','quant_executed', 'task_to_execute', 'total_revenue'
        ]
        csvwriter.writerow(row_title)
        csvfile.flush() 
    
        with open(dir+paramsFile, 'rb') as f:
            trainstate_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        rng = jax.random.PRNGKey(0)
        rng, key_reset, key_step = jax.random.split(rng, 3)
        obs,state=env.reset(key_reset,env_params)
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=ppo_config)
        hstate = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)
        done = False
        
        checkpoint = int(paramsFile.split("_")[1].split(".")[0])
        network_config = (network,trainstate_params,checkpoint,env,env_params,key_step)
        device = jax.devices()[-1]
        evaluate_jit = jax.jit(make_evaluation(network_config), device=device)    
        
        rng = jax.device_put(jax.random.PRNGKey(0), device)
        for i in range(1,10000):
            print(i)

            # start = time.time()
            rng, _rng = jax.random.split(rng)
            row_data,(obs, done, hstate,state,reward,info) = evaluate_jit(rng, obs,done,hstate,state)
            # print(f"time taken: {time.time()-start}")
            
            csvwriter.writerow(row_data)
            csvfile.flush() 
            if done:
                break
            
def twap_evaluation(paramsFile,window_idx):
    env=ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx)
    env_params=env.default_params
    assert env.task_size == 500
    # Automatically create the directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)    
    with open(csv_dir+paramsFile.split(".")[0]+f'_twap_{window_idx}.csv', 'w', newline='') as csvfile:
        print(paramsFile)
        csvwriter = csv.writer(csvfile)
        # Add a header row if needed
        row_title = [
            'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','raw_delta_aggressive','raw_delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','quant_executed', 'task_to_execute', 'total_revenue'
        ]
        csvwriter.writerow(row_title)
        csvfile.flush() 
    
        rng = jax.random.PRNGKey(0)
        rng, key_reset, key_step = jax.random.split(rng, 3)
        obs,state=env.reset(key_reset,env_params)
        done = False
        for i in range(1,10000):
            print(i)
            raw_action = jnp.array([0, 0])
            # action = raw_action.round().astype(jnp.int32)[0,0,:].clip(0, None)
            action = raw_action
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            row_data = [
                paramsFile.split("_")[1].split(".")[0], info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],raw_action[0],raw_action[1],
                info['done'], info['slippage'], info['price_drift'], info['advantage_reward'], 
                info['drift_reward'], info['quant_executed'], 
                info['task_to_execute'], info['total_revenue']
            ]
            csvwriter.writerow(row_data)
            csvfile.flush() 
            if done:
                break
    
import re
import argparse

def main(idx=-1):
    start_time =time.time()
    for window_idx in range(13):
        start=time.time()
        print(f">>> window_idx: {window_idx}")
        def extract_number_from_filename(filename):
            match = re.search(r'_(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0  # default if no number is found

        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
        onlyfiles = sorted(onlyfiles, key=extract_number_from_filename)
        paramsFile = onlyfiles[idx]
        evaluate_savefile(paramsFile,window_idx)
        print(f"Time for evaluation: \n",time.time()-start)
    print(f"Total time for evaluation ppo : \n",time.time()-start_time)
    
def main2(idx=-1):
    start_time =time.time()
    for window_idx in range(13):
        start=time.time()
        print(f">>> window_idx: {window_idx}")        
        def extract_number_from_filename(filename):
            match = re.search(r'_(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0  # default if no number is found

        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
        onlyfiles = sorted(onlyfiles, key=extract_number_from_filename)
        paramsFile = onlyfiles[idx]
        twap_evaluation(paramsFile,window_idx)
        print(f"Time for evaluation: \n",time.time()-start)
    print(f"Total time for evaluation twap: \n",time.time()-start_time)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify index of the file to evaluate.')
    parser.add_argument('--idx', metavar='idx', type=int, default=-1, help='Index of the file to evaluate.')
    
    args = parser.parse_args()
    main(args.idx)
    # main2(args.idx)
    # /bin/python3 /homes/80/kang/AlphaTrade/gymnax_exchange/test_scripts/evaluation.py 2
    # /bin/python3 /homes/80/kang/AlphaTrade/gymnax_exchange/test_scripts/evaluation.py -1
