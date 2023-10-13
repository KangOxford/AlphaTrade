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
    "LR": 2.5e-4,
    "ENT_COEF": 0.1,
    "NUM_ENVS": 200,
    "TOTAL_TIMESTEPS": 5e7,
    "NUM_MINIBATCHES": 2,
    "UPDATE_EPOCHS": 5,
    "NUM_STEPS": 455,
    "CLIP_EPS": 0.2,
    
    # "LR": 2.5e-6,
    # "NUM_ENVS": 1,
    # "NUM_STEPS": 1,
    # "NUM_MINIBATCHES": 1,
    # "NUM_ENVS": 1000,
    # "NUM_STEPS": 10,
    # "NUM_MINIBATCHES": 4,
    # "TOTAL_TIMESTEPS": 1e7,
    # "UPDATE_EPOCHS": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    # "CLIP_EPS": 0.2,
    # "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 2.0,
    "ANNEAL_LR": True,
    "NORMALIZE_ENV": True,
    
    "ENV_NAME": "alphatradeExec-v0",
    "ENV_LENGTH": "oneWindow",
    # "ENV_LENGTH": "allWindows",
    "DEBUG": True,
    "ATFOLDER": "/homes/80/kang/AlphaTrade/training_oneDay",
    # "ATFOLDER": "/homes/80/kang/AlphaTrade/testing_oneDay",
    "TASKSIDE":'sell',
    # "LAMBDA":0.1,
    # "GAMMA":10.0,
    "LAMBDA":0.0,
    "GAMMA":0.0,
    "TASK_SIZE":500,
    "RESULTS_FILE":"/homes/80/kang/AlphaTrade/results_file_"+f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
    "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-07_09-09/",
    "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-07_09-09/csv/",
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
        rng, _rng = jax.random.split(rng)
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
        return row_data
    return step


    
    
def evaluate_savefile(paramsFile,window_idx):
    env=ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx)
    env_params=env.default_params
    assert env.task_size == 500
    # Automatically create the directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    with open(csv_dir+paramsFile.split(".")[0]+f'_wdw_idx_{window_idx}.csv', 'w', newline='') as csvfile:
        print(paramsFile)
        csvwriter = csv.writer(csvfile)
        # Add a header row if needed
        row_title = [
            'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','raw_delta_aggressive','raw_delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','step_reward','quant_executed', 'task_to_execute', 'total_revenue'
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
        
        for i in range(1,10000):
            print(i)

            rng = jax.device_put(jax.random.PRNGKey(0), device)
            # start = time.time()
            row_data = evaluate_jit(rng,obs,done,hstate,state)
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
            'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','raw_delta_aggressive','raw_delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','step_reward','quant_executed', 'task_to_execute', 'total_revenue'
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
