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
    "TASKSIDE":'sell',
    # "LAMBDA":0.1,
    # "GAMMA":10.0,
    "LAMBDA":0.0,
    "GAMMA":0.0,
    "TASK_SIZE":500,
    "RESULTS_FILE":"/homes/80/kang/AlphaTrade/results_file_"+f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
    "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-06_12-57/",
    "CHECKPOINT_CSV_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-06_12-57/csv/",
}

env=ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"])
env_params=env.default_params
print(env_params.message_data.shape, env_params.book_data.shape)
assert env.task_size == 500
import time; timestamp = str(int(time.time()))

dir = ppo_config['CHECKPOINT_DIR']
csv_dir = ppo_config["CHECKPOINT_CSV_DIR"]
# Automatically create the directory if it doesn't exist
os.makedirs(csv_dir, exist_ok=True)


    
def evaluate_savefile(paramsFile):
    with open(csv_dir+paramsFile.split(".")[0]+'.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Add a header row if needed
        row_title = [
            'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','step_reward','quant_executed', 'task_to_execute', 'total_revenue'
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
        # done = jnp.array([False]*1)
        # ac_in = (obs[np.newaxis, np.newaxis, :], init_done[np.newaxis, :])
        # assert len(ac_in[0].shape) == 3
        # hstate, pi, value = network.apply(trainstate_params, init_hstate, ac_in)
        
        # action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip(0, None) # CAUTION about the [0,0,:], only works for num_env=1
        # obs,state,reward,done,info=env.step(key_step, state, action, env_params)
        # # excuted_list = []
        
        for i in range(1,10000):
            print(i)
            # ==================== ACTION ====================
            # ---------- acion from trained network ----------
            ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
            assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
            assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
            hstate, pi, value = network.apply(trainstate_params, hstate, ac_in) 
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0,0,:].clip( 0, None)
            # ---------- acion from trained network ----------
            # ==================== ACTION ====================    
            # print(f"-------------\nPPO {i}th delta are: {action} with sum {action.sum()}")
            # start=time.time()
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            # print(f"Time for {i} step: \n",time.time()-start)
            # print("{" + ", ".join([f"'{k}': {v}" for k, v in info.items()]) + "}")
            # excuted_list.append(info["quant_executed"])
            # # Write the data
            row_data = [
                paramsFile.split("_")[1].split(".")[0], info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],
                info['done'], info['slippage'], info['price_drift'], info['advantage_reward'], 
                info['drift_reward'], info['step_reward'], info['quant_executed'], 
                info['task_to_execute'], info['total_revenue']
            ]
            csvwriter.writerow(row_data)
            csvfile.flush() 
            if done:
                break
            
def main():
    idx = 0
    while True:
        onlyfiles = sorted([f for f in listdir(dir) if isfile(join(dir, f))])
        paramsFile = onlyfiles[idx]
        evaluate_savefile(paramsFile)
        idx += 1
        
if __name__=="__main__":
    main()