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

# idx = 0
# while True:
    # onlyfiles = sorted([f for f in listdir(dir) if isfile(join(dir, f))])
    # paramsFile = onlyfiles[idx]
    


# Main function to evaluate and save file
def evaluate_savefile(paramsFile, dir, env_params, ppo_config, csv_dir, env, ssm_size, n_layers):
    # The computation part
    print(f"Task started for file: {paramsFile}")
    def compute_rows(paramsFile, dir, env_params, ppo_config, env, ssm_size, n_layers):
        rows = []
        with open(dir + paramsFile, 'rb') as f:
            trainstate_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print("params restored")
        
        rng = jax.random.PRNGKey(0)
        rng, key_reset, key_step = jax.random.split(rng, 3)

        obs, state = env.reset(key_reset, env_params)
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=ppo_config)
        hstate = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)

        done = False
        for i in range(1, 10000):
            jax.debug.print(">>> {}",i)
            ac_in = (obs[np.newaxis, np.newaxis, :], jnp.array([done])[np.newaxis, :])
            hstate, pi, value = network.apply(trainstate_params, hstate, ac_in)
            action = pi.sample(seed=rng).round().astype(jnp.int32)[0, 0, :].clip(0, None)
            obs, state, reward, done, info = env.step(key_step, state, action, env_params)

            row_data = [
                paramsFile.split("_")[1].split(".")[0], info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],
                reward, info['done'], info['slippage'], info['price_drift'], info['advantage_reward'], 
                info['drift_reward'], info['step_reward'], info['quant_executed'], 
                info['task_to_execute'], info['total_revenue']
            ]
            rows.append(row_data)

            if done:
                break
        return rows

    # The file writing part
    def write_to_csv(rows, csv_dir, paramsFile):
        with open(csv_dir + paramsFile.split(".")[0] + '.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            row_title = [
                'checkpoint_name', 'window_index', 'current_step', 'average_price', 
                'delta_sum', 'delta_aggressive', 'delta_passive', 'reward', 'done', 
                'slippage', 'price_drift', 'advantage_reward', 'drift_reward',
                'step_reward', 'quant_executed', 'task_to_execute', 'total_revenue'
            ]
            csvwriter.writerow(row_title)
            for row in rows:
                csvwriter.writerow(row)
    rows = compute_rows(paramsFile, dir, env_params, ppo_config, env, ssm_size, n_layers)
    write_to_csv(rows, csv_dir, paramsFile)

# Main call
# evaluate_savefile(paramsFile, dir, env_params, ppo_config, csv_dir, env, ssm_size, n_layers)

# idx += 1
import concurrent
from concurrent.futures import ProcessPoolExecutor
# idx = 0
onlyfiles = sorted([f for f in listdir(dir) if isfile(join(dir, f))])
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(evaluate_savefile, onlyfiles[i], dir, env_params, ppo_config, csv_dir, env, ssm_size, n_layers) for i in range(len(onlyfiles))]
    for future in concurrent.futures.as_completed(futures):
        print(f"Completed {future.result()}")