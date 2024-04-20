import os
import sys
import time
import csv
import datetime
import dataclasses
import jax
import flax
import numpy as np
import jax.numpy as jnp
import pandas as pd
import re
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxrl.actorCritic import ActorCriticRNN, ScannedRNN
from gymnax_exchange.jaxrl import actorCriticS5, run_twap


timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
config = {
    "ENV_NAME": "alphatradeExec-v0",
    "DEBUG": True,
    "TASKSIDE": "random", #"random",
    "RNN_TYPE": "S5",  # "GRU", "S5"
    "HIDDEN_SIZE": 64,  # 128
    "JOINT_ACTOR_CRITIC_NET": True, 
    "NUM_ENVS": 1,

    "REWARD_LAMBDA": 1.0,
    "ACTION_TYPE": "pure",
    "CONT_ACTIONS": False,  # True
    "TASK_SIZE": 100, #500,
    "EPISODE_TIME": 60 * 5,
    "DATA_TYPE": "fixed_time", 
    "MAX_TASK_SIZE": 100,
    "TASK_SIZE": 100, # 500,
    "REDUCE_ACTION_SPACE_BY": 10,
    
    "WINDOW_INDEX": -1,  # gets overwritten by the evaluation function
    # "ATFOLDER": "./testing_oneDay/",
    "ATFOLDER": "./testing_oneWeek/",
    # "ATFOLDER": "./training_oneDay/",
    "RESULTS_FILE": "./training_runs/results_file_"+f"{timestamp}",

    "CHECKPOINT_DIR": "./training_runs/checkpoints_04-12_16-02/",
    "HEURISTIC_DIR": "./policy_evals/",
    "CHECKPOINT_ID": 16574000,  # which checkpoint to use from the checkpoint directory (-1 for latest)
}

# dir = config['CHECKPOINT_DIR']
# csv_dir = config['CHECKPOINT_DIR'] + "csv/"


def _init_network(env, env_params, config):
    if config['RNN_TYPE'] == "GRU":
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)

        if config['JOINT_ACTOR_CRITIC_NET']:
            hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"])
        else:
            hstate = (
                ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"]),
                ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"])
            )
    elif config['RNN_TYPE'] == "S5":
        network = actorCriticS5.ActorCriticS5(env.action_space(env_params).shape[0], config=config)

        if config['JOINT_ACTOR_CRITIC_NET']:
            hstate = actorCriticS5.ActorCriticS5.initialize_carry(
                config["NUM_ENVS"], actorCriticS5.ssm_size, actorCriticS5.n_layers)
        else:
            raise NotImplementedError('Separate actor critic nets not supported for S5 yet.')
    else:
        raise NotImplementedError('Only GRU and S5 RNN types supported for now.')

    return network, hstate

def _extract_number_from_filename(filename):
    filename = filename.rsplit('/', 1)[-1]
    match = re.search(r'_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # default if no number is found

def get_checkpoint_file_name(checkpoint_dir, checkpoint_num):
    # use latest checkpoint if checkpoint_num is -1
    if checkpoint_num == -1:
        files = glob.glob(f"{checkpoint_dir}/*.ckpt")
        files = sorted(files, key=_extract_number_from_filename)[-1:]
    else:
        files = glob.glob(f"{checkpoint_dir}/*{checkpoint_num}.ckpt")
    
    assert len(files) == 1, f"Found {len(files)} files with checkpoint number {checkpoint_num}"
    return files[0]

def load_eval_csv(csv_dir, checkpoint, window_idx):
    data = pd.read_csv(csv_dir + f'checkpoint_{checkpoint}_wdw_idx_{window_idx}.csv')
    # data.action = data.action.apply(lambda x: [float(num) for num in x[1:-1].split()])
    return data

def load_all_eval_csvs(csv_dir, checkpoint):
    files = glob.glob(f"{csv_dir}*{checkpoint}_wdw_idx_*.csv")
    data = pd.concat([pd.read_csv(file) for file in files])
    data.sort_values(by=['window_index', 'current_step'], inplace=True)
    return data

def _load_network_params(params_file):
    with open(params_file, 'rb') as f:
        trainstate_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
    return trainstate_params

def make_heuristic_step(env, policy_fn):
    def step(rng, obs, done, hstate, state, env_params):
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        action = policy_fn(env, obs, rng_action)
        _, state, reward, done, info = env.step(rng, state, action, env_params)
        obs = env._get_obs(state, env_params, normalize=False, flatten=False)

        row_data = info | obs
        for i in range(action.shape[0]):
            row_data[f'action_{i}'] = action[i]
        row_data['reward'] = reward

        return row_data, (obs, done, hstate, state, reward, info)
    return step

def make_step(network_config):
    network, trainstate_params, env = network_config
    
    def step(rng, obs, done, hstate, state, env_params):
        ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
        assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
        assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
        hstate, pi, value = network.apply(trainstate_params, hstate, ac_in)
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        raw_action = pi.sample(seed=rng_action)
        # action = raw_action.round().astype(jnp.int32)[0,0,:].clip(0, None)
        # action = raw_action[0,0,:]
        action = jnp.squeeze(raw_action)
        obs, state, reward, done, info = env.step(rng_step, state, action, env_params)
        obs_dict = env._get_obs(state, env_params, normalize=False, flatten=False)

        row_data = info | obs_dict
        for i in range(action.shape[0]):
            row_data[f'action_{i}'] = action[i]
        row_data['reward'] = reward

        return row_data, (obs, done, hstate, state, reward, info)
    return step

def get_eval_fn(checkpoint_num, config, heuristic_fn=None, csv_dir=None):
    if csv_dir is None:
        csv_dir = config['CHECKPOINT_DIR'] + "csv/"

    env = ExecutionEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=-1,  # TODO: check this
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        rewardLambda=config["REWARD_LAMBDA"],
        ep_type=config["DATA_TYPE"],
    )
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=config["REWARD_LAMBDA"],
        task_size=config["TASK_SIZE"],
        episode_time=config["EPISODE_TIME"],
    )
    device = jax.devices()[-1]
    
    # network evaluation
    if heuristic_fn is None:
        params_path = get_checkpoint_file_name(config['CHECKPOINT_DIR'], checkpoint_num)
        print('loading params from', params_path)
        trainstate_params = _load_network_params(params_path)

        network, hstate = _init_network(env, env_params, config)
        network_config = (network, trainstate_params, env)
        step_jit = jax.jit(make_step(network_config), device=device)

        file_prefix = params_path.rsplit("/", 1)[-1].split('.')[0]
    
    # heuristic fn evaluation
    else:
        step_jit = jax.jit(make_heuristic_step(env, heuristic_fn), device=device)
        hstate = None
        file_prefix = "heuristic"
    
    def evaluate(window_idx, rng, hstate=hstate, env_params=env_params):
        print('evaluating window_idx', window_idx)
        rng, rng_reset = jax.random.split(rng)
        env_params = dataclasses.replace(
            env_params,
            window_selector=window_idx
        )
        obs, state = env.reset(rng_reset, env_params)
        if heuristic_fn is not None:
            obs = env._get_obs(state, env_params, normalize=False, flatten=False)

        os.makedirs(csv_dir, exist_ok=True) 
        with open(csv_dir + file_prefix + f'_wdw_idx_{window_idx}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            done = False
            for i in range(0, 10_000):
                # print("step", i)

                # start = time.time()
                rng, rng_eval = jax.random.split(rng)
                row_data, (obs, done, hstate, state, reward, info) = step_jit(rng_eval, obs, done, hstate, state, env_params)
                # print(f"time taken: {time.time()-start}")

                # Add a header row if needed
                if i == 0:
                    row_title = list(row_data.keys())
                    csvwriter.writerow(row_title)
                
                csvwriter.writerow(list(row_data.values()))
                csvfile.flush() 
                if done:
                    break

    return evaluate, env.n_windows

def eval_ppo(config, checkpoint_num=-1, level_stride=1, csv_dir=None):
    rng = jax.random.PRNGKey(0)
    print('evaluating checkpoint', checkpoint_num)
    ppo_eval_fn, n_windows = get_eval_fn(checkpoint_num, config, csv_dir=csv_dir)
    print(f"n_windows: {n_windows}")

    # for window_idx in range(0, n_windows, 10):
    for window_idx in range(0, n_windows, level_stride):
        rng, _rng = jax.random.split(rng)
        ppo_eval_fn(window_idx, _rng)

def eval_heuristic(config, csv_dir, heuristic_fn, level_stride=1):
    rng = jax.random.PRNGKey(0)
    # heuristic TWAP eval
    # csv_dir = config['CHECKPOINT_DIR'] + "twap_pass_buy/"
    twap_eval_fn, n_windows = get_eval_fn(
        None, config, heuristic_fn=heuristic_fn, csv_dir=csv_dir)
    for window_idx in range(0, n_windows, level_stride):
        rng, _rng = jax.random.split(rng)
        twap_eval_fn(window_idx, _rng)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify index of the file to evaluate.')
    parser.add_argument('--idx', metavar='idx', type=int, default=-1, help='Index of the file to evaluate.')
    parser.add_argument('--heuristic_eval', action='store_true', default=True, help='Whether to evaluate the heuristic policies.')
    args = parser.parse_args()
    
    # how many levels to skip over each eval step (so that the time periods aren't overlapping)
    level_stride = 5

    if args.idx != -1:
        config["CHECKPOINT_ID"] = args.idx
    
    ################## PPO eval ##################

    print("Evaluating PPO SELL")
    config['TASKSIDE'] = 'sell'
    eval_ppo(
        config,
        config["CHECKPOINT_ID"],
        level_stride,
        csv_dir=config['CHECKPOINT_DIR'] + "csv_sell_week/"
    )

    print("Evaluating PPO BUY")
    config['TASKSIDE'] = 'buy'
    eval_ppo(
        config,
        config["CHECKPOINT_ID"],
        level_stride,
        csv_dir=config['CHECKPOINT_DIR'] + "csv_buy_week/"
    )

    ################## TWAP eval ##################

    if args.heuristic_eval:

        # Heuristic evaluation
        print("Evaluating TWAP SELL")
        config['TASKSIDE'] = 'sell'
        eval_heuristic(
            config,
            csv_dir=config['HEURISTIC_DIR'] + "twap_pass_sell/",
            heuristic_fn=run_twap.twap_pass,
            level_stride=level_stride
        )
        print("Evaluating TWAP BUY")
        config['TASK_SIDE'] = 'buy'
        eval_heuristic(
            config,
            csv_dir=config['HEURISTIC_DIR'] + "twap_pass_buy/",
            heuristic_fn=run_twap.twap_pass,
            level_stride=level_stride
        )

    print("Done evaluating.")
