import os
import sys
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
import time
import dataclasses
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gymnax_exchange.jaxen.BeforeJAXLOBMARL.exec_env import ExecutionEnv
import faulthandler
import pandas as pd  
from gymnax_exchange.jaxob.jaxob_config import EnvironmentExecutionConfig

faulthandler.enable()
''''
Script use:
Run mm env in debug mode for full logging test. Saves all messages, order book states and trades objects
as well as the normal info, so a full epsiode can be traced out

'''

if __name__ == "__main__":
    ATFolder = "/home/duser/AlphaTrade/training_oneDay/val"
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 10,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 30,
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    env_config_hps = [ {"task":"random",
                        "action_type":"pure",
                        "action_space":"fixed_quants",
                        "end_fn":"unwind_FT",
                        "max_task_size":100,
                        "n_actions":8,
                        "fixed_quant_value":10,
                        "num_messages_by_agent":8,
                        "debug_mode":True}
                        ]
    trader_id=10
    env_cfg=EnvironmentExecutionConfig(**env_config_hps[0])
    env = ExecutionEnv(
        cfg=env_cfg,
        key=key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        trader_unique_id=trader_id,
        ep_type=config["EP_TYPE"],
    )

    env_params = dataclasses.replace(
        env.default_params,
        episode_time=config["EPISODE_TIME"],
    )

    obs, state = env.reset(key_reset, env_params)
    test_steps = 15000


    rewards = np.zeros((test_steps, 1))
    total_revenue = np.zeros((test_steps, 1))
    quant_executed = np.zeros((test_steps, 1))
    average_price = np.zeros((test_steps, 1))
    mid_price=np.zeros((test_steps, 1))
    vwap_rm=np.zeros((test_steps, 1))
    slippage_rm=np.zeros((test_steps, 1))
    price_drift_rm=np.zeros((test_steps, 1))
    price_adv_rm=np.zeros((test_steps, 1))
    avantage_reward=np.zeros((test_steps, 1))
    drift_reward=np.zeros((test_steps, 1))
    drift=np.zeros((test_steps, 1))
    trade_duration=np.zeros((test_steps, 1))
    advantage_reward=np.zeros((test_steps, 1))

    
    #Now also log: all messages, all trades, the L2 state..
    total_messages=np.zeros((test_steps,100+env_cfg.num_messages_by_agent,8),dtype=int) #100 is fixed, then num messages by agent extra
    total_trades=np.zeros((test_steps,100,8),dtype=int) #fixed 100 a step
    lob_states=np.zeros((test_steps,40),)#getting 10 levels of l2 state, each gives price and quant



    output_dir = 'gymnax_exchange/jaxen/Testing/full_logging_tests/data/exec'
    valid_steps = 0

    
    for i in range(test_steps):
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        test_action = env.action_space().sample(key_policy)
        
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        rewards[i] = reward
        total_revenue[i] = info["total_revenue"]
        quant_executed[i] = info["quant_executed"]
        average_price[i] = info["average_price"]
        vwap_rm[i] = info["vwap_rm"]
        mid_price[i] = info["mid_price"]
        slippage_rm[i] = info["slippage_rm"]
        price_adv_rm[i] = info["price_adv_rm"]
        price_drift_rm[i] = info["price_drift_rm"]
        advantage_reward[i] = info["advantage_reward"]
        drift_reward[i] = info["drift_reward"]
        drift[i]=info["drift"]
        trade_duration[i] = info["trade_duration"]

        #===============================#
        #=====Store the full logging data==#
        #================================#
        total_messages[i,:,:]=info["total_msgs"]
        total_trades[i,:,:]=info["trades"]
        lob_states[i,:]=info["lob_state"]       
        

        valid_steps += 1
        if done:
            break


    #==================================================================#
    #----------------------Save data to CSVs---------------------------#
    #==================================================================#
    #Trim the arrays
 
    total_revenue = total_revenue[:valid_steps]
    quant_executed = quant_executed[:valid_steps]
    vwap_rm = vwap_rm[:valid_steps]
    average_price = average_price[:valid_steps]
    trade_duration = trade_duration[:valid_steps]
    drift = drift[:valid_steps]
    drift_reward = drift_reward[:valid_steps]
    advantage_reward=advantage_reward[:valid_steps]
    price_drift_rm=price_drift_rm[:valid_steps]
    slippage_rm=slippage_rm[:valid_steps]
    price_adv_rm=price_adv_rm[:valid_steps]
    mid_price=mid_price[:valid_steps]
    total_messages = total_messages[:valid_steps]
    total_trades = total_trades[:valid_steps]
    lob_states = lob_states[:valid_steps]
    reward = rewards[:valid_steps]
  

    #Make CSVs

    #Reward
    reward = np.hstack([reward,drift_reward,advantage_reward])
    # Add column headers
    reward_column_names = ['Reward','drift_reward','advantage_reward']

    reward_df = pd.DataFrame(reward, columns=reward_column_names)
    reward_df['step'] = np.arange(1, len(reward_df) + 1)#add step column
    reward_df.to_csv(os.path.join(output_dir, 'reward_data.csv'), index=False)

    #Environment stats
    env_data = np.hstack([total_revenue, quant_executed, vwap_rm, average_price, trade_duration, drift, price_drift_rm,slippage_rm,price_adv_rm,mid_price])
    # Add column headers
    env_data_column_names = ['total_revenue', 'quant_executed', 'vwap_rm', 'average_price', 'trade_duration', 'drift', 'price_drift_rm','slippage_rm','price_adv_rm','mid_price']

    # Save data using pandas to handle CSV easily
    env_data_df = pd.DataFrame(env_data, columns=env_data_column_names)
    env_data_df['step'] = np.arange(1, len(env_data_df) + 1)##add step column
    env_data_df.to_csv(os.path.join(output_dir, 'env_data_df.csv'), index=False)

    #Simulator level data
     # ==== Message & Trade formatting ====
    msg_headers = ["Type", "Side", "Quantity", "Price", "OID", "TID", "Ts", "Tns"]
    trade_headers = ["Price", "Quantity", "OIDs", "OIDa", "T", "Ts", "TIDs", "TIDa"]

    def vertical_stack_with_step(data: np.ndarray, headers: list[str]) -> pd.DataFrame:
        '''Process the trades and messages'''
        steps, rows, cols = data.shape

        output = []
        step_col = []

        for step in range(steps):
            chunk = data[step].astype(object)
            output.append(chunk)
            step_col.extend([step + 1] * rows)  # 1-based step index

        stacked = np.vstack(output)
        df = pd.DataFrame(stacked, columns=headers)
        df['step'] = step_col
        return df
    

    msg_df = vertical_stack_with_step(total_messages, msg_headers)
    trade_df = vertical_stack_with_step(total_trades, trade_headers)

    msg_df.to_csv(os.path.join(output_dir, "total_messages.csv"), index=False)
    trade_df.to_csv(os.path.join(output_dir, "total_trades.csv"), index=False)

    ##Lob data
    def parse_lob_rearranged(lob_states: np.ndarray, valid_steps: int, output_dir: str):
        '''Take the L2 state and make it build out from the middle.'''
        lob_snapshots = []
        price_levels = []

        for step in range(valid_steps):
            snapshot = []

            # For each level, extract ask and bid prices and quantities
            asks = []
            bids = []

            for level in range(10):
                ask_p = lob_states[step][level * 4 + 0]
                ask_q = lob_states[step][level * 4 + 1]
                bid_p = lob_states[step][level * 4 + 2]
                bid_q = lob_states[step][level * 4 + 3]

                #Handel -1 prices if our LOB has <10 levels (dont include)
                if ask_p > 0:
                    asks.append((ask_p, ask_q, "ask", step))
                if bid_p > 0:
                    bids.append((bid_p, bid_q, "bid", step))

            # Sort asks ascending and bids descending
            asks_sorted = sorted(asks, key=lambda x: x[0])  # Ascending asks
            bids_sorted = sorted(bids, key=lambda x: -x[0])  # Descending bids

            # Combine them: best bid to worst bid, best ask to worst ask
            # We want to reverse the bids to have best bid in the middle
            full_lob = bids_sorted + asks_sorted  # Bids first (highest to lowest), then asks (lowest to highest)

            # Store in a snapshot
            lob_snapshots.extend(full_lob)

        # Create a DataFrame and save the results
        lob_df = pd.DataFrame(lob_snapshots, columns=["Price", "Quantity", "Side", "Step"])
        return lob_df
    lob_df = parse_lob_rearranged(lob_states, valid_steps, output_dir)
    lob_df.to_csv(os.path.join(output_dir, "lob_states.csv"), index=False)

