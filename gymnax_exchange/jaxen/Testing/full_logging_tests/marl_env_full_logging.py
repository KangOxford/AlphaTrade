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
from gymnax_exchange.jaxen.marl_env import MARLEnv

import faulthandler
import pandas as pd  
import chex

from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig

faulthandler.enable()

# ============================
# Configuration
# ============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""""
TODO:
Turn marl onto debug to run this!!!!
Please turn off after!!!!!
"""


if __name__ == "__main__":
    import sys
    import time
    import dataclasses

    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        print("Using default folder:", ATFolder)

    config = {
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 300,  # for example, 5 minutes
        "WINDOW_INDEX": 1,
        # sub–env parameters:
        "MM_TRADER_ID": -4999991,
        "MM_REWARD_LAMBDA": 0.0001,
        "MM_ACTION_TYPE": "pure",
        "MM_MAX_TASK_SIZE": 500,
        "EXE_TRADER_ID": -9999992,
        "EXE_REWARD_LAMBDA": 1.0,
        "EXE_TASK_SIZE": 5000,
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the MARL environment.
    env = MARLEnv(
        key = key_reset,
        alphatradePath=ATFolder,
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        ep_type=config["EP_TYPE"],
        mm_trader_id=config["MM_TRADER_ID"],
        exe_trader_id=config["EXE_TRADER_ID"],
    )
    # Get the default combined parameters.
    print("starting default parameters")
    env_params = env.default_params

    env_params = dataclasses.replace(env_params, episode_time=config["EPISODE_TIME"])

    # Reset the environment.
    obs, state = env.reset_env(key_reset, env_params)
    print("Reset done. Market maker obs:", obs["market_maker"])
    print("Execution obs:", obs["execution"])

    test_steps = 15000 # Adjusted for your test case; make sure this isn't too high
    # ============================
    # Initialize data storage
    # ===========================
    mm_rewards = np.zeros((test_steps, 1), dtype=int)
    mm_reward_portfolio_value = np.zeros((test_steps, 1), dtype=int)
    mm_reward_complex = np.zeros((test_steps, 1), dtype=int)
    mm_reward_spooner = np.zeros((test_steps, 1), dtype=int)
    mm_reward_spooner_scaled = np.zeros((test_steps, 1), dtype=int)
    mm_reward_spooner_damped = np.zeros((test_steps, 1), dtype=int)
    mm_reward_delta_netWorth = np.zeros((test_steps, 1), dtype=int)
    mm_inventory = np.zeros((test_steps, 1), dtype=int)
    mm_total_PnL = np.zeros((test_steps, 1), dtype=int)
    mm_buyQuant = np.zeros((test_steps, 1), dtype=int)
    mm_sellQuant = np.zeros((test_steps, 1), dtype=int)
    mm_bid_price = np.zeros((test_steps, 1), dtype=int)
    mm_ask_price = np.zeros((test_steps, 1), dtype=int)
    mm_netWorth = np.zeros((test_steps, 1), dtype=int)
    mm_averageMidprice = np.zeros((test_steps, 1), dtype=int)
    mm_midprice=np.zeros((test_steps, 1), dtype=int)
    mm_average_best_bid=np.zeros((test_steps, 1), dtype=int)
    mm_average_best_ask=np.zeros((test_steps, 1), dtype=int)


    exe_rewards = np.zeros((test_steps, 1))
    exe_total_revenue = np.zeros((test_steps, 1))
    exe_quant_executed = np.zeros((test_steps, 1))
    exe_average_price = np.zeros((test_steps, 1))
    exe_mid_price=np.zeros((test_steps, 1))
    exe_vwap_rm=np.zeros((test_steps, 1))
    exe_slippage_rm=np.zeros((test_steps, 1))
    exe_price_drift_rm=np.zeros((test_steps, 1))
    exe_price_adv_rm=np.zeros((test_steps, 1))
    exe_avantage_reward=np.zeros((test_steps, 1))
    exe_drift_reward=np.zeros((test_steps, 1))
    exe_drift=np.zeros((test_steps, 1))
    exe_trade_duration=np.zeros((test_steps, 1))
    exe_advantage_reward=np.zeros((test_steps, 1))

    #Now also log: all messages, all trades, the L2 state..
    total_messages=np.zeros((test_steps,100+env.mm_env.cfg.num_messages_by_agent+env.exe_env.cfg.num_messages_by_agent,8),dtype=int) #100 is fixed, then num messages by agent extra
    total_trades=np.zeros((test_steps,100,8),dtype=int) #fixed 100 a step
    lob_states=np.zeros((test_steps,40),)#getting 10 levels of l2 state, each gives price and quant


    output_dir = 'gymnax_exchange/jaxen/Testing/full_logging_tests/data/marl'
    valid_steps = 0
    test_steps = 15000

 

    # run a loop that samples random actions for each agent.
    for i in range(test_steps):      
        print(f"Step {i}")

        key_step, _ = jax.random.split(key_step, 2)

        #key_policy, _ = jax.random.split(key_policy, 2)
        # Get random actions from each agent's action space.

        key_policy, subkey_mm = jax.random.split(key_policy)
        action_mm = env.mm_env.action_space().sample(subkey_mm)

        key_policy, subkey_exe = jax.random.split(key_policy)
        action_exe = env.exe_env.action_space().sample(subkey_exe)
        #action_mm = env.mm_env.action_space().sample(key_policy)
        #action_exe = env.exe_env.action_space().sample(key_policy)
        actions = {"market_maker": action_mm, "execution": action_exe}
        obs, state, rewards, done, info = env.step(key_step, state, actions, env_params)

        # Store data
        mm_rewards[i] = rewards["market_maker"]
        mm_reward_portfolio_value[i] = info["market_maker"]["reward_portfolio_value"]
        mm_reward_complex[i] = info["market_maker"]["reward_complex"]
        mm_reward_spooner[i] = info["market_maker"]["reward_spooner"]
        mm_reward_spooner_scaled[i] = info["market_maker"]["reward_spooner_scaled"]
        mm_reward_spooner_damped[i] = info["market_maker"]["reward_spooner_damped"]
        mm_reward_delta_netWorth[i] = info["market_maker"]["reward_delta_netWorth"]
        mm_inventory[i] = info["market_maker"]["inventory"]
        mm_total_PnL[i] = info["market_maker"]["total_PnL"]
        mm_buyQuant[i] = info["market_maker"]["buyQuant"]
        mm_sellQuant[i] = info["market_maker"]["sellQuant"]
        mm_bid_price[i] = info["market_maker"]["action_prices"][0]  # Store best ask
        mm_ask_price[i] = info["market_maker"]["action_prices"][1]
        mm_averageMidprice[i] = info["market_maker"]["averageMidprice"]  # Store mid price
        mm_midprice[i]=info["market_maker"]["end_mid_price"]
        mm_netWorth[i]=info["market_maker"]["netWorth"]
        mm_average_best_bid[i]=info["market_maker"]["average_best_bid"]
        mm_average_best_ask[i]=info["market_maker"]["average_best_ask"]


        exe_rewards[i] = rewards["execution"]
        exe_total_revenue[i] = info["execution"]["total_revenue"]
        exe_quant_executed[i] = info["execution"]["quant_executed"]
        exe_average_price[i] = info["execution"]["average_price"]
        exe_vwap_rm[i] = info["execution"]["vwap_rm"]
        exe_mid_price[i] = info["execution"]["mid_price"]
        exe_slippage_rm[i] = info["execution"]["slippage_rm"]
        exe_price_adv_rm[i] = info["execution"]["price_adv_rm"]
        exe_price_drift_rm[i] = info["execution"]["price_drift_rm"]
        exe_advantage_reward[i] = info["execution"]["advantage_reward"]
        exe_drift_reward[i] = info["execution"]["drift_reward"]
        exe_drift[i] = info["execution"]["drift"]
        exe_trade_duration[i] = info["execution"]["trade_duration"]


        #===============================#
        #=====Store the full logging data==#
        #================================#
        total_messages[i,:,:]=info["total_msgs"]
        total_trades[i,:,:]=info["trades"]
        lob_states[i,:]=info["lob_state"] 
        
        # Increment valid steps
        valid_steps += 1
        if done["__all__"]:
            print("Episode finished!")
            break
    total_messages = total_messages[:valid_steps]
    total_trades = total_trades[:valid_steps]
    lob_states = lob_states[:valid_steps]
  
    mm_rewards=mm_rewards[:valid_steps]
    mm_reward_portfolio_value = mm_reward_portfolio_value[:valid_steps]
    mm_reward_complex = mm_reward_complex[:valid_steps]
    mm_reward_spooner = mm_reward_spooner[:valid_steps]
    mm_reward_spooner_damped = mm_reward_spooner_damped[:valid_steps]
    mm_reward_spooner_scaled = mm_reward_spooner_scaled[:valid_steps]
    mm_reward_delta_netWorth = mm_reward_delta_netWorth[:valid_steps]
    mm_inventory = mm_inventory[:valid_steps]
    mm_total_PnL = mm_total_PnL[:valid_steps]
    mm_buyQuant = mm_buyQuant[:valid_steps]
    mm_sellQuant = mm_sellQuant[:valid_steps]
    mm_bid_price =mm_bid_price[:valid_steps]
    mm_ask_price = mm_ask_price[:valid_steps]
    mm_averageMidprice =mm_averageMidprice[:valid_steps]
    mm_midprice=mm_midprice[:valid_steps]
    mm_netWorth = mm_netWorth[:valid_steps]
    mm_average_best_bid=mm_average_best_bid[:valid_steps]
    mm_average_best_ask=mm_average_best_ask[:valid_steps]

    
    exe_rewards = exe_rewards[:valid_steps]
    exe_total_revenue =exe_total_revenue[:valid_steps]
    exe_quant_executed =exe_quant_executed[:valid_steps]
    exe_average_price =exe_average_price[:valid_steps]
    exe_vwap_rm =exe_vwap_rm[:valid_steps]
    exe_mid_price =exe_mid_price[:valid_steps]
    exe_slippage_rm =exe_slippage_rm[:valid_steps]
    exe_price_adv_rm =exe_price_adv_rm[:valid_steps]
    exe_price_drift_rm =exe_price_drift_rm[:valid_steps]
    exe_advantage_reward =exe_advantage_reward[:valid_steps]
    exe_drift_reward =exe_drift_reward[:valid_steps]
    exe_drift =exe_drift[:valid_steps]
    exe_trade_duration =exe_trade_duration[:valid_steps]

    ##log to csvs
    exe_reward = np.hstack([exe_rewards,exe_drift_reward,exe_advantage_reward])
    # Add column headers
    exe_reward_column_names = ['Reward','drift_reward','advantage_reward']

    exe_reward_df = pd.DataFrame(exe_reward, columns=exe_reward_column_names)
    exe_reward_df['step'] = np.arange(1, len(exe_reward_df) + 1)#add step column
    exe_reward_df.to_csv(os.path.join(output_dir, 'exe_reward_data.csv'), index=False)

    #Environment stats
    exe_env_data = np.hstack([exe_total_revenue, exe_quant_executed, exe_vwap_rm, exe_average_price, exe_trade_duration, exe_drift, exe_price_drift_rm,exe_slippage_rm,exe_price_adv_rm,exe_mid_price])
    # Add column headers
    exe_env_data_column_names = ['total_revenue', 'quant_executed', 'vwap_rm', 'average_price', 'trade_duration', 'drift', 'price_drift_rm','slippage_rm','price_adv_rm','mid_price']

    # Save data using pandas to handle CSV easily
    exe_env_data_df = pd.DataFrame(exe_env_data, columns=exe_env_data_column_names)
    exe_env_data_df['step'] = np.arange(1, len(exe_env_data_df) + 1)##add step column
    exe_env_data_df.to_csv(os.path.join(output_dir, 'exe_env_data_df.csv'), index=False)


    ##mm
    mm_reward = np.hstack([mm_rewards, mm_reward_portfolio_value, mm_reward_complex, mm_reward_spooner,mm_reward_spooner_damped, mm_reward_spooner_scaled, mm_reward_delta_netWorth])
    # Add column headers
    mm_reward_column_names = ['Reward', 'Portfolio Value Reward', 'Complex Reward', 'Spooner Reward','Spooner Damped Reward', 'Spooner Scaled Reward', 'Delta Net Worth Reward']

    mm_reward_df = pd.DataFrame(mm_reward, columns=mm_reward_column_names)
    mm_reward_df['step'] = np.arange(1, len(mm_reward_df) + 1)#add step column
    mm_reward_df.to_csv(os.path.join(output_dir, 'mm_reward_data.csv'), index=False)

    #Environment stats
    mm_env_data = np.hstack([mm_inventory, mm_total_PnL, mm_buyQuant, mm_sellQuant, mm_bid_price, mm_ask_price, mm_averageMidprice,mm_midprice,mm_average_best_bid,mm_average_best_ask,mm_netWorth])
    # Add column headers
    mm_env_data_column_names = ['Inventory', 'Total PnL', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice','midprice','average_best_bid','average_best_ask', 'netWorth']

    # Save data using pandas to handle CSV easily
    mm_env_data_df = pd.DataFrame(mm_env_data, columns=mm_env_data_column_names)
    mm_env_data_df['step'] = np.arange(1, len(mm_env_data_df) + 1)##add step column
    mm_env_data_df.to_csv(os.path.join(output_dir, 'mm_env_data_df.csv'), index=False)


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
