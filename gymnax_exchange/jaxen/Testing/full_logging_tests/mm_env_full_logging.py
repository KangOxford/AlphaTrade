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
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
import faulthandler
import pandas as pd  
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig

faulthandler.enable()
''''
Script use:
Run mm env in debug mode for full logging test. Saves all messages, order book states and trades objects
as well as the normal info, so a full epsiode can be traced out

'''

#############################################################
#===================Full logging test=======================#

if __name__ == "__main__":
    #================#
    # Load the Files#
    #================#
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
    

    #Set keys
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    #=================#
    # Define Env Config
    #==================#
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 13,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*5,  
    }

    env_config_hps = [{"observation_space":"engineered",
                         "reward_space":"spooner_scaled",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"mid",
                         "action_space":"fixed_quants",
                         "debug_mode":True ########ENSURE THIS IS TRUE FOR FULL LOGGING TEST
                         }]
   
    env_cfg=EnvironmentConfig(**env_config_hps[0])
    trader_id=10
    env = MarketMakingEnv(
        cfg = env_cfg,
        key = key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
        trader_unique_id=trader_id,
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )

    # Initialize the environment state
    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {state.start_index}")
    print("Time for reset: \n", time.time() - start)
    print("Inventory after reset: \n", state.inventory)
    print(f"Number of available windows: {env.n_windows}")

    test_steps = 15000 #Max as check, aim to finish the episode
    # ============================
    # Initialize data storage
    # ============================
    output_dir = 'gymnax_exchange/jaxen/Testing/full_logging_tests/data/mm'

    #Log all the same as before
    rewards = np.zeros((test_steps, 1), dtype=int)
    reward_portfolio_value = np.zeros((test_steps, 1), dtype=int)
    reward_complex = np.zeros((test_steps, 1), dtype=int)
    reward_spooner = np.zeros((test_steps, 1), dtype=int)
    reward_spooner_scaled = np.zeros((test_steps, 1), dtype=int)
    reward_spooner_damped = np.zeros((test_steps, 1), dtype=int)
    reward_delta_netWorth = np.zeros((test_steps, 1), dtype=int)
    inventory = np.zeros((test_steps, 1), dtype=int)
    total_PnL = np.zeros((test_steps, 1), dtype=int)
    buyQuant = np.zeros((test_steps, 1), dtype=int)
    sellQuant = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    ask_price = np.zeros((test_steps, 1), dtype=int)
    netWorth = np.zeros((test_steps, 1), dtype=int)
    averageMidprice = np.zeros((test_steps, 1), dtype=int)
    midprice=np.zeros((test_steps, 1), dtype=int)
    average_best_bid=np.zeros((test_steps, 1), dtype=int)
    average_best_ask=np.zeros((test_steps, 1), dtype=int)

    #Now also log: all messages, all trades, the L2 state..
    total_messages=np.zeros((test_steps,100+env_cfg.num_messages_by_agent,8),dtype=int) #100 is fixed, then num messages by agent extra
    total_trades=np.zeros((test_steps,100,8),dtype=int) #fixed 100 a step
    lob_states=np.zeros((test_steps,40),)#getting 10 levels of l2 state, each gives price and quant

    # ============================
    # Track the number of valid steps
    # ============================
    valid_steps = 0

    # ============================
    # Run the test loop
    # ============================
    for i in range(test_steps):
        # ==================== ACTION ====================
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        test_action = env.action_space().sample(key_policy) 
        #test_action= 2
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)


        #====================#
        #== Store standard data#
        #======================#
        rewards[i] = reward
        reward_portfolio_value[i] = info["reward_portfolio_value"]
        reward_complex[i] = info["reward_complex"]
        reward_spooner[i] = info["reward_spooner"]
        reward_spooner_scaled[i] = info["reward_spooner_scaled"]
        reward_spooner_damped[i] = info["reward_spooner_damped"]
        reward_delta_netWorth[i] = info["reward_delta_netWorth"]
        inventory[i] = info["inventory"]
        total_PnL[i] = info["total_PnL"]
        buyQuant[i] = info["buyQuant"]
        sellQuant[i] = info["sellQuant"]
        bid_price[i] = info["action_prices"][0]  # Store best ask
        ask_price[i] = info["action_prices"][1]
        averageMidprice[i] = info["averageMidprice"]  # Store mid price
        midprice[i]=info["end_mid_price"]
        netWorth[i]=info["netWorth"]
        average_best_bid[i]=info["average_best_bid"]
        average_best_ask[i]=info["average_best_ask"]


        #===============================#
        #=====Store the full logging data==#
        #================================#
        total_messages[i,:,:]=info["total_msgs"]
        total_trades[i,:,:]=info["trades"]
        lob_states[i,:]=info["lob_state"]       
        
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            print(f"Episode ended at step {valid_steps}")
            break
    
    #==================================================================#
    #----------------------Save data to CSVs---------------------------#
    #==================================================================#
    #Trim the arrays
    total_messages = total_messages[:valid_steps]
    total_trades = total_trades[:valid_steps]
    lob_states = lob_states[:valid_steps]
    reward = rewards[:valid_steps]
    reward_portfolio_value = reward_portfolio_value[:valid_steps]
    reward_complex = reward_complex[:valid_steps]
    reward_spooner = reward_spooner[:valid_steps]
    reward_spooner_damped = reward_spooner_damped[:valid_steps]
    reward_spooner_scaled = reward_spooner_scaled[:valid_steps]
    reward_delta_netWorth = reward_delta_netWorth[:valid_steps]
    inventory = inventory[:valid_steps]
    total_PnL = total_PnL[:valid_steps]
    buyQuant = buyQuant[:valid_steps]
    sellQuant = sellQuant[:valid_steps]
    bid_price = bid_price[:valid_steps]
    ask_price = ask_price[:valid_steps]
    averageMidprice = averageMidprice[:valid_steps]
    midprice=midprice[:valid_steps]
    netWorth = netWorth[:valid_steps]
    average_best_bid=average_best_bid[:valid_steps]
    average_best_ask=average_best_ask[:valid_steps]

    #Make CSVs

    #Reward
    reward = np.hstack([reward, reward_portfolio_value, reward_complex, reward_spooner,reward_spooner_damped, reward_spooner_scaled, reward_delta_netWorth])
    # Add column headers
    reward_column_names = ['Reward', 'Portfolio Value Reward', 'Complex Reward', 'Spooner Reward','Spooner Damped Reward', 'Spooner Scaled Reward', 'Delta Net Worth Reward']

    reward_df = pd.DataFrame(reward, columns=reward_column_names)
    reward_df['step'] = np.arange(1, len(reward_df) + 1)#add step column
    reward_df.to_csv(os.path.join(output_dir, 'reward_data.csv'), index=False)

    #Environment stats
    env_data = np.hstack([inventory, total_PnL, buyQuant, sellQuant, bid_price, ask_price, averageMidprice,midprice,average_best_bid,average_best_ask,netWorth])
    # Add column headers
    env_data_column_names = ['Inventory', 'Total PnL', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice','midprice','average_best_bid','average_best_ask', 'netWorth']

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

