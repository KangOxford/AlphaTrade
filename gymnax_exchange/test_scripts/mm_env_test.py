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
import chex

from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig

faulthandler.enable()

# ============================
# Configuration
# ============================
test_steps = 15000 # Adjusted for your test case; make sure this isn't too high

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        #ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        ATFolder= "/home/duser/AlphaTrade/testing"

        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 0,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*30,  
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)
    
    # env=MarketMakingEnv(ATFolder,"sell",1)

    env_cfg = EnvironmentConfig()

    env = MarketMakingEnv(
        cfg = env_cfg,
        key = key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
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

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    
    #ask_raw_orders_history = np.zeros((test_steps, 100, 6), dtype=int)
    #bid_raw_orders_history = np.zeros((test_steps, 100,6), dtype=int)
    rewards = np.zeros((test_steps, 1), dtype=int)
    inventory = np.zeros((test_steps, 1), dtype=int)
    total_PnL = np.zeros((test_steps, 1), dtype=int)
    buyQuant = np.zeros((test_steps, 1), dtype=int)
    sellQuant = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    ask_price = np.zeros((test_steps, 1), dtype=int)
    netWorth = np.zeros((test_steps, 1), dtype=int)
    

    averageMidprice = np.zeros((test_steps, 1), dtype=int)
 
   


    output_dir = 'gymnax_exchange/test_scripts/test_outputs/'
   
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
        #jax.debug.print("action{}",test_action)
        #test_action=7
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
        rewards[i] = reward
        inventory[i] = info["inventory"]
        total_PnL[i] = info["total_PnL"]
        buyQuant[i] = info["buyQuant"]
        sellQuant[i] = info["sellQuant"]
        bid_price[i] = info["action_prices"][0]  # Store best ask
        ask_price[i] = info["action_prices"][1]
        averageMidprice[i] = info["averageMidprice"]  # Store mid price
        netWorth[i]=info["netWorth"]
        
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            break

    # ============================
    # Clip the arrays to remove trailing zeros
    # ============================

    plot_until_step = valid_steps 

    rewards = rewards[:plot_until_step]
    inventory = inventory[:plot_until_step]
    total_PnL = total_PnL[:plot_until_step] 
    buyQuant = buyQuant[:plot_until_step]
    sellQuant = sellQuant[:plot_until_step]
    bid_price = bid_price[:plot_until_step]
    ask_price = ask_price[:plot_until_step]
    averageMidprice = averageMidprice[:plot_until_step]

    netWorth=netWorth[:plot_until_step]


    # ============================
    # Save all data to CSV
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([rewards, inventory, total_PnL, buyQuant, sellQuant, bid_price, ask_price, averageMidprice,netWorth])
    
    # Add column headers
    column_names = ['Reward', 'Inventory', 'Total PnL', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice','netWorth']
    
    # Save data using pandas to handle CSV easily
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(reward_file, index=False)
    
    print(f"Data saved to {reward_file}")
    print(f"Last valid step {valid_steps}")
    print(f"Last PnL: {total_PnL[-1]}")
    
    # ============================
    # Plotting all metrics on one page
    # ============================
    # Create a figure with subplots (3 rows and 3 columns to fit the new data)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust the grid as needed

    # Plot each metric on a separate subplot
    axes[0, 0].plot(range(plot_until_step), rewards, label="Reward", color='blue')
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Rewards Over Steps")
    
    axes[0, 1].plot(range(plot_until_step), inventory, label="Inventory", color='green')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Inventory")
    axes[0, 1].set_title("Inventory Over Steps")
    
    axes[0, 2].plot(range(plot_until_step), total_PnL, label="Total PnL", color='orange')
    axes[0, 2].set_xlabel("Steps")
    axes[0, 2].set_ylabel("Total PnL")
    axes[0, 2].set_title("Total PnL Over Steps")
    
    axes[1, 0].plot(range(plot_until_step), buyQuant, label="Buy Quantity", color='red')
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Buy Quantity")
    axes[1, 0].set_title("Buy Quantity Over Steps")
    
    axes[1, 1].plot(range(plot_until_step), sellQuant, label="Sell Quantity", color='purple')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Sell Quantity")
    axes[1, 1].set_title("Sell Quantity Over Steps")
    
    # Combined plot for Bid Price, Ask Price, and Average Mid Price
    axes[1, 2].plot(range(plot_until_step), bid_price, label="Bid Price", color='pink')
    axes[1, 2].plot(range(plot_until_step), ask_price, label="Ask Price", color='cyan')
    axes[1, 2].plot(range(plot_until_step), averageMidprice, label="Average Mid Price", color='magenta')

    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask, Mid,Agr, and PP Prices Over Steps")
    axes[1, 2].legend()

    axes[2, 0].plot(range(plot_until_step), netWorth, label="netWorth", color='gold')
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Net Worth ")
    axes[2, 0].set_title("Net Worth  Over Steps")


    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/reward_symmetrically_dampened_0.0001_lambda_all_steps.png'
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")



    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Ensure output directory exists

    os.makedirs(output_dir, exist_ok=True)





   