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


def generate_plots(
    rewards,
    reward_portfolio_value,
    reward_complex,
    reward_spooner,
    reward_spooner_damped,
    reward_spooner_scaled,
    reward_delta_netWorth,
    inventory,
    total_PnL,
    buyQuant,
    sellQuant,
    bid_price,
    ask_price,
    averageMidprice,
    netWorth,
    valid_steps,
    reward_file,
    output_dir,
):
    """
    Generates plots and saves data to CSV for various metrics.

    Args:
        rewards: List of reward values.
        reward_portfolio_value: List of portfolio value reward values.
        reward_complex: List of complex reward values.
        reward_spooner: List of spooner reward values.
        reward_spooner_damped: List of spooner damped reward values.
        reward_spooner_scaled: List of spooner scaled reward values.
        reward_delta_netWorth: List of delta net worth reward values.
        inventory: List of inventory values.
        total_PnL: List of total PnL values.
        buyQuant: List of buy quantity values.
        sellQuant: List of sell quantity values.
        bid_price: List of bid price values.
        ask_price: List of ask price values.
        averageMidprice: List of average mid-price values.
        netWorth: List of net worth values.
        valid_steps: Number of valid steps.
        reward_file: Path to save the CSV data.
        output_dir: Path to save the plot images.
    """

    plot_until_step = valid_steps

    rewards = rewards[:plot_until_step]
    reward_portfolio_value = reward_portfolio_value[:plot_until_step]
    reward_complex = reward_complex[:plot_until_step]
    reward_spooner = reward_spooner[:plot_until_step]
    reward_spooner_damped = reward_spooner_damped[:plot_until_step]
    reward_spooner_scaled = reward_spooner_scaled[:plot_until_step]
    reward_delta_netWorth = reward_delta_netWorth[:plot_until_step]
    inventory = inventory[:plot_until_step]
    total_PnL = total_PnL[:plot_until_step]
    buyQuant = buyQuant[:plot_until_step]
    sellQuant = sellQuant[:plot_until_step]
    bid_price = bid_price[:plot_until_step]
    ask_price = ask_price[:plot_until_step]
    averageMidprice = averageMidprice[:plot_until_step]
    netWorth = netWorth[:plot_until_step]

    # ============================
    # Save all data to CSV including all rewards
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([
        rewards, reward_portfolio_value, reward_complex, reward_spooner,
        reward_spooner_damped, reward_spooner_scaled, reward_delta_netWorth,
        inventory, total_PnL, buyQuant, sellQuant, bid_price, ask_price, averageMidprice, netWorth
    ])
    # Add column headers
    column_names = [
        'Reward', 'Portfolio Value Reward', 'Complex Reward', 'Spooner Reward',
        'Spooner Damped Reward', 'Spooner Scaled Reward', 'Delta Net Worth Reward',
        'Inventory', 'Total PnL', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice', 'netWorth'
    ]

    # Save data using pandas to handle CSV easily
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(reward_file, index=False)
 
    print(f"Data saved to {reward_file}")
    print(f"Last valid step {valid_steps}")
    print(f"Last NetWorth: {netWorth[-1]}")
    print(f"Last PnL: {total_PnL[-1]}")

    # ============================
    # Plotting each reward type separately
    # ============================

    reward_types = {
        "Reward": rewards,
        "Portfolio Value Reward": reward_portfolio_value,
        "Complex Reward": reward_complex,
        "Spooner Reward": reward_spooner,
        "Spooner Damped Reward": reward_spooner_damped,
        "Spooner Scaled Reward": reward_spooner_scaled,
        "Delta Net Worth Reward": reward_delta_netWorth,
    }

    for reward_name, reward_data in reward_types.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(plot_until_step), reward_data, label=reward_name, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')  # Add dashed line at y=0
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title(f"{reward_name} Over Steps")
        plt.legend()

        # Save each plot to a separate file
        plot_file = f"gymnax_exchange/jaxen/Testing/output/reward_{reward_name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_file)
        plt.close()

        print(f"Plot saved to {plot_file}")

    # ============================
    # Plotting all metrics on one page
    # ============================
    # Create a figure with subplots (3 rows and 3 columns to fit the new data)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust the grid as needed

    # Plot each metric on a separate subplot
    axes[0, 0].plot(range(plot_until_step), rewards, label="Reward", color='green')
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Reward Over Steps")
    axes[0, 0].legend()


    axes[0, 1].plot(range(plot_until_step), inventory, label="Inventory", color='green')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Inventory")
    axes[0, 1].set_title("Inventory Over Steps")
    axes[0, 1].legend()

    axes[0, 2].plot(range(plot_until_step), total_PnL, label="Total PnL", color='orange')
    axes[0, 2].set_xlabel("Steps")
    axes[0, 2].set_ylabel("Total PnL")
    axes[0, 2].set_title("Total PnL Over Steps")
    axes[0, 2].legend()

    axes[1, 0].plot(range(plot_until_step), buyQuant, label="Buy Quantity", color='red')
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Buy Quantity")
    axes[1, 0].set_title("Buy Quantity Over Steps")
    axes[1, 0].legend()

    axes[1, 1].plot(range(plot_until_step), sellQuant, label="Sell Quantity", color='purple')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Sell Quantity")
    axes[1, 1].set_title("Sell Quantity Over Steps")
    axes[1, 1].legend()

    # Combined plot for Bid Price, Ask Price, and Average Mid Price
    axes[1, 2].plot(range(plot_until_step), bid_price, label="Bid Price", color='pink')
    axes[1, 2].plot(range(plot_until_step), ask_price, label="Ask Price", color='cyan')
    axes[1, 2].plot(range(plot_until_step), averageMidprice, label="Average Mid Price", color='magenta')

    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask, Mid Prices Over Steps")
    axes[1, 2].legend()

    axes[2, 0].plot(range(plot_until_step), netWorth, label="netWorth", color='gold')
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Net Worth ")
    axes[2, 0].set_title("Net Worth  Over Steps")
    axes[2, 0].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = os.path.join(output_dir, 'combined_all_steps.png')
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")




if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/val"
        #ATFolder= "/home/duser/AlphaTrade/testing"

        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 6,
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

    test_steps = 15000 # Adjusted for your test case; make sure this isn't too high
    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/jaxen/Testing/outputdata.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
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
 
   


    output_dir = 'gymnax_exchange/jaxen/Testing/output'
   
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
        #test_action=8
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
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
        netWorth[i]=info["netWorth"]
        
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            break

    # ============================
    #Plot
    # ============================
    generate_plots(
     rewards,
     reward_portfolio_value,
     reward_complex,
     reward_spooner,
     reward_spooner_damped,
     reward_spooner_scaled,
     reward_delta_netWorth,
     inventory,
     total_PnL,
     buyQuant,
     sellQuant,
     bid_price,
     ask_price,
     averageMidprice,
     netWorth,
     valid_steps,
     reward_file,
     output_dir,
 )


   