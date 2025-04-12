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



import pandas as pd
import seaborn as sns
import os

from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig

faulthandler.enable()
''''
Script use:
Run mm env in debug mode for full logging test. Saves all messages, order book states and trades objects
as well as the normal info, so a full epsiode can be traced out

'''


###==================================================#
#=========Old plotting FN: Plots all standard info stuff#
#====================================================#
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
     midprice,
     netWorth,
     average_best_bid,
     average_best_ask,
     valid_steps,
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
    midprice=midprice[:plot_until_step]
    netWorth = netWorth[:plot_until_step]
    average_best_bid=average_best_bid[:plot_until_step]
    average_best_ask=average_best_ask[:plot_until_step]

    # ============================
    # Save all data to CSV including all rewards
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([
        rewards, reward_portfolio_value, reward_complex, reward_spooner,
        reward_spooner_damped, reward_spooner_scaled, reward_delta_netWorth,
        inventory, total_PnL, buyQuant, sellQuant, bid_price, ask_price, averageMidprice,midprice,average_best_bid,average_best_ask,netWorth
    ])
    # Add column headers
    column_names = [
        'Reward', 'Portfolio Value Reward', 'Complex Reward', 'Spooner Reward',
        'Spooner Damped Reward', 'Spooner Scaled Reward', 'Delta Net Worth Reward',
        'Inventory', 'Total PnL', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice','midprice','average_best_bid','average_best_ask', 'netWorth'
    ]

    # Save data using pandas to handle CSV easily
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(os.path.join(output_dir, 'data.csv'), index=False)

 
    
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
        plot_file_name = f"reward_{reward_name.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(output_dir, plot_file_name))
        plt.close()


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
    
    midprice
    axes[1, 2].plot(range(plot_until_step), average_best_bid, label="average_best_bid Price", color='yellow')
    axes[1, 2].plot(range(plot_until_step), average_best_ask, label="average_best_ask Price", color='orange')
    axes[1, 2].plot(range(plot_until_step), averageMidprice, label="Average Mid Price", color='magenta')

    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask, Average Prices Over Steps")
    axes[1, 2].legend()

    axes[2, 0].plot(range(plot_until_step), netWorth, label="netWorth", color='gold')
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Net Worth ")
    axes[2, 0].set_title("Net Worth  Over Steps")
    axes[2, 0].legend()

    axes[2, 1].plot(range(plot_until_step), bid_price, label="Bid Price", color='pink')
    axes[2, 1].plot(range(plot_until_step), ask_price, label="Ask Price", color='cyan')
    axes[2, 1].plot(range(plot_until_step), midprice, label="midprice Price", color='yellow')
    axes[2, 1].set_title("End Step Prices  Over Steps")
    axes[2, 1].legend()
    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = os.path.join(output_dir, 'combined_all_steps.png')
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")




#==============================================================#
#-------Plotting function for L2 State, messages and Trades---#
#==============================================================#


#helper fns

def parse_lob_rearranged(lob_states: np.ndarray, valid_steps: int, output_dir: str):
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
    lob_df.to_csv(os.path.join(output_dir, "lob_states_rearranged.csv"), index=False)

    return lob_df



def plot_lob_heatmap_from_rearranged_df_with_midprice(lob_df: pd.DataFrame, output_dir: str):
    # Ensure Step is an integer, which will properly organize the y-axis (time steps)
    lob_df["Step"] = lob_df["Step"].astype(int)

    # Pivot the data: Each price level (x-axis), each time step (y-axis), values are quantity
    pivot_data = lob_df.pivot_table(index="Step", columns="Price", values="Quantity", aggfunc="first").fillna(0)


    # Plotting the heatmap: y = time step, x = price level, color = quantity
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(pivot_data, cmap="viridis", cbar_kws={'label': 'Quantity'})
    plt.title("Rearranged Order Book Heatmap (Price vs. Time)")
    plt.xlabel("Price")
    plt.ylabel("Step")
    
    # Reduce the number of y-axis ticks to avoid overlap
    max_steps = len(pivot_data.index)
    step_interval = max(1, max_steps // 10)  # Adjust this to your preference

    ax.set_yticks(range(0, max_steps, step_interval))  # Set custom tick positions
    ax.set_yticklabels(range(0, max_steps, step_interval))  # Set custom tick labels

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lob_heatmap_with_midprice.png"))
    plt.close()





def analyze_and_visualize_lob_data(
    total_messages: np.ndarray,
    total_trades: np.ndarray,
    lob_states: np.ndarray,
    valid_steps: int,
    output_dir: str,
):
    """
    Saves total_messages, total_trades, and lob_states to CSVs and plots their evolution over time.
    
    Parameters:
        total_messages: np.ndarray, shape (steps, 104, 8)
        total_trades: np.ndarray, shape (steps, 100, 8)
        lob_states: np.ndarray, shape (steps, 40)
        valid_steps: int, number of valid time steps to use for plots
        output_dir: str, directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Trim data to valid steps
    total_messages = total_messages[:valid_steps]
    total_trades = total_trades[:valid_steps]
    lob_states = lob_states[:valid_steps]

    # ==== Message & Trade formatting ====
    msg_headers = ["Type", "Side", "Quantity", "Price", "OID", "TID", "Ts", "Tns"]
    trade_headers = ["Price", "Quantity", "OIDs", "OIDa", "T", "Ts", "TIDs", "TIDa"]

    def vertical_stack_with_spacer(data: np.ndarray, headers: list[str]) -> pd.DataFrame:
        steps, rows, cols = data.shape
        spacer = np.full((1, cols), "00000000", dtype=object)

        output = []
        for step in range(steps):
            chunk = data[step].astype(object)
            output.append(chunk)
            output.append(spacer)  # Add separator row
        stacked = np.vstack(output[:-1])  # Drop last spacer
        return pd.DataFrame(stacked, columns=headers)

    msg_df = vertical_stack_with_spacer(total_messages, msg_headers)
    trade_df = vertical_stack_with_spacer(total_trades, trade_headers)

    msg_df.to_csv(os.path.join(output_dir, "total_messages.csv"), index=False)
    trade_df.to_csv(os.path.join(output_dir, "total_trades.csv"), index=False)

    # ==== LOB state formatting ====
    lob_df = parse_lob_rearranged(lob_states, valid_steps, output_dir)
    lob_df.to_csv(os.path.join(output_dir, "lob_states.csv"), index=False)

     # ---- Plotting ----
    plot_lob_heatmap_from_rearranged_df_with_midprice(lob_df, output_dir)


   
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
    output_dir = 'gymnax_exchange/jaxen/Testing/output/mm/full_logging'

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
        test_action= 1
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
    #Old Plotting Function
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
     midprice,
     netWorth,
     average_best_bid,
     average_best_ask,
     valid_steps,
     output_dir,
    )

    #full logging plots
    analyze_and_visualize_lob_data(
    total_messages,
    total_trades,
    lob_states,
    valid_steps,
    output_dir,
    )


   