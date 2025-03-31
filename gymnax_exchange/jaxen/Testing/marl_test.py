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
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
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

def generate_plots(
    mm_rewards,
    mm_reward_portfolio_value,
    mm_reward_complex,
    mm_reward_spooner,
    mm_reward_spooner_damped,
    mm_reward_spooner_scaled,
    mm_reward_delta_netWorth,
    mm_inventory,
    mm_total_PnL,
    mm_buyQuant,
    mm_sellQuant,
    mm_bid_price,
    mm_ask_price,
    mm_averageMidprice,
    mm_netWorth,
    exe_rewards,
    exe_total_revenue,
    exe_quant_executed,
    exe_average_price,
    exe_vwap_rm,
    exe_mid_price,
    exe_slippage_rm,
    exe_price_adv_rm,
    exe_price_drift_rm,
    exe_advantage_reward,
    exe_drift_reward,
    exe_trade_duration,
    valid_steps,
    output_dir,
):
    """
    Generates plots and saves data to CSV for both MM and EXE environments.
    """

    plot_until_step = valid_steps

    # Slice all data to valid steps
    datasets = [
        mm_rewards, mm_reward_portfolio_value, mm_reward_complex, mm_reward_spooner,
        mm_reward_spooner_damped, mm_reward_spooner_scaled, mm_reward_delta_netWorth,
        mm_inventory, mm_total_PnL, mm_buyQuant, mm_sellQuant, mm_bid_price, mm_ask_price,
        mm_averageMidprice, mm_netWorth, exe_rewards, exe_total_revenue, exe_quant_executed,
        exe_average_price, exe_vwap_rm, exe_mid_price, exe_slippage_rm, exe_price_adv_rm,
        exe_price_drift_rm, exe_advantage_reward, exe_drift_reward, exe_trade_duration
    ]
    datasets = [data[:plot_until_step].squeeze() for data in datasets]

    mm_rewards = mm_rewards[:plot_until_step]
    mm_reward_portfolio_value = mm_reward_portfolio_value[:plot_until_step]
    mm_reward_complex = mm_reward_complex[:plot_until_step]
    mm_reward_spooner = mm_reward_spooner[:plot_until_step]
    mm_reward_spooner_scaled = mm_reward_spooner_scaled[:plot_until_step]
    mm_reward_spooner_damped = mm_reward_spooner_damped[:plot_until_step]
    mm_reward_delta_netWorth = mm_reward_delta_netWorth[:plot_until_step]
    mm_inventory = mm_inventory[:plot_until_step]
    mm_total_PnL = mm_total_PnL[:plot_until_step]
    mm_buyQuant = mm_buyQuant[:plot_until_step]
    mm_sellQuant = mm_sellQuant[:plot_until_step]
    mm_bid_price = mm_bid_price[:plot_until_step]
    mm_ask_price = mm_ask_price[:plot_until_step]
    mm_averageMidprice = mm_averageMidprice[:plot_until_step]
    mm_netWorth = mm_netWorth[:plot_until_step]
    exe_rewards = exe_rewards[:plot_until_step]
    exe_total_revenue = exe_total_revenue[:plot_until_step]
    exe_quant_executed = exe_quant_executed[:plot_until_step]
    exe_average_price = exe_average_price[:plot_until_step]
    exe_vwap_rm = exe_vwap_rm[:plot_until_step]
    exe_mid_price = exe_mid_price[:plot_until_step]
    exe_slippage_rm = exe_slippage_rm[:plot_until_step]
    exe_price_adv_rm = exe_price_adv_rm[:plot_until_step]
    exe_price_drift_rm = exe_price_drift_rm[:plot_until_step]
    exe_advantage_reward = exe_advantage_reward[:plot_until_step]
    exe_drift_reward = exe_drift_reward[:plot_until_step]
    exe_trade_duration = exe_trade_duration[:plot_until_step]

    # ============================
    # Save all data to CSV
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([
        mm_rewards, mm_reward_portfolio_value, mm_reward_complex, mm_reward_spooner,
        mm_reward_spooner_damped, mm_reward_spooner_scaled, mm_reward_delta_netWorth,
        mm_inventory, mm_total_PnL, mm_buyQuant, mm_sellQuant, mm_bid_price, mm_ask_price, mm_averageMidprice, mm_netWorth,
        exe_rewards, exe_total_revenue, exe_quant_executed, exe_average_price, exe_vwap_rm, exe_mid_price,
        exe_slippage_rm, exe_price_adv_rm, exe_price_drift_rm, exe_advantage_reward, exe_drift_reward, exe_trade_duration
    ])
    # Column headers for all metrics
    column_names = [
        'MM Reward', 'MM Portfolio Value Reward', 'MM Complex Reward', 'MM Spooner Reward',
        'MM Spooner Damped Reward', 'MM Spooner Scaled Reward', 'MM Delta Net Worth Reward',
        'MM Inventory', 'MM Total PnL', 'MM Buy Quantity', 'MM Sell Quantity', 'MM Bid Price', 'MM Ask Price', 'MM Average Midprice', 'MM Net Worth',
        'Exe Reward', 'Exe Total Revenue', 'Exe Quant Executed', 'Exe Average Price', 'Exe VWAP RM', 'Exe Mid Price',
        'Exe Slippage RM', 'Exe Price Adv RM', 'Exe Price Drift RM', 'Exe Advantage Reward', 'Exe Drift Reward', 'Exe Trade Duration'
    ]
    # Save data as CSV
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(os.path.join(output_dir, 'data.csv'), index=False)

    print(f"Last valid step {valid_steps}")
    print(f"Last NetWorth: {mm_netWorth[-1]}")
    print(f"Last PnL: {mm_total_PnL[-1]}")

    # ============================
    # Plotting All Metrics
    # ============================
    fig, axes = plt.subplots(6, 5, figsize=(18, 18))

    # MM Rewards Plots
    axes[0, 0].plot(range(plot_until_step), mm_rewards, label="MM Reward", color='green')
    axes[0, 0].set_title("MM Reward Over Steps")
    axes[0, 0].legend()

    axes[0, 1].plot(range(plot_until_step), mm_reward_portfolio_value, label="MM Portfolio Value Reward", color='blue')
    axes[0, 1].set_title("MM Portfolio Value Reward")
    axes[0, 1].legend()

    axes[0, 2].plot(range(plot_until_step), mm_reward_complex, label="MM Complex Reward", color='orange')
    axes[0, 2].set_title("MM Complex Reward")
    axes[0, 2].legend()

    axes[0, 3].plot(range(plot_until_step), mm_reward_spooner, label="MM Spooner Reward", color='purple')
    axes[0, 3].set_title("MM Spooner Reward")
    axes[0, 3].legend()

    axes[0, 4].plot(range(plot_until_step), mm_reward_spooner_scaled, label="MM Spooner Scaled Reward", color='red')
    axes[0, 4].set_title("MM Spooner Scaled Reward")
    axes[0, 4].legend()

    # EXE Rewards Plots
    axes[1, 0].plot(range(plot_until_step), exe_rewards, label="Exe Reward", color='green')
    axes[1, 0].set_title("Exe Reward Over Steps")
    axes[1, 0].legend()

    axes[1, 1].plot(range(plot_until_step), exe_total_revenue, label="Exe Total Revenue", color='blue')
    axes[1, 1].set_title("Exe Total Revenue")
    axes[1, 1].legend()

    axes[1, 2].plot(range(plot_until_step), exe_quant_executed, label="Exe Quant Executed", color='orange')
    axes[1, 2].set_title("Exe Quant Executed")
    axes[1, 2].legend()

    axes[1, 3].plot(range(plot_until_step), exe_average_price, label="Exe Average Price", color='purple')
    axes[1, 3].set_title("Exe Average Price")
    axes[1, 3].legend()

    axes[1, 4].plot(range(plot_until_step), exe_vwap_rm, label="Exe VWAP RM", color='red')
    axes[1, 4].set_title("Exe VWAP RM")
    axes[1, 4].legend()

    # MM/EXE Inventory & PnL Plots
    axes[2, 0].plot(range(plot_until_step), mm_inventory, label="MM Inventory", color='green')
    axes[2, 0].set_title("MM Inventory")
    axes[2, 0].legend()

    axes[2, 1].plot(range(plot_until_step), mm_total_PnL, label="MM Total PnL", color='blue')
    axes[2, 1].set_title("MM Total PnL")
    axes[2, 1].legend()

    axes[2, 2].plot(range(plot_until_step), mm_buyQuant, label="MM Buy Quantity", color='orange')
    axes[2, 2].set_title("MM Buy Quantity")
    axes[2, 2].legend()

    axes[2, 3].plot(range(plot_until_step), mm_sellQuant, label="MM Sell Quantity", color='purple')
    axes[2, 3].set_title("MM Sell Quantity")
    axes[2, 3].legend()


    axes[3, 1].plot(range(plot_until_step), mm_averageMidprice, label="MM Average Midprice", color='purple')
    axes[3, 1].plot(range(plot_until_step), mm_ask_price, label="MM Ask Price", color='blue')
    axes[3, 1].plot(range(plot_until_step), mm_bid_price, label="MM Bid Price", color='red')
    axes[3, 1].set_title("MM Prices Midprice")
    axes[3, 1].legend()

    axes[3, 2].plot(range(plot_until_step), exe_mid_price, label="Exe Mid Price", color='green')
    axes[3, 2].set_title("Exe Mid Price")
    axes[3, 2].legend()

    axes[3, 3].plot(range(plot_until_step), exe_slippage_rm, label="Exe Slippage RM", color='blue')
    axes[3, 3].set_title("Exe Slippage RM")
    axes[3, 3].legend()

    axes[3, 4].plot(range(plot_until_step), exe_price_adv_rm, label="Exe Price Adv RM", color='orange')
    axes[3, 4].set_title("Exe Price Adv RM")
    axes[3, 4].legend()

    # Remaining Metrics Plots
    axes[4, 0].plot(range(plot_until_step), exe_price_drift_rm, label="Exe Price Drift RM", color='purple')
    axes[4, 0].set_title("Exe Price Drift RM")
    axes[4, 0].legend()

    axes[4, 1].plot(range(plot_until_step), exe_advantage_reward, label="Exe Advantage Reward", color='red')
    axes[4, 1].set_title("Exe Advantage Reward")
    axes[4, 1].legend()

    axes[4, 2].plot(range(plot_until_step), exe_drift_reward, label="Exe Drift Reward", color='green')
    axes[4, 2].set_title("Exe Drift Reward")
    axes[4, 2].legend()

    axes[4, 3].plot(range(plot_until_step), exe_trade_duration, label="Exe Trade Duration", color='blue')
    axes[4, 3].set_title("Exe Trade Duration")
    axes[4, 3].legend()

    axes[4, 4].plot(range(plot_until_step), mm_netWorth, label="MM Net Worth", color='orange')
    axes[4, 4].set_title("MM Net Worth")
    axes[4, 4].legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = os.path.join(output_dir, 'combined_all_steps.png')
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")




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
        "EXE_TASK_SIZE": 100,
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
    exe_trade_duration=np.zeros((test_steps, 1))
    exe_advantage_reward=np.zeros((test_steps, 1))
    

    
    output_dir = 'gymnax_exchange/jaxen/Testing/output/marl'
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
        mm_netWorth[i]=info["market_maker"]["netWorth"]


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
        exe_trade_duration[i] = info["execution"]["trade_duration"]
        
        # Increment valid steps
        valid_steps += 1
        
      



        if done["__all__"]:
            print("Episode finished!")
            break


        
    # ============================
    #Plot
    # ============================
    generate_plots(
     mm_rewards,
    mm_reward_portfolio_value,
    mm_reward_complex,
    mm_reward_spooner,
    mm_reward_spooner_damped,
    mm_reward_spooner_scaled,
    mm_reward_delta_netWorth,
    mm_inventory,
    mm_total_PnL,
    mm_buyQuant,
    mm_sellQuant,
    mm_bid_price,
    mm_ask_price,
    mm_averageMidprice,
    mm_netWorth,
    exe_rewards,
    exe_total_revenue,
    exe_quant_executed,
    exe_average_price,
    exe_mid_price,
    exe_vwap_rm,
    exe_slippage_rm,
    exe_price_adv_rm,
    exe_price_drift_rm,
    exe_advantage_reward,
    exe_drift_reward,
    exe_trade_duration,
    valid_steps,
    output_dir,
 )


   