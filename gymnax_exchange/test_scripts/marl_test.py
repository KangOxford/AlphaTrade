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
    exe_mid_price,
    exe_vwap_rm,
    exe_slippage_rm,
    exe_price_adv_rm,
    exe_price_drift_rm,
    exe_advantage_reward,
    exe_drift_reward,
    exe_trade_duration,
    valid_steps,
    reward_file,
    output_dir,
):
    """
    Generates plots and saves data to CSV for both MM and EXE environments.
    """

    plot_until_step = valid_steps

    # Trim data to valid steps
    datasets = [
        mm_rewards, mm_reward_portfolio_value, mm_reward_complex, mm_reward_spooner,
        mm_reward_spooner_damped, mm_reward_spooner_scaled, mm_reward_delta_netWorth,
        mm_inventory, mm_total_PnL, mm_buyQuant, mm_sellQuant, mm_bid_price, mm_ask_price,
        mm_averageMidprice, mm_netWorth, exe_rewards, exe_total_revenue, exe_quant_executed,
        exe_average_price, exe_mid_price, exe_vwap_rm, exe_slippage_rm, exe_price_adv_rm,
        exe_price_drift_rm, exe_advantage_reward, exe_drift_reward, exe_trade_duration
    ]
    datasets = [data[:plot_until_step].squeeze() for data in datasets]

    mm_rewards= mm_rewards[:plot_until_step]
    mm_reward_portfolio_value = mm_reward_portfolio_value[:plot_until_step]
    mm_reward_complex = mm_reward_complex[:plot_until_step]
    mm_reward_spooner = mm_reward_spooner[:plot_until_step]
    mm_reward_spooner_scaled = mm_reward_spooner_scaled[:plot_until_step]
    mm_reward_spooner_damped= mm_reward_spooner_damped[:plot_until_step]
    mm_reward_delta_netWorth = mm_reward_delta_netWorth[:plot_until_step]
    mm_inventory = mm_inventory[:plot_until_step]
    mm_total_PnL = mm_total_PnL[:plot_until_step]
    mm_buyQuant = mm_buyQuant[:plot_until_step]
    mm_sellQuant = mm_sellQuant[:plot_until_step]
    mm_bid_price = mm_bid_price[:plot_until_step]  # Store best ask
    mm_ask_price = mm_ask_price[:plot_until_step]
    mm_averageMidprice = mm_averageMidprice[:plot_until_step]  # Store mid price
    mm_netWorth=mm_netWorth[:plot_until_step]
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

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))

    # First row
    axes[0, 0].plot(range(valid_steps), exe_total_revenue, label="Total Revenue", color='green')
    axes[0, 0].set_title("Total Revenue Over Steps")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Total Revenue")
    axes[0, 0].legend()

    axes[0, 1].plot(range(valid_steps), exe_quant_executed, label="Quantity Executed", color='purple')
    axes[0, 1].set_title("Quantity Executed Over Steps")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Quantity Executed")
    axes[0, 1].legend()

    axes[0, 2].plot(range(valid_steps), mm_total_PnL, label="Total PnL", color='orange')
    axes[0, 2].set_title("Total PnL Over Steps")
    axes[0, 2].set_xlabel("Steps")
    axes[0, 2].set_ylabel("Total PnL")
    axes[0, 2].legend()

    # Second row
    axes[1, 0].plot(range(valid_steps), exe_average_price, label="Average Price", color='orange')
    axes[1, 0].plot(range(valid_steps), exe_vwap_rm, label="VWAP", color='blue', linestyle='dashed')
    axes[1, 0].set_title("Average Price & VWAP Over Steps")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Price")
    axes[1, 0].legend()

    axes[1, 1].plot(range(valid_steps), exe_rewards, label="Reward", color='red')
    axes[1, 1].set_title("Reward Over Steps")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].legend()

    # Combined plot for Bid Price, Ask Price, and Average Mid Price
    axes[1, 2].plot(range(valid_steps), mm_bid_price, label="Bid Price", color='pink')
    axes[1, 2].plot(range(valid_steps), mm_ask_price, label="Ask Price", color='cyan')
    axes[1, 2].plot(range(valid_steps), mm_averageMidprice, label="Average Mid Price", color='magenta')
    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask, Mid Prices Over Steps")
    axes[1, 2].legend()

    # Third row
    axes[2, 0].plot(range(valid_steps), exe_mid_price, label="Mid Price", color='brown')
    axes[2, 0].set_title("Mid Price Over Steps")
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Mid Price")
    axes[2, 0].legend()

    axes[2, 1].plot(range(valid_steps), exe_slippage_rm, label="Slippage RM", color='cyan')
    axes[2, 1].set_title("Slippage RM Over Steps")
    axes[2, 1].set_xlabel("Steps")
    axes[2, 1].set_ylabel("Slippage RM")
    axes[2, 1].legend()

    axes[2, 2].plot(range(valid_steps), mm_netWorth, label="Net Worth", color='gold')
    axes[2, 2].set_xlabel("Steps")
    axes[2, 2].set_ylabel("Net Worth")
    axes[2, 2].set_title("Net Worth Over Steps")
    axes[2, 2].legend()

    # Fourth row
    axes[3, 0].plot(range(valid_steps), exe_trade_duration, label="Trade Duration", color='black')
    axes[3, 0].set_title("Trade Duration Over Steps")
    axes[3, 0].set_xlabel("Steps")
    axes[3, 0].set_ylabel("Trade Duration")
    axes[3, 0].legend()

    axes[3, 1].plot(range(valid_steps), mm_buyQuant, label="Buy Quantity", color='red')
    axes[3, 1].set_title("Buy Quantity Over Steps")
    axes[3, 1].set_xlabel("Steps")
    axes[3, 1].set_ylabel("Buy Quantity")
    axes[3, 1].legend()

    axes[3, 2].plot(range(valid_steps), mm_sellQuant, label="Sell Quantity", color='purple')
    axes[3, 2].set_title("Sell Quantity Over Steps")
    axes[3, 2].set_xlabel("Steps")
    axes[3, 2].set_ylabel("Sell Quantity")
    axes[3, 2].legend()

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
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
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
    

    
    output_dir = 'gymnax_exchange/test_scripts/test_outputs/'
    valid_steps = 0

 

    # run a loop that samples random actions for each agent.
    for i in range(1, 20000):       
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
    reward_file,
    output_dir,
 )


   