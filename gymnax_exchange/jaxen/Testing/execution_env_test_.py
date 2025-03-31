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
from gymnax_exchange.jaxen.exec_env import ExecutionEnv  
import faulthandler
import pandas as pd  
import chex

from gymnax_exchange.jaxob.jaxob_config import EnvironmentExecutionConfig

faulthandler.enable()

# ============================
# Configuration
# ============================

# ============================
# Configuration
# ============================

def generate_plots(
    rewards,
    total_revenue,
    quant_executed,
    average_price,
    vwap_rm,
    mid_price,
    slippage_rm,
    price_adv_rm,
    price_drift_rm,
    advantage_reward,
    drift_reward,
    trade_duration,
    valid_steps,
    output_dir
):
    """
    Generates plots and saves data to CSV for execution environment metrics.
    """
    # Trim data to valid steps
    variables = [
        rewards, total_revenue, quant_executed, average_price, vwap_rm, mid_price,
        slippage_rm, price_adv_rm, price_drift_rm, advantage_reward, drift_reward, trade_duration
    ]
    trimmed_data = [var[:valid_steps] for var in variables]
    
    # Save data to CSV
    column_names = [
        'Reward', 'Total Revenue', 'Quantity Executed', 'Average Price', 'VWAP', 'Mid Price',
        'Slippage RM', 'Price Advantage RM', 'Price Drift RM', 'Advantage Reward', 'Drift Reward', 'Trade Duration'
    ]
    df = pd.DataFrame(np.column_stack(trimmed_data), columns=column_names)
    df.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    
    # Plot setup
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    # Data and Titles
    titles = column_names
    colors = ['red', 'green', 'purple', 'orange', 'blue', 'brown', 'cyan', 'magenta', 'gray', 'pink', 'navy', 'black']
    
    # Generate subplots
    for i, (data, title, color) in enumerate(zip(trimmed_data, titles, colors)):
        axes[i].plot(range(valid_steps), data, label=title, color=color)
        axes[i].set_title(f"{title} Over Steps")
        axes[i].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_plot.png'))
    plt.close()
    print("All plots saved.")


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
    
    env_cfg = EnvironmentExecutionConfig()
    env = ExecutionEnv(
        cfg=env_cfg,
        key=key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
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
    trade_duration=np.zeros((test_steps, 1))
    advantage_reward=np.zeros((test_steps, 1))


    output_dir = 'gymnax_exchange/jaxen/Testing/output/exec'
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
        trade_duration[i] = info["trade_duration"]

        valid_steps += 1
        if done:
            break


    generate_plots(
         rewards,
    total_revenue,
    quant_executed,
    average_price,
    vwap_rm,
    mid_price,
    slippage_rm,
    price_adv_rm,
    price_drift_rm,
    advantage_reward,
    drift_reward,
    trade_duration,

    valid_steps,
    output_dir
    )
