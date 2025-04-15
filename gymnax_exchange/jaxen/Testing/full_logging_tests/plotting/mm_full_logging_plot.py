import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig
import matplotlib.gridspec as gridspec
import dataclasses
import jax
from matplotlib.lines import Line2D



# Load data
data_path = 'gymnax_exchange/jaxen/Testing/full_logging_tests/data/mm'
output_dir='gymnax_exchange/jaxen/Testing/full_logging_tests/plotting/mm'

env_data = pd.read_csv(os.path.join(data_path, 'env_data_df.csv'))
trades_df=pd.read_csv(os.path.join(data_path, 'total_trades.csv'))
messages_df =pd.read_csv(os.path.join(data_path, 'total_messages.csv'))
reward_df=pd.read_csv(os.path.join(data_path, 'reward_data.csv'))
lob_states = pd.read_csv(os.path.join(data_path, 'lob_states.csv'))

#Load trhe env for env stats like tick size, keep this the same as how we define the env for the run
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
                        "debug_mode":True
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
trader_id=10

#=========================================================================#
#------------------------------Plotting Functions-------------------------#
#=========================================================================#



# --- 1. Plot averageMidprice, average_best_bid, and average_best_ask over steps ---
def plot_env_midprice_stats(env_data, output_dir):
    """Creates a plot of the average mid price and average best bid/ ask across steps
    Bid/ ask are rounded to nearest tick (down), mid price is a float"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(env_data['step'], env_data['averageMidprice'], label='Average Mid Price')
    ax.plot(env_data['step'], env_data['average_best_bid'], label='Average Best Bid')
    ax.plot(env_data['step'], env_data['average_best_ask'], label='Average Best Ask')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')
    ax.set_title('Midprice and Best Bids/Asks over Steps')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "prices.png"))
    plt.close(fig)


# --- 2. Plot cumulative LOB for a specific step ---
def plot_cumulative_lob(lob_states, output_dir, step):
    """Creates a Cumsum LOB represenation of the first 10 levels each side at the specified step"""
    lob_at_step = lob_states[lob_states['Step'] == step]
    bids = lob_at_step[lob_at_step['Side'] == 'bid'].sort_values(by='Price', ascending=False)
    asks = lob_at_step[lob_at_step['Side'] == 'ask'].sort_values(by='Price', ascending=True)

    bid_cum = bids['Quantity'].cumsum()
    ask_cum = asks['Quantity'].cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(bids['Price'], bid_cum, width=env.tick_size, label='Cumulative Bids', color='blue', alpha=0.7, align='center')
    ax.bar(asks['Price'], ask_cum, width=env.tick_size, label='Cumulative Asks', color='red', alpha=0.7, align='center')

    ax.set_xlabel('Price')
    ax.set_ylabel('Cumulative Quantity')
    ax.set_title(f'Cumulative Order Book (Block View) at Step {step}')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"Cumulative_Block_Order_Book_Step_{step}.png"))
    plt.close(fig)

# --- 3. Heatmap of LOB states across steps ---
def plot_lob_heatmap(lob_states, output_dir):
    """Creates a heat map for the LOB for whole episode"""
    # Convert sides to +1/-1
    lob_states['signed_quantity'] = lob_states.apply(
        lambda row: row['Quantity'] if row['Side'] == 'bid' else -row['Quantity'], axis=1)

    pivot = lob_states.pivot_table(index='Price', columns='Step', values='signed_quantity', aggfunc='sum').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(pivot, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Heatmap of LOB Quantities (bids positive, asks negative)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "lob_heatmap.png"))
    plt.close(fig)

#------------4. plot trader activity---------#


def plot_buy_sell_inventory_networth(env_data_df, trades_df,output_dir, trader_id=10):
    """Plots buy/sell activity with midprice (arrows) with legend, inventory, and net worth,
    determining buy/sell directly from the trades dataframe."""
    trader_trades = trades_df[((trades_df['TIDa'] == trader_id) | (trades_df['TIDs'] == trader_id))].copy()

    buy_steps = trader_trades[((trader_trades['TIDs'] == trader_id) & (trader_trades['Quantity'] > 0)) |
                               ((trader_trades['TIDa'] == trader_id) & (trader_trades['Quantity'] < 0))][['step', 'Price']]
    sell_steps = trader_trades[((trader_trades['TIDs'] == trader_id) & (trader_trades['Quantity'] < 0)) |
                                ((trader_trades['TIDa'] == trader_id) & (trader_trades['Quantity'] > 0))][['step', 'Price']]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, height_ratios=[2, 1])

    # Top subplot: Buy/Sell Activity with Midprice
    ax1 = axs[0]
    ax1.plot(env_data_df['step'], env_data_df['averageMidprice'], label='Average Midprice', color='green', linewidth=2)
    for _, row in buy_steps.iterrows():
        ax1.annotate('', xy=(row['step'], row['Price']), xytext=(row['step'], row['Price'] + 0.5),
                     arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=1.5))
    for _, row in sell_steps.iterrows():
        ax1.annotate('', xy=(row['step'], row['Price']), xytext=(row['step'], row['Price'] - 0.5),
                     arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=1.5))
    ax1.set_ylabel('Price')
    ax1.set_title(f'Trader {trader_id} Buy/Sell Activity with Midprice')
    ax1.grid(True)

    # Create custom legend elements
    buy_legend = Line2D([0], [0], marker='>', color='w', markerfacecolor='blue', markersize=10, label='Buy')
    sell_legend = Line2D([0], [0], marker='>', color='w', markerfacecolor='red', markersize=10, label='Sell')
    ax1.legend(handles=[buy_legend, sell_legend], loc='upper right')

    # Bottom subplot: Inventory and Net Worth
    ax2 = axs[1]
    ax2.plot(env_data_df['step'], env_data_df['Inventory'], label='Inventory', color='blue', linewidth=2)
    ax2.plot(env_data_df['step'], env_data_df['netWorth'], label='Net Worth', color='orange', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Value')
    ax2.set_title('Inventory and Net Worth Over Time')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"Trader_{trader_id}_BuySell_Inventory_NetWorth.png"))
    plt.close(fig)

def plot_agent_prices_reward(env_data_df, messages_df, reward_df, output_dir, trader_id=10):
    """Plots agent bid/ask offers with midprice, best bid/ask, and reward."""
    agent_messages = messages_df[messages_df['TID'] == trader_id]
    buy_offers = agent_messages[agent_messages['Side'] == 1]
    sell_offers = agent_messages[agent_messages['Side'] == -1]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, height_ratios=[2, 1])

    # Top subplot: Agent Bid/Ask Offers with Midprice 
    ax1 = axs[0]
    ax1.plot(env_data_df['step'], env_data_df['averageMidprice'], label='Average Midprice', color='green', linewidth=2)
    ax1.plot(buy_offers['step'], buy_offers['Price'], color='blue', label='Agent Bid Offer', linewidth=2, linestyle='--')
    ax1.plot(sell_offers['step'], sell_offers['Price'], color='red', label='Agent Ask Offer', linewidth=2, linestyle='--')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Trader {trader_id} Bid and Ask Offers with LOB')
    ax1.legend()
    ax1.grid(True)

    # Bottom subplot: Reward Over Time
    ax2 = axs[1]
    ax2.plot(reward_df['step'], reward_df['Reward'], label='Reward', color='purple', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Over Time')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"Trader_{trader_id}_AgentOffers_Reward.png"))
    plt.close(fig)

# ---- Run visualizations ----
plot_env_midprice_stats(env_data,output_dir)
plot_cumulative_lob(lob_states, output_dir,step=25)
plot_buy_sell_inventory_networth(env_data, trades_df, output_dir, trader_id)
plot_agent_prices_reward(env_data, messages_df, reward_df, output_dir, trader_id)





