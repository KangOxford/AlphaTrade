import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
from gymnax_exchange.jaxen.marl_env import MARLEnv
import dataclasses
import jax
from matplotlib.lines import Line2D



# Load data
data_path = 'gymnax_exchange/jaxen/Testing/full_logging_tests/data/marl'
output_dir='gymnax_exchange/jaxen/Testing/full_logging_tests/plotting/marl'

exec_env_data = pd.read_csv(os.path.join(data_path, 'exe_env_data_df.csv'))
mm_env_data = pd.read_csv(os.path.join(data_path, 'mm_env_data_df.csv'))
trades_df=pd.read_csv(os.path.join(data_path, 'total_trades.csv'))
messages_df =pd.read_csv(os.path.join(data_path, 'total_messages.csv'))
exe_reward_df=pd.read_csv(os.path.join(data_path, 'exe_reward_data.csv'))
mm_reward_df=pd.read_csv(os.path.join(data_path, 'mm_reward_data.csv'))
lob_states = pd.read_csv(os.path.join(data_path, 'lob_states.csv'))

#Load trhe env for env stats like tick size, keep this the same as how we define the env for the run

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


def plot_buy_sell_inventory_networth(env_data_df, trades_df, messages_df, output_dir, trader_id=10):
    """Plots buy/sell activity with midprice (arrows) with legend, inventory, and net worth."""
    mask_trades = (trades_df['TIDa'] == trader_id) | (trades_df['TIDs'] == trader_id)
    trader_trades = trades_df[mask_trades].copy()
    trader_trades['OID'] = trader_trades.apply(
        lambda row: row['OIDs'] if row['TIDs'] == trader_id else row['OIDa'], axis=1)
    merged_trades_messages = pd.merge(trader_trades, messages_df, how='left', on=['OID', 'step'])
    if 'Price' not in merged_trades_messages.columns:
        if 'Price_x' in merged_trades_messages.columns:
            merged_trades_messages['Price'] = merged_trades_messages['Price_x']
            merged_trades_messages.drop(columns=['Price_x'], inplace=True)
        elif 'Price_y' in merged_trades_messages.columns:
            merged_trades_messages['Price'] = merged_trades_messages['Price_y']
            merged_trades_messages.drop(columns=['Price_y'], inplace=True)
    merged_trades_messages = merged_trades_messages[~merged_trades_messages['Side'].isna()]
    merged_trades_messages['Side'] = merged_trades_messages['Side'].astype(int)
    buy_steps = merged_trades_messages[merged_trades_messages['Side'] == 1][['step', 'Price']]
    sell_steps = merged_trades_messages[merged_trades_messages['Side'] == -1][['step', 'Price']]

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




def plot_buy_sell_task_size(env_data_df, trades_df, messages_df, output_dir, trader_id=10):
    """Plots buy/sell activity with midprice (arrows) with legend, inventory, and net worth."""
    mask_trades = (trades_df['TIDa'] == trader_id) | (trades_df['TIDs'] == trader_id)
    trader_trades = trades_df[mask_trades].copy()
    trader_trades['OID'] = trader_trades.apply(
        lambda row: row['OIDs'] if row['TIDs'] == trader_id else row['OIDa'], axis=1)
    merged_trades_messages = pd.merge(trader_trades, messages_df, how='left', on=['OID', 'step'])
    if 'Price' not in merged_trades_messages.columns:
        if 'Price_x' in merged_trades_messages.columns:
            merged_trades_messages['Price'] = merged_trades_messages['Price_x']
            merged_trades_messages.drop(columns=['Price_x'], inplace=True)
        elif 'Price_y' in merged_trades_messages.columns:
            merged_trades_messages['Price'] = merged_trades_messages['Price_y']
            merged_trades_messages.drop(columns=['Price_y'], inplace=True)
    merged_trades_messages = merged_trades_messages[~merged_trades_messages['Side'].isna()]
    merged_trades_messages['Side'] = merged_trades_messages['Side'].astype(int)
    buy_steps = merged_trades_messages[merged_trades_messages['Side'] == 1][['step', 'Price']]
    sell_steps = merged_trades_messages[merged_trades_messages['Side'] == -1][['step', 'Price']]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, height_ratios=[2, 1])

    # Top subplot: Buy/Sell Activity with Midprice
    ax1 = axs[0]
    ax1.plot(env_data_df['step'], env_data_df['mid_price'], label='Midprice', color='green', linewidth=2)
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
    cumulative_quant = env_data_df['quant_executed'].cumsum()
    ax2.plot(env_data_df['step'], cumulative_quant, label='Cumulative Quant Executed', color='blue', linewidth=2)
    ###Add a straight line for the max task aim
    x_values = env_data_df['step']
  

    # Get the constant value
    constant_value = env.exe_env.cfg.max_task_size
    # Create a list or NumPy array of the constant value with the same length as x_values
    y_values = [constant_value] * len(x_values)

    ax2.plot(env_data_df['step'], y_values, label='Task Size', color='orange', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Value')
    ax2.set_title('Quant Executed with Task size')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"Trader_{trader_id}_BuySell_quant_exec_tasksize.png"))
    plt.close(fig)

def plot_agent_prices_reward_exe(env_data_df, messages_df, reward_df, output_dir, trader_id=10):
    """Plots agent bid/ask offers with midprice, best bid/ask, and reward."""
    agent_messages = messages_df[messages_df['TID'] == trader_id]
    buy_offers = agent_messages[agent_messages['Side'] == 1]
    sell_offers = agent_messages[agent_messages['Side'] == -1]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, height_ratios=[2, 1])

    # Top subplot: Agent Bid/Ask Offers with Midprice 
    ax1 = axs[0]
    ax1.plot(env_data_df['step'], env_data_df['mid_price'], label=' Midprice', color='green', linewidth=2)
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
    
def plot_agent_interactions(output_dir,exec_env_data, mm_env_data, trades_df, env_data_df, config):
    exe_id = config["EXE_TRADER_ID"]
    mm_id = config["MM_TRADER_ID"]

    # Find trades between the two agents
    inter_trades_mask = (
        ((trades_df['TIDa'] == exe_id) & (trades_df['TIDs'] == mm_id)) |
        ((trades_df['TIDa'] == mm_id) & (trades_df['TIDs'] == exe_id))
    )
    inter_trades = trades_df[inter_trades_mask].copy()

    # Identify trade direction from EXE's point of view
    def label_trade(row):
        if row['TIDa'] == exe_id:
            return 'EXE_sell_MM_buy'
        else:
            return 'EXE_buy_MM_sell'

    inter_trades['direction'] = inter_trades.apply(label_trade, axis=1)

    # Merge with env_data to get midprice at that step
    if 'step' not in inter_trades.columns:
        raise ValueError("trades_df must contain 'step' column for alignment")

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, height_ratios=[2, 1])

    # === Top subplot: Trade Arrows over Midprice ===
    ax1 = axs[0]
    ax1.plot(env_data_df['step'], env_data_df['mid_price'], label='Midprice', color='green', linewidth=2)

    for _, row in inter_trades.iterrows():
        step = row['step']
        price = row['Price']
        if row['direction'] == 'EXE_sell_MM_buy':
            ax1.annotate('', xy=(step, price), xytext=(step, price + 0.5),
                         arrowprops=dict(facecolor='purple', edgecolor='purple', arrowstyle='->', lw=1.5))
        else:
            ax1.annotate('', xy=(step, price), xytext=(step, price - 0.5),
                         arrowprops=dict(facecolor='orange', edgecolor='orange', arrowstyle='->', lw=1.5))

    ax1.set_ylabel('Price')
    ax1.set_title('EXE-MM Trades on Midprice')
    ax1.grid(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='>', color='w', markerfacecolor='purple', markersize=10, label='EXE Sell / MM Buy'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='orange', markersize=10, label='EXE Buy / MM Sell')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # === Bottom subplot: MM Inventory and EXE Quant Executed ===
    ax2 = axs[1]

    if 'Inventory' in mm_env_data.columns:
        ax2.plot(mm_env_data['step'], mm_env_data['Inventory'], label='MM Inventory', color='blue', linewidth=2)
    if 'quant_executed' in exec_env_data.columns:
        ax2.plot(exec_env_data['step'], exec_env_data['quant_executed'], label='EXE quant_executed', color='red', linewidth=2)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Value')
    ax2.set_title('MM Inventory and EXE Quant Executed')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"EXE_MM_Trades_Midprice_and_States.png"))
    plt.close(fig)


# ---- Run visualizations ----
plot_env_midprice_stats(mm_env_data,output_dir)
plot_cumulative_lob(lob_states, output_dir,step=25)
plot_buy_sell_inventory_networth(mm_env_data, trades_df, messages_df, output_dir, config["MM_TRADER_ID"])
plot_agent_prices_reward(mm_env_data, messages_df, mm_reward_df, output_dir, config["MM_TRADER_ID"])

plot_buy_sell_task_size(exec_env_data, trades_df, messages_df, output_dir, config["EXE_TRADER_ID"])
plot_agent_prices_reward_exe(exec_env_data, messages_df, exe_reward_df, output_dir, config["EXE_TRADER_ID"])

plot_agent_interactions(output_dir,
    exec_env_data=exec_env_data,
    mm_env_data=mm_env_data,
    trades_df=trades_df,
    env_data_df=exec_env_data,  # Use EXE's env as source of midprice
    config=config
)




