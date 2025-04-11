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

import distrax

from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig
from flax import serialization
from flax.core import frozen_dict

from dataclasses import dataclass

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString




##Special class to handel the flag list, Jax String.
@jax.tree_util.register_pytree_node_class
@dataclass
class JString:
    tokens: jnp.ndarray
    length: jnp.ndarray

    def __init__(self, tokens, length=None):
        self.tokens = jnp.array(tokens)
        self.length = (
            length if length is not None
            else jnp.ones_like(tokens[:, 0]) * tokens.shape[1]
        )

    def tree_flatten(self):
        # The children are the arrays that JAX can trace.
        children = (self.tokens, self.length)
        # No auxiliary static data.
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tokens, length = children
        return cls(tokens, length)
    
#Function to process the obsveration
def handle_continuous(observation):
        return jnp.array(observation).astype(jnp.float8_e4m3b11fnuz).view(jnp.uint8).astype(jnp.int32)



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




if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"


    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 13,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*2,  
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)


    env_config_hps = [{"observation_space":"engineered",
                         "reward_space":"spooner_scaled",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":10,
                         "reference_price_portfolio_value":"mid",
                         "action_space":"fixed_quants"
                         }]
   
    env_cfg=EnvironmentConfig(**env_config_hps[0])

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
    obs, env_state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {env_state.start_index}")
    print("Time for reset: \n", time.time() - start)
    print("Inventory after reset: \n", env_state.inventory)
    print(f"Number of available windows: {env.n_windows}")


    
    #===========================================#
    #Init the pre trained model
    #======================================#
    # Load the trained model parameters 
    params_filename = "/home/duser/AlphaTrade/params_file_upbeat-sweep-1_04-08_11-16"
    with open(params_filename, 'rb') as f:
        params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())
        
    # Initialize the model
    num_tokens = 1 + env.action_space(env_params).n + 256
    config["MIN_ACTION_TOK"] = 1
    config["MAX_ACTION_TOK"] = env_cfg.n_actions

    #Load the RWKV
    RWKV, _ = get_rand_model(0, "6", 3, 256, num_tokens, dtype=jnp.float32, rwkv_type="ScanRWKV")
    #Define the forward function and jit version
    forward, params = get_ppo_agent(RWKV, params, seed=1)
    v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))


    #Get init state for training
    init_state = RWKV.default_state(params)
    #returns 0s for weights, in /home/duser/AlphaTrade/jax_rwkv/src/jax_rwkv/base_rwkv.py
    if isinstance(init_state, tuple):
        init_state = tuple([jnp.repeat(s[None], 1, axis=0) for s in init_state])
    else:
        init_state = jnp.repeat(init_state[None], 1, axis=0)
    rwkv_state = init_state

    test_steps = 15000 # Adjusted for your test case; make sure this isn't too high
    # ============================
    # Initialize data storage
    # ============================
    output_dir = 'gymnax_exchange/jaxen/Testing/output/mm'

    #Init storage
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

        #=========================#
        #tokenizer the obvs space
        #=======================#
        tokenized = handle_continuous(obs)

        #====================#
        #Evaluate policy from model
        #=====================#
        tokenized_batched = tokenized[None, :] 
        pi, _, rwkv_state =v_forward_jit(tokenized_batched, rwkv_state, params, jnp.ones(1, dtype=jnp.int32) * tokenized_batched.shape[-1])
        pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
        action = pi.sample(seed=key_policy)
        action_batched=action[None, :] 

        ##Roll net forward
        _, _, rwkv_state = v_forward_jit(action_batched, rwkv_state, params, jnp.ones(1, dtype=jnp.int32))

        #Step Env
        action= action.item()
        obs, env_state, reward, done, info = env.step(key_step, env_state, action, env_params)
        
        
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
        midprice[i]=info["end_mid_price"]
        netWorth[i]=info["netWorth"]
        average_best_bid[i]=info["average_best_bid"]
        average_best_ask[i]=info["average_best_ask"]
 
        
        
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            print(f"Episode ended at step {valid_steps}")
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
     midprice,
     netWorth,
     average_best_bid,
     average_best_ask,
     valid_steps,
     output_dir,
 )


   