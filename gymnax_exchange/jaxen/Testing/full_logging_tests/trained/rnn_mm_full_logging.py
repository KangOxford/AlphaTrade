import jax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
import jax.numpy as jnp
import flax.linen as nn
import datetime
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Callable, Optional
import distrax
import gymnax
import functools
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)
import dataclasses

from purejaxrl.purejaxrl.wrappers import LogWrapper, FlattenObservationWrapper


#=============import the policy==============#
import jax
import jax.numpy as jnp
from flax import serialization


# ================= imports =================#


from ast import Dict
from contextlib import nullcontext
# from email import message
# from random import sample
# from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, flatten_util
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
import pandas as pd 
import matplotlib.pyplot as plt 
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.base_env import EnvParams as BaseEnvParams
from gymnax_exchange.jaxen.base_env import EnvState as BaseEnvState
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv as MarketMakingEnv
from gymnax_exchange.utils import utils
import dataclasses
from flax.core import frozen_dict
from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig





# ============================
# Configuration
# ============================




#----Define the RNN model----#

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        # Initialize the GRUCell with the hidden size.
        gru_cell = nn.GRUCell(features=ins.shape[-1])  # Use the last dimension of `ins` as the hidden size.
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[-1]),  # Use the last dimension of `ins` as the hidden size.
            rnn_state,
        )
        new_rnn_state, y = gru_cell(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Initialize the GRUCell with the hidden_size as the features argument.
        gru_cell = nn.GRUCell(features=hidden_size)
        # Use a dummy key since the default state init fn is just zeros.
        return gru_cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )
class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"


    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": -1,
        "EP_TYPE": "fixed_time",
         "NUM_ENVS": 256, 
        "EPISODE_TIME": 60*30,  
    }

    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    params_filename = "/home/duser/AlphaTrade/params_file_mild-sweep-1_04-15_12-35"
    with open(params_filename, 'rb') as f:
        params = serialization.from_bytes(frozen_dict.FrozenDict, f.read())

    env_config_hps = [{"observation_space":"engineered",
                         "reward_space":"spooner_scaled",
                         "inv_penalty":"none",
                         "n_actions":8,
                         "end_fn":"unwind_ref_price",
                         "fixed_quant_value":15,
                         "reference_price_portfolio_value":"near_touch",
                         "action_space":"fixed_quants",
                         "asymmetrically_dampened_lambda":1,
                         "inventoryPnL_lambda":0.8,
                          "debug_mode":True ########ENSURE THIS IS TRUE FOR FULL LOGGING TEST
                         }]
   
    env_cfg=EnvironmentConfig(**env_config_hps[0])

    env = MarketMakingEnv(
        cfg = env_cfg,
        key = key_reset,
        alphatradePath=config["ATFOLDER"],
        window_index=config["WINDOW_INDEX"],
        episode_time=config["EPISODE_TIME"],
         trader_unique_id = 10,
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
    # Initialize the model

    network = ActorCriticRNN(env.action_space(env_params).n, config=config)
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)

    reset_rng = jax.random.split(key_reset, config["NUM_ENVS"])


   
   

    test_steps = 15000 # Adjusted for your test case; make sure this isn't too high
  

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
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
    episode_reward=0
    done = jnp.array([False])
    for i in range(test_steps):
        # ==================== ACTION ====================
        rng, _rng = jax.random.split(rng)
        #test action
        ac_in = (obsv[jnp.newaxis, :], done[jnp.newaxis, :])
        init_hstate, pi, value = network.apply(params, init_hstate, ac_in)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
        
        # Take a step in the environment
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
    
        episode_reward += reward[1].sum()

        
        #====================#
        #== Store standard data#
        #======================#
        rewards[i] = reward[1]
        reward_portfolio_value[i] = info["reward_portfolio_value"][1]
        reward_complex[i] = info["reward_complex"][1]
        reward_spooner[i] = info["reward_spooner"][1]
        reward_spooner_scaled[i] = info["reward_spooner_scaled"][1]
        reward_spooner_damped[i] = info["reward_spooner_damped"][1]
        reward_delta_netWorth[i] = info["reward_delta_netWorth"][1]
        inventory[i] = info["inventory"][1]
        total_PnL[i] = info["total_PnL"][1]
        buyQuant[i] = info["buyQuant"][1]
        sellQuant[i] = info["sellQuant"][1]
        bid_price[i] = info["action_prices"][1][0]  # Store best ask
        ask_price[i] = info["action_prices"][1][1]
        averageMidprice[i] = info["averageMidprice"][1]  # Store mid price
        midprice[i]=info["end_mid_price"][1]
        netWorth[i]=info["netWorth"][1]
        average_best_bid[i]=info["average_best_bid"][1]
        average_best_ask[i]=info["average_best_ask"][1]


        #===============================#
        #=====Store the full logging data==#
        #================================#
        total_messages[i,:,:]=info["total_msgs"][1]
        total_trades[i,:,:]=info["trades"][1]
        lob_states[i,:]=info["lob_state"][1]     
        
        # Increment valid steps
        valid_steps += 1
        
        if done[1]:
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

