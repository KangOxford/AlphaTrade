"""
Market Making Environment for Limit Order Book with variable start time for episodes. 

University of Oxford
Corresponding Author: 
Kang Li     (kang.li@keble.ox.ac.uk)
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
Peer Nagy   (peer.nagy@reuben.ox.ac.uk)
V1.0 



Module Description
This module extends the base simulation environment for limit order books 
 using JAX for high-performance computations, specifically tailored for 
 execution tasks in financial markets. It is particularly designed for 
 reinforcement learning applications focusing on 
 optimal trade execution strategies.

Key Components
EnvState:   Dataclass to encapsulate the current state of the environment, 
            including the raw order book, trades, and time information.
EnvParams:  Configuration class for environment-specific parameters, 
            such as task details, message and book data, and episode timing.
MarketMakingEnv: Environment class inheriting from BaseLOBEnv, 
              offering specialized methods for order placement and 
              execution tasks in trading environments. 


Functionality Overview
__init__:           Initializes the execution environment, setting up paths 
                    for data, action types, and task details. 
                    It includes pre-processing and initialization steps 
                    specific to execution tasks.
default_params:     Returns the default parameters for execution environment,
                    adjusting for tasks such as buying or selling.
step_env:           Advances the environment by processing actions and market 
                    messages. It updates the state and computes the reward and 
                    termination condition based on execution-specific criteria.
reset_env:          Resets the environment to a state appropriate for a new 
                    execution task. Initializes the order book and sets initial
                    state specific to the execution context.
is_terminal:        Checks whether the current state is terminal, based on 
                    the number of steps executed or tasks completed.

action_space:       Defines the action space for execution tasks, including 
                    order types and quantities.
observation_space:  Define the observation space for execution tasks.
state_space:        Describes the state space of the environment, tailored 
                    for execution tasks with components 
                    like bids, asks, and trades.
reset_env:          Resets the environment to a specific state for execution. 
                    It selects a new data window, initializes the order book, 
                    and sets the initial state for execution tasks.
_getActionMsgs:      Generates action messages based on 
                    the current state and action. 
                    It determines the type, side, quantity, 
                    and price of orders to be executed.
                    including detailed order book information and trade history
_get_obs:           Constructs and returns the current observation for the 
                    execution environment, derived from the state.
_get_state_from_data:
_reshape_action:
_best_prices_impute
_get_reward:
name, num_actions:  Inherited methods providing the name of the environment 
                    and the number of possible actions.


                
_get_data_messages: Inherited method to fetch market messages for a given 
                    step from all available messages.
"""

# from jax import config
# config.update("jax_enable_x64",True)
# ============== testing scripts ===============
import os
import sys
import time 
import timeit
import random
import dataclasses
from ast import Dict
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import lax, flatten_util
# ----------------------------------------------
import gymnax
from gymnax.environments import environment, spaces
# sys.path.append('/Users/sasrey/AlphaTrade')
# sys.path.append('/homes/80/kang/AlphaTrade')

sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
#sys.path.append('.')
print(os.getcwd())
#print(os.listdir('/home/duser/AlphaTrade/training_oneDay/data/Flow_10'))
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
# ---------------------------------------------- 
import chex
from jax import config
import faulthandler
faulthandler.enable()
chex.assert_gpu_available(backend=None)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64",True)
config.update("jax_disable_jit", False) # use this during training
#jax.config.update("jax_log_compiles", True) use this to see when he is recompiling
# config.update("jax_disable_jit", True) # Code snippet to disable all jitting.
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())
jax.numpy.set_printoptions(linewidth=183)
# ================= imports ==================

import pandas as pd
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
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxen.base_env import EnvParams as BaseEnvParams
from gymnax_exchange.jaxen.base_env import EnvState as BaseEnvState
from gymnax_exchange.utils import utils
import dataclasses

import jax.tree_util as jtu


from gymnax_exchange.jaxob.jaxob_config import EnvironmentConfig

# Uncomment when obtained access to repository.
# from lobgen.data_processing.data_config import set_config, TokenizerConfig, get_config
# set_config(TokenizerConfig(split_vocab=True))



@struct.dataclass
class EnvState(BaseEnvState):
    best_asks: chex.Array
    best_bids: chex.Array
    init_price: int
    inventory:int
    mid_price:float
    total_PnL: float
    price_bid_passive :int
    quant_bid_passive :int
    price_ask_passive:int
    quant_ask_passive:int
    delta_time: float
    cash_balance: float

@struct.dataclass
class EnvParams(BaseEnvParams):
    pass

class MarketMakingEnv(BaseLOBEnv):
    def __init__(
            self,cfg:EnvironmentConfig, key, alphatradePath, window_index,  episode_time,
              trader_unique_id=-9999997, ep_type="fixed_time"):
        self.cfg=cfg
        super().__init__(
            cfg = cfg,
            key = key,
            alphatradePath = alphatradePath,
            window_selector = window_index,
            sliceTimeWindow = episode_time,
            trader_unique_id = trader_unique_id,
            ep_type = ep_type,
        )
        

        ##Choose observation space based on config.
        if self.cfg.observation_space == "engineered":
            self.observation_fn = self._get_obs_engineered
        elif self.cfg.observation_space == "messages":
            self.observation_fn = self._get_obs_msg
        elif self.cfg.observation_space == "messages_new_tokenizer":
            self.observation_fn = self._get_obs_msg_new_tokenizer
        else:
            raise ValueError("Invalid observation_space specified.")
        
        ##Choose get action message function based on config
        if self.cfg.action_space == "fixed_quants":
            self.action_fn = self._getActionMsgs_fixedQuant
        elif self.cfg.action_space == "fixed_prices":
            self.action_fn = self._getActionMsgs_fixedPrice
        elif self.cfg.action_space == "AvSt":
            self.action_fn = self._getActionMsgs_AvSt
        elif self.cfg.action_space == "spread_skew":
            self.action_fn = self._getActionMsgs_spread_skew
        elif self.cfg.action_space == "directional_trading":
            self.action_fn = self._getActionMsgs_directional_trading
        else:
            raise ValueError("Invalid action_space specified.")
        
        ##Choose an end function from theconfig
        if self.cfg.end_fn=="force_market_order":
            self.end_fn =self._force_market_order_if_done
        elif self.cfg.end_fn=="unwind_ref_price":
            self.end_fn=self.unwind_ref_price
        elif self.cfg.end_fn=="do_nothing":
            self.end_fn=self.end_fn_pass
      

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        base_params = super().default_params

        flat_tree = jtu.tree_flatten(base_params)[0]
        #TODO: Clean this up to not have a magic number
        # BaseEnvParams
        base_vals = flat_tree[0:5] #Considers the base parameter values other than init state.
        state_vals = flat_tree[5:] #Considers the state values

        #jax.debug.print("state_vals shapes: {}", [getattr(leaf, "shape", None) for leaf in state_vals])

        return EnvParams(
            *base_vals,
            EnvState(*state_vals),
        )




    def step_env(
        self, key: chex.PRNGKey, state: EnvState, input_action: jax.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        #=======================================#
        #====Load data messages for next step===#
        #=======================================#      
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )
   
        #=======================================#s
        #======Process agent actions ===========#
        #=======================================#
        #action = self._reshape_action(input_action, state, params,key)
        action=input_action
        action_msgs = self.get_action(action, state, params)
        action_prices = action_msgs[:, 3] #price is position 3 of msg


        #Cancel all previous agent orders each step, send fresh
        #jax.debug.print(f"time: {state.time}   ")

      
        cnl_msg_bid = job.getCancelMsgs(
            state.bid_raw_orders,
            self.trader_unique_id,
            self.cfg.num_action_messages_by_agent//2, 
            1,  # bid
            state.time[0],  # cancel_time
            state.time[1],  # cancel_time_ns
        )
        cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            self.trader_unique_id,
            self.cfg.num_action_messages_by_agent//2,
            -1,  # ask
            state.time[0],  # cancel_time
            state.time[1],  # cancel_time_ns
        )
        ##Does not work for directional trading space. Probably need to call some config checks to do this.
        
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        #jax.debug.print(f"Market Maker action msg: {action_msgs}")

        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self._filter_messages(action_msgs, cnl_msgs)
        
        #=======================================#
        #===Process all messages through book===#
        #=======================================#
        # Add to the top of the data messages
        total_messages = jnp.concatenate([cnl_msgs, action_msgs, data_messages], axis=0)
        # Save time of final message to add to state
        time = total_messages[-1, -2:]
        # To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)

        #jax.debug.print(f"Number of overall messages: {self.stepLines + self.cfg.num_messages_by_agent}")

        # Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestbids, bestasks) = job.scan_through_entire_array_save_bidask(self.cfg,key,
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines + self.cfg.num_messages_by_agent # to include our action messages increase this by cfg.num_messages_by_agent
        )
        # If best price is not available in the current step, use the last available price
        # TODO: check if we really only want the most recent stepLines prices (+1 for the additional market order)
        bestasks = self._ffill_best_prices(bestasks[-self.stepLines-self.cfg.num_messages_by_agent:], state.best_asks[-1, 0])
        bestbids = self._ffill_best_prices(bestbids[-self.stepLines-self.cfg.num_messages_by_agent:], state.best_bids[-1, 0])

        #jax.debug.print(f"bestasks shape in function: {bestasks.shape}")

        #bestasks = self._ffill_best_prices(bestasks, state.best_asks[-1, 0])
        #bestbids = self._ffill_best_prices(bestbids, state.best_bids[-1, 0])
        ##jax.debug.print(f"bestasks: {bestasks}")
        #jax.debug.print(f"bestbids: {bestbids}")
        agent_trades = job.get_agent_trades(trades, self.trader_unique_id)
        executions = self._get_executed_by_action(agent_trades, action, state,action_prices)
        executions=jnp.abs(executions)
        #=======================================#
        #===force inventory sale at episode end=#
        #=======================================#
        (asks, bids, trades), new_id_counter, new_time=self.get_episode_end_fn(key,
            bestasks, bestbids, time, asks, bids, trades, state, params)
        #bestasks = jnp.concatenate([bestasks,bestasks[:,:] ], axis=0, dtype=jnp.int32)
        #bestbids = jnp.concatenate([bestbids, bestbids[:,:]], axis=0, dtype=jnp.int32)
    
        price_bid_passive,quant_bid_passive,price_ask_passive,quant_ask_passive = self._get_pass_price_quant(state)
        # TODO: consider adding quantity before (in priority) to each price / level

        # TODO: use the agent quant identification from the separate function _get_executed_by_level instead of _get_reward
        reward, extras = self._get_reward(state, params, trades,bestasks,bestbids)
        old_time=state.time
        old_mid_price=state.mid_price
        state = EnvState(
            ask_raw_orders = asks,
            bid_raw_orders = bids,
            trades = trades,
            init_time = state.init_time,
            time = new_time,
            customIDcounter = new_id_counter,
            window_index = state.window_index,
            step_counter = state.step_counter + 1,
            max_steps_in_episode = state.max_steps_in_episode,
            start_index = state.start_index,
            best_asks = bestasks,
            best_bids = bestbids,
            init_price = state.init_price,
            mid_price=extras["mid_price"],
            inventory=extras["end_inventory"],
            total_PnL = state.total_PnL + extras["PnL"],
            cash_balance= extras["cash_balance"],
            price_bid_passive = price_bid_passive,
            quant_bid_passive = quant_bid_passive,
            price_ask_passive=price_ask_passive,
            quant_ask_passive=quant_ask_passive,            
            delta_time = new_time[0] + new_time[1]/1e9 - state.time[0] - state.time[1]/1e9,
        )
        done = self.is_terminal(state, params)
        average_best_ask = state.best_asks[-100:].mean(axis=0)[0] #// self.tick_size) * self.tick_size)
        average_best_bid = state.best_bids[-100:].mean(axis=0)[0] #// self.tick_size) * self.tick_size)
        if self.cfg.debug_mode==False:
        #### Standard logging####
            info = {
            "reward":reward,
            "reward_portfolio_value":extras["reward_portfolio_value"],
            "reward_complex":extras["reward_complex"],
            "reward_spooner":extras[ "reward_spooner"],
            "reward_spooner_damped":extras["reward_spooner_damped"],
            "reward_spooner_scaled":extras[ "reward_spooner_scaled"],
            "reward_delta_netWorth":extras["reward_delta_netWorth"],
            "window_index": state.window_index,
            "total_PnL": state.total_PnL,                           
            "current_step": state.step_counter,
            "done": done,
            "time_seconds":state.time[0],
            "inventory": state.inventory,
            "market_share":extras["market_share"],
            "buyPnL":extras["buyPnL"],
            "scaledInventoryPnL":extras["scaledInventoryPnL"],
            "netWorth":extras["netWorth"],
            "sellPnL":extras["sellPnL"],
            "buyQuant":extras["buyQuant"],
            "sellQuant":extras["sellQuant"],
            "window_index": state.window_index,
            "inventoryValue":extras["inventoryValue"],
            "other_exec_quants":extras["other_exec_quants"],
            "averageMidprice":extras["averageMidprice"],
            "end_mid_price":extras["mid_price"],
            "Step_PnL":extras["PnL"],
            "action_prices":action_prices,
            "InventoryPnL":extras["InventoryPnL"],
            "approx_realized_pnl":extras["approx_realized_pnl"],
            "approx_unrealized_pnl": extras["approx_unrealized_pnl"],
            "average_best_bid":average_best_bid,
            "average_best_ask":average_best_ask
            }  
        #Debug mode logging, log all messages, trades and the L2 state every step##
        elif self.cfg.debug_mode==True:
            lob_state = job.get_L2_state(
                                state.ask_raw_orders,  # Current ask orders
                                state.bid_raw_orders,  # Current bid orders
                                10,  # Number of levels
                                self.cfg  
                                )
            
           # jax.debug.print("l2:{}",lob_state)
            info={
                "trades":trades,
                "total_msgs":total_messages,
                "lob_state":lob_state,
            "reward":reward,
            "reward_portfolio_value":extras["reward_portfolio_value"],
            "reward_complex":extras["reward_complex"],
            "reward_spooner":extras[ "reward_spooner"],
            "reward_spooner_damped":extras["reward_spooner_damped"],
            "reward_spooner_scaled":extras[ "reward_spooner_scaled"],
            "reward_delta_netWorth":extras["reward_delta_netWorth"],
            "window_index": state.window_index,
            "total_PnL": state.total_PnL,                           
            "current_step": state.step_counter,
            "done": done,
            "time_seconds":state.time[0],
            "inventory": state.inventory,
            "market_share":extras["market_share"],
            "buyPnL":extras["buyPnL"],
            "scaledInventoryPnL":extras["scaledInventoryPnL"],
            "netWorth":extras["netWorth"],
            "sellPnL":extras["sellPnL"],
            "buyQuant":extras["buyQuant"],
            "sellQuant":extras["sellQuant"],
            "window_index": state.window_index,
            "inventoryValue":extras["inventoryValue"],
            "other_exec_quants":extras["other_exec_quants"],
            "averageMidprice":extras["averageMidprice"],
            "end_mid_price":extras["mid_price"],
            "Step_PnL":extras["PnL"],
            "action_prices":action_prices,
            "InventoryPnL":extras["InventoryPnL"],
            "approx_realized_pnl":extras["approx_realized_pnl"],
            "approx_unrealized_pnl": extras["approx_unrealized_pnl"],
            "average_best_bid":average_best_bid,
            "average_best_ask":average_best_ask,
            "best_asks":bestasks,
            "best_bids":bestbids
            }   
        else:
            raise ValueError("invalid mode")                     
        return self.get_observation(state, params, total_messages,action_prices,executions,old_time,old_mid_price), state, reward, done, info
    
    def reset_env(
            self,
            key : chex.PRNGKey,
            params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
        """ Reset the environment to init state (pre computed from data)."""
        key_, key = jax.random.split(key)
        _, state = super().reset_env(key, params)
        state = dataclasses.replace(state, cash_balance=0.0)
        # Pad best_bids and best_asks to correct shape
        num_total_msgs = self.stepLines + self.cfg.num_messages_by_agent
        best_bid = state.best_bids[-1]  # or whatever is the current best bid
        best_ask = state.best_asks[-1]
        bestbids = jnp.tile(best_bid[None, :], (num_total_msgs, 1))
        bestasks = jnp.tile(best_ask[None, :], (num_total_msgs, 1))
        state = dataclasses.replace(state, best_bids=bestbids, best_asks=bestasks)
        ##remove....
        price_bid_passive,quant_bid_passive,price_ask_passive,quant_ask_passive = self._get_pass_price_quant(state)
        state = dataclasses.replace(state, price_bid_passive=price_bid_passive, quant_bid_passive=quant_bid_passive,price_ask_passive=price_ask_passive,quant_ask_passive=quant_ask_passive)
        ##...
        blank_messages = jnp.zeros((104, 8), dtype=jnp.int32) ##Reset for the message based obs space.
        ##FIXME: The size here needs to be size of messages sent, could change.
        if self.cfg.action_space=="fixed_quants" or self.cfg.action_space=="AvSt":
            action_prices=jnp.zeros((2,1),dtype=jnp.int32) #2 trades
            exections=jnp.zeros((2,2),dtype=jnp.int32)
        elif self.cfg.action_space=="fixed_prices":
            action_prices=jnp.zeros((self.cfg.n_actions,1),dtype=jnp.int32) #2 trades
            exections=jnp.zeros((self.cfg.n_actions,2),dtype=jnp.int32)
        elif self.cfg.action_space=="spread_skew":
            action_prices=jnp.zeros((2,1),dtype=jnp.int32) #2 trades (bid and ask)
            exections=jnp.zeros((2,2),dtype=jnp.int32)
        elif self.cfg.action_space=="directional_trading":
            action_prices=jnp.zeros((2,1),dtype=jnp.int32) #1 trade (bid or ask)
            exections=jnp.zeros((2,2),dtype=jnp.int32)
        else:
            raise ValueError("Other action spaces not finished")
        
        obs = self.get_observation(state, params,blank_messages,action_prices,exections,state.time,state.mid_price)
        return obs, state
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """ Check whether state is terminal.
         For a market making task, we run untill time completes. This is hardcoded 
          as 5 seconds before the end of the episode or one step before """
        if self.ep_type == 'fixed_time':
            # TODO: make the 5 sec a function of the step size
            time_left=(params.episode_time - (state.time - state.init_time)[0] )
            #jax.debug.print("time_left :{}",time_left)
            #jax.debug.print("time :{}",state.time)
            #jax.debug.print("init_time :{}",state.init_time)
            #jax.debug.print("start_index :{}",state.start_index)
            return (
                (params.episode_time - (state.time - state.init_time)[0] <= 5)  # time over (last 5 seconds)
            )
        elif self.ep_type == 'fixed_steps':
            return (
                (state.max_steps_in_episode - state.step_counter <= 1)  # last step  
            )
        else:
            raise ValueError(f"Unknown episode type: {self.ep_type}")
   
    def _get_pass_price_quant(self, state):
        """Get price and quanitity n_ticks into books"""
        bid_passive_2=state.best_bids[-1, 0] - self.tick_size * self.cfg.n_ticks_in_book
        ask_passive_2=state.best_asks[-1, 0] + self.tick_size * self.cfg.n_ticks_in_book
        quant_bid_passive_2 = job.get_volume_at_price(state.bid_raw_orders, bid_passive_2)
        quant_ask_passive_2 = job.get_volume_at_price(state.ask_raw_orders, ask_passive_2)
        return bid_passive_2,quant_bid_passive_2,ask_passive_2,quant_ask_passive_2
    
    def _get_state_from_data(self,key,first_message,book_data,max_steps_in_episode,window_index,start_index):
        """Reset state from data"""
        base_state = super()._get_state_from_data(key,first_message, book_data, max_steps_in_episode, window_index, start_index)
        base_vals = jtu.tree_flatten(base_state)[0]
        best_bid, best_ask = job.get_best_bid_and_ask_inclQuants(self.cfg,base_state.ask_raw_orders,base_state.bid_raw_orders)
        M =jnp.float32((best_bid[0] + best_ask[0]) / 2)
        return EnvState(
            ##This is reset
            *base_vals,
            best_asks=jnp.resize(best_ask,(self.stepLines,2)),
            best_bids=jnp.resize(best_bid,(self.stepLines,2)),
            init_price=M,
            mid_price=M,
            inventory=0,
            total_PnL=0.,
            # updated on reset:
            price_bid_passive = 0,
            quant_bid_passive = 0,
            price_ask_passive=0,
            quant_ask_passive=0,
            delta_time=0.,
            cash_balance=0.0
        )
     
    def _filter_messages(
            self, 
            action_msgs: jax.Array,
            cnl_msgs: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
        """ Filter out cancelation messages, when same actions should be placed again.
            NOTE: only simplifies cancellations if new action size <= old action size.
                  To prevent multiple split orders, new larger orders still cancel the entire old order.
            TODO: consider allowing multiple split orders
            ex: at one level, 3 cancel & 1 action --> 2 cancel, 0 action
        """
        @partial(jax.vmap, in_axes=(0, None))
        def p_in_cnl(p, prices_cnl):
            return jnp.where((prices_cnl == p) & (p != 0), True, False)
        def matching_masks(prices_a, prices_cnl):
            res = p_in_cnl(prices_a, prices_cnl)
            return jnp.any(res, axis=1), jnp.any(res, axis=0)
        @jax.jit
        def argsort_rev(arr):
            """ 'arr' sorted in descending order (LTR priority tie-breaker) """
            return (arr.shape[0] - 1 - jnp.argsort(arr[::-1]))[::-1]
        @jax.jit
        def rank_rev(arr):
            """ Rank array in descending order, with ties having left-to-right priority. """
            return jnp.argsort(argsort_rev(arr))
        
        # jax.debug.print("action_msgs\n {}", action_msgs)
        # jax.debug.print("cnl_msgs\n {}", cnl_msgs)

        a_mask, c_mask = matching_masks(action_msgs[:, 3], cnl_msgs[:, 3])
        # jax.debug.print("a_mask \n{}", a_mask)
        # jax.debug.print("c_mask \n{}", c_mask)
        # jax.debug.print("MASK DIFF: {}", a_mask.sum() - c_mask.sum())
        
        a_i = jnp.where(a_mask, size=a_mask.shape[0], fill_value=-1)[0]
        a = jnp.where(a_i == -1, 0, action_msgs[a_i][:, 2])
        c_i = jnp.where(c_mask, size=c_mask.shape[0], fill_value=-1)[0]
        c = jnp.where(c_i == -1, 0, cnl_msgs[c_i][:, 2])
        
        # jax.debug.print("a_i \n{}", a_i)
        # jax.debug.print("a \n{}", a)
        # jax.debug.print("c_i \n{}", c_i)
        # jax.debug.print("c \n{}", c)

        rel_cnl_quants = (c >= a) * a
        # rel_cnl_quants = jnp.maximum(0, c - a)
        # jax.debug.print("rel_cnl_quants {}", rel_cnl_quants)
        # reduce both cancel and action message quantities to simplify
        action_msgs = action_msgs.at[:, 2].set(
            action_msgs[:, 2] - rel_cnl_quants[rank_rev(a_mask)])
            # action_msgs[:, 2] - rel_cnl_quants[utils.rank_rev(a_mask)])
        # set actions with 0 quant to dummy messages
        action_msgs = jnp.where(
            (action_msgs[:, 2] == 0).T,
            0,
            action_msgs.T,
        ).T
        cnl_msgs = cnl_msgs.at[:, 2].set(cnl_msgs[:, 2] - rel_cnl_quants[rank_rev(c_mask)])
            # cnl_msgs[:, 2] - rel_cnl_quants[utils.rank_rev(c_mask)])
        # jax.debug.print("action_msgs NEW \n{}", action_msgs)
        # jax.debug.print("cnl_msgs NEW \n{}", cnl_msgs)

        return action_msgs, cnl_msgs

    def _ffill_best_prices(self, prices_quants, last_valid_price):
        def ffill(arr, inval=-1):
            """ Forward fill array values `inval` with previous value """
            def f(prev, x):
                new = jnp.where(x != inval, x, prev)
                return (new, new)
            # initialising with inval in case first value is already invalid
            _, out = jax.lax.scan(f, inval, arr)
            return out

        # if first new price is invalid (-1), copy over last price
        prices_quants = prices_quants.at[0, 0:2].set(
            jnp.where(
                # jnp.repeat(prices_quants[0, 0] == -1, 2),
                prices_quants[0, 0] == -1,
                jnp.array([last_valid_price, 0]),
                prices_quants[0, 0:2]
            )
        )
        # set quantity to 0 if price is invalid (-1)
        prices_quants = prices_quants.at[:, 1].set(
            jnp.where(prices_quants[:, 0] == -1, 0, prices_quants[:, 1])
        )
        # forward fill new prices if some are invalid (-1)
        prices_quants = prices_quants.at[:, 0].set(ffill(prices_quants[:, 0]))
        # jax.debug.print("prices_quants\n {}", prices_quants)
        return prices_quants
    

    ###########Functions for the new tokenizer#####################
    def locate_type_4(self, total_messages, trades):
        """
        Replace values in column a0 of total_messages(type) with 4 (execution) if the value in column 4 (OID)
        appears in column 3 (OID agr) of the trades array. Also resizes the array to accommodate both messages and trades.
        
        Args:
            total_messages: JAX array
            trades: JAX array 
            
        Returns:
            messages: JAX array with masked values in column 0 and proper sizing
        """
        # Initialize result with maximum possible size
        max_output_size = total_messages.shape[0]# + trades.shape[0]
        result = jnp.zeros((max_output_size, total_messages.shape[1]))
        
        # Copy the total_messages into the result array
        result = result.at[:total_messages.shape[0], :].set(total_messages)
        
        # Extract the relevant columns
        trades_col3 = trades[:, 3]  # Column 3=aggresive OID
        messages_col4 = total_messages[:, 4]  # Column 4 =OID
        
        # Create a mask for each message indicating whether its column 4 value 
        # is in trades column 3
        comparison_matrix = messages_col4[:, None] == trades_col3[None, :]
        mask = jnp.any(comparison_matrix, axis=1)
        
        # Update column 0 values to 4 where mask is True
        result = result.at[:total_messages.shape[0], 0].set(
            jnp.where(mask, 4, total_messages[:, 0])
        )
        
        return jnp.array(result, dtype=jnp.int32)
        
    


    def locate_type_4_real(self, total_messages, trades):
        """
        JIT-compatible implementation for processing trades and creating type 4 messages.
        
        Args:
            total_messages: JAX array (shape: [M, 8])
            trades: JAX array (shape: [N, 8])
            
        Returns:
            JAX array with updated messages and additional type 4 messages.
        """
        # Define the maximum possible size of the output
        max_output_size = total_messages.shape[0] + trades.shape[0]
        result = jnp.zeros((max_output_size, total_messages.shape[1]))
        
        # Initialize with original messages
        result = result.at[:total_messages.shape[0]].set(total_messages)
        
        def outer_loop(carry, i):
            result, current_idx = carry
            msg = total_messages[i]
            message_oid = msg[4]
            original_pos = i
            current_pos = current_idx
            
            # Copy the original message to the output position
            result = result.at[current_pos].set(msg)
            current_pos += 1
            
            # Track remaining quantity
            remaining_qty = msg[2]
            
            def inner_loop(carry, j):
                result, current_pos, remaining_qty = carry
                trade = trades[j]
                
                # Check if this trade matches our message
                is_match = trade[3] == message_oid
                
                # Calculate the quantity to trade
                trade_qty = trade[2]
                used_qty = jnp.minimum(remaining_qty, trade_qty)
                
                # Create a type 4 message for this trade
                type4_msg = msg.at[0].set(4.0).at[2].set(used_qty * is_match)
                
                # Conditionally add the type 4 message
                result = result.at[current_pos].set(
                    jnp.where(is_match, type4_msg, result[current_pos])
                )
                
                # Update position and remaining quantity conditionally
                current_pos = current_pos + is_match
                remaining_qty = remaining_qty - (used_qty * is_match)
                
                return (result, current_pos, remaining_qty), None
            
            # Process all trades against this message
            (result, current_pos, remaining_qty), _ = jax.lax.scan(
                inner_loop,
                (result, current_pos, remaining_qty),
                jnp.arange(trades.shape[0])
            )
            
            # Update the original message quantity
            result = result.at[original_pos, 2].set(remaining_qty)
            
            return (result, current_pos), None
        
        # Process all messages
        (result, _), _ = jax.lax.scan(
            outer_loop,
            (result, total_messages.shape[0]),
            jnp.arange(total_messages.shape[0])
        )
        
        # Remove zero rows - in a JIT-compatible way
        is_nonzero = jnp.any(result != 0, axis=1)
        valid_count = jnp.sum(is_nonzero)
        
        # Create a properly sized output array
        final_result = jnp.zeros((valid_count, total_messages.shape[1]))
        
        def copy_valid_rows(carry, idx):
            final_result, count = carry
            
            def check_row(carry, row_idx):
                count, valid_idx, is_valid = carry
                row_valid = is_nonzero[row_idx]
                next_valid_idx = valid_idx + row_valid
                next_count = count - row_valid
                
                # When we find a valid row and this is the one we want
                is_target = (count == 1) & row_valid
                
                return (next_count, next_valid_idx, is_target), row_idx
            
            # Find the row we want
            (_, _, _), row_idx = jax.lax.scan(
                check_row,
                (idx + 1, 0, False),  # Start count at idx+1 to find the idx-th valid row
                jnp.arange(max_output_size)
            )
            
            # Extract the last value which should be our target row
            target_row = row_idx[-1]
            
            # Copy the row
            final_result = final_result.at[idx].set(result[target_row])
            
            return (final_result, count), None
        
        # Fill the output array with valid rows
        (final_result, _), _ = jax.lax.scan(
            copy_valid_rows,
            (final_result, valid_count),
            jnp.arange(valid_count)
        )
        
        return final_result



    def calculate_row_wise_differences_time(self,input_array, old_ts,old_tns):
            """
            Calculate row-wise differences for columns 6 and 7, 
            with first row difference relative to initial time.
            
            Args:
                input_array: JAX array 
                initial_time: Initial time to calculate first row's difference
            
            Returns:
                Updated input array with row-wise differences
            """
           
            # Create a copy of the input array to avoid modifying the original
            updated_array = input_array.copy()

            
            # Extract columns 6 and 7 (indices 6 and 7)
            col6_values = input_array[:, 6]
            col7_values = input_array[:, 7]
            
            # Calculate row-wise differences for column 6
            col6_differences = jnp.zeros_like(col6_values)
            col6_differences = col6_differences.at[0].set(col6_values[0] - old_ts)
            col6_differences = col6_differences.at[1:].set(col6_values[1:] - col6_values[:-1])
            
            # Calculate row-wise differences for column 7
            col7_differences = jnp.zeros_like(col7_values)
            col7_differences = col7_differences.at[0].set(col7_values[0] - old_tns)
            col7_differences = col7_differences.at[1:].set(col7_values[1:] - col7_values[:-1])
            
            # Replace columns 6 and 7 with the calculated differences
            updated_array = updated_array.at[:, 6].set(col6_differences)
            updated_array = updated_array.at[:, 7].set(col7_differences)
            
            return updated_array
    
    

    def calculate_row_wise_differences_midprice(self, mid_price_array, start_price,n_cancels):
        """
        Calculate row-wise differences for a 1D price array,
        with the first difference relative to the initial price.

        Args:
            mid_price_array: JAX 1D array of prices
            start_price: Initial price to calculate the first row's difference

        Returns:
            Updated JAX 1D array with row-wise differences
        """
        # Create a copy of the input array to avoid modifying the original
        updated_array = mid_price_array.copy()

        # Calculate row-wise differences
        updated_array = updated_array.at[:n_cancels].set(0) #Cancels by agent dont shift mid price
        updated_array = updated_array.at[n_cancels].set(mid_price_array[n_cancels] - start_price)
        updated_array = updated_array.at[n_cancels + 1:].set(mid_price_array[n_cancels + 1:] - mid_price_array[n_cancels:-1])  # Subsequent differences

        return updated_array

    def fill_trailing_zeros(self,arr):
        """Helper funcition to fill padding with the last value before padding"""
        # Find indices of non-zero elements
        non_zero_indices = jnp.where(arr != 0, size=arr.size, fill_value=-1)[0]
        
        # Get the last non-zero index
        last_non_zero_index = jnp.max(non_zero_indices)
        
        # Retrieve the last non-zero value
        last_non_zero_value = arr[last_non_zero_index]
        
        # Create a mask for trailing zeros
        trailing_zeros_mask = jnp.arange(arr.size) > last_non_zero_index
        
        # Replace trailing zeros with the last non-zero value
        filled_arr = jnp.where(trailing_zeros_mask, last_non_zero_value, arr)
        
        return filled_arr



    
    def renumber_order_ids(self, data_messages, customIDcounter):
        """
        Renumber columns 4 and 5 of data_messages with incrementing IDs.
        
        Args:
            data_messages: JAX array of messages
            customIDcounter: Integer counter that changes between steps
        
        Returns:
            Updated data_messages with renumbered columns
        """            
        ##Keep counting from the start, inclusive of padding...
        next_order_ID =  customIDcounter * (data_messages.shape[0])
        num_messages = data_messages.shape[0]
        
        # Create the sequence by adding the offset to a range
        new_order_ids = jnp.arange(num_messages) + next_order_ID
        
        # Update the messages
        updated_messages = data_messages.at[:, 4].set(new_order_ids)
        updated_messages = updated_messages.at[:, 4].set(new_order_ids)
        return updated_messages
 
    def _get_executed_by_price(self, agent_trades: jax.Array) -> jax.Array:
        """ 
        Get executed quantity by price from trades. Results are sorted by increasing price. 
        NOTE: this will not work for aggressive orders eating through the book (size limited by actions)
        TODO: make this more general for aggressive actions?
        """
        if self.cfg.action_type=="fixed_quants":
            num_trades=2
        elif self.cfg.action_type =="fixed_price":
            num_trades=self.cfg.n_actions+1
        else:
            raise ValueError("Other Action spaces not yet implemented")
        price_levels, r_idx = jnp.unique(
            agent_trades[:, 0], return_inverse=True, size=num_trades+1, fill_value=0)
        quant_by_price = jax.ops.segment_sum(jnp.abs(agent_trades[:, 1]), r_idx, num_segments=num_trades+1)
        price_quants = jnp.vstack((price_levels[1:], quant_by_price[1:])).T
        return price_quants
    
    def _get_executed_by_level(self, agent_trades: jax.Array, actions: jax.Array, state: EnvState) -> jax.Array:
        """ Get executed quantity by level from trades. Results are sorted from aggressive to passive
            using previous actions. (0 actions are skipped)
            NOTE: this will not work for aggressive orders eating through the book (size limited by actions)
            TODO: make this more general for aggressive actions?
            UPDATE FOR MM_Env: leave in order?
        """
       # is_sell_task = state.is_sell_task
        price_quants = self._get_executed_by_price(agent_trades)
        # sort from aggr to passive
        #price_quants = jax.lax.cond(
           # is_sell_task,
        #    lambda: price_quants,
         #   lambda: price_quants[::-1],  # for buy task, most aggressive is highest price
        #)
        #put executions in non-zero action places (keeping the order)
        price_quants = price_quants[jnp.argsort(jnp.argsort(actions <= 0))]
        return price_quants
    
    def _get_executed_by_action(self, agent_trades: jax.Array, actions: jax.Array, state: EnvState,action_prices:jax.Array) -> jax.Array:
        """ Get executed quantity by level from trades. 
        """
        #TODO: This will have an issue if we buy and sell at the same price. This should be avoided anyway.
        #TODO: Put in a safe guard for that.
        def find_index_safe(x, action_prices):
            # Create a mask for matching prices
            match_mask = action_prices == x
            has_match = jnp.any(match_mask)
            first_match = jnp.argmax(match_mask)  # Returns the first index of True, or 0 if no match
            return jax.lax.cond(
                has_match,
                lambda _: first_match,  # Return the index if a match exists
                lambda _: -1,           # Return -1 otherwise
                operand=None
            )

        # Map prices to indices
        price_to_index = jax.vmap(lambda x: find_index_safe(x, action_prices))(agent_trades[:, 0])
        #jax.debug.print("action_prices:{}",action_prices)
        #jax.debug.print("agent_trades :{}",agent_trades)

        # Create masks for valid indices
        valid_indices = price_to_index >= 0
        num_prices=self.cfg.num_action_messages_by_agent
        #if self.cfg.action_space == "fixed_quants" or self.cfg.action_space=="AvSt":
        #    num_prices = 2 #2 trades for this setup.
        #elif self.cfg.action_space=="fixed_prices":
        #    num_prices=self.cfg.n_actions
        #elif self.cfg.action_space=="spread_skew":
        #    num_prices = 2  # 2 trades (bid and ask)
        #elif self.cfg.action_space=="directional_trading":
        #    num_prices = 1  # 1 trade (bid or ask)
        #else:
        #    raise ValueError("Invalid action space specified")

        # Mask trades and indices instead of boolean indexing
        valid_trades = jnp.where(valid_indices, agent_trades[:, 1], 0)
        #jax.debug.print("valid_trades:{}",valid_trades)
        valid_price_to_index = jnp.where(valid_indices, price_to_index, 0)

        # Sum trades by price level
        executions = jax.ops.segment_sum(valid_trades, valid_price_to_index, num_segments=num_prices)
       # Create a 2D array with price levels and corresponding trade quantities
        price_quantity_pairs = jnp.stack([action_prices, executions], axis=-1)

        # Optionally, you can print or debug the final result
        #jax.debug.print("Price and Quantity Pairs: {}", price_quantity_pairs)

        return price_quantity_pairs
      
    
    def _getActionMsgs_fixedQuant(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''Transform discrete action into bid and ask order messages based on current best prices.'''
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((state.best_asks[-1][0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-1][0] // self.tick_size) * self.tick_size)
        
        # Define mappings for each action: [0-7]
        bid_offsets = jnp.array([0, 1, 2, 3, 0, 2, 1, 4], dtype=jnp.int32)
        ask_offsets = jnp.array([0, 1, 2, 3, 2, 0, 4, 1], dtype=jnp.int32)
        bid_quants = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
        ask_quants = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)##config quant....
       
        tick_offset = self.cfg.n_ticks_in_book * self.tick_size  # Total price offset per direction
        
        # Get parameters for current action
        bid_offset = bid_offsets[action]
        ask_offset = ask_offsets[action]
        bid_quant = bid_quants[action]*self.cfg.fixed_quant_value
        ask_quant = ask_quants[action]*self.cfg.fixed_quant_value
        
        # Calculate prices with bounds checking
        bid_price = best_bid - bid_offset * tick_offset
        ask_price = best_ask + ask_offset * tick_offset
        bid_price = jnp.maximum(bid_price, 0) 
        ask_price = jnp.maximum(bid_price+self.cfg.n_ticks_in_book * self.tick_size, ask_price)
        
        
        # --------------- Construct messages ---------------#
        # Message components (2 messages: bid then ask)
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, self.trader_unique_id, dtype=jnp.int32)
        
        # Generate unique order IDs
        base_id = self.trader_unique_id + state.customIDcounter
        order_ids = base_id + jnp.array([0, 1], dtype=jnp.int32)
        
        # Time fields (replicated for each message)
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )
        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        return action_msgs
    
    def _getActionMsgs_AvSt(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''AvST action space: Discrete selections to paramterise K in the AvSt forumla.
        0-7, with lower giving more aggresive bid and asks
        '''
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((state.best_asks[-1][0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-1][0] // self.tick_size) * self.tick_size)
        mid_price = (best_ask + best_bid) // 2

        #Select aaggresion parameter
        gamma_values = jnp.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20], dtype=jnp.float32)  # Risk aversion
        gamma = gamma_values[action]

        #Estimate K paramter from data
        executed = jnp.where((state.trades[:, 0] >= 0)[:, jnp.newaxis], state.trades, 0)
        market_order=executed.shape[0]
        k = (market_order)/state.delta_time+0.1#ensure non zero for div later

        # Market volatility estimation (rolling standard deviation of mid-price)
        mid_price_history = ((state.best_asks[-100:]+state.best_bids[-100:])/2)
        returns= jnp.log(mid_price_history[1:] / mid_price_history[:-1])
        vol = jnp.std(returns)
        vol=jnp.clip(vol,0.001,0.2)#clip for large data point smoothing
        varaince=vol**2 #variance
        #jax.debug.print("vol:{}",vol)
        
        #Get time until ep end
        time_left = params.episode_time - (state.time - state.init_time)[0]
        normalized_time = time_left / params.episode_time

        #Reservation price
        res_price = (mid_price - ((state.inventory)) * gamma * (varaince) * normalized_time)

        #Spread
        spread = (gamma*varaince*normalized_time + (2/gamma) * jnp.log(1 + gamma/k))*self.tick_size
        spread=jnp.clip(spread,self.tick_size,self.cfg.maxint)#make sure spread is at least a tick

        bid_price= res_price-spread
        ask_price= res_price+spread

        # Ensure valid price bound 
        bid_price = jnp.clip(bid_price, 0, self.cfg.maxint) 
        ask_price = jnp.clip(ask_price,  0, self.cfg.maxint) 

        #Ensure ints of tick_size
        bid_price=((bid_price) // self.tick_size * self.tick_size).astype(jnp.int32)
        ask_price=((ask_price) // self.tick_size * self.tick_size).astype(jnp.int32)

        # Set fixed quantities
        bid_quant = self.cfg.fixed_quant_value
        ask_quant = self.cfg.fixed_quant_value

        # Construct order messages
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1 = limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1 = bid, -1 = ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, self.trader_unique_id, dtype=jnp.int32)

        # Generate order IDs
        base_id = self.trader_unique_id + state.customIDcounter
        order_ids = base_id + jnp.array([0, 1], dtype=jnp.int32)

        # Time fields
        times = jnp.resize(state.time + params.time_delay_obs_act, (2, 2))

        # Stack messages
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids, trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)

        #Debug prints:
        #jax.debug.print("vol:{}",vol)
        #jax.debug.print("Inv:{}",state.inventory)
        #jax.debug.print("best bid:{}",state.best_bids[-1][0])
        #jax.debug.print("best ask:{}",state.best_asks[-1][0])
        #jax.debug.print("res price:{}",res_price)
        #jax.debug.print("spread:{}",spread)
        #jax.debug.print("mid price :{}",mid_price)
        #jax.debug.print("msg:{}",action_msgs)
        return action_msgs
    
    def _getActionMsgs_fixedPrice(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''Shape the action quantities in to messages sent the order book at the 
        prices levels determined from the orderbook'''
        def normal_quant_price(price_levels: jax.Array, action: jax.Array):
            def combine_mid_nt(quants, prices):
                quants = quants \
                    .at[2].set(quants[2] + quants[1]) \
                    .at[1].set(0)
                prices = prices.at[1].set(-1)
                return quants, prices

            quants = action.astype(jnp.int32)          

            if self.cfg.n_actions == 4:
                # if mid_price == near_touch_price: combine orders into one
                return jax.lax.cond(
                    price_levels[1] == price_levels[2],
                    combine_mid_nt,
                    lambda q, p: (q, p),
                    quants, prices
                )
            else:
                return quants, prices
        
            
        def buy_task_prices(best_ask, best_bid):
            FT = ((best_ask) // self.tick_size * self.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            BI = best_bid + self.tick_size*self.cfg.n_ticks_in_book #BID inside, slightly more aggresive buying
            NT = best_bid
            PP = best_bid - self.tick_size*self.cfg.n_ticks_in_book
            MKT = self.cfg.maxint
            if action.shape[0]//2 == 4:
                return FT, M, NT, PP, MKT
            elif action.shape[0]//2 == 3:
                return BI, NT, PP, MKT
            elif action.shape[0]//2 == 2:
                return NT, PP, MKT
            elif action.shape[0]//2 == 1:
                return NT, MKT

        def sell_task_prices(best_ask, best_bid):
            # FT = best_bid
            FT = ((best_bid) // self.tick_size * self.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            AI = best_ask - self.tick_size*self.cfg.n_ticks_in_book #Ask inside, slightly more aggresive selling
            NT = best_ask
            PP = best_ask + self.tick_size*self.cfg.n_ticks_in_book
            MKT = 0
            if action.shape[0]//2 == 4:
                return FT, M, NT, PP, MKT
            elif action.shape[0]//2 == 3:
                return AI, NT, PP, MKT
            elif action.shape[0]//2 == 2:
                return NT, PP, MKT
            elif action.shape[0]//2 == 1:
                return NT, MKT

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types = jnp.ones((self.cfg.n_actions,), jnp.int32)
        sides_bids = jnp.ones((self.cfg.n_actions // 2,), jnp.int32)  # Use integer division to ensure result is an int
        sides_asks = (-1) * jnp.ones((self.cfg.n_actions // 2,), jnp.int32)
        sides = jnp.concatenate([sides_bids, sides_asks])
        trader_ids = jnp.ones((self.cfg.n_actions,), jnp.int32) * self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids = (jnp.ones((self.cfg.n_actions,), jnp.int32) *
                    (self.trader_unique_id + state.customIDcounter)) \
                    + jnp.arange(0, self.cfg.n_actions) #Each message has a unique ID
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (self.cfg.n_actions, 2)
        )
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
   
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((state.best_asks[-1][0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-1][0] // self.tick_size) * self.tick_size)


        sell_levels=sell_task_prices(best_ask, best_bid)
        sell_levels = jnp.array(sell_levels[:-1]) #Drop Market price

        buy_levels=buy_task_prices(best_ask, best_bid)
        buy_levels = jnp.array(buy_levels[:-1])

        price_levels=jnp.concatenate([buy_levels,sell_levels])
        

        # --------------- 02 info for deciding prices ---------------
        quants = action.astype(jnp.int32)
        prices=price_levels
     
        #quants, prices = normal_quant_price(price_levels, action)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        #jax.debug.print('action_msgs\n {}', action_msgs)
        return action_msgs
        # ============================== Get Action_msgs ==============================

    def _getActionMsgs_spread_skew(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''Transform discrete action into bid and ask order messages based on spread and skew parameters.
        Actions [0-5] map to combinations of:
        spread: 0 = tight spread, 1 = wide spread
        skew: 0 = bid skew, 1 = neutral, 2 = ask skew
        
        Mapping:
        0: tight spread, bid skew
        1: tight spread, neutral 
        2: tight spread, ask skew
        3: wide spread, bid skew
        4: wide spread, neutral
        5: wide spread, ask skew
        '''
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((state.best_asks[-1][0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-1][0] // self.tick_size) * self.tick_size)
        mid_price = (best_ask + best_bid) / 2


        #jax.debug.print("Best Ask: {}, Best Bid: {}, Mid Price: {}", best_ask, best_bid, mid_price)
        #jax.debug.print("best asks: {}", state.best_asks)
        #jax.debug.print("best bids: {}", state.best_bids)
        
        # Get current spread
        current_spread = best_ask - best_bid
        
        # Map action to spread and skew parameters
        # action = spread_type * 3 + skew_type
        spread_type = action // 3  # 0 = tight, 1 = wide
        skew_type = action % 3     # 0 = neutral, 1 = ask skew, 2 = bid skew
        
        # Define spread multipliers
        # Tight spread = 1.0 * current_spread
        # Wide spread = 2.0 * current_spread
        spread_multiplier = jnp.where(spread_type == 0, 1.0, 2.0)
        new_spread = current_spread * spread_multiplier
        
        # Define skew amounts (in ticks)
        # The skew will shift the mid price by this many ticks
        skew_ticks = jnp.where(skew_type == 0, -self.cfg.n_ticks_in_book,   # bid skew - shift down by n_ticks_in_book ticks
                            jnp.where(skew_type == 1, 0,  # neutral - no skew
                            self.cfg.n_ticks_in_book))   # ask skew - shift up by n_ticks_in_book ticks


        # Calculate skewed mid price
        skewed_mid = mid_price + skew_ticks 
        
        # Calculate final bid and ask prices
        half_spread = new_spread // 2
        bid_price = skewed_mid - half_spread
        ask_price = skewed_mid + half_spread
        
        # Ensure prices are a multiple of tick size
        bid_price = (bid_price // self.tick_size) * self.tick_size
        ask_price = (ask_price // self.tick_size) * self.tick_size
        
        # Set fixed quantities
        bid_quant = self.cfg.fixed_quant_value
        ask_quant = self.cfg.fixed_quant_value
        
        # Construct order messages
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1 = limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1 = bid, -1 = ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, self.trader_unique_id, dtype=jnp.int32)
        
        # Generate order IDs
        base_id = self.trader_unique_id + state.customIDcounter
        order_ids = base_id + jnp.array([0, 1], dtype=jnp.int32)
        
        # Time fields
        times = jnp.resize(state.time + params.time_delay_obs_act, (2, 2))
        
        # Stack messages
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids, trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        
        # Debug prints
        #jax.debug.print("Action: {}", action)
        #jax.debug.print("Best Ask: {}, Best Bid: {}, Mid Price: {}", best_ask, best_bid, mid_price)
        #jax.debug.print("Spread Type: {}, Skew Type: {}", spread_type, skew_type)
        #jax.debug.print("Current Spread: {}, New Spread: {}", current_spread, new_spread)
        #jax.debug.print("Skew Ticks: {}, Skewed Mid: {}", skew_ticks, skewed_mid)
        #jax.debug.print("Final Bid Price: {}, Final Ask Price: {}", bid_price, ask_price)
        #jax.debug.print("Final Messages:\n{}", action_msgs)
        
        return action_msgs



    def _getActionMsgs_directional_trading(self, action: jax.Array, state: EnvState, params: EnvParams):
        '''Action space for directional trading. The agent can either:
            - Do nothing (action = 0)
            - Buy at best ask (action = 1)
            - Sell at best bid (action = 2)
        
        Always sends two messages for compatibility with message filtering
        '''
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((state.best_asks[-1][0] // self.tick_size) * self.tick_size)
        best_bid = jnp.int32((state.best_bids[-1][0] // self.tick_size) * self.tick_size)
        
        # Debug prints
        #jax.debug.print("Directional Trading Action: {}", action)
        #jax.debug.print("Best Ask: {}, Best Bid: {}", best_ask, best_bid)
        
        quant = self.cfg.fixed_quant_value
        
        # Define mappings for each action to bid/ask orders
        # For action 0 (do nothing): no orders
        # For action 1 (buy at ask): only buy order
        # For action 2 (sell at bid): only sell order
        
        # Define which actions should place orders on each side
        bid_active = jnp.array([0, 1, 0], dtype=jnp.int32)[action]
        ask_active = jnp.array([0, 0, 1], dtype=jnp.int32)[action]
        
        # Message components (always 2 messages: bid then ask)
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        
        # Set quantities based on action - zero quantity for inactive sides
        bid_quant = bid_active * quant
        ask_quant = ask_active * quant
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        
        # Set prices
        prices = jnp.array([best_ask, best_bid], dtype=jnp.int32)
        trader_ids = jnp.full(2, self.trader_unique_id, dtype=jnp.int32)
        
        # Generate unique order IDs
        base_id = self.trader_unique_id + state.customIDcounter
        order_ids = base_id + jnp.array([0, 1], dtype=jnp.int32)
        
        # Time fields (replicated for each message)
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )
        
        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids, trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        
        # Debug print final messages
        #jax.debug.print("Final Action Messages:\n{}", action_msgs)
        return action_msgs



    #===================End Episode Functions=============================================#
    def end_fn_pass(self,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: EnvState,
            params: EnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:
        if self.cfg.action_space=="fixed_quants"or self.cfg.action_space=="AvSt":
            id_counter = state.customIDcounter + 2 + 1 ## we send 2 messages here
        elif self.cfg.action_space=="fixed_prices":
            id_counter = state.customIDcounter + self.cfg.n_actions + 1 ## we send n_messages here
        elif self.cfg.action_space=="spread_skew":
            id_counter = state.customIDcounter + 2 + 1  # 2 messages for bid and ask
        elif self.cfg.action_space=="directional_trading":
            id_counter = state.customIDcounter + 1 + 1  # 1 message
        else:
            raise ValueError("Action space not implemented yet")
        time = time + params.time_delay_obs_act
        return (asks, bids, trades),  id_counter, time

    def unwind_ref_price(self,
            bestasks: jax.Array,
            bestbids: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: EnvState,
            params: EnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:   
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        '''Function to create an artifical trade which liquidates the agent's position.
            cfg.rerefernce price sets the price of the trade


            NOTE: The prices in the trade here are NOT normalised by tick size. This is correct, as it is "as if" we sent
            and order with these prices. The get_reward, will see the trade, and normalsie the prices following. No change needed.
        '''
             
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = (self.trader_unique_id == executed[:, 6]) | (self.trader_unique_id == executed[:, 7]) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0) 

        #Find agent Buys and Agent sells from agent Trades:
        #The below mask puts passive buys or aggresive buys into "agent buys".
        #Logic: Q>0, TIDs=BUY; Q<0 TIDa= BUY
        mask_buy = (((agentTrades[:, 1] >= 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] < 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        mask_sell = (((agentTrades[:, 1] < 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] >= 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        agent_buys=jnp.where(mask_buy[:, jnp.newaxis], agentTrades, 0)
        agent_sells=jnp.where(mask_sell[:, jnp.newaxis], agentTrades, 0)

        #Find amount bought and sold in the step
        buyQuant=jnp.abs(agent_buys[:, 1]).sum()
        sellQuant=jnp.abs(agent_sells[:, 1]).sum()

        #Calculate the change in inventory & the new inventory
        inventory_delta = buyQuant - sellQuant
        new_inventory=state.inventory+inventory_delta
        
        #-----check if ep over-----#
        if self.ep_type == 'fixed_time':
            remainingTime = params.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= 5  # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1
        averageMidprice = ((bestbids[:, 0] + bestasks[:, 0]) / 2).mean() #should be a float

        
        new_time = time + params.time_delay_obs_act


        is_sell_task = jnp.where(new_inventory > 0, 1, 0)
        FT_price = jax.lax.cond(
            is_sell_task,
            lambda: ((bestbids[-1, 0]) // self.tick_size * self.tick_size).astype(jnp.int32),
            lambda: (( bestasks[-1, 0])// self.tick_size * self.tick_size).astype(jnp.int32),
        )

        def place_refprice_trade(trades, price, quant, time):
            '''Place a doom trade at a trade at specified price to close out our mm agent at the end of the episode.'''
            trade = job.create_trade(
                price, quant, -666666,  self.trader_unique_id + state.customIDcounter+ 1 +self.cfg.num_action_messages_by_agent, *time, -666666, self.trader_unique_id) #-66666 is an artifical OID for the artifical person we "traded with" to close our position
            trades = job.add_trade(trades, trade)
            return trades

        ##Get the price to unwind at based on the config
        if self.cfg.reference_price_portfolio_value == "mid":
            reference_price = averageMidprice
        elif self.cfg.reference_price_portfolio_value == "best_bid_ask":
            reference_price=FT_price
        elif self.cfg.reference_price_portfolio_value == "near_touch":
            # Even if we value our at the near touch price, we still want to unwind at the far touch price to be realistic
            reference_price=FT_price
        else:
            raise ValueError("Invalid reference price type.")
        
        trades = jax.lax.cond(
            ep_is_over & (jnp.abs(new_inventory) > 0),  # Check if episode is over and we still have remaining quantity
            place_refprice_trade,  # Place a midprice trade
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, reference_price, jnp.sign(new_inventory) * jnp.abs(new_inventory), new_time  # Inv +ve means incoming is sell so standing buy.
        )

        #OID logic based on config
        num_messages=self.cfg.num_action_messages_by_agent
        id_counter=state.customIDcounter +num_messages+1
        return (asks, bids, trades),  id_counter, new_time
    
    
    def _force_market_order_if_done(
            self,
            key: chex.PRNGKey,
            #quant_left: jax.Array,
            bestask: jax.Array,
            bestbid: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: EnvState,
            params: EnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:
        """ Force a market order if episode is over (either in terms of time or steps).
         Cancel all agent trades and place a market trade. If this is unmatched, cancel any remaing volume
          and place an artificial trade at a bad price.
           NOTICE,NOT REALLY USED FOR MARKET MAKING """
        
        def create_mkt_order():
            '''Create a market order by either placing a limit
            order at 0 or max int. Buy if inventory is less than zero and
            visa versa'''
            is_sell_task = jnp.where(state.inventory > 0, 1, 0)
            mkt_p = (1 - is_sell_task) * self.cfg.maxint // self.tick_size * self.tick_size
            side = (1 - is_sell_task*2)
            # TODO: this addition wouldn't work if the ns time at index 1 increases to more than 1 sec
            new_time = time + params.time_delay_obs_act
            mkt_msg = jnp.array([
                # type, side, quant, price
                #NOTE: MAKING ZERO TO TEST SELL AT MID PRICE jnp.abs(state.inventory)
                1, side, 0 , mkt_p,
                self.trader_unique_id,
                self.trader_unique_id + state.customIDcounter + self.cfg.n_actions,  # unique order ID for market order
                *new_time,  # time of message
            ])
            if self.cfg.action_space=="fixed_quants"or self.cfg.action_space=="AvSt":
                id_counter = state.customIDcounter + 2 + 1 ## we send 2 messages here
            elif self.cfg.action_space=="fixed_prices":
                id_counter = state.customIDcounter + self.cfg.n_actions + 1 ## we send n_messages here
            elif self.cfg.action_space=="spread_skew":
                id_counter = state.customIDcounter + 2 + 1  # 2 messages for bid and ask
            elif self.cfg.action_space=="directional_trading":
                id_counter = state.customIDcounter + 1 + 1  # 1 message
            else:
                raise ValueError("Action space not implemented yet")
            return mkt_msg, id_counter, new_time

        def create_dummy_order():
            '''To comply with fixed array constraints, 
            create a dummy trade when the episode is not over'''
            next_id = state.customIDcounter + self.cfg.n_actions
            return jnp.zeros((8,), dtype=jnp.int32), next_id, time 
        

        def place_doom_trade(trades, price, quant, time):
            '''Place a doom trade at a punishment price for any unmatched
            market order. If this is placed, the orderbook will be completly drained.'''
            doom_trade = job.create_trade(
                price, quant, -666666,  self.trader_unique_id + state.customIDcounter+ 1 +self.cfg.n_actions, *time, -666666, self.trader_unique_id)
            trades = job.add_trade(trades, doom_trade)
            return trades
         
        #-----check if ep over-----#
        if self.ep_type == 'fixed_time':
            remainingTime = params.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= 5  # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1

        #----filter the market or dummy order through---#
        order_msg, id_counter, time = jax.lax.cond(
            ep_is_over,
            create_mkt_order,
            create_dummy_order
        )
        #==============Cancel previous orders by the agent prior to the market order=========###
        #Cancel all previous agent orders before the market order so that we do not trade with ourselves.
        
        cnl_msg_bid = job.getCancelMsgs(
                state.bid_raw_orders,
                self.trader_unique_id,
                self.cfg.num_action_messages_by_agent//2,
                1  # bid
            )
        cnl_msg_ask = job.getCancelMsgs(
                state.ask_raw_orders,
                self.trader_unique_id,
                self.cfg.num_action_messages_by_agent//2,
                -1  # ask
            )
        
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)
        
        (asks, bids, trades), (new_bestbid, new_bestask) = job.scan_through_entire_array_save_bidask(self.cfg,key,
            cnl_msgs, 
            (asks, bids, trades),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines
        )
   
        #Filter our new message through the orderbook#
        (asks, bids, trades), (new_bestbid, new_bestask) = job.cond_type_side_save_bidask(self.cfg,
            (asks, bids, trades),
            (key,order_msg)
        )
        
        # make sure best prices use the most recent available price and are not negative
        bestask = jax.lax.cond(
            new_bestask[0] <= 0,
            lambda: jnp.array([bestask[0], 0]),
            lambda: new_bestask,
        )
        bestbid = jax.lax.cond(
            new_bestbid[0] <= 0,
            lambda: jnp.array([bestbid[0], 0]),
            lambda: new_bestbid,
        )

        #==============Cancel previous orders by the agent prior to the market order=========###
        #Cancel all previous agent orders before the doom order. This avoids the "best bid" or " best ask"
        #corresponding to the left over market price#
        cnl_msg_bid = job.getCancelMsgs(
            bids,
            self.trader_unique_id,
            1, 
            1  # bids
        )
        cnl_msg_ask = job.getCancelMsgs(
            asks,
            self.trader_unique_id,
            1,
            -1  # ask side
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        (asks, bids, trades), (new_bestbid, new_bestask) = job.scan_through_entire_array_save_bidask(self.cfg,key,
            cnl_msgs, 
            (asks, bids, trades),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines
        )
       
        # make sure best prices use the most recent available price and are not negative
        bestask = jax.lax.cond(
            new_bestask[1][0] <= 0, #Price after second cancel message
            lambda: jnp.array([bestask[0], 0]),
            lambda: new_bestask[1],
        )
        bestbid = jax.lax.cond(
            new_bestbid[1][0] <= 0,
            lambda: jnp.array([bestbid[0], 0]),
            lambda: new_bestbid[1],
        )     

        ###TODO: check matching
        mkt_exec_quant = jnp.where(
            trades[:, 3] == order_msg[5],
            jnp.abs(trades[:, 1]),  # executed quantity
            0
        ).sum()        
        # assume execution at really unfavorable price if market order doesn't execute (worst case)
        # create artificial trades for this
        quant_still_left = jnp.abs(state.inventory) - mkt_exec_quant
       # jax.debug.print('quant_still_left: {}', quant_still_left)
        # assume doom price with 25% extra cost
        is_sell_task = jnp.where(state.inventory > 0, 1, 0)

        
        doom_price = jax.lax.cond(
            is_sell_task,
            #lambda: ((0.75 * bestbid[0]) // self.tick_size * self.tick_size).astype(jnp.int32),
            #lambda: ((1.25 * bestask[0]) // self.tick_size * self.tick_size).astype(jnp.int32),
            lambda: ((bestbid[0]+bestask[0])//2 // self.tick_size * self.tick_size).astype(jnp.int32),
            lambda: ((bestbid[0]+bestask[0])//2 // self.tick_size * self.tick_size).astype(jnp.int32), #For sell at opposite test
        )
        #jax.debug.print('ep_is_over: {}; quant_still_left: {}; remainingTime: {}; doom price :{}', ep_is_over, quant_still_left, remainingTime,doom_price)
        trades = jax.lax.cond(
            ep_is_over & (quant_still_left > 0),  # Check if episode is over and we still have remaining quantity
            place_doom_trade,  # Place a doom trade with unfavorable price
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, doom_price, 0, time  # Inv +ve means incoming is sell so standing buy.
        )#jnp.sign(state.inventory) * quant_still_left
        agent_trades = job.get_agent_trades(trades, self.trader_unique_id)
       # price_quants = self._get_executed_by_price(agent_trades)
        doom_quant = ep_is_over * quant_still_left

        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant

    def _get_reward(self, state: EnvState, params: EnvParams, trades: chex.Array,bestasks :chex.Array, bestbids: chex.Array) -> jnp.int32:
        '''Return the reward. There are a few options for reward funciton and assocaited hyper parameters:
        '''
        # ====================01 get reward stats ==========================================#
        #Notice, normalise prices in reward by tick size. On state prices are not normalised 
        #Being constient with exec. Cash balance and pnl etc are normalised in state, also consitent

        # Find trades by agent vs by others
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        mask2 = (self.trader_unique_id == executed[:, 6]) | (self.trader_unique_id == executed[:, 7]) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        otherTrades = jnp.where(mask2[:, jnp.newaxis], 0, executed)
    
        #Find agent Buys and Agent sells from agent Trades:
        #The below mask puts passive buys or aggresive buys into "agent buys".
        #Logic: Q>0, TIDs=BUY; Q<0 TIDa= BUY
        mask_buy = (((agentTrades[:, 1] >= 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] < 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        mask_sell = (((agentTrades[:, 1] < 0) & (self.trader_unique_id == agentTrades[:, 6]))|((agentTrades[:, 1] >= 0)  & (self.trader_unique_id == agentTrades[:, 7])))
        agent_buys=jnp.where(mask_buy[:, jnp.newaxis], agentTrades, 0)
        agent_sells=jnp.where(mask_sell[:, jnp.newaxis], agentTrades, 0)

        #Find amount bought and sold in the step
        buyQuant=jnp.abs(agent_buys[:, 1]).sum()
        sellQuant=jnp.abs(agent_sells[:, 1]).sum()

        #Find total traded volume
        TradedVolume=buyQuant+sellQuant

        #Calculate the change in inventory & the new inventory
        inventory_delta = buyQuant - sellQuant
        new_inventory=state.inventory+inventory_delta

        #Find the new obsvered mid price at the end of the step.
        #non normalized=> going on state
        mid_price_end = (bestbids[-1][0] + bestasks[-1][0]) / 2# * self.tick_size) * self.tick_size

        #Real Revenue calcs: (actual cash flow+actual value of portfolio)
        income=(agent_sells[:, 0]* jnp.abs(agent_sells[:, 1])).sum()
        outgoing=(agent_buys[:, 0] * jnp.abs(agent_buys[:, 1])).sum() 

        #PnL,== cash balance== normalised by tick size
        PnL=(income-outgoing)/self.tick_size

        # Compute a reference price based on the config
        if self.cfg.reference_price_portfolio_value == "mid":
            reference_price = mid_price_end/self.tick_size
        elif self.cfg.reference_price_portfolio_value == "best_bid_ask":
            # For a long position, use the best bid; for a short, the best ask.
            reference_price = jax.lax.cond(new_inventory > 0,
                                        lambda: bestbids[-1][0]/self.tick_size,
                                        lambda: bestasks[-1][0]/self.tick_size)
        elif self.cfg.reference_price_portfolio_value == "near_touch":
            # For a long position, use the best ask; for a short, the best bid.
            reference_price = jax.lax.cond(new_inventory > 0,
                                        lambda: bestasks[-1][0]/self.tick_size,
                                        lambda: bestbids[-1][0]/self.tick_size)
        else:
            raise ValueError("Invalid reference price type.")

        # Keep track of overall cash balance (same as overall PnL)
        new_cash_balance = state.cash_balance + PnL
        inventoryValue=new_inventory*(reference_price)
        netWorth=new_cash_balance+inventoryValue  

        #calculate a fraction of total market activity attributable to us.
        other_exec_quants = jnp.abs(otherTrades[:, 1]).sum()
        market_share = TradedVolume / (TradedVolume + other_exec_quants)

        #=========02 Get rewards============================##

        #------------A) spooner Rewards-------------------------#       
        #Inventory PnL: 
        InventoryPnL= state.inventory*(mid_price_end-state.mid_price)/self.tick_size 
    
        #Market Making PNL:     
        averageMidprice = ((bestbids[:, 0] + bestasks[:, 0]) / 2).mean() #should be a float
        buyPnL = ((averageMidprice - agent_buys[:, 0]) * jnp.abs(agent_buys[:, 1])).sum() /self.tick_size
        sellPnL = ((agent_sells[:, 0] - averageMidprice) * jnp.abs(agent_sells[:, 1])).sum() /self.tick_size

        ##aggresive
        aggresive_buyPnL = ((bestasks[-1][0] - agent_buys[:, 0]) * jnp.abs(agent_buys[:, 1])).sum() /self.tick_size
        aggresive_sellPnL = ((agent_sells[:, 0] - bestbids[-1][0]) * jnp.abs(agent_sells[:, 1])).sum() /self.tick_size



        #A1)Spooner paper reward
        reward_spooner = aggresive_buyPnL + aggresive_sellPnL + InventoryPnL - jnp.maximum(0,InventoryPnL)

        #A2)spooner_damped
        reward_spooner_damped = aggresive_buyPnL + aggresive_sellPnL + InventoryPnL - (1-self.cfg.inventoryPnL_lambda)*jnp.maximum(0,InventoryPnL)

        #A3) Spooner Scaled
        scaledInventoryPnL=InventoryPnL//(jnp.abs(state.inventory)+1)
        reward_spooner_scaled=aggresive_buyPnL + aggresive_sellPnL+ self.cfg.inventoryPnL_lambda*(InventoryPnL - (1-self.cfg.asymmetrically_dampened_lambda)*jnp.maximum(0,InventoryPnL) )
        #----------------------B) Complex reward---------------------------------------------#
        inventoryPnL_lambda = self.cfg.inventoryPnL_lambda
        unrealizedPnL_lambda = self.cfg.unrealizedPnL_lambda
        asymmetrically_dampened_lambda = self.cfg.asymmetrically_dampened_lambda
        avg_buy_price = jnp.where(buyQuant > 0, (agent_buys[:, 0]/ buyQuant * jnp.abs(agent_buys[:, 1])).sum(), 0)  
        avg_sell_price = jnp.where(sellQuant > 0, (agent_sells[:, 0]/ sellQuant * jnp.abs(agent_sells[:, 1])).sum(), 0)
        approx_realized_pnl = jnp.minimum(buyQuant, sellQuant) * (avg_sell_price - avg_buy_price) /self.tick_size
        approx_unrealized_pnl = jnp.where( 
            inventory_delta > 0,
            inventory_delta * (averageMidprice - avg_buy_price)/self.tick_size,  # Excess buys
            jnp.abs(inventory_delta) * (avg_sell_price - averageMidprice)/self.tick_size  # Excess sells
        )
  
        reward_complex = approx_realized_pnl + unrealizedPnL_lambda * approx_unrealized_pnl +  inventoryPnL_lambda * jnp.minimum(InventoryPnL,InventoryPnL*asymmetrically_dampened_lambda) #Last term adds negative inventory PnL without dampening
    
        #--------------------C) Portfolilo Value--------------#
        reward_portfolio_value=new_inventory*(reference_price)+new_cash_balance

        #-----------------d) delta Portfolio Value--------#
        #Get old ref price
        if self.cfg.reference_price_portfolio_value == "mid":
            old_reference_price = state.mid_price/self.tick_size
        elif self.cfg.reference_price_portfolio_value == "best_bid_ask":
            # For a long position, use the best bid; for a short, the best ask. (this is realistic)
            old_reference_price = jax.lax.cond(state.inventory > 0,
                                        lambda: state.best_bids[-1][0]/self.tick_size,
                                        lambda: state.best_asks[-1][0]/self.tick_size)
        elif self.cfg.reference_price_portfolio_value == "near_touch":
            # For a long position, use the best ask; for a short, the best bid. (this is not realistic, but might be useful for training)
            old_reference_price = jax.lax.cond(state.inventory > 0,
                                        lambda: state.best_asks[-1][0]/self.tick_size,
                                        lambda: state.best_bids[-1][0]/self.tick_size)
        else:
            raise ValueError("Invalid reference price type.")
        #old net worth
        old_netWorth=old_reference_price*state.inventory+state.cash_balance
        delta_netWorth=netWorth-old_netWorth
        reward_delta_netWorth=delta_netWorth
        

        #===================== 03) Set reward based on config file==================#
        if self.cfg.reward_space == "portfolio_value":
            reward = reward_portfolio_value
        elif self.cfg.reward_space == "portfolio_value_scaled":
            reward = reward_portfolio_value/100
        elif self.cfg.reward_space == "pnl":
            reward = PnL
        elif self.cfg.reward_space == "complex":
            reward =reward_complex
        elif self.cfg.reward_space == "zero_inv":
            reward = -jnp.abs(new_inventory)
        elif self.cfg.reward_space=="spooner":
            reward=reward_spooner
        elif self.cfg.reward_space=="spooner_damped":
            reward=reward_spooner_damped
        elif self.cfg.reward_space=="spooner_scaled":
            reward=reward_spooner_scaled/10
        elif self.cfg.reward_space=="delta_netWorth":
            reward=reward_delta_netWorth
        else:
            raise ValueError("Invalid reward_space specified.")
        
        # Set inventory penalty based on config file
        if self.cfg.inv_penalty == "none":
            inv_pen = 0.0
        elif self.cfg.inv_penalty == "linear":
            inv_pen = (-1) * jnp.abs(new_inventory)
        elif self.cfg.inv_penalty == "quadratic":
            inv_pen = (-1) * (new_inventory ** 2)
        else:
            raise ValueError("Invalid inventory penalty specified.")
        reward = reward + inv_pen

        # ----------04) normalize the reward ----------#
        

        return reward, {
            "reward_portfolio_value":reward_portfolio_value,
            "reward_complex":reward_complex,
            "reward_spooner":reward_spooner,
            "reward_spooner_damped":reward_spooner_damped,
            "reward_spooner_scaled":reward_spooner_scaled,
            "reward_delta_netWorth":reward_delta_netWorth,
            "market_share": market_share,
            "inventoryValue":inventoryValue,
            "buyPnL":buyPnL,
            "sellPnL":sellPnL,
            "PnL": PnL, 
            "cash_balance" : new_cash_balance,
            "netWorth":netWorth,
            "end_inventory":new_inventory,
            "mid_price":mid_price_end,
            "agentQuant":inventory_delta,
            "buyQuant":buyQuant,
            "sellQuant":sellQuant,
            "approx_realized_pnl":approx_realized_pnl,
            "approx_unrealized_pnl" : approx_unrealized_pnl,
            "InventoryPnL":InventoryPnL,
            "scaledInventoryPnL":scaledInventoryPnL,
            "other_exec_quants":other_exec_quants,
            "averageMidprice": averageMidprice
        }

    #======================Wrappers to choose funcitons=========================================#    
    def get_episode_end_fn(self,key,bestasks, bestbids, time, asks, bids, trades, state, params):
        """
        Wrapper function to call the appropriate episode end function.
        """
        if self.cfg.end_fn == "unwind_ref_price":
            return self.end_fn(bestasks, bestbids, time, asks, bids, trades, state, params)
        elif self.cfg.end_fn == "force_market_order":
            return self.end_fn(key,bestasks, bestbids, time, asks, bids, trades, state, params)
        elif self.cfg.end_fn =="do_nothing":
            return self.end_fn(time, asks, bids, trades, state, params)
        else:
            raise ValueError("Invalid end_fn specified.")

    def get_observation(self, state, params, total_messages, action_prices, executions,old_time,old_mid_price):
        """
        Wrapper function to call the appropriate observation function.
        """
        if self.cfg.observation_space == "engineered":
            return self.observation_fn(state, params, action_prices, executions)
        elif self.cfg.observation_space == "messages":
            return self.observation_fn(state, total_messages) 
        elif self.cfg.observation_space == "messages_new_tokenizer":
            return self.observation_fn(state, total_messages,old_time,old_mid_price) 
        else:
            raise ValueError("Invalid observation_space specified.")
        
    def get_action(self,action, state, params):
        """
        Wrapper function to call the appropriate action function.
        """
        if self.cfg.action_space == "fixed_quants":
            return self.action_fn(action, state, params)
        elif self.cfg.action_space == "fixed_prices":
            return self.action_fn(action, state, params)
        elif self.cfg.action_space == "AvSt":
            return self.action_fn(action, state, params)
        elif self.cfg.action_space == "spread_skew":
            return self.action_fn(action, state, params)
        elif self.cfg.action_space == "directional_trading":
            return self._getActionMsgs_directional_trading(action, state, params)
        else:
            raise ValueError("Invalid action sspace specified.")
    #=================observation functions========================#    
    def _get_obs_msg(self, state, total_msgs: chex.Array):
        return total_msgs
    

    def _get_obs_msg_new_tokenizer_old(self, state, total_msgs: chex.Array, old_time,old_mid_price):
        # We have to find n_msgs for some features. This is inclusive of cancel here

        ##Give a time to the cancels, the time of the agent actions
        total_msgs = total_msgs.at[:self.cfg.num_messages_by_agent//2, 6].set(total_msgs[self.cfg.num_messages_by_agent//2+1, 6])
        total_msgs = total_msgs.at[:self.cfg.num_messages_by_agent//2, 7].set(total_msgs[self.cfg.num_messages_by_agent//2+1, 7])

        #1. Process message features
        ###Reinstate TYPE 4 to messages if we are doing the new tokenizer
        #jax.debug.print("total_msgs start:{}",total_msgs)
        #total_messages_T4=self.locate_type_4(total_msgs,state.trades)
        #jax.debug.print("total_msgs with t4:{}",total_messages_T4)

        #Replace the time columns with delta times
        old_ts=old_time[0]
        old_tns=old_time[1]/1e9
        total_messages_T4 = self.calculate_row_wise_differences_time(total_messages_T4, old_ts,old_tns)
        #jax.debug.print("total_msgs with delta time:{}",total_messages_T4)

        msg_type = total_msgs[:,0]  # type
        msg_direction = total_msgs[:,1]  # direction
        
        #Combine type and direction into event_dir 
        event_dir = msg_direction * 4 + msg_type

        ##Renumber OID:
        #total_messages_T4 = self.renumber_order_ids(total_messages_T4, state.customIDcounter)
        #jax.debug.print("total_msgs with oid order:{}",total_messages_T4)

        #jax.debug.print('prices {}', total_msgs[:,3])

    
        # Compute the raw mid prices from the state (assuming state.best_bids and state.best_asks have matching shapes)
        raw_mid_prices = (state.best_bids[:, 0] + state.best_asks[:, 0]) // 2

        # Create a padding of num_messages_by_agent copies of the first mid price
        start_padding = jnp.full((self.cfg.num_messages_by_agent,), raw_mid_prices[0])

        ##Pad for the ending?+>replace 0 with the end mid price?
        raw_mid_prices=self.fill_trailing_zeros(raw_mid_prices)

        # Append the padded values to the beginning of the raw mid prices array
        mid_prices = jnp.concatenate([start_padding, raw_mid_prices], axis=0)

        delta_mid_prices=self.calculate_row_wise_differences_midprice(mid_prices,old_mid_price,self.cfg.num_messages_by_agent//2)
        
        #Extract other message features
        msg_features = jnp.array([
           event_dir,  # Combined event_dir
           total_msgs[:,4],  # order_id (Done in step, we would need to change that for orders from the day before)
           total_msgs[:,3] - mid_prices,  # normalized price (should this not be to some fixed value=> they do SOD...)
           total_msgs[:,2],  # size
           total_msgs[:,6],  # delta_time_s 
           total_msgs[:,7],  # delta_time_ns 
           delta_mid_prices,  # possibly working, need to check how our trades are handled (first mid ok?).
        ])
        
        #2. Get LOB state
        lob_state = job.get_L2_state(
           state.ask_raw_orders,  # Current ask orders
           state.bid_raw_orders,  # Current bid orders
           10,  # Number of levels
           self.cfg  
        )
        new_ts=state.time[0] 
        new_tns=state.time[1]/1e9
        #Add time_s and time_ns at the start (add the values from the state as others changes by above)
        lob_state_with_time = jnp.concatenate([
           jnp.array([new_ts,new_tns]),  # time_s, time_ns
           lob_state
        ])
        
        # Transpose (7x104) to (104x7) - puts each row's features together
        msg_features_transposed = jnp.transpose(msg_features)

        # Flatten to get [event_dir[0], total_msgs[0,4], ..., delta_mid_prices[0], event_dir[1], ...]
        msg_features_flat = msg_features_transposed.reshape(-1)

        
        # Concatenate with lob_state_with_time
        return jnp.concatenate([msg_features_flat, lob_state_with_time])



    def _get_obs_msg_new_tokenizer(self, state, total_msgs: chex.Array, old_time, old_mid_price):
        """
        Construct a tokenized observation matching the pretraining format:
        [orderbook_tokens..., message_tokens...]
        """

        # cfg = get_config()  ## Removed so that we don't need get_config from lobgen. 
        num_agent_msgs = 4  # 2 cancels + 2 actions for directional trading
        num_msgs = 100      # total messages in obs
        num_data_msgs = num_msgs - num_agent_msgs


        #jax.debug.print("total_msgs:{}",total_msgs.shape)
        #jax.debug.print("Best bids:{}",state.best_bids[:, 0].shape)
        #jax.debug.print("Best asks:{}",state.best_asks[:, 0].shape)

        # Extract fields
        event = total_msgs[:, 0]
        direction = total_msgs[:, 1]
        order_id = total_msgs[:, 4]
        price = total_msgs[:, 3]
        size = total_msgs[:, 2]
        time_s = total_msgs[:, 6]
        time_ns = total_msgs[:, 7]

        # event_dir
        event_dir = direction.astype(jnp.uint8) * 4 + event.astype(jnp.uint8)

        # delta_time: difference between consecutive time_s/time_ns
        delta_time_s = jnp.zeros_like(time_s)
        delta_time_ns = jnp.zeros_like(time_ns)
        delta_time_s = delta_time_s.at[0].set(time_s[0] - old_time[0])
        delta_time_ns = delta_time_ns.at[0].set(time_ns[0] - old_time[1])
        delta_time_s = delta_time_s.at[1:].set(time_s[1:] - time_s[:-1])
        delta_time_ns = delta_time_ns.at[1:].set(time_ns[1:] - time_ns[:-1])

        # delta_price: difference in best price after each message
        # For this, you need the best bid/ask after each message. If you have them, do:
        #   mid_prices = (best_bids[:, 0] + best_asks[:, 0]) // 2
        #   delta_price = jnp.zeros_like(mid_prices)
        #   delta_price = delta_price.at[0].set(mid_prices[0] - old_mid_price)
        #   delta_price = delta_price.at[1:].set(mid_prices[1:] - mid_prices[:-1])
        # If not, you can set to zero or compute from state.

        # Stack into [num_msgs, 7] array
        #msg_array = jnp.stack([
        #    event_dir,
       #     order_id,
        #    price,
         #   size,
          #  delta_time_s,
           # delta_time_ns,
           # delta_price
        #], axis=1).astype(jnp.int32)

        # 3. Get orderbook snapshot (L2 state)
        #l2_state = job.get_L2_state(
        #    state.ask_raw_orders,
        #    state.bid_raw_orders,
        #    10,
        #    self.cfg
        #)  # shape: (42,)

        # 4. Tokenize orderbook and messages
        #    You need to port the logic from single_tokenized.py to JAX.
        #    For each int32 field (except event_dir), split into two uint16 tokens and add the correct offset.
        #    For event_dir, just add the offset.

        # Example for messages:
        #   - event_dir: [num_msgs, 1]
        #   - all other fields: [num_msgs, 6] -> split each int32 into two uint16, then flatten
        #   - add field-specific offsets (from cfg)

        # 5. Concatenate tokenized orderbook and messages into 1D array
        #    obs = jnp.concatenate([orderbook_tokens, message_tokens], axis=0)

        # 6. Return obs

        

        return 0
      



    
    def _get_obs_engineered(
            self,
            state: EnvState,
            params: EnvParams,
            action_prices: chex.Array,
            executions: chex.Array,
            normalize: bool = True,
            flatten: bool = True,
        ) -> chex.Array:
        """ Return observation from raw state trafo. """
        # NOTE: only uses most recent observation from state
        time = state.time[0] + state.time[1]/1e9
        time_elapsed = time - (state.init_time[0] + state.init_time[1]/1e9)
        obs = {
            "p_bid" : state.best_bids[-1][0],  
            "p_ask":state.best_asks[-1][0], 
            "spread": jnp.abs(state.best_asks[-1][0] - state.best_bids[-1][0]),
            "q_bid": state.best_bids[-1][1],
            "q_ask": state.best_asks[-1][1],
            "price_bid_passive":state.price_bid_passive,
            "quant_bid_passive":state.quant_bid_passive,
            "price_ask_passive":state.price_ask_passive,
            "quant_ask_passive":state.quant_ask_passive,
            "time": time,
            "delta_time": state.delta_time,
            "time_remaining": params.episode_time - time_elapsed,
            "inventory" : state.inventory,
            "mid_price":state.mid_price,
            "total_PnL" : state.total_PnL,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
            "prev_action": action_prices,  # the prices of our action messages
            "prev_executed":executions,  # the quant that we executed (traded) at each action price
            #"prev_executed_ratio": jnp.where(executions==0., 0., executions /10)# state.prev_action[:, 1]), Hard code size of normal trade
            
        }

        # TODO: put this into config somewhere?
        #       also check if we can get rid of manual normalization
        #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
        p_mean = 3.5e7
        p_std = 1e6
        means = {
            "p_bid": state.mid_price,
            "p_ask": state.mid_price,
            "spread": 0,
            "q_bid": 0,
            "q_ask": 0,
            "price_bid_passive":0,
            "quant_bid_passive":0,
            "price_ask_passive":0,
            "quant_ask_passive":0,
            "time": 0,
            "delta_time": 0,
            "time_remaining": 0,
            "inventory" : 0,
            "mid_price":0,
            "total_PnL" : 0,
            "step_counter": 0,
            "max_steps": 0,
            #"remaining_ratio": 0,
            "prev_action": 0,
            "prev_executed": 0,
        
        }
        stds = {
            "p_bid": 1e5, #p_std,
            "p_ask": 1e5, #p_std,
            "spread": 1e4,
            "q_bid": 100,
            "q_ask": 100,
            "price_bid_passive":100,
            "quant_bid_passive":100,
            "price_ask_passive":100,
            "quant_ask_passive":100,
            "time": 1e5,
            "delta_time": 10,
            "time_remaining": self.sliceTimeWindow, # 10 minutes = 600 seconds
            "mid_price": 1e7, #p_std,
            "inventory" : 10,
            "total_PnL" : 100,
            "step_counter": 30,  # TODO: find way to make this dependent on episode length
            "max_steps": 30,
            "prev_action": 10,
            "prev_executed": 10,
            #"prev_executed_ratio": 1,
        }
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)
        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        return obs

    def normalize_obs(
            self,
            obs: Dict[str, jax.Array],
            means: Dict[str, jax.Array],
            stds: Dict[str, jax.Array]
        ) -> Dict[str, jax.Array]:
        """ normalized observation by substracting 'mean' and dividing by 'std'
            (config values don't need to be actual mean and std)
        """
        obs = jax.tree_map(lambda x, m, s: (x - m) / s, obs, means, stds)
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """ Action space of the environment. """
        if self.cfg.action_space == "directional_trading":
            return spaces.Discrete(3)  # [0: do nothing, 1: buy at ask, 2: sell at bid]
        elif self.cfg.action_space == "fixed_prices":
            return spaces.Box(0, 100, (self.cfg.n_actions,), dtype=jnp.int32)
        elif self.cfg.action_space == "fixed_quants" or self.cfg.action_space == "AvSt":
            return spaces.Discrete(8)
        elif self.cfg.action_space == "spread_skew":
            return spaces.Discrete(6)  # 6 possible combinations (2 spreads × 3 skews)
        else:
            raise ValueError("Invalid action_space specified.")
       

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        if self.cfg.observation_space =="engineered":
             return spaces.Box(-10, 10, (17+3*self.cfg.num_action_messages_by_agent,), dtype=jnp.float32) # Obvs space is hard coded as size 17. We then add an object size n_trades plus an object size 2 by n_trades. (total =+3*n_trades)
        elif self.cfg.observation_space =="messages":
                num_messages_total=self.cfg.num_messages_by_agent+self.stepLines
                return spaces.Box(low=-1*self.cfg.maxint, high=self.cfg.maxint ,shape=(num_messages_total, 8), dtype=jnp.int32)
        elif self.cfg.observation_space == "messages_new_tokenizer":
            # cfg               = get_config()
            raise NotImplementedError("Cannot use tokenizer without lobgen repo, which is private and not a dependency.")
            num_messages      = self.cfg.num_messages_by_agent + self.n_data_msg_per_step + self.nTradesLogged
            toks_per_message  = 13      # we now split each int32 message‐field into two 16-bit tokens
            toks_per_book     = 84      # 42 book fields × 2 halves
            # vocab_size        = cfg.TOTAL_NUM_TOKENS
            return spaces.Box(
                low=0,
                high=vocab_size - 1,
                shape=(1, num_messages * toks_per_message + toks_per_book),
                dtype=jnp.int32,
            )
        else:
            raise ValueError("Invalid observation_space specified.")

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return NotImplementedError



# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        #ATFolder = "/home/duser/AlphaTrade/training_oneDay/train"
        ATFolder= "/home/duser/AlphaTrade/training_oneDay/train"

        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    
    config = {
        "ATFOLDER": ATFolder,
        "WINDOW_INDEX": 15,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60*30,  
        "TRADERID":10
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
        trader_unique_id=config["TRADERID"],
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )
    # print(env_params.message_data.shape, env_params.book_data.shape)


    start=time.time()
    obs,state=env.reset(key_reset, env_params)
    print("Time for reset: \n",time.time()-start)

    #print("State after reset: \n",state)
    print("Inventory after reset: \n",state.inventory)
    

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,3):
         # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*200)
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        #test_action=env.action_space().sample(key_policy)
        test_action = env.action_space().sample(key_policy) 
        jax.debug.print("test_action :{}",test_action)
        env.action_space().sample(key_policy) 
        # test_action = jnp.array([100, 10])
        print(f"Sampled {i}th actions are: ", test_action)

        start=time.time()
        obs, state, reward, done, info = env.step(
            key_step, state, test_action, env_params)
        #print(obs)

        #print(f"Orderbook: {info['lob_state']}")
        #print(f"action message: {info['total_msgs']}")
        #print(f"trades: {info['trades']}")
        #print(f"best_asks: {info['best_asks']}")
        #print(f"best_bids: {info['best_bids']}")
       # print("Step reward:", reward)
        #print("Step info:", info)
        #print("time",info["time_seconds"])
        #print("obs:", obs)

        print("Intial Time \n", state.init_time)
        print("Time \n", state.time)
     #   print('revenue',state.total_revenue)
        #print('revenue', state.total_revenue)
        #print('inventory',state.inventory)
        #print('reward',reward)
        #
       # print("Reward: \n",reward)
       # print("Time \n", state.time)
        #print("Intial Time \n", state.init_time)
        #for key, value in info.items():
           #print(key, value)
            
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
            exit()
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================




    # # ####### Testing the vmap abilities ########
    
    enable_vmap=False
    if enable_vmap:
        # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 1024
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)


        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys,
         state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)

        #=======================================#
        #===============Timing Test=============#
        #=======================================#
        # ========== VMAP TIMING TEST LOOP ==========

        print("\n" + "="*60)
        print("Starting VMAP timing test loop with detailed timing")
        print("="*60)

        num_envs = 1024
        vmap_keys = jax.random.split(rng, num_envs)

        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_sample_action = jax.vmap(env.action_space().sample, in_axes=(0))

        # -----------------------------------
        # Time Full Reset + Episode Rollout
        # -----------------------------------
        full_start = time.time()

        # RESET
        reset_start = time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        reset_end = time.time()
        reset_time = reset_end - reset_start

        # ROLLOUT (track only stepping)
        step_start = time.time()

        done_flags = jnp.zeros(num_envs, dtype=bool)
        step_counter = jnp.zeros(num_envs, dtype=int)

        def cond_fn(val):
            _, _, done_flags, _ = val
            return jnp.any(~done_flags)

        def body_fn(val):
            state, rng, done_flags, step_counter = val
            rng, key_action, key_step = jax.random.split(rng, 3)
            keys_action = jax.random.split(key_action, num_envs)
            keys_step = jax.random.split(key_step, num_envs)

            actions = vmap_sample_action(keys_action)
            obs, next_state, reward, done, info = vmap_step(keys_step, state, actions, env_params)

            # Masked update for unfinished envs
            def masked_update(s, ns):
                mask = done_flags
                while mask.ndim < s.ndim:
                    mask = mask[..., None]
                return jnp.where(mask, s, ns)

            state = jax.tree_map(masked_update, state, next_state)

            # Update done flags and step count
            done_flags = jnp.logical_or(done_flags, done)
            step_counter += jnp.where(done_flags, 0, 1)

            return (state, rng, done_flags, step_counter)


        state, rng, done_flags, step_counter = jax.lax.while_loop(
            cond_fn, body_fn, (state, rng, done_flags, step_counter)
        )

        step_end = time.time()
        step_time = step_end - step_start
        full_end = time.time()
        full_time = full_end - full_start

        avg_steps_per_env = jnp.mean(step_counter)
        avg_step_time = step_time / jnp.sum(step_counter)

        # -----------------------------------
        # Print results
        # -----------------------------------
        print(f"\nCompleted VMAP run with {num_envs} environments.")
        print(f"Reset time:           {reset_time:.4f} seconds")
        print(f"Rollout (steps) time: {step_time:.4f} seconds")
        print(f"Total time:           {full_time:.4f} seconds")
        print(f"Avg steps per env:    {avg_steps_per_env:.2f}")
        print(f"Avg time per step:    {avg_step_time:.6f} seconds")
        print("="*60)
