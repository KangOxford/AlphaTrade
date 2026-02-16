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
MMEnvState:   Dataclass to encapsulate the current state of the environment, 
            including the raw order book, trades, and time information.
MMEnvParams:  Configuration class for environment-specific parameters, 
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
# chex.assert_gpu_available(backend=None)
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

from gymnax_exchange.utils import utils
import dataclasses

import jax.tree_util as jtu


from gymnax_exchange.jaxob.jaxob_config import MarketMaking_EnvironmentConfig
# from lobgen.data_processing.data_config import set_config, TokenizerConfig, get_config
# set_config(TokenizerConfig(split_vocab=True)) 
from gymnax_exchange.jaxen.StatesandParams import MMEnvState, MMEnvParams, LoadedEnvParams, LoadedEnvState, WorldState
from gymnax_exchange.jaxen.StatesandParams import MultiAgentState
from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig
#from gymnax_exchange.jaxen.from_JAXMARL import spaces


class MarketMakingAgent():
    def __init__(
            self,cfg:MarketMaking_EnvironmentConfig, world_config: World_EnvironmentConfig):
        
        self.cfg=cfg
        self.world_config = world_config


        
        ##Choose get action message function based on config
        if self.cfg.action_space == "fixed_quants":
            self.action_fn = self._getActionMsgs_fixedQuant
        elif self.cfg.action_space == "fixed_prices":
            self.action_fn = self._getActionMsgs_fixedPrice
        elif self.cfg.action_space == "AvSt":
            self.action_fn = self._getActionMsgs_AvSt
        elif self.cfg.action_space == "bobStrategy":
            self.action_fn = self._getActionMsgs_BobStrategy
        elif self.cfg.action_space == "bobRL":
            self.action_fn = self._getActionMsgs_BobRL
        elif self.cfg.action_space == "spread_skew":
            self.action_fn = self._getActionMsgs_spread_skew
        elif self.cfg.action_space == "directional_trading":
            self.action_fn = self._getActionMsgs_directional_trading
        elif self.cfg.action_space == "simple":
            self.action_fn = self._getActionMsgs_simple
        else:
            raise ValueError("Invalid action_space specified.")
        
        ##Choose an end function from theconfig
        # if self.cfg.end_fn=="force_market_order":
        #     self.end_fn =self._force_market_order_if_done
        # elif self.cfg.end_fn=="unwind_ref_price":
        #     self.end_fn=self.unwind_ref_price
        # elif self.cfg.end_fn=="do_nothing":
        #     self.end_fn=self.end_fn_pass
      

    def default_params(self,
                       agent_config:MarketMaking_EnvironmentConfig,
                       trader_id_range_start:int,
                        number_of_agents_per_type:int) -> MMEnvParams:
        next_trader_id_range_start = trader_id_range_start - number_of_agents_per_type
        # Return array (one for each agent)
        trader_id = jnp.arange(trader_id_range_start, next_trader_id_range_start, -1)
        time_delay_obs_act = jnp.full((number_of_agents_per_type,), agent_config.time_delay_obs_act)
        normalize = jnp.full((number_of_agents_per_type,), agent_config.normalize)

        # print("normalize.shape:", normalize.shape)
        # print(f"trader_id: {trader_id}")
        # print(f"next_trader_id_range_start: {next_trader_id_range_start}")
        return MMEnvParams(trader_id=trader_id, time_delay_obs_act=time_delay_obs_act, normalize=normalize), next_trader_id_range_start



    def step_env(
        self, key: chex.PRNGKey, state: MMEnvState, input_action: jax.Array, params: MMEnvParams
    ) -> Tuple[chex.Array, MMEnvState, float, bool, dict]:

        #=======================================#
        #====Load data messages for next step===#
        #=======================================#      
        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + self.world_config.episode_time
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
            agent_params.trader_id,
            self.cfg.num_action_messages_by_agent//2, 
            1,  # bid
            state.time[0],  # cancel_time
            state.time[1],  # cancel_time_ns
        )
        cnl_msg_ask = job.getCancelMsgs(
            state.ask_raw_orders,
            agent_params.trader_id,
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

        #jax.debug.print(f"Number of overall messages: {self.n_data_msg_per_step + self.cfg.num_messages_by_agent}")

        # Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(self.cfg,key,
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            # TODO: this returns bid/ask for last n_data_msg_per_step only, could miss the direct impact of actions
            self.n_data_msg_per_step + self.cfg.num_messages_by_agent # to include our action messages increase this by cfg.num_messages_by_agent
        )
        # If best price is not available in the current step, use the last available price
        # TODO: check if we really only want the most recent n_data_msg_per_step prices (+1 for the additional market order)
        bestasks = self._ffill_best_prices(bestasks[-self.n_data_msg_per_step-self.cfg.num_messages_by_agent:], state.best_asks[-1, 0])
        bestbids = self._ffill_best_prices(bestbids[-self.n_data_msg_per_step-self.cfg.num_messages_by_agent:], state.best_bids[-1, 0])

        #jax.debug.print(f"bestasks shape in function: {bestasks.shape}")

        #bestasks = self._ffill_best_prices(bestasks, state.best_asks[-1, 0])
        #bestbids = self._ffill_best_prices(bestbids, state.best_bids[-1, 0])
        ##jax.debug.print(f"bestasks: {bestasks}")
        #jax.debug.print(f"bestbids: {bestbids}")
        agent_trades,_ = job.get_agent_trades(trades, agent_params.trader_id)
        executions = self._get_executed_by_action(agent_trades, action, state,action_prices)
        executions=jnp.abs(executions)
        #=======================================#
        #===force inventory sale at episode end=#
        #=======================================#
        (asks, bids, trades), new_id_counter, new_time=self.get_episode_end_fn(key,
            bestasks, bestbids, time, asks, bids, trades, state, params)
        #bestasks = jnp.concatenate([bestasks,bestasks[:,:] ], axis=0, dtype=jnp.int32)
        #bestbids = jnp.concatenate([bestbids, bestbids[:,:]], axis=0, dtype=jnp.int32)
    
        # TODO: consider adding quantity before (in priority) to each price / level

        # TODO: use the agent quant identification from the separate function _get_executed_by_level instead of _get_reward
        reward, extras = self._get_reward(state, params, trades,bestasks,bestbids)
        old_time=state.time
        old_mid_price=state.mid_price
        state = MMEnvState(
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
            mid_price=extras["mid_price"],
            inventory=extras["end_inventory"],
            total_PnL = state.total_PnL + extras["PnL"],
            cash_balance= extras["cash_balance"],          
            delta_time = new_time[0] + new_time[1]/1e9 - state.time[0] - state.time[1]/1e9,
        )
        done = self.is_terminal(state, params)
        average_best_ask = state.best_asks[-100:].mean(axis=0)[0] #// self.world_config.tick_size) * self.world_config.tick_size)
        average_best_bid = state.best_bids[-100:].mean(axis=0)[0] #// self.world_config.tick_size) * self.world_config.tick_size)
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
    

    @partial(jax.jit, static_argnames=("self", "num_msgs_per_step"))
    def reset_env(
            self,
            agent_param: MMEnvParams,
            key : chex.PRNGKey,
            world_state: WorldState,
            num_msgs_per_step: int
        ) -> Tuple[chex.Array, MMEnvState]:
        """ Reset the environment state to the initial state."""

        agent_state = MMEnvState(
            posted_distance_bid=0,
            posted_distance_ask=0,
            inventory=0,
            total_PnL=0.0,
            cash_balance=0.0
        )

        # Calculate things for the message obs space
        if self.cfg.observation_space == "messages_new_tokenizer":
            lob_state_before = job.get_L2_state(
                world_state.ask_raw_orders,  # Current ask orders
                world_state.bid_raw_orders,  # Current bid orders
                10,  # Number of levels
                self.world_config  
            )
            blank_messages = jnp.zeros((num_msgs_per_step, 8), dtype=jnp.int32) # Reset for the message based obs space.
        else:
            lob_state_before = None
            blank_messages = None


        obs = self.get_observation(world_state = world_state, 
                                   agent_state = agent_state, 
                                   agent_param = agent_param, 
                                   total_messages = blank_messages, 
                                   old_time = world_state.time, 
                                   old_mid_price = world_state.mid_price, 
                                   lob_state_before = lob_state_before,
                                   normalize = self.cfg.normalize,
                                   flatten  = True)

        return obs, agent_state



    def is_terminal(self, world_state: WorldState) -> bool:
        """ Episode termination handled by MarlEnv, market maker never stops making markets. """
        # if self.world_config.ep_type == 'fixed_time':
        #     # TODO: make the 5 sec a function of the step size
        #     time_left=(self.world_config.episode_time - (world_state.time - world_state.init_time)[0] )
        #     #jax.debug.print("time_left :{}",time_left)
        #     #jax.debug.print("time :{}",world_state.time)
        #     #jax.debug.print("init_time :{}",world_state.init_time)
        #     #jax.debug.print("start_index :{}",world_state.start_index)
        #     done = (time_left <= self.cfg.seconds_before_episode_end)  # time over (last 5 seconds)


        #     #jax.debug.print("episode_time :{}", self.world_config.episode_time)
        #     #jax.debug.print("world_state.init_time :{}",world_state.init_time)
        #     #jax.debug.print("world_state.time :{}",world_state.time)
        #     #jax.debug.print("time_left :{}",time_left)
        #     #jax.debug.print("done : {}" , done)


        #     #jax.debug.print("done :{}",done)
        return False
        
        # elif self.world_config.ep_type == 'fixed_steps':
        #     return (
        #         (world_state.max_steps_in_episode - world_state.step_counter <= 1)  # last step  
        #     )
        # else:
        #     raise ValueError(f"Unknown episode type: {self.world_config.ep_type}")
   
    def _get_pass_price_quant(self, state):
        """Get price and quanitity n_ticks into books"""
        bid_passive_2=state.best_bids[-1, 0] - self.world_config.tick_size * self.cfg.n_ticks_offset
        ask_passive_2=state.best_asks[-1, 0] + self.world_config.tick_size * self.cfg.n_ticks_offset
        quant_bid_passive_2 = job.get_volume_at_price(state.bid_raw_orders, bid_passive_2)
        quant_ask_passive_2 = job.get_volume_at_price(state.ask_raw_orders, ask_passive_2)
        return bid_passive_2,quant_bid_passive_2,ask_passive_2,quant_ask_passive_2
    
    def _get_state_from_data(self,key,first_message,book_data,max_steps_in_episode,window_index,start_index):
        """Reset state from data"""
        base_state = super()._get_state_from_data(key,first_message, book_data, max_steps_in_episode, window_index, start_index)
        base_vals = jtu.tree_flatten(base_state)[0]
        best_bid, best_ask = job.get_best_bid_and_ask_inclQuants(self.cfg,base_state.ask_raw_orders,base_state.bid_raw_orders)
        M =jnp.float32((best_bid[0] + best_ask[0]) / 2)
        #TODO: we could do an array of all those values that we only add on reset (not loaded)
        return MMEnvState(
            ##This is reset
            *base_vals,
            best_asks=jnp.resize(best_ask,(self.n_data_msg_per_step,2)),
            best_bids=jnp.resize(best_bid,(self.n_data_msg_per_step,2)),
            mid_price=M,
            inventory=0,
            total_PnL=0.,
            # updated on reset:
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
    
    def _get_executed_by_level(self, agent_trades: jax.Array, actions: jax.Array, state: MMEnvState) -> jax.Array:
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
    
    def _get_executed_by_action(self, agent_trades: jax.Array, actions: jax.Array, state: MMEnvState,action_prices:jax.Array) -> jax.Array:
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
      
    
    def _getActionMsgs_fixedQuant(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
        '''Transform discrete action into bid and ask order messages based on current best prices.'''
        if self.cfg.fixed_action_setting == True:
            action = jnp.asarray([self.cfg.fixed_action])
        
        
        
        # Use the most recent best_ask and best_bid values
        #These values may be my own orders... I clearly don't want to base myself off them. Get from world state directly.
        ask_mask=(world_state.ask_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        bid_mask=(world_state.bid_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        
        masked_asks=jnp.where(ask_mask[:, jnp.newaxis], world_state.ask_raw_orders, -1)
        masked_bids=jnp.where(bid_mask[:, jnp.newaxis], world_state.bid_raw_orders, -1)
    
        best_ask, best_bid = job.get_best_bid_and_ask(self.world_config,masked_asks,masked_bids)
        #If the book is empty here, we get -1 back.

        empty_book = jnp.where((best_ask == -1) | (best_bid == -1),True, False)
        #We then replace with the last known bbid, bask, which in turn should have been forward filled, but is most likely our own order which will be v far from the last true market price. 
        best_ask = jnp.int32((best_ask // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((best_bid // self.world_config.tick_size) * self.world_config.tick_size)
        #The world state will have the mid-price propagated. Doing this just for the sake of logging to have reasonable averages. 
        #If the book is empty, the quants are put to 0 anyway. 
        best_bid = jnp.where(empty_book, world_state.best_bids[-1,0], best_bid)
        best_ask = jnp.where(empty_book, world_state.best_asks[-1,0], best_ask)

        # best_ask_old = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        # best_bid_old = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        def bid_ask_callback(delta, best_ask, best_bid,old_ask,old_bid,bids_raw,asks_raw,bids_masked,asks_masked):
            if delta!=0:
                print("The best bid and ask are changing! from {}-{} to {}-{}".format(old_bid,old_ask,best_bid,best_ask))
                print("Raw bids: {}".format(bids_raw))
                print("Masked bids: {}".format(bids_masked))
                print("Raw asks: {}".format(asks_raw))
                print("Masked asks: {}".format(asks_masked))
        # jax.debug.callback(bid_ask_callback,jnp.abs(best_ask-best_ask_old)+jnp.abs(best_bid-best_bid_old), best_ask, best_bid, best_ask_old, best_bid_old, world_state.bid_raw_orders, world_state.ask_raw_orders, masked_bids, masked_asks)
        #jax.debug.print("old best ask: {}", best_ask)
       # jax.debug.print("old best bid: {}", best_bid)
        if self.cfg.sell_buy_all_option==False:
            # Define mappings for each action: [0-8]
            # WARNING: Be very careful when changing the dimension of these arrays, they must match num_actions in the action space, and in the config and will fail silently if not changed.
            bid_offsets = jnp.array([0, 1, 2, 3, 4, 0, 2, 5, 1,0], dtype=jnp.float32)
            ask_offsets = jnp.array([0, 1, 2, 3, 4, 2, 0, 1, 5,0], dtype=jnp.float32)
            bid_quants = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1,0], dtype=jnp.int32)
            ask_quants = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1,0], dtype=jnp.int32)##config quant....
            # bid_quants = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
            # ask_quants = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)##config quant....
        elif self.cfg.sell_buy_all_option==True:
        #New option to sell and buy whole inventory
            inventory=agent_state.inventory
            bid_offsets = jnp.array([10, 2, 4, -1, 0, 2, -20, 0,0], dtype=jnp.float32)
            ask_offsets = jnp.array([10, 2, 4, -1, 2, 0, 0, -20,0], dtype=jnp.float32)
            bid_quants = jnp.array([1, 1, 1, 1, 1, 1,inventory//self.cfg.fixed_quant_value, 0,0], dtype=jnp.int32)
            ask_quants = jnp.array([1, 1, 1, 1, 1, 1, 0, inventory//self.cfg.fixed_quant_value,0], dtype=jnp.int32)##config quant....

       
        tick_offset = self.cfg.n_ticks_offset * self.world_config.tick_size  # Total price offset per direction
        half_spread_prev = jnp.maximum((best_ask - best_bid) / 2, self.world_config.tick_size/2)
        half_spread= (half_spread_prev//self.world_config.tick_size+1) * self.world_config.tick_size

        # Get parameters for current action
        bid_offset = bid_offsets[action]
        ask_offset = ask_offsets[action]
        bid_quant = bid_quants[action]*self.cfg.fixed_quant_value
        ask_quant = ask_quants[action]*self.cfg.fixed_quant_value
        
        #If the book is empty (aside from our own order), we exit the market by not posting anything.
        bid_quant=jnp.where(empty_book, 0, bid_quant)
        ask_quant=jnp.where(empty_book, 0, ask_quant)
        

        # Calculate prices with bounds checking
        bid_price = best_bid - bid_offset  * half_spread
        ask_price = best_ask + ask_offset  * half_spread

        #jax.debug.print("bid_price before:{}",bid_price)
        #jax.debug.print("ask_price before:{}",ask_price)

        bid_price = jnp.maximum(bid_price, 0) // self.world_config.tick_size * self.world_config.tick_size
        bid_price = bid_price.astype(jnp.int32)
        ask_price = jnp.maximum(bid_price + self.world_config.tick_size, ask_price) // self.world_config.tick_size * self.world_config.tick_size
        ask_price = ask_price.astype(jnp.int32)
        def print_posting_distance(best_bid, best_ask, bid_price, ask_price, window_idx,step,half_spread,prev_half_spread):
            if 1080 <= window_idx <= 1090:
                print(f"Window {window_idx}: Posted bid from best bid: {best_bid - bid_price},step {step}, half_spread {half_spread}, prev_half_spread {prev_half_spread}")
                print(f"Window {window_idx}: Posted ask from best ask: {ask_price - best_ask},step {step}, half_spread {half_spread}, prev_half_spread {prev_half_spread}")
        
        # jax.debug.callback(print_posting_distance, best_bid, best_ask, bid_price, ask_price, world_state.window_index,world_state.step_counter,half_spread,half_spread_prev)

    
        #jax.debug.print("bid_price after:{}",bid_price)
        #jax.debug.print("ask_price after:{}",ask_price)
        
        # --------------- Construct messages ---------------#
        # Message components (2 messages: bid then ask)
        types = jnp.asarray([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.asarray([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        quants = jnp.asarray([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.asarray([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)


        if self.cfg.tenth_action== "MarketOrder":
            liq_types = jnp.asarray([4, 4], dtype=jnp.int32)  # 4=IOC order
            liq_sides = jnp.asarray([-1, 1], dtype=jnp.int32)  # -1=exec on ask, buy order, 1=exec on bid, sell order
            liq_quants = jnp.asarray([self.cfg.auto_liquidate_alpha*jnp.maximum(-agent_state.inventory,0), self.cfg.auto_liquidate_alpha*jnp.maximum(agent_state.inventory,0)], dtype=jnp.int32)
            liq_prices = jnp.asarray([best_ask+half_spread*10, best_bid-half_spread*10], dtype=jnp.int32)
            types=jnp.where(action==9, liq_types, types)
            sides=jnp.where(action==9, liq_sides, sides)
            quants=jnp.where(action==9, liq_quants, quants)
            prices=jnp.where(action==9, liq_prices, prices)
        if self.cfg.tenth_action== "NA":
            pass


        if self.cfg.auto_liquidate_threshold !=0:
            liq_types = jnp.asarray([4, 4], dtype=jnp.int32)  # 4=IOC order
            liq_sides = jnp.asarray([-1, 1], dtype=jnp.int32)  # -1=exec on ask, buy order, 1=exec on bid, sell order
            liq_quants = jnp.asarray([self.cfg.auto_liquidate_alpha*jnp.maximum(-agent_state.inventory,0), self.cfg.auto_liquidate_alpha*jnp.maximum(agent_state.inventory,0)], dtype=jnp.int32)
            liq_prices = jnp.asarray([best_ask+half_spread*10, best_bid-half_spread*10], dtype=jnp.int32)
            types=jnp.where(jnp.abs(agent_state.inventory)>self.cfg.auto_liquidate_threshold, liq_types, types)
            sides=jnp.where(jnp.abs(agent_state.inventory)>self.cfg.auto_liquidate_threshold, liq_sides, sides)
            quants=jnp.where(jnp.abs(agent_state.inventory)>self.cfg.auto_liquidate_threshold, liq_quants, quants)
            prices=jnp.where(jnp.abs(agent_state.inventory)>self.cfg.auto_liquidate_threshold, liq_prices, prices)

        quants = quants.flatten() # Flatten so they have the same shape
        prices = prices.flatten()
        
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)

        
        # Time fields (replicated for each message)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )



        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)

        # action_msgs = jnp.where((action==9) & (~empty_book), jnp.ones_like(action_msgs)*9, action_msgs)  
        #jax.debug.print("action_msgs mm:{}",action_msgs)

        return action_msgs,{"posted_bid_price":bid_price,"posted_ask_price":ask_price,"bid_distance_from_best":best_bid - bid_price,"ask_distance_from_best":ask_price - best_ask,"empty_book":empty_book,"bid_quant":bid_quant,"ask_quant":ask_quant}



    
    def _getActionMsgs_simple(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
        '''Transform discrete action into bid and ask order messages based on current best prices.'''
        # Use the most recent best_ask and best_bid values
        #There
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)

        #jax.debug.print("old best ask: {}", best_ask)
        #jax.debug.print("old best bid: {}", best_bid)
        if self.cfg.sell_buy_all_option==False:
            if self.cfg.simple_nothing_action==True:
                # Define mappings for each action: [0-7]
                bid_offsets = jnp.array([0, -2000, 0,0], dtype=jnp.float32)
                ask_offsets = jnp.array([0, 0,  -2000,0], dtype=jnp.float32)
                bid_quants = jnp.array([1,  1,  0,0], dtype=jnp.int32)
                ask_quants = jnp.array([1,  0,  1,0], dtype=jnp.int32)##config quant....
            else:
                bid_offsets = jnp.array([0, -2000, 0], dtype=jnp.float32)
                ask_offsets = jnp.array([0, 0,  -2000], dtype=jnp.float32)
                bid_quants = jnp.array([1,  1,  0], dtype=jnp.int32)
                ask_quants = jnp.array([1,  0,  1], dtype=jnp.int32)##config quant....
        elif self.cfg.sell_buy_all_option==True:
             #New option to sell and buy whole inventory
            inventory=agent_state.inventory
            def quants_positive_inventory(inventory):
                bid_quant = self.cfg.fixed_quant_value
                ask_quant = jnp.maximum(jnp.abs(inventory),self.cfg.fixed_quant_value)
                return ask_quant, bid_quant
            def quants_negative_inventory(inventory):
                bid_quant = jnp.maximum(jnp.abs(inventory),self.cfg.fixed_quant_value)
                ask_quant = self.cfg.fixed_quant_value
                return ask_quant, bid_quant
            ask_quant , bid_quant = jax.lax.cond(
                inventory > 0,
                quants_positive_inventory,
                quants_negative_inventory,
                inventory
            )
            if self.cfg.simple_nothing_action==True:
                bid_offsets = jnp.array([0, -2000, 0, 0 ], dtype=jnp.float32)
                ask_offsets = jnp.array([0,  0, -2000, 0], dtype=jnp.float32)
                bid_quants = jnp.array([self.cfg.fixed_quant_value,  bid_quant, 0, 0], dtype=jnp.int32)
                ask_quants = jnp.array([self.cfg.fixed_quant_value,  0, ask_quant, 0], dtype=jnp.int32)##config quant....
            else:
                bid_offsets = jnp.array([0, -2000, 0 ], dtype=jnp.float32)
                ask_offsets = jnp.array([0,  0, -2000], dtype=jnp.float32)
                bid_quants = jnp.array([self.cfg.fixed_quant_value,  bid_quant, 0], dtype=jnp.int32)
                ask_quants = jnp.array([self.cfg.fixed_quant_value,  0, ask_quant], dtype=jnp.int32)##config quant....
        #jax.debug.print("bid_quants: {}", bid_quants)
        #jax.debug.print("ask_quants: {}", ask_quants)
        tick_offset = self.cfg.n_ticks_offset * self.world_config.tick_size  # Total price offset per direction

        #jax.debug.print("tick_offset: {}", tick_offset)
        #if self.fixed_action_setting == True:
            
        #jax.debug.print("action: {}", action)
        if self.cfg.fixed_action_setting == True:
            action = jnp.array([self.cfg.fixed_action])

        #jax.debug.print("action after: {}", action)

        # Get parameters for current action
        bid_offset = bid_offsets[action]
        ask_offset = ask_offsets[action]

        #jax.debug.print("bid_offset: {}", bid_offset)
        #jax.debug.print("ask_offset: {}", ask_offset)

        if self.cfg.sell_buy_all_option==True:
            bid_quant = bid_quants[action]
            ask_quant = ask_quants[action]
        else:
            bid_quant = bid_quants[action]*self.cfg.fixed_quant_value
            ask_quant = ask_quants[action]*self.cfg.fixed_quant_value
        
        # Calculate prices with bounds checking
        bid_price = best_bid - bid_offset * tick_offset
        ask_price = best_ask + ask_offset * tick_offset

        #jax.debug.print("bid_price before:{}",bid_price)
        #jax.debug.print("ask_price before:{}",ask_price)

        bid_price = jnp.maximum(bid_price, 0) // self.world_config.tick_size * self.world_config.tick_size
        bid_price = bid_price.astype(jnp.int32)
        ask_price =  ask_price // self.world_config.tick_size * self.world_config.tick_size
        ask_price = ask_price.astype(jnp.int32)
        
        #jax.debug.print("bid_price after:{}",bid_price)
        #jax.debug.print("ask_price after:{}",ask_price)
        
        # --------------- Construct messages ---------------#
        # Message components (2 messages: bid then ask)
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)

        quants = quants.flatten() # Flatten so they have the same shape
        prices = prices.flatten()
        
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)

        
        # Time fields (replicated for each message)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )



        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)


        #jax.debug.print("action_msgs mm:{}",action_msgs)

        return action_msgs,{"bid_quant":bid_quant,"ask_quant":ask_quant,"empty_book":False,"bid_distance_from_best":0,"ask_distance_from_best":0,"posted_bid_price":0,"posted_ask_price":0}



    
    def _getActionMsgs_AvSt(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
        '''AvST action space: Discrete selections to paramterise K in the AvSt forumla.
        0-7, with lower giving more aggresive bid and asks
        '''
        # Use the most recent best_ask and best_bid values
        # best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        # best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        # Use the most recent best_ask and best_bid values
        #These values may be my own orders... I clearly don't want to base myself off them. Get from world state directly.
        ask_mask=(world_state.ask_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        bid_mask=(world_state.bid_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        
        masked_asks=jnp.where(ask_mask[:, jnp.newaxis], world_state.ask_raw_orders, -1)
        masked_bids=jnp.where(bid_mask[:, jnp.newaxis], world_state.bid_raw_orders, -1)
    
        best_ask, best_bid = job.get_best_bid_and_ask(self.world_config,masked_asks,masked_bids)
        #If the book is empty here, we get -1 back.

        empty_book = jnp.where((best_ask == -1) | (best_bid == -1),True, False)
        #We then replace with the last known bbid, bask, which in turn should have been forward filled, but is most likely our own order which will be v far from the last true market price. 
        best_ask = jnp.int32((best_ask // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((best_bid // self.world_config.tick_size) * self.world_config.tick_size)
        #The world state will have the mid-price propagated. Doing this just for the sake of logging to have reasonable averages. 
        #If the book is empty, the quants are put to 0 anyway. 
        best_bid = jnp.where(empty_book, world_state.best_bids[-1,0], best_bid)
        best_ask = jnp.where(empty_book, world_state.best_asks[-1,0], best_ask)




        mid_price = (best_ask + best_bid) // 2

        #Select aaggresion parameter
        gamma_values = jnp.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20], dtype=jnp.float32)  # Risk aversion
        gamma = gamma_values[action]

        k = self.cfg.avst_k_parameter  # Market depth parameter

        # Market volatility estimation (rolling standard zeviation of mid-price)
        variance = self.cfg.avst_var_parameter #* mid_price
        

        #Get time until ep end
        if self.world_config.ep_type == "fixed_time":
            time_left = self.world_config.episode_time - (world_state.time - world_state.init_time)[0]
        else:
            time_left = self.world_config.episode_time - world_state.step_counter  # Placeholder for other ep types
       
        normalized_time = time_left / self.world_config.episode_time

        #Reservation price
        res_price = (mid_price - ((agent_state.inventory)) * gamma * (variance) * normalized_time)

        #Spread
        spread = (gamma*variance*normalized_time + (2/gamma) * jnp.log(1 + gamma/k))#*self.world_config.tick_size
        spread=jnp.clip(spread,self.world_config.tick_size,self.world_config.maxint)#make sure spread is at least a tick

        bid_price= res_price-spread/2
        ask_price= res_price+spread/2
        def print_distances(best_bid, best_ask, bid_price, ask_price, mid_price, res_price, spread,window_idx,step,position):
            print(f"Window {window_idx}:")
            print("best ask: {}", best_ask)
            print("best bid: {}", best_bid)
            print("mid price: {}", mid_price)
            print("reservation price: {}", res_price)
            print("spread: {}", spread)
            print("bid price before clipping: {}", bid_price)
            print("ask price before clipping: {}", ask_price)
            print("agent inventory: {}", position)
            print("step: {}",step)
        # jax.debug.callback(print_distances, best_bid, best_ask, bid_price, ask_price, mid_price, res_price, spread, world_state.window_index, world_state.step_counter, agent_state.inventory,)


        # Ensure valid price bound 
        bid_price = jnp.clip(bid_price, 0, self.world_config.maxint) 
        ask_price = jnp.clip(ask_price,  0, self.world_config.maxint) 

        #Ensure ints of tick_size
        bid_price=((bid_price) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32)
        ask_price=((ask_price) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32)

        # Ensure that the quote prices are not crossing the midprice. Allows for entrance into spread, but not if the spread is one ticckc
        def round_down(x, multiple):
            """Round down to the nearest multiple of X (strictly less than x) - JAX compatible"""
            return (x // multiple - jnp.where(x % multiple == 0, 1, 0)) * multiple

        def round_up(x, multiple):
            """Round up to the nearest multiple of X (strictly greater than x) - JAX compatible"""
            return (x // multiple + 1) * multiple

        bid_price = jnp.minimum(bid_price, round_down(mid_price, self.world_config.tick_size))
        ask_price = jnp.maximum(ask_price, round_up(mid_price, self.world_config.tick_size))
        # jax.debug.print("bid price : {}", bid_price)
        # jax.debug.print("ask price: {}", ask_price)
        # Set fixed quantities
        bid_quant = self.cfg.fixed_quant_value
        ask_quant = self.cfg.fixed_quant_value

        # Construct order messages
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1 = limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1 = bid, -1 = ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)

        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)

        # Time fields
        times = jnp.resize(world_state.time + self.cfg.time_delay_obs_act, (2, 2))

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

        def debug_neg_distances(best_bid, best_ask, bid_price, ask_price, mid_price, res_price, spread,window_idx,step,position):
            if bid_price>best_bid:
                print(f"Window {window_idx}: Posted bid inside the spread! distance from best bid: {best_bid - bid_price}, step {step}, position {position}")
                print("best ask: {}", best_ask)
                print("best bid: {}", best_bid)
                print("mid price: {}", mid_price)
                print("reservation price: {}", res_price)
                print("spread: {}", spread)
                print("bid price before clipping: {}", bid_price)
                print("ask price before clipping: {}", ask_price)
                print("agent inventory: {}", position)
                print("step: {}",step)
            if ask_price<best_ask:
                print(f"Window {window_idx}: Posted ask inside the spread! distance from best ask: {ask_price - best_ask}, step {step}, position {position}")
                print("best ask: {}", best_ask)
                print("best bid: {}", best_bid)
                print("mid price: {}", mid_price)
                print("reservation price: {}", res_price)
                print("spread: {}", spread)
                print("bid price before clipping: {}", bid_price)
                print("ask price before clipping: {}", ask_price)
                print("agent inventory: {}", position)
                print("step: {}",step)
        # jax.debug.callback(debug_neg_distances, best_bid, best_ask, bid_price, ask_price, mid_price, res_price, spread, world_state.window_index, world_state.step_counter, agent_state.inventory,)

        return action_msgs,{"bid_quant":bid_quant,"ask_quant":ask_quant,"empty_book":False,"bid_distance_from_best":best_bid-bid_price,"ask_distance_from_best":ask_price-best_ask,"posted_bid_price":bid_price,"posted_ask_price":ask_price}

    def _getActionMsgs_BobStrategy(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
        '''Transform discrete action into bid and ask order messages based on current best prices.'''
        if self.cfg.fixed_action_setting == True:
            action = jnp.asarray([self.cfg.fixed_action])
        
        kappa= (action+1)/(self.cfg.bob_v0*5)
        
        
        # Use the most recent best_ask and best_bid values
        #These values may be my own orders... I clearly don't want to base myself off them. Get from world state directly.
        ask_mask=(world_state.ask_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        bid_mask=(world_state.bid_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        
        masked_asks=jnp.where(ask_mask[:, jnp.newaxis], world_state.ask_raw_orders, -1)
        masked_bids=jnp.where(bid_mask[:, jnp.newaxis], world_state.bid_raw_orders, -1)
    
        best_ask, best_bid = job.get_best_bid_and_ask(self.world_config,masked_asks,masked_bids)
        #If the book is empty here, we get -1 back.

        empty_book = jnp.where((best_ask == -1) | (best_bid == -1),True, False)
        #We then replace with the last known bbid, bask, which in turn should have been forward filled, but is most likely our own order which will be v far from the last true market price. 
        best_ask = jnp.int32((best_ask // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((best_bid // self.world_config.tick_size) * self.world_config.tick_size)
        #The world state will have the mid-price propagated. Doing this just for the sake of logging to have reasonable averages. 
        #If the book is empty, the quants are put to 0 anyway. 
        best_bid = jnp.where(empty_book, world_state.best_bids[-1,0], best_bid)
        best_ask = jnp.where(empty_book, world_state.best_asks[-1,0], best_ask)

        position=agent_state.inventory

        v_0=self.cfg.bob_v0
        bid_quant=jnp.round(v_0*jnp.maximum(1 - kappa*position,0)).astype(jnp.int32) #epsilon = -1
        ask_quant=jnp.round(v_0*jnp.maximum(1 + kappa*position,0)).astype(jnp.int32) #epsilon = 1
        bid_quant=jnp.where(empty_book, 0, bid_quant)
        ask_quant=jnp.where(empty_book, 0, ask_quant)
    
        #jax.debug.print("bid_price after:{}",bid_price)
        #jax.debug.print("ask_price after:{}",ask_price)
        
        # --------------- Construct messages ---------------#
        # Message components (2 messages: bid then ask)
        types = jnp.asarray([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.asarray([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        quants = jnp.asarray([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.asarray([best_bid, best_ask], dtype=jnp.int32)
        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)



        quants = quants.flatten() # Flatten so they have the same shape
        prices = prices.flatten()
        
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)

        
        # Time fields (replicated for each message)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )



        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)

        # action_msgs = jnp.where((action==9) & (~empty_book), jnp.ones_like(action_msgs)*9, action_msgs)  
        #jax.debug.print("action_msgs mm:{}",action_msgs)

        return action_msgs,{"bid_quant":bid_quant,"ask_quant":ask_quant,"empty_book":empty_book,"bid_distance_from_best":0,"ask_distance_from_best":0,"posted_bid_price":0,"posted_ask_price":0}
    
    
    def _getActionMsgs_BobRL(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
        '''Transform discrete action into bid and ask order messages based on current best prices.'''
        if self.cfg.fixed_action_setting == True:
            action = jnp.asarray([self.cfg.fixed_action])
        
        
        
        # Use the most recent best_ask and best_bid values
        #These values may be my own orders... I clearly don't want to base myself off them. Get from world state directly.
        ask_mask=(world_state.ask_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        bid_mask=(world_state.bid_raw_orders[:,job.cst.OrderSideFeat.TID.value]!=agent_params.trader_id)
        
        masked_asks=jnp.where(ask_mask[:, jnp.newaxis], world_state.ask_raw_orders, -1)
        masked_bids=jnp.where(bid_mask[:, jnp.newaxis], world_state.bid_raw_orders, -1)
    
        best_ask, best_bid = job.get_best_bid_and_ask(self.world_config,masked_asks,masked_bids)
        #If the book is empty here, we get -1 back.

        empty_book = jnp.where((best_ask == -1) | (best_bid == -1),True, False)
        #We then replace with the last known bbid, bask, which in turn should have been forward filled, but is most likely our own order which will be v far from the last true market price. 
        best_ask = jnp.int32((best_ask // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((best_bid // self.world_config.tick_size) * self.world_config.tick_size)
        #The world state will have the mid-price propagated. Doing this just for the sake of logging to have reasonable averages. 
        #If the book is empty, the quants are put to 0 anyway. 
        best_bid = jnp.where(empty_book, world_state.best_bids[-1,0], best_bid)
        best_ask = jnp.where(empty_book, world_state.best_asks[-1,0], best_ask)

        
        if self.cfg.bob_v0==1:
            bid_quants = jnp.array([1, 2, 0,], dtype=jnp.int32)
            ask_quants = jnp.array([1, 0, 2], dtype=jnp.int32)
        elif self.cfg.bob_v0==2:
            bid_quants = jnp.array([2, 3, 1, 4, 0], dtype=jnp.int32)
            ask_quants = jnp.array([2, 1, 3, 0, 4], dtype=jnp.int32)
        elif self.cfg.bob_v0==5:
            bid_quants = jnp.array([5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0], dtype=jnp.int32)
            ask_quants = jnp.array([5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10], dtype=jnp.int32)
        elif self.cfg.bob_v0==10:
            bid_quants = jnp.array([10, 11, 9, 12, 8, 13, 7, 14, 6, 15,
                                         5, 16, 4, 17, 3, 18, 2, 19, 1, 20, 0], dtype=jnp.int32)
            ask_quants = jnp.array([10, 9 , 11, 8, 12, 7, 13, 6, 14, 5,
                                         15, 4, 16, 3, 17, 2, 18, 1, 19, 0, 20], dtype=jnp.int32)
        else:
            raise ValueError("cfg.bob_v0 must be one of [1,2,5,10]")

        scale=self.cfg.fixed_quant_value # Typically 1 
        bid_quant = bid_quants[action]*scale
        ask_quant = ask_quants[action]*scale
        bid_quant=jnp.where(empty_book, 0, bid_quant)
        ask_quant=jnp.where(empty_book, 0, ask_quant)
    
        #jax.debug.print("bid_price after:{}",bid_price)
        #jax.debug.print("ask_price after:{}",ask_price)
        
        # --------------- Construct messages ---------------#
        # Message components (2 messages: bid then ask)
        types = jnp.asarray([1, 1], dtype=jnp.int32)  # 1=limit order
        sides = jnp.asarray([1, -1], dtype=jnp.int32)  # 1=bid, -1=ask
        quants = jnp.asarray([bid_quant, ask_quant], dtype=jnp.int32)
        prices = jnp.asarray([best_bid, best_ask], dtype=jnp.int32)
        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)




        quants = quants.flatten() # Flatten so they have the same shape
        prices = prices.flatten()
        
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)

        
        # Time fields (replicated for each message)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )



        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)

        # action_msgs = jnp.where((action==9) & (~empty_book), jnp.ones_like(action_msgs)*9, action_msgs)  
        #jax.debug.print("action_msgs mm:{}",action_msgs)

        return action_msgs,{"bid_quant":bid_quant,"ask_quant":ask_quant,"empty_book":empty_book,"bid_distance_from_best":0,"ask_distance_from_best":0,"posted_bid_price":0,"posted_ask_price":0}
    
    def _getActionMsgs_fixedPrice(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
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
            FT = ((best_ask) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                 * self.world_config.tick_size).astype(jnp.int32)
            BI = best_bid + self.world_config.tick_size*self.cfg.n_ticks_offset #BID inside, slightly more aggresive buying
            NT = best_bid
            PP = best_bid - self.world_config.tick_size*self.cfg.n_ticks_offset
            if action.shape[0]//2 == 4:
                return FT, M, NT, PP
            elif action.shape[0]//2 == 3:
                return BI, NT, PP
            elif action.shape[0]//2 == 2:
                return NT, PP
            elif action.shape[0]//2 == 1:
                return NT

        def sell_task_prices(best_ask, best_bid):
            # FT = best_bid
            FT = ((best_bid) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                 * self.world_config.tick_size).astype(jnp.int32)
            AI = best_ask - self.world_config.tick_size*self.cfg.n_ticks_offset #Ask inside, slightly more aggresive selling
            NT = best_ask
            PP = best_ask + self.world_config.tick_size*self.cfg.n_ticks_offset
            if action.shape[0]//2 == 4:
                return FT, M, NT, PP
            elif action.shape[0]//2 == 3:
                return AI, NT, PP
            elif action.shape[0]//2 == 2:
                return NT, PP
            elif action.shape[0]//2 == 1:
                return NT

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types = jnp.ones((self.cfg.n_actions,), jnp.int32)
        sides_bids = jnp.ones((self.cfg.n_actions // 2,), jnp.int32)  # Use integer division to ensure result is an int
        sides_asks = (-1) * jnp.ones((self.cfg.n_actions // 2,), jnp.int32)
        sides = jnp.concatenate([sides_bids, sides_asks])
        trader_ids = jnp.ones((self.cfg.n_actions,), jnp.int32) * agent_params.trader_id #This agent will always have the same (unique) trader ID
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.n_actions,), self.world_config.placeholder_order_id, dtype=jnp.int32)

        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (self.cfg.n_actions, 2)
        )
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
   
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)


        sell_levels=sell_task_prices(best_ask, best_bid)
        sell_levels = jnp.asarray(sell_levels) 
    
        buy_levels=buy_task_prices(best_ask, best_bid)
        buy_levels = jnp.asarray(buy_levels)

        price_levels=jnp.concatenate([buy_levels,sell_levels])
        

        # --------------- 02 info for deciding prices ---------------
        quants = action.astype(jnp.int32)
        prices=price_levels
     
        #quants, prices = normal_quant_price(price_levels, action)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        #jax.debug.print('action_msgs\n {}', action_msgs)
        return action_msgs,{"bid_quant":0,"ask_quant":0,"empty_book":False,"bid_distance_from_best":0,"ask_distance_from_best":0,"posted_bid_price":0,"posted_ask_price":0}
        # ============================== Get Action_msgs ==============================

    def _getActionMsgs_spread_skew(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
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
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        mid_price = (best_ask + best_bid) / 2


        #jax.debug.print("Best Ask: {}, Best Bid: {}, Mid Price: {}", best_ask, best_bid, mid_price)
        #jax.debug.print("action: {}", action)
        #jax.debug.print("best asks: {}",best_ask)
        #jax.debug.print("best bids: {}", best_bid)
        
        # Get current spread
        current_spread = best_ask - best_bid
        
        # Map action to spread and skew parameters
        # action = spread_type * 3 + skew_type
        spread_type = action // 3  # 0 = tight, 1 = wide
        skew_type = action % 3     # 0 = neutral, 1 = ask skew, 2 = bid skew
        
        # Define spread multipliers
        # Tight spread = 1.0 * current_spread
        # Wide spread = 2.0 * current_spread
        spread_multiplier = jnp.where(spread_type == 0, 1.0, self.cfg.spread_multiplier)
        new_spread = current_spread * spread_multiplier
        
        # Define skew amounts (in ticks)
        # The skew will shift the mid price by this many ticks
        skew_ticks = jnp.where(skew_type == 0, -self.cfg.skew_multiplier,   # bid skew - shift down by skew_multiplier ticks
                            jnp.where(skew_type == 1, 0,  # neutral - no skew
                            self.cfg.skew_multiplier))   # ask skew - shift up by skew_multiplier ticks


        # Calculate skewed mid price
        if self.cfg.multiplier_type == "spread":
            skewed_mid = mid_price + skew_ticks * new_spread
        elif self.cfg.multiplier_type == "tick":
            skewed_mid = mid_price + skew_ticks * self.world_config.tick_size

        #jax.debug.print("mid price: {}", mid_price)
        #jax.debug.print("skew ticks: {}", skew_ticks)
        #jax.debug.print("skewed mid: {}", skewed_mid)
        
        # Calculate final bid and ask prices
        half_spread = new_spread // 2
        bid_price = skewed_mid - half_spread
        ask_price = skewed_mid + half_spread
        






        #spread4 = current_spread * self.cfg.spread_multiplier
        #half_spread4 = spread4 // 2
        #bid_price_4 = mid_price - half_spread4
        #ask_price_4 = mid_price + half_spread4
        #bid_price = bid_price_4
        #ask_price = ask_price_4

        #jax.debug.print("bid price: agent {}", bid_price)
        #jax.debug.print("ask price: agent{}", ask_price)





        #jax.debug.print("best asks: {}", best_ask)
        # Ensure prices are a multiple of tick size
        bid_price = (bid_price // self.world_config.tick_size) * self.world_config.tick_size
        ask_price = (ask_price // self.world_config.tick_size) * self.world_config.tick_size
        
        #jax.debug.print("bid price: agent {}", bid_price)
        #jax.debug.print("ask price: agent{}", ask_price)


        # Set fixed quantities
        bid_quant = self.cfg.fixed_quant_value
        ask_quant = self.cfg.fixed_quant_value
        
        # Construct order messages
        types = jnp.array([1, 1], dtype=jnp.int32)  # 1 = limit order
        sides = jnp.array([1, -1], dtype=jnp.int32)  # 1 = bid, -1 = ask
        quants = jnp.array([bid_quant, ask_quant], dtype=jnp.int32)


        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32)
        #print("prices:{}",prices)

        prices = jnp.array([bid_price, ask_price], dtype=jnp.int32).reshape(-1)
        #print("prices:{}",prices)

        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)
        
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        
        # Time fields
        times = jnp.resize(world_state.time + self.cfg.time_delay_obs_act, (2, 2))


        # jax.debug.print("types shape: {}", types.shape)
        # jax.debug.print("sides shape: {}", sides.shape)
        # jax.debug.print("quants shape: {}", quants.shape)
        # jax.debug.print("prices shape: {}", prices.shape)
        # jax.debug.print("order_ids shape: {}", order_ids.shape)
        # jax.debug.print("trader_ids shape: {}", trader_ids.shape)
        # jax.debug.print("times shape: {}", times.shape)
    



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
        
        return action_msgs,{"bid_quant":bid_quant,"ask_quant":ask_quant,"empty_book":False,"bid_distance_from_best":0,"ask_distance_from_best":0,"posted_bid_price":0,"posted_ask_price":0}



    def _getActionMsgs_directional_trading(self, action: jax.Array, world_state: WorldState, agent_state: MMEnvState, agent_params: MMEnvParams):
        '''Action space for directional trading. The agent can either:
            - Do nothing (action = 0)
            - Buy at best ask (action = 1)
            - Sell at best bid (action = 2)
        
        Always sends two messages for compatibility with message filtering
        '''
        # Use the most recent best_ask and best_bid values
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        
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
        trader_ids = jnp.full(2, agent_params.trader_id, dtype=jnp.int32)
        
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        
        # Time fields (replicated for each message)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (2, 2)  # Shape (2 messages, 2 time fields)
        )
        
        # Stack components into message array
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids, trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)
        
        # Debug print final messages
        #jax.debug.print("Final Action Messages:\n{}", action_msgs)
        return action_msgs,{"bid_quant":bid_quant,"ask_quant":ask_quant,"empty_book":False,"bid_distance_from_best":0,"ask_distance_from_best":0,"posted_bid_price":0,"posted_ask_price":0}



    def get_messages(self, action: jax.Array, world_state: WorldState, agent_state:MMEnvState, agent_params: MMEnvParams):
        '''Get the action and cancel messages'''
        def doNothing_callback(action,action_msgs,cancel_msgs,empty_book):
            if action==9 & empty_book==True:
                print("Market Maker doing nothing this step")
                print("Action messages sent: ",action_msgs)
                print("Cancel messages sent: ",cancel_msgs)
                print("Empty book: ",empty_book)
        
        action_msgs,extras = self.action_fn(action,
                                    world_state,
                                    agent_state,
                                    agent_params)

        cancel_msgs_bid = job.getCancelMsgs(
            bookside = world_state.bid_raw_orders,
            agentID = agent_params.trader_id,
            size = self.cfg.num_messages_by_agent//4,
            side = 1,
            cancel_time = world_state.time[0],
            cancel_time_ns = world_state.time[1]
        )

        cancel_msgs_ask = job.getCancelMsgs(
            bookside = world_state.ask_raw_orders,
            agentID = agent_params.trader_id,
            size = self.cfg.num_messages_by_agent//4,
            side = -1,
            cancel_time = world_state.time[0],
            cancel_time_ns = world_state.time[1]
        )
        cancel_msgs = jnp.concatenate([cancel_msgs_bid, cancel_msgs_ask], axis=0)

        #jax.debug.print(f"Market Maker action msg: {action_msgs}")
        #jax.debug.print(f"Market Maker cancel msg: {mm_cnl_msgs}")

        # Do filtering to net cancellations in MM)
        action_msgs, cancel_msgs = self._filter_messages(action_msgs, cancel_msgs)

        #jax.debug.print("action messages order mm: {}", action_msgs)
        #jax.debug.print("cancel messages order mm: {}", cancel_msgs)
        # cancel_msgs = jnp.where(action_msgs==9*jnp.ones_like(action_msgs), cancel_msgs*0, cancel_msgs)
        # action_msgs = jnp.where(action_msgs==9*jnp.ones_like(action_msgs), action_msgs*0, action_msgs)  
        # jax.debug.callback(doNothing_callback,action,action_msgs,cancel_msgs,extras["empty_book"])
        return action_msgs, cancel_msgs,extras











    def unwind_ref_price(self,
            bestasks: jax.Array,
            bestbids: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: MMEnvState,
            agent_params: MMEnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:   
        
        '''Function to create an artifical trade which liquidates the agent's position.
            cfg.rerefernce price sets the price of the trade


            NOTE: The prices in the trade here are NOT normalised by tick size. This is correct, as it is "as if" we sent
            and order with these prices. The get_reward, will see the trade, and normalsie the prices following. No change needed.
        '''


        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0) 
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = (agent_params.trader_id == executed[:, 6]) | (agent_params.trader_id == executed[:, 7]) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0) 

        #Find agent Buys and Agent sells from agent Trades:
        #The below mask puts passive buys or aggresive buys into "agent buys".
        #Logic: Q>0, TIDs=BUY; Q<0 TIDa= BUY
        mask_buy = (((agentTrades[:, 1] >= 0) & (agent_params.trader_id == agentTrades[:, 6]))|
                    ((agentTrades[:, 1] < 0)  & (agent_params.trader_id == agentTrades[:, 7])))
        mask_sell = (((agentTrades[:, 1] < 0) & (agent_params.trader_id == agentTrades[:, 6]))|
                     ((agentTrades[:, 1] >= 0)  & (agent_params.trader_id == agentTrades[:, 7])))
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
            remainingTime = self.world_config.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= 5  # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1

        averageMidprice = ((bestbids[:, 0] + bestasks[:, 0]) / 2).mean() #should be a float

        new_time = time + self.cfg.time_delay_obs_act


        is_sell_task = jnp.where(new_inventory > 0, 1, 0)
        FT_price = jax.lax.cond(
            is_sell_task,
            lambda: ((bestbids[-1, 0]) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            lambda: (( bestasks[-1, 0])// self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
        )

        def place_refprice_trade(trades, price, quant, time):
            '''Place a doom trade at a trade at specified price to close out our mm agent at the end of the episode.'''
            trade = job.create_trade(
                price, quant, -666666,  agent_params.trader_id + state.customIDcounter+ 1 +self.cfg.num_action_messages_by_agent, *time, -666666, agent_params.trader_id) #-66666 is an artifical OID for the artifical person we "traded with" to close our position
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
            state: MMEnvState,
            params: MMEnvParams,
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
            mkt_p = (1 - is_sell_task) * self.world_config.maxint // self.world_config.tick_size * self.world_config.tick_size
            side = (1 - is_sell_task*2)
            # TODO: this addition wouldn't work if the ns time at index 1 increases to more than 1 sec
            new_time = time + self.cfg.time_delay_obs_act
            mkt_msg = jnp.array([
                # type, side, quant, price
                #NOTE: MAKING ZERO TO TEST SELL AT MID PRICE jnp.abs(state.inventory)
                1, side, 0 , mkt_p,
                agent_params.trader_id,
                agent_params.trader_id + state.customIDcounter + self.cfg.n_actions,  # unique order ID for market order
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
                price, quant, -666666,  agent_params.trader_id + state.customIDcounter+ 1 +self.cfg.n_actions, *time, -666666, agent_params.trader_id)
            trades = job.add_trade(trades, doom_trade)
            return trades
         
        #-----check if ep over-----#
        if self.ep_type == 'fixed_time':
            remainingTime = self.world_config.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
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
                agent_params.trader_id,
                self.cfg.num_action_messages_by_agent//2,
                1  # bid
            )
        cnl_msg_ask = job.getCancelMsgs(
                state.ask_raw_orders,
                agent_params.trader_id,
                self.cfg.num_action_messages_by_agent//2,
                -1  # ask
            )
        
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)
        
        (asks, bids, trades), (new_bestask, new_bestbid) = job.scan_through_entire_array_save_bidask(self.cfg,key,
            cnl_msgs, 
            (asks, bids, trades),
            # TODO: this returns bid/ask for last n_data_msg_per_step only, could miss the direct impact of actions
            self.n_data_msg_per_step
        )
   
        #Filter our new message through the orderbook#
        (asks, bids, trades), (new_bestask, new_bestbid) = job.cond_type_side_save_bidask(self.cfg,
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
            agent_params.trader_id,
            1, 
            1  # bids
        )
        cnl_msg_ask = job.getCancelMsgs(
            asks,
            agent_params.trader_id,
            1,
            -1  # ask side
        )
        cnl_msgs = jnp.concatenate([cnl_msg_bid, cnl_msg_ask], axis=0)

        (asks, bids, trades), (new_bestask, new_bestbid) = job.scan_through_entire_array_save_bidask(self.cfg,key,
            cnl_msgs, 
            (asks, bids, trades),
            # TODO: this returns bid/ask for last n_data_msg_per_step only, could miss the direct impact of actions
            self.n_data_msg_per_step
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
            #lambda: ((0.75 * bestbid[0]) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            #lambda: ((1.25 * bestask[0]) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            lambda: ((bestbid[0]+bestask[0])//2 // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            lambda: ((bestbid[0]+bestask[0])//2 // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32), #For sell at opposite test
        )
        #jax.debug.print('ep_is_over: {}; quant_still_left: {}; remainingTime: {}; doom price :{}', ep_is_over, quant_still_left, remainingTime,doom_price)
        trades = jax.lax.cond(
            ep_is_over & (quant_still_left > 0),  # Check if episode is over and we still have remaining quantity
            place_doom_trade,  # Place a doom trade with unfavorable price
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, doom_price, 0, time  # Inv +ve means incoming is sell so standing buy.
        )#jnp.sign(state.inventory) * quant_still_left
        agent_trades = job.get_agent_trades(trades, agent_params.trader_id)
       # price_quants = self._get_executed_by_price(agent_trades)
        doom_quant = ep_is_over * quant_still_left

        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant




    def _extract_agent_trade_stats(self, trades :jax.Array, trader_id : jax.typing.ArrayLike) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # Find trades by agent vs by others
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)

        # print(f"agent_params.trader_id: {agent_params.trader_id}")
        
        mask2 = ((trader_id == executed[:, job.cst.TradesFeat.PASS_TID.value]) |
                    (trader_id == executed[:, job.cst.TradesFeat.AGRS_TID.value])) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        otherTrades = jnp.where(mask2[:, jnp.newaxis], 0, executed)
    

        #Find agent Buys and Agent sells from agent Trades:
        #The below mask puts passive buys or aggresive buys into "agent buys".
        #Logic: Q>0, TIDs=BUY; Q<0 TIDa= BUY
        mask_buy = (((agentTrades[:, 1] >= 0) & (trader_id == agentTrades[:, job.cst.TradesFeat.PASS_TID.value])) |
                    ((agentTrades[:, 1] < 0)  & (trader_id == agentTrades[:, job.cst.TradesFeat.AGRS_TID.value])))
        mask_sell = (((agentTrades[:, 1] < 0) & (trader_id == agentTrades[:, job.cst.TradesFeat.PASS_TID.value])) |
                     ((agentTrades[:, 1] >= 0)  & (trader_id == agentTrades[:, job.cst.TradesFeat.AGRS_TID.value])))
        agent_buys=jnp.where(mask_buy[:, jnp.newaxis], agentTrades, 0)
        agent_sells=jnp.where(mask_sell[:, jnp.newaxis], agentTrades, 0)


        #TODO: Not very optimised calculating these masks and arrays twice, good enough for now. 
        #For the purpose of assigning rebates, differentiate between passive and aggressive trades
        mask_buy_passive = ((agentTrades[:, 1] >= 0) & (trader_id == agentTrades[:, job.cst.TradesFeat.PASS_TID.value]))
        mask_sell_passive = ((agentTrades[:, 1] < 0) & (trader_id == agentTrades[:, job.cst.TradesFeat.PASS_TID.value]))
        agent_passive_buys=jnp.where(mask_buy_passive[:, jnp.newaxis], agentTrades, 0)
        agent_passive_sells=jnp.where(mask_sell_passive[:, jnp.newaxis], agentTrades, 0)
        return agentTrades, otherTrades, agent_buys, agent_sells, agent_passive_buys, agent_passive_sells



    def get_reward(self, 
                    world_state: WorldState, 
                    agent_state: MMEnvState, 
                    agent_params: MMEnvParams, 
                    trades: jax.Array,
                    bestasks: jax.Array,
                    bestbids: jax.Array,
                    ep_done_time: bool) -> Tuple[jax.Array,dict]:
        '''Return the reward. There are a few options for reward funciton and assocaited hyper parameters:
        '''
        # ====================01 get reward stats ==========================================#
        #Notice, normalise prices in reward by tick size. On state prices are not normalised 
        #Being constient with exec. Cash balance and pnl etc are normalised in state, also consitent


        #########################################################
        # Get reward stats before unwind
        #########################################################

        _, _, agent_buys_before_unwind, agent_sells_before_unwind,_,_ = \
                self._extract_agent_trade_stats(trades, agent_params.trader_id)
        
        #Find amount bought and sold in the step
        buyQuant=jnp.abs(agent_buys_before_unwind[:, job.cst.TradesFeat.Q.value]).sum()
        sellQuant=jnp.abs(agent_sells_before_unwind[:, job.cst.TradesFeat.Q.value]).sum()

        new_inventory_before_final_trade=agent_state.inventory+buyQuant - sellQuant

        #These values may be used throughout, they do not change even in the event of a fictional trade
        averageMidprice = ((bestbids[:, 0] + bestasks[:, 0]) / 2).mean() #should be a float
        last_mid_price = (bestbids[-1,0] + bestasks[-1,0]) / 2


        #########################################################################################
        # Add artificial trade if episode is done
        # Important: this artificial trade is not saved, its just used to calculate the reward
        #########################################################################################

        def add_fictional_trade(trades, price, quant):
            '''Place a doom trade at a trade at specified price to close out our mm agent at the end of the episode.'''
            trade = job.create_trade(
                price, quant, self.world_config.artificial_order_id_end_episode,  self.world_config.placeholder_order_id, 0,0, self.world_config.artificial_trader_id_end_episode, agent_params.trader_id) #-66666 is an artifical OID for the artifical person we "traded with" to close our position
            trades = job.add_trade(trades, trade)
            return trades


        ##Get the price to unwind at based on the config
        if self.cfg.unwind_price == "mid_avg":
            unwind_price = averageMidprice
        elif self.cfg.unwind_price == "mid":
            unwind_price = last_mid_price
        elif self.cfg.unwind_price == "far_touch":
            unwind_price = jax.lax.cond(new_inventory_before_final_trade > 0,
                    lambda: bestbids[-1][0],
                    lambda: bestasks[-1][0])
        else:
            raise ValueError("Invalid unwind price type.")
        penalty=self.cfg.unwind_price_penalty * self.world_config.tick_size
        penalty=jax.lax.cond(
            new_inventory_before_final_trade >0,
            lambda: penalty,
            lambda: -penalty
        )

        trades = jax.lax.cond(
            ep_done_time & (jnp.abs(new_inventory_before_final_trade) > 0),  # Check if episode is over and we still have remaining quantity
            add_fictional_trade,  # Place a midprice trade
            lambda trades, b, c: trades,  # If not, return the existing trades
            trades, unwind_price-penalty, jnp.sign(new_inventory_before_final_trade) * jnp.abs(new_inventory_before_final_trade) # Inv +ve means incoming is sell so standing buy.
        )
        forced_unwind=new_inventory_before_final_trade * ep_done_time

        #jax.debug.print("trades mm env: {}", trades)


        #########################################################
        # Get trades after fictional trade
        #########################################################

        _, otherTrades, agent_buys, agent_sells, passive_buys, passive_sells = \
                self._extract_agent_trade_stats(trades, agent_params.trader_id)




        def large_reward_callback(reward, abs_reward, trades, window_index, inventory_pnl, buy_pnl, sell_pnl, delta_mid, inventory):
            if abs_reward > 100_000:
                print(f"Large reward: {reward}")
                print(f"Trades: {trades}")
                print(f"Window index: {window_index}")
                print(f"Inventory PnL: {inventory_pnl}")
                print(f"Buy PnL: {buy_pnl}")
                print(f"Sell PnL: {sell_pnl}")
                print(f"Delta Mid: {delta_mid}")
                print(f"Inventory: {inventory}")

        #########################################################
        # Get reward
        #########################################################


        #Find the new obsvered mid price at the end of the step.
        #non normalized=> going on state
        mid_price_end = (bestbids[-1][0] + bestasks[-1][0]) / 2

        #Real Revenue calcs: (actual cash flow+actual value of portfolio)
        income=(agent_sells[:, job.cst.TradesFeat.P.value].astype(jnp.float32)/self.world_config.tick_size * 
                jnp.abs(agent_sells[:, job.cst.TradesFeat.Q.value])).sum()
        outgoing=(agent_buys[:, job.cst.TradesFeat.P.value].astype(jnp.float32)/self.world_config.tick_size * 
                  jnp.abs(agent_buys[:, job.cst.TradesFeat.Q.value])).sum() 

        buyQuant=jnp.abs(agent_buys[:, job.cst.TradesFeat.Q.value]).sum()
        sellQuant=jnp.abs(agent_sells[:, job.cst.TradesFeat.Q.value]).sum()

        new_inventory=agent_state.inventory+buyQuant - sellQuant

        rebate_value = (
            (passive_buys[:, job.cst.TradesFeat.P.value].astype(jnp.float32)/self.world_config.tick_size * 
                jnp.abs(passive_buys[:, job.cst.TradesFeat.Q.value])).sum() + 
            (passive_sells[:, job.cst.TradesFeat.P.value].astype(jnp.float32)/self.world_config.tick_size * 
                jnp.abs(passive_sells[:, job.cst.TradesFeat.Q.value])).sum()
        )
        rebate_income = rebate_value * (self.cfg.rebate_bps / 10_000)
        

        # Compute a reference price based on the config
        if self.cfg.reference_price == "mid_avg":
            ref_buy = averageMidprice
            ref_sell = averageMidprice
            reference_price = averageMidprice
        elif self.cfg.reference_price == "mid":
            ref_buy = last_mid_price
            ref_sell = last_mid_price
            reference_price = last_mid_price
        elif self.cfg.reference_price == "far_touch":
            # For a long position, use the best bid; for a short, the best ask.
            ref_buy = bestasks[-1][0]
            ref_sell = bestbids[-1][0]
            reference_price = jax.lax.cond(new_inventory > 0,
                                        lambda: ref_buy,
                                        lambda: ref_sell)
        elif self.cfg.reference_price == "near_touch":
            # For a long position, use the best ask; for a short, the best bid.
            ref_buy = bestbids[-1][0]
            ref_sell = bestasks[-1][0]
            reference_price = jax.lax.cond(new_inventory > 0,
                                        lambda: ref_buy,
                                        lambda: ref_sell)
        else:
            raise ValueError("Invalid reference price type.")

        #PnL,== cash balance change
        PnL=(income-outgoing+rebate_income)
        # Keep track of overall cash balance (same as overall PnL)
        new_cash_balance = agent_state.cash_balance + PnL
        inventoryValue=new_inventory*(reference_price)/self.world_config.tick_size#Mark to market inventory value
        netWorth=new_cash_balance+inventoryValue

        #calculate a fraction of total market activity attributable to us.
        other_exec_quants = jnp.abs(otherTrades[:, job.cst.TradesFeat.Q.value]).sum()
        TradedVolume = buyQuant + sellQuant
        market_share = TradedVolume / (TradedVolume + other_exec_quants)

        #=========02 Get rewards============================##

        #------------A) spooner Rewards-------------------------#       
        #Inventory PnL: The value obtained due to the midprice changing and us holding inventory
        InventoryPnL= agent_state.inventory*(mid_price_end-world_state.mid_price)/self.world_config.tick_size

        buyPnL = (((ref_buy - agent_buys[:, 0])/self.world_config.tick_size * jnp.abs(agent_buys[:, 1])).sum())
        sellPnL = (((agent_sells[:, 0] - ref_sell)/self.world_config.tick_size * jnp.abs(agent_sells[:, 1])).sum())
        

        #A1)Spooner paper reward
        reward_spooner = buyPnL + sellPnL +rebate_income+ InventoryPnL 

        #A2)spooner_damped
        reward_spooner_damped = buyPnL + sellPnL + rebate_income + InventoryPnL - (self.cfg.inventoryPnL_eta*InventoryPnL)

        #A2.5 Spooner Asym Dampened
        reward_spooner_asym_damped = buyPnL + sellPnL + rebate_income + InventoryPnL - jnp.maximum(0,(self.cfg.inventoryPnL_eta*InventoryPnL))

        #A2.75 Spooner Asym Damped, actually
        reward_spooner_asym_damped2 = buyPnL + sellPnL + rebate_income + self.cfg.inventoryPnL_gamma*(InventoryPnL - jnp.maximum(0,self.cfg.inventoryPnL_eta*InventoryPnL))

        #A3) Spooner Scaled
        scaledInventoryPnL=InventoryPnL//(jnp.abs(agent_state.inventory)+1)
        reward_spooner_scaled=buyPnL + sellPnL + rebate_income + self.cfg.inventoryPnL_eta*(InventoryPnL - (1-self.cfg.inventoryPnL_eta)*jnp.maximum(0,InventoryPnL) )
        
        #----------------------B) Complex reward---------------------------------------------#
        inventory_change= buyQuant - sellQuant
        inventoryPnL_eta = self.cfg.inventoryPnL_eta
        unrealizedPnL_lambda = self.cfg.unrealizedPnL_lambda
        asymmetrically_dampened_lambda = self.cfg.inventoryPnL_eta
        avg_buy_price = jnp.where(buyQuant > 0, (agent_buys[:, 0]/ buyQuant * jnp.abs(agent_buys[:, 1])).sum(), 0)  
        avg_sell_price = jnp.where(sellQuant > 0, (agent_sells[:, 0]/ sellQuant * jnp.abs(agent_sells[:, 1])).sum(), 0)
        approx_realized_pnl = jnp.minimum(buyQuant, sellQuant) * (avg_sell_price - avg_buy_price) 
        approx_unrealized_pnl = jnp.where( 
            inventory_change > 0,
            inventory_change * (averageMidprice - avg_buy_price),  # Excess buys
            jnp.abs(inventory_change) * (avg_sell_price - averageMidprice)  # Excess sells
        )
  
        reward_complex = approx_realized_pnl + unrealizedPnL_lambda * approx_unrealized_pnl +  inventoryPnL_eta * jnp.minimum(InventoryPnL,InventoryPnL*asymmetrically_dampened_lambda) #Last term adds negative inventory PnL without dampening
    
        #--------------------C) Portfolio Value--------------#
        reward_portfolio_value=new_inventory*(reference_price/self.world_config.tick_size)+new_cash_balance
        def debug_callback_times(world_state,agent_state, reward_portfolio_value,new_inventory, reference_price, new_cash_balance,income,outgoing,rebate_income):
            if world_state.step_counter in [44,45,46,47]:
                print("Reward PV:", reward_portfolio_value)
                print("new_inventory: ", new_inventory)
                print("ref_price: ", reference_price)
                print("new_cash_balance: ", new_cash_balance)
                print("old cash balance: ", agent_state.cash_balance)
                print("income: ", income)
                print("outgoing: ", outgoing)
                print("rebate_income: ", rebate_income)


        # jax.debug.callback(debug_callback_times, world_state,agent_state, reward_portfolio_value,new_inventory, reference_price, new_cash_balance,income,outgoing,rebate_income)
        #----------------- D) Delta Portfolio Value--------#
        #Get old ref price
        if self.cfg.reference_price in ("mid","mid_avg"):
            old_reference_price = world_state.mid_price
        elif self.cfg.reference_price == "far_touch":
            # For a long position, use the best bid; for a short, the best ask. (this is realistic)
            old_reference_price = jax.lax.cond(agent_state.inventory > 0,
                                        lambda: world_state.best_asks[-1][0],
                                        lambda: world_state.best_bids[-1][0])
        elif self.cfg.reference_price == "near_touch":
            # For a long position, use the best ask; for a short, the best bid. (this is not realistic, but might be useful for training)
            old_reference_price = jax.lax.cond(agent_state.inventory > 0,
                                        lambda: world_state.best_bids[-1][0],
                                        lambda: world_state.best_asks[-1][0])
        else:
            raise ValueError("Invalid reference price type.")
        #old net worth
        old_netWorth=old_reference_price/self.world_config.tick_size*agent_state.inventory+agent_state.cash_balance
        delta_netWorth=netWorth-old_netWorth
        

        #===================== 03) Set reward based on config file==================#
        if self.cfg.reward_function == "portfolio_value": #Cash balance + value of portfolio at midprice (or BB/BA)
            reward = reward_portfolio_value
        elif self.cfg.reward_function == "buy_sell_pnl":
            reward = (buyPnL + sellPnL)
        elif self.cfg.reward_function == "complex":
            #Skip
            reward =reward_complex
        elif self.cfg.reward_function == "zero_inv":
            #Skip, debugging only
            reward = -jnp.abs(new_inventory)
        elif self.cfg.reward_function=="spooner":
            reward=reward_spooner
        elif self.cfg.reward_function=="spooner_damped":
            reward=reward_spooner_damped
        elif self.cfg.reward_function=="spooner_asym_damped":
            reward=reward_spooner_asym_damped
        elif self.cfg.reward_function=="spooner_asym_damped2":
            reward=reward_spooner_asym_damped2
        elif self.cfg.reward_function=="spooner_scaled":
            #Skip
            reward=reward_spooner_scaled
        elif self.cfg.reward_function=="delta_portfolio_value":
            reward=delta_netWorth
        else:
            raise ValueError("Invalid reward_space specified.")
        
        # Set inventory penalty based on config file
        if self.cfg.inv_penalty == "none":
            inv_pen = 0.0
        elif self.cfg.inv_penalty == "linear":
            inv_pen = (-1) * jnp.abs(new_inventory)
            #jax.debug.print("inv_pen: {}", inv_pen)
        elif self.cfg.inv_penalty == "quadratic":
            inv_pen = (-1) * (new_inventory ** 2) / self.cfg.inv_penalty_quadratic_factor
            #jax.debug.print("new_inventory: {}", new_inventory)
            #jax.debug.print("inv_pen: {}", inv_pen)
        elif self.cfg.inv_penalty == "exp4":
            inv_pen = (-1) * (jnp.exp(new_inventory*4))
            #jax.debug.print("new_inventory: {}", new_inventory)
            #jax.debug.print("inv_pen: {}", inv_pen)
        elif self.cfg.inv_penalty == "threshold":
            inv_pen = jax.lax.cond(
                jnp.abs(new_inventory) > self.cfg.inv_penalty_threshold,
                lambda: (-1.0) * ((new_inventory ** 2)/self.cfg.inv_penalty_quadratic_factor),
                lambda: 0.0
            )
        else:
            raise ValueError("Invalid inventory penalty specified.")
        reward = reward + self.cfg.inv_penalty_lambda * inv_pen

        if self.cfg.clip_reward:
            reward = jnp.clip(reward, -10000, 10000)
        
        if self.cfg.volume_traded_bonus == "market_share":
            reward = reward + jnp.abs(reward) * market_share

        if self.cfg.exclude_extreme_spreads==True:
            #jax.debug.print("reward before: {}", reward)
            # If spread is larger than 0.1 (which would most likely only happen if the book is empty)
            all_spreads = (world_state.best_asks[:, 0] - world_state.best_bids[:, 0]) 
            mid_prices = (world_state.best_asks[:, 0] + world_state.best_bids[:, 0]) / 2
            #jax.debug.print("all_spreads: {}", all_spreads)
            spread_ratio = all_spreads / mid_prices
            #jax.debug.print("spread_ratio: {}", spread_ratio)
            any_large_spread = jnp.any(spread_ratio > 0.1)
            #jax.debug.print("any_large_spread: {}", any_large_spread)
            reward = jax.lax.cond(
                any_large_spread,
                lambda: 0.0,
                lambda: reward
            )
            #jax.debug.print("reward: {}", reward)
        # jax.debug.callback(large_reward_callback,reward,
        #                                     jnp.abs(reward),
        #                                     trades,
        #                                     world_state.window_index,
        #                                     InventoryPnL,
        #                                     buyPnL,
        #                                     sellPnL,
        #                                     mid_price_end - world_state.mid_price,
        #                                     agent_state.inventory)
        def large_pv_callback(reward, abs_reward, ep_done_time, window_index, netWorth,
                        delta_netWorth,
                        new_inventory,
                        new_cash_balance,
                        buyQuant,
                        sellQuant,
                        PnL,
                        reference_price,
                        old_reference_price,
                        delta_ref_price,
                        delta_mid_price,
                        mid_price_end,
                        mid_price,trades,agent_buys,agent_sells,
                        bidside,
                        askside,
                        bestbids,
                        bestasks):
            if abs_reward>100000:
                print(f"PV: {reward}")
                print(f"Abs Reward: {abs_reward}")
                print(f"Episode done: {ep_done_time}")
                print(f"Window index: {window_index}")
                print(f"Net Worth: {netWorth}")
                print(f"Delta Net Worth: {delta_netWorth}")
                print(f"Inventory: {new_inventory}")
                print(f"Cash Balance: {new_cash_balance}")
                print(f"Buy Quantity: {buyQuant}")
                print(f"Sell Quantity: {sellQuant}")
                print(f"PnL: {PnL}")
                print(f"Reference Price: {reference_price}")
                print(f"Old Reference Price: {old_reference_price}")
                print(f"Delta Reference Price: {delta_ref_price}")
                print(f"Delta Mid Price: {delta_mid_price}")
                print(f"Mid Price End: {mid_price_end}")
                print(f"Mid Price: {mid_price}")
                print(f"Trades: {trades}")
                print(f"Agent Buys: {agent_buys}")
                print(f"Agent Sells: {agent_sells}")
                print(f"Best Bids: {bestbids}")
                print(f"Bid Side: {bidside}")
                print(f"Best Asks: {bestasks}")
                print(f"Ask Side: {askside}")

        

        # jax.debug.callback(large_pv_callback,
        #                 reward_portfolio_value,
        #                 jnp.abs(reward),
        #                 ep_done_time,
        #                 world_state.window_index,
        #                 netWorth,
        #                 delta_netWorth,
        #                 new_inventory,
        #                 new_cash_balance,
        #                 buyQuant,
        #                 sellQuant,
        #                 PnL,
        #                 reference_price,
        #                 old_reference_price,
        #                 reference_price - old_reference_price,
        #                 mid_price_end - world_state.mid_price,
        #                 mid_price_end,
        #                 world_state.mid_price,
        #                 trades,
        #                 agent_buys,
        #                 agent_sells,
        #                 world_state.ask_raw_orders,
        #                 world_state.bid_raw_orders,
        #                 bestbids,
        #                 bestasks)
            
            
        return reward/self.cfg.reward_scaling_quo, {
            "reward":reward,
            "reward_portfolio_value":reward_portfolio_value,
            "end_of_ep_pv":reward_portfolio_value*ep_done_time,
            "reward_complex":reward_complex,
            "reward_spooner":reward_spooner,
            "reward_spooner_damped":reward_spooner_damped,
            "reward_spooner_asym_damped":reward_spooner_asym_damped,
            "reward_spooner_asym_damped2":reward_spooner_asym_damped2,
            "reward_spooner_scaled":reward_spooner_scaled,
            "reward_delta_portfolio_value":delta_netWorth,
            "forced_unwind":forced_unwind,
            "market_share": market_share,
            "inventoryValue":inventoryValue,
            "delta_mid_price": mid_price_end - world_state.mid_price,
            "buyPnL":buyPnL,
            "sellPnL":sellPnL,
            "invPnL":InventoryPnL,
            "PnL": PnL, 
            "cash_balance" : new_cash_balance,
            "netWorth":netWorth,
            "end_inventory":new_inventory,
            "mid_price":mid_price_end,
            "buyQuant":buyQuant,
            "sellQuant":sellQuant,
            "approx_realized_pnl":approx_realized_pnl,
            "approx_unrealized_pnl" : approx_unrealized_pnl,
            "InventoryPnL":InventoryPnL,
            "scaledInventoryPnL":scaledInventoryPnL,
            "other_exec_quants":other_exec_quants,
            "averageMidprice": averageMidprice # this should be on world info
        }



    def update_state_and_get_done_and_info(self, world_state:WorldState, agent_state_old: MMEnvState, extras) -> Tuple[MMEnvState, Dict]: 
        # Get new state
        new_inventory = extras["end_inventory"]
        new_PnL = agent_state_old.total_PnL + extras["PnL"]
        new_cash_balance = extras["cash_balance"]

        agent_state = MMEnvState(
            posted_distance_bid = extras["bid_distance_from_best"],
            posted_distance_ask = extras["ask_distance_from_best"],
            inventory = new_inventory,
            total_PnL = new_PnL,
            cash_balance= new_cash_balance    
        )
        
        # Get done
        done = self.is_terminal(world_state)

        # Get info
        info = {
            "reward":extras["reward"],
            "reward_portfolio_value":extras["reward_portfolio_value"],
            # "reward_complex":extras["reward_complex"],
            "reward_spooner":extras[ "reward_spooner"],
            "end_of_ep_pv":extras["end_of_ep_pv"],
            "reward_spooner_damped":extras["reward_spooner_damped"],
            "reward_spooner_asym_damped":extras[ "reward_spooner_asym_damped"],
            "reward_spooner_asym_damped2":extras[ "reward_spooner_asym_damped2"],
            "reward_delta_pv":extras["reward_delta_portfolio_value"],
            "total_PnL": agent_state.total_PnL,                           
            "done": done,
            "inventory": agent_state.inventory,
            "delta_mid_price":extras["delta_mid_price"],
            "market_share":extras["market_share"],
            "buyPnL":extras["buyPnL"],
            "forced_unwind":extras["forced_unwind"],
            "invPnL":extras["invPnL"],
            "posted_bid_price":extras["posted_bid_price"],
            "posted_ask_price":extras["posted_ask_price"],
            "bid_distance_from_best":extras["bid_distance_from_best"],
            "ask_distance_from_best":extras["ask_distance_from_best"],
            "ask_quant":extras["ask_quant"],
            "bid_quant":extras["bid_quant"],
            # "scaledInventoryPnL":extras["scaledInventoryPnL"],
            # "netWorth":extras["netWorth"],
            "sellPnL":extras["sellPnL"],
            # "buyQuant":extras["buyQuant"],
            # "sellQuant":extras["sellQuant"],
            "inventoryValue":extras["inventoryValue"],
            # "other_exec_quants":extras["other_exec_quants"],
            # "Step_PnL":extras["PnL"],
            # "InventoryPnL":extras["InventoryPnL"],
            # "approx_realized_pnl":extras["approx_realized_pnl"],
            # "approx_unrealized_pnl": extras["approx_unrealized_pnl"]
        }


        #jax.debug.print("info mm env: {}", info)


        return agent_state, done, info




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

    def get_observation(self, world_state,
                         agent_state,
                           agent_param,
                             total_messages,
                               old_time,
                                 old_mid_price,
                                   lob_state_before,
                                     normalize,
                                     flatten):
        """
        Wrapper function to call the appropriate observation function.
        """
        if self.cfg.observation_space == "engineered":
            return self._get_obs_engineered(world_state=world_state, 
                                       agent_state=agent_state,
                                       agent_param=agent_param,
                                       normalize=normalize,
                                       flatten=flatten)
        elif self.cfg.observation_space == "messages":
            return self._get_obs_msg(total_msgs=total_messages) 
        elif self.cfg.observation_space == "messages_new_tokenizer":
            return self._get_obs_msg_new_tokenizer(world_state=world_state,  
                                       total_msgs=total_messages, 
                                       old_time=old_time, 
                                       old_mid_price=old_mid_price, 
                                       lob_state_before=lob_state_before) 
        elif self.cfg.observation_space == "basic":
            return self._get_obs_basic(world_state=world_state, 
                                       agent_state=agent_state,
                                       agent_param=agent_param,
                                       normalize=normalize,
                                       flatten=flatten)
        else:
            raise ValueError("Invalid observation_space specified.")

        



    def get_action(self, action: jax.Array, world_state: MultiAgentState, agent_state: MMEnvState, agent_params: MMEnvParams):
        """
        Wrapper function to call the appropriate action function.
        """
        if self.cfg.action_space == "fixed_quants":
            return self.action_fn(action=action, world_state=world_state, agent_state=agent_state, agent_params=agent_params)
        elif self.cfg.action_space == "fixed_prices":
            return self.action_fn(action=action, world_state=world_state, agent_params=agent_params)
        elif self.cfg.action_space == "AvSt":
            return self.action_fn(action=action, world_state=world_state, agent_state=agent_state, agent_params=agent_params)
        elif self.cfg.action_space == "bobStrategy":
            return self.action_fn(action=action, world_state=world_state, agent_state=agent_state, agent_params=agent_params)
        elif self.cfg.action_space == "bobRL":
            return self.action_fn(action=action, world_state=world_state, agent_state=agent_state, agent_params=agent_params)
        elif self.cfg.action_space == "spread_skew":
            return self.action_fn(action=action, world_state=world_state, agent_params=agent_params)
        elif self.cfg.action_space == "directional_trading":
            return self._getActionMsgs_directional_trading(action=action, world_state=world_state, agent_params=agent_params)
        elif self.cfg.action_space == "simple":
            return self.action_fn(action=action, world_state=world_state, agent_state=agent_state, agent_params=agent_params)
        else:
            raise ValueError("Invalid action sspace specified.")
        


    #=================observation functions========================#    
    def _get_obs_msg(self, total_msgs: chex.Array):
        return total_msgs
    

    def _get_obs_msg_new_tokenizer(self, world_state: WorldState, total_msgs: chex.Array, old_time, old_mid_price, lob_state_before):
        """
        Construct a tokenized observation matching the pretraining format:
        [orderbook_tokens, message_tokens]
        """
        cfg_pretraining = get_config()

        #jax.debug.print("total_msgs:{}",total_msgs.shape)
        #jax.debug.print("Best bids:{}",state.best_bids[:, 0].shape)
        #jax.debug.print("Best asks:{}",state.best_asks[:, 0].shape)

        # Extract fields
        event = total_msgs[:, 0]
        direction = total_msgs[:, 1]
        order_id = total_msgs[:, 4]
        price = total_msgs[:, 3] // 100  # divide by 100 cause its also done in pretraining
        size = total_msgs[:, 2]
        time_s = total_msgs[:, 6]
        time_ns = total_msgs[:, 7]

        # delta_time: difference between consecutive time_s/time_ns
        delta_time_s = jnp.zeros_like(time_s)
        delta_time_ns = jnp.zeros_like(time_ns)
        delta_time_s = delta_time_s.at[0].set(0) #for now just set it to 0 because the messages are initialized with 0 but the time is with the actual time => large difference for very first value
        delta_time_ns = delta_time_ns.at[0].set(0)
        # For all other values
        ds = time_s[1:] - time_s[:-1]
        dns = time_ns[1:] - time_ns[:-1]
        ds = ds - (dns < 0)
        dns = jnp.where(dns < 0, dns + int(1e9), dns)
        delta_time_s = delta_time_s.at[1:].set(ds)
        delta_time_ns = delta_time_ns.at[1:].set(dns)

        #jax.debug.print("time_s: {}", time_s)
        #jax.debug.print("time_ns: {}", time_ns)
        #jax.debug.print("delta_time_s: {}", delta_time_s)
        #jax.debug.print("delta_time_ns: {}", delta_time_ns)

        ############################
        # Get the delta prices
        ############################

        # Extract best bid/ask prices (shape: [num_msgs])
        best_bid_prices = world_state.best_bids[:, 0] // 100 # Divide by 100 as in pretraining preprocessing
        best_ask_prices = world_state.best_asks[:, 0] // 100
        old_mid_price = old_mid_price // 100

        #jax.debug.print("old_mid_price: {}", old_mid_price)

        # Compute mid prices (shape: [num_msgs])
        mid_prices = (best_bid_prices + best_ask_prices) // 2  # integer division
        #jax.debug.print("mid_prices: {}", mid_prices[0])

        # Initialize delta_price array
        delta_price = jnp.zeros_like(mid_prices)

        # For the very first message, use the old mid price and compare it to that (delta price is 2* best mid price change)
        delta_price = delta_price.at[0].set(2 * (mid_prices[0] - old_mid_price))

        # Subsequent messages:
        # Interesting fact: one message can actually change both the best bid and the best ask (because we can consume a level and then be added at the level on our side)
        delta_price = delta_price.at[1:].set(
            (best_ask_prices[1:] - best_ask_prices[:-1]) + (best_bid_prices[1:] - best_bid_prices[:-1])
        )

        #############################
        # Tokenization 
        #############################

        ######
        # Messages
        ######

        event_dir_tok = direction.astype(jnp.uint8) * 4 + event.astype(jnp.uint8)
        event_dir_tok = event_dir_tok.astype(jnp.uint32) + cfg_pretraining.EVENT_START

        def split_and_offset(x, offset):
            x = x.astype(jnp.int32)  # Ensure input is really int32
            low = (x & 0xFFFF).astype(jnp.uint16) + offset      #  Lower 16 bits + offset
            high = ((x >> 16) & 0xFFFF).astype(jnp.uint16) + offset  #Upper 16 bits + offset
            return jnp.stack([low, high], axis=-1)  # Shape: (..., 2)

        order_id_tok      = split_and_offset(order_id,      cfg_pretraining.ORDER_ID_B_START)
        price_tok         = split_and_offset(price,         cfg_pretraining.PRICE_B_START)
        size_tok          = split_and_offset(size,          cfg_pretraining.SIZE_B_START)
        delta_time_s_tok  = split_and_offset(delta_time_s,  cfg_pretraining.TIME_B_START)
        delta_time_ns_tok = split_and_offset(delta_time_ns, cfg_pretraining.TIME_B_START)
        delta_price_tok   = split_and_offset(delta_price,   cfg_pretraining.PRICE_B_START)

        message_tokens = jnp.concatenate([
            event_dir_tok[:, None],  # (num_msgs, 1)
            order_id_tok,            # (num_msgs, 2)
            price_tok,               # (num_msgs, 2)
            size_tok,                # (num_msgs, 2)
            delta_time_s_tok,        # (num_msgs, 2)
            delta_time_ns_tok,       # (num_msgs, 2)
            delta_price_tok          # (num_msgs, 2)
        ], axis=-1)
        message_tokens_flat = message_tokens.reshape(-1)

        ######
        # Book
        ######

        #print("lob_state_before: {}", lob_state_before)

        # add time to the lob_state_before
        time_s = world_state.time[0]
        time_ns = world_state.time[1]

        lob_state_with_time = jnp.concatenate([jnp.array([time_s, time_ns]), lob_state_before]) # shape (42,)

        #  Split into uint16 tokens
        x_split = jax.lax.bitcast_convert_type(lob_state_with_time, jnp.uint16).reshape(-1)  # shape (84,)

        #print("x_split: {}", x_split.shape)

        #  Build offset array
        orderbook_shift = jnp.array(
            [cfg_pretraining.TIME_B_START] * 4
            + [cfg_pretraining.PRICE_B_START, cfg_pretraining.PRICE_B_START, cfg_pretraining.SIZE_B_START, cfg_pretraining.SIZE_B_START] * 2 * 10
        )  # shape (84,)

        #  Add offset
        orderbook_tokens = x_split.astype(jnp.uint32) + orderbook_shift  # shape (84,)


        ###################
        #  Concatenate orderbook and message tokens
        ###################

        obs = jnp.concatenate([orderbook_tokens, message_tokens_flat], axis=0)

        #jax.debug.print("obs: {}", obs.shape)

        return obs
      


    def _get_obs_basic(
            self,
            world_state: WorldState,
            agent_state: MMEnvState,
            agent_param: MMEnvParams,
            normalize: bool,
            flatten: bool = True,
        ) -> chex.Array:
        """ Return observation from raw state trafo. """
        # NOTE: only uses most recent observation from state

        spread=jnp.abs(world_state.best_asks[-1][0] - world_state.best_bids[-1][0])

        obs = {
            "spread": spread,
            "inventory" : agent_state.inventory,
        }

        # TODO: put this into config somewhere?
        #       also check if we can get rid of manual normalization
        #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?

        means = {
            "spread": 0,
            "inventory" : 0,
        }

        stds = {
            "spread": 1e4,
            "inventory" : 10,
        }
        
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)
        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        return obs
    


    def _get_obs_engineered(
            self,
            world_state: WorldState,
            agent_state: MMEnvState,
            agent_param: MMEnvParams,
            normalize: bool,
            flatten: bool = True,
        ) -> chex.Array:
        """ Return observation from raw state trafo. """
        # NOTE: only uses most recent observation from state
        time = world_state.time[0] + world_state.time[1]/1e9
        time_elapsed = time - (world_state.init_time[0] + world_state.init_time[1]/1e9)

        bid_vol_tot= job.get_volume(world_state.bid_raw_orders)
        ask_vol_tot= job.get_volume(world_state.ask_raw_orders)
        spread=jnp.abs(world_state.best_asks[-1][0] - world_state.best_bids[-1][0])

        # posted_ask= job.get_order_by_tid(world_state.ask_raw_orders,agent_param.trader_id)
        # dist_of_posted_ask =(posted_ask[job.cst.OrderSideFeat.P.value] - world_state.best_asks[-1][0])/spread
        # dist_of_posted_ask = jnp.where(posted_ask[job.cst.OrderSideFeat.P.value]>0, dist_of_posted_ask, -1.0) # if no order is posted, set distance to -1
        # posted_bid= job.get_order_by_tid(world_state.bid_raw_orders,agent_param.trader_id)
        # dist_of_posted_bid =(world_state.best_bids[-1][0]- posted_bid[job.cst.OrderSideFeat.P.value])/spread
        # dist_of_posted_bid = jnp.where(posted_bid[job.cst.OrderSideFeat.P.value]>0, dist_of_posted_bid, -1.0) # if no order is posted, set distance to -1
        # jax.debug.print("dist_of_posted_ask: {}", dist_of_posted_ask)
        # jax.debug.print("dist_of_posted_bid: {}", dist_of_posted_bid)
        # jax.debug.print("Posted ask price: {}", posted_ask[job.cst.OrderSideFeat.P.value])
        # jax.debug.print("Posted bid price: {}", posted_bid[job.cst.OrderSideFeat.P.value])
        
        if self.world_config.ep_type == "fixed_time":
            obs = {
                # "dist_of_posted_ask": agent_state.posted_distance_ask,
                # "dist_of_posted_bid": agent_state.posted_distance_bid,
                "p_bid" : world_state.best_bids[-1][0],  
                "p_ask":world_state.best_asks[-1][0], 
                "spread": spread,
                "q_bid": bid_vol_tot, #world_state.best_bids[-1][1],
                "q_ask": ask_vol_tot, #world_state.best_asks[-1][1],
                "delta_time": world_state.delta_time,
                "time_remaining": self.world_config.episode_time - time_elapsed,
                "mid_price":world_state.mid_price,
                "step_counter": world_state.step_counter,
                # Set Agent specific stuff
                # "total_PnL" : agent_state.total_PnL,
                # "cash_balance" : agent_state.cash_balance,
                "inventory" : agent_state.inventory,
            }

            # TODO: put this into config somewhere?
            #       also check if we can get rid of manual normalization
            #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?

            means = {
                # "dist_of_posted_ask": 0,
                # "dist_of_posted_bid": 0,
                "p_bid" : 0,
                "p_ask":0, 
                "spread": 0,
                "q_bid": 0,
                "q_ask": 0,
                "delta_time": 0,
                "time_remaining": 0,
                "mid_price":0,
                "step_counter": 0,
                # Set Agent specific stuff
                # "total_PnL" : 0,
                # "cash_balance" : 0,
                "inventory" : 0,
            }

            stds = {
                # "dist_of_posted_ask": 1.0,
                # "dist_of_posted_bid": 1.0,
                "p_bid" : 1e6,
                "p_ask":1e6, 
                "spread": 1e4,
                "q_bid": 1000,
                "q_ask": 1000,
                "delta_time": 10,
                "time_remaining": self.world_config.episode_time,
                "mid_price":1e6,
                "step_counter": 10,

                # Set Agent specific stuff
                # "total_PnL" : 1000,
                # "cash_balance" : 1000,
                "inventory" : 10,
            }

        elif self.world_config.ep_type == "fixed_steps": # leave away time related stuff
            obs = {
                # "dist_of_posted_ask": agent_state.posted_distance_ask,
                # "dist_of_posted_bid": agent_state.posted_distance_bid,
                "p_bid" : world_state.best_bids[-1][0],  
                "p_ask":world_state.best_asks[-1][0], 
                "spread": spread,
                "q_bid": bid_vol_tot,#world_state.best_bids[-1][1],
                "q_ask": ask_vol_tot,#world_state.best_asks[-1][1],
                "mid_price":world_state.mid_price,
                "step_counter": world_state.step_counter,
                # Set Agent specific stuff
                # "total_PnL" : agent_state.total_PnL,
                # "cash_balance" : agent_state.cash_balance,
                "inventory" : agent_state.inventory,
            }

            # TODO: put this into config somewhere?
            #       also check if we can get rid of manual normalization
            #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?

            means = {
                # "dist_of_posted_ask": 0,
                # "dist_of_posted_bid": 0,
                "p_bid" : 0,
                "p_ask":0, 
                "spread": 0,
                "q_bid": 0,
                "q_ask": 0,
                "mid_price":0,
                "step_counter": 0,

                # Set Agent specific stuff
                # "total_PnL" : 0,
                # "cash_balance" : 0,
                "inventory" : 0,
            }

            stds = {
                # "dist_of_posted_ask": 1,
                # "dist_of_posted_bid": 1,
                "p_bid" : 1e6,
                "p_ask":1e6, 
                "spread": 1e4,
                "q_bid": 1000,
                "q_ask": 1000,
                "mid_price":1e6,
                "step_counter": 10,

                # Set Agent specific stuff
                # "total_PnL" : 1000,
                # "cash_balance" : 1000,
                "inventory" : 10,
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
        obs = jax.tree.map(lambda x, m, s: (x - m) / s, obs, means, stds)
        return obs


    def action_space(self) -> spaces.Box:
        
        """ Action space of the environment. """
        if self.cfg.action_space == "directional_trading":
            return spaces.Discrete(self.cfg.n_actions)  # [0: do nothing, 1: buy at ask, 2: sell at bid]
        elif self.cfg.action_space == "fixed_prices":
            return spaces.Box(0, 100, (self.cfg.n_actions,), dtype=jnp.int32)
        elif self.cfg.action_space == "fixed_quants" or self.cfg.action_space == "AvSt":
            return spaces.Discrete(self.cfg.n_actions) #TODO change back to 8
        if self.cfg.action_space == "bobStrategy":
            return spaces.Discrete(self.cfg.n_actions)  # [0: do nothing, 1: buy at ask, 2: sell at bid]
        elif self.cfg.action_space == "bobRL":
            return spaces.Discrete(self.cfg.n_actions)  # depending on v_0 bob in config. 
        elif self.cfg.action_space == "spread_skew":
            return spaces.Discrete(self.cfg.n_actions)  # 6 possible combinations (2 spreads × 3 skews)
        elif self.cfg.action_space == "simple":
            if self.cfg.simple_nothing_action==True:
                return spaces.Discrete(self.cfg.n_actions)
            else:
                return spaces.Discrete(self.cfg.n_actions)
        else:
            raise ValueError("Invalid action_space specified.")
       

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self):
        """Observation space of the environment."""
        if self.cfg.observation_space =="engineered":
            if self.world_config.ep_type == "fixed_time":
             return spaces.Box(-1000, 1000, (10,), dtype=jnp.float32)
            elif self.world_config.ep_type == "fixed_steps":
                return spaces.Box(-1000, 1000, (8,), dtype=jnp.float32)
        elif self.cfg.observation_space =="messages":
                num_messages_total=self.cfg.num_messages_by_agent+self.world_config.n_data_msg_per_step
                return spaces.Box(low=-1*self.world_config.maxint, high=self.world_config.maxint ,shape=(num_messages_total, 8), dtype=jnp.int32)
        elif self.cfg.observation_space == "messages_new_tokenizer":
            cfg_pretraining               = get_config()
            num_messages      = self.cfg.num_messages_by_agent + self.world_config.n_data_msg_per_step
            toks_per_message  = 13      # we now split each int32 message‐field into two 16-bit tokens
            toks_per_book     = 84      # 42 book fields × 2 halves
            vocab_size        = cfg_pretraining.TOTAL_NUM_TOKENS
            return spaces.Box(
                low=0,
                high=vocab_size - 1,
                shape=(1, num_messages * toks_per_message + toks_per_book),
                dtype=jnp.int32,
            )
        elif self.cfg.observation_space == "basic":
            if self.world_config.ep_type == "fixed_time":
                return spaces.Box(-1000, 1000, (2,), dtype=jnp.float32)
            elif self.world_config.ep_type == "fixed_steps":
                return spaces.Box(-1000, 1000, (2,), dtype=jnp.float32)
        else:
            raise ValueError("Invalid observation_space specified.")


    def state_space(self, params: MMEnvParams) -> spaces.Dict:
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

    env_cfg = MarketMaking_EnvironmentConfig()

    env = MarketMakingAgent(
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
