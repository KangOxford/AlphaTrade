"""
Execution Environment for Limit Order Book  with variable start time for episodes. 

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
ExecutionEnv: Environment class inheriting from BaseLOBEnv, 
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
sys.path.append('.')
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
# config.update("jax_disable_jit", True) # Code snippet to disable all jitting.
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())
jax.numpy.set_printoptions(linewidth=183)
# ================= imports ==================


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

@struct.dataclass
class EnvState(BaseEnvState):
    prev_action: chex.Array
    prev_executed: chex.Array
    # Potentially could be moved to base,
    # so long as saving of best ask/bids is base behaviour. 
    best_asks: chex.Array
    best_bids: chex.Array
    # Execution specific stuff
    init_price: int
    task_to_execute: int
    quant_executed: int
    # Execution specific rewards. 
    total_revenue: float
    drift_return: float
    advantage_return: float
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float
    is_sell_task: int
    trade_duration: float
    quant_passive_2: int
    price_passive_2: int
    delta_time: float

@struct.dataclass
class EnvParams(BaseEnvParams):
    task_size: int 
    reward_lambda: float = 1.0

class ExecutionEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, task, window_index, action_type, episode_time,
            max_task_size = 500, rewardLambda=1., ep_type="fixed_time"):
        
        #Define Execution-specific attributes.
        self.task = task # "random", "buy", "sell"
        self.n_ticks_in_book = 2 # Depth of PP actions
        self.action_type = action_type # 'delta' or 'pure'
        self.max_task_size = max_task_size
        self.rewardLambda = rewardLambda
        # TODO: fix!! this can be overwritten in the base class
        self.n_actions = 2 # 4: (FT, M, NT, PP), 3: (FT, NT, PP), 2 (FT, NT), 1 (FT)

        #Call base-class init function
        super().__init__(
            alphatradePath,
            window_index,
            episode_time,
            ep_type,
        )

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        base_params = super().default_params
        flat_tree = jtu.tree_flatten(base_params)[0]
        #TODO: Clean this up to not have a magic number
        # BaseEnvParams
        base_vals = flat_tree[0:5] #Considers the base parameter values other than init state.
        state_vals = flat_tree[5:] #Considers the state values
        return EnvParams(
            *base_vals,
            EnvState(*state_vals),
            self.max_task_size,
            reward_lambda=self.rewardLambda
        )


    def step_env(
        self, key: chex.PRNGKey, state: EnvState, input_action: jax.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + params.episode_time
        )
        
        action = self._reshape_action(input_action, state, params,key)
        action_msgs = self._getActionMsgs(action, state, params)
        action_prices = action_msgs[:, 3]
        # jax.debug.print('action_msgs\n {}', action_msgs)

        raw_order_side = jax.lax.cond(
            state.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        cnl_msgs = job.getCancelMsgs(
            raw_order_side,
            job.INITID + 1,
            self.n_actions,  # max number of orders to cancel
            1 - state.is_sell_task * 2
        )
        
        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self._filter_messages(action_msgs, cnl_msgs)
        # jax.debug.print('filtered action_msgs\n {}', action_msgs)
        
        # Add to the top of the data messages
        total_messages = jnp.concatenate([cnl_msgs, action_msgs, data_messages], axis=0)
        # Save time of final message to add to state
        time = total_messages[-1, -2:]
        # To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit = (jnp.ones((self.nTradesLogged, 6)) * -1).astype(jnp.int32)
        # Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            # TODO: this returns bid/ask for last stepLines only, could miss the direct impact of actions
            self.stepLines
        )

        # If best price is not available in the current step, use the last available price
        # TODO: check if we really only want the most recent stepLines prices (+1 for the additional market order)
        bestasks, bestbids = (
            self._ffill_best_prices(
                bestasks[-self.stepLines+1:],
                state.best_asks[-1, 0]
            ),
            self._ffill_best_prices(
                bestbids[-self.stepLines+1:],
                state.best_bids[-1, 0]
            )
        )
        # jax.debug.print('agent_id {}, trades {}', self.trader_unique_id, trades)
        # filter to trades by our agent (rest are 0s)

        agent_trades = job.get_agent_trades(trades, self.trader_unique_id)
        # executions = self._get_executed_by_level(agent_trades, action, state)
        executions = self._get_executed_by_action(agent_trades, action, state)
        quant_executed_this_step = executions.sum()
        quant_left = state.task_to_execute - (state.quant_executed + quant_executed_this_step)
        
        # jax.debug.print('agent_trades\n {}', agent_trades[:30])
        # jax.debug.print('executions: {}', executions)
        # jax.debug.print(
        #     "quant_executed_this_step: {}, quant_left: {}, quant_executed_this_step {}",
        #     quant_executed_this_step, quant_left, quant_executed_this_step)

        # TODO: check if episode time is over and force market order if necessary
        (asks, bids, trades), (new_bestask, new_bestbid), new_id_counter, new_time, mkt_exec_quant, doom_quant = \
            self._force_market_order_if_done(
                quant_left, bestasks[-1], bestbids[-1], time, asks, bids, trades, state, params)

        bestasks = jnp.concatenate([bestasks, jnp.resize(new_bestask, (1, 2))], axis=0, dtype=jnp.int32)
        bestbids = jnp.concatenate([bestbids, jnp.resize(new_bestbid, (1, 2))], axis=0, dtype=jnp.int32)

        # jax.debug.print("bestasks\n {}", bestasks)
        
        price_passive_2, quant_passive_2 = self._get_pass_price_quant(state)
        # jax.debug.print('price_passive_2: {}, quant_passive_2: {}', price_passive_2, quant_passive_2)
        # TODO: consider adding quantity before (in priority) to each price / level

        # TODO: use the agent quant identification from the separate function _get_executed_by_level instead of _get_reward
        reward, extras = self._get_reward(state, params, trades)
        quant_executed = state.quant_executed + extras["agentQuant"]
        # CAVE: uses seconds only (not ns)
        trade_duration_step = (agent_trades[:, 1] / state.task_to_execute * (agent_trades[:, -2] - state.init_time[0])).sum()
        trade_duration = state.trade_duration + trade_duration_step
        # jax.debug.print('trade_duration_step: {}, trade_duration: {}', trade_duration_step, trade_duration)
        # jax.debug.print('left before mkt: {}, left after mkt {}', quant_left, state.task_to_execute - state.quant_executed - extras["agentQuant"])
        state = EnvState(
            #jnp.vstack([jnp.arange(3), jnp.arange(3)]).T
            prev_action = jnp.vstack([action_prices, action]).T,  # includes prices and quantitites  
            prev_executed = executions, # include prices and quantities 
            ask_raw_orders = asks,
            bid_raw_orders = bids,
            trades = trades,
            init_time = state.init_time,
            # time = time,
            time = new_time,
            # customIDcounter = state.customIDcounter + self.n_actions + 1,
            customIDcounter = new_id_counter,
            window_index = state.window_index,
            step_counter = state.step_counter + 1,
            max_steps_in_episode = state.max_steps_in_episode,
            start_index = state.start_index,
            best_asks = bestasks,
            best_bids = bestbids,
            init_price = state.init_price,
            task_to_execute = state.task_to_execute,
            quant_executed = quant_executed,
            total_revenue = state.total_revenue + extras["revenue"],
            drift_return = state.drift_return + extras["drift"],
            advantage_return = state.advantage_return + extras["advantage"],
            slippage_rm = extras["slippage_rm"],
            price_adv_rm = extras["price_adv_rm"],
            price_drift_rm = extras["price_drift_rm"],
            vwap_rm = extras["vwap_rm"],
            is_sell_task = state.is_sell_task,
            trade_duration = trade_duration,
            price_passive_2 = price_passive_2,
            quant_passive_2 = quant_passive_2,
            delta_time = new_time[0] + new_time[1]/1e9 - state.time[0] - state.time[1]/1e9,
        )
        done = self.is_terminal(state, params)
        info = {
            "window_index": state.window_index,
            "total_revenue": state.total_revenue,
            "quant_executed": state.quant_executed,
            "task_to_execute": state.task_to_execute,
            "average_price": jnp.nan_to_num(state.total_revenue 
                                            / state.quant_executed, 0.0),
            "current_step": state.step_counter,
            "done": done,
            "slippage_rm": state.slippage_rm,
            "price_adv_rm": state.price_adv_rm,
            "price_drift_rm": state.price_drift_rm,
            "vwap_rm": state.vwap_rm,
            "advantage_reward": state.advantage_return,
            "drift_reward": state.drift_return,
            "trade_duration": state.trade_duration,
            "mkt_forced_quant": mkt_exec_quant + doom_quant,
            "doom_quant": doom_quant, 
        }
        return self._get_obs(state, params), state, reward, done, info
    

    def reset_env(
            self,
            key : chex.PRNGKey,
            params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
        """ Reset the environment to init state (pre computed from data)."""
        key_, key = jax.random.split(key)
        _, state = super().reset_env(key, params)
        if self.task == 'random':
            direction = jax.random.randint(key_, minval=0, maxval=2, shape=())
        else:
            direction = 0 if self.task == 'buy' else 1
            
        state = dataclasses.replace(state, is_sell_task=direction)

        # update passive prices and quants depending on task direction
        # (other features are independent)
        # TODO: save passive prices and quants on both sides and handle this in _get_obs
        price_passive_2, quant_passive_2 = self._get_pass_price_quant(state)
        state = dataclasses.replace(state, price_passive_2=price_passive_2, quant_passive_2=quant_passive_2)

        obs = self._get_obs(state, params)
        return obs, state
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """ Check whether state is terminal. """
        if self.ep_type == 'fixed_time':
            # TODO: make the 5 sec a function of the step size
            return (
                (params.episode_time - (state.time - state.init_time)[0] <= 5)  # time over (last 5 seconds)
                |  (state.task_to_execute - state.quant_executed <= 0)  # task done
            )
        elif self.ep_type == 'fixed_steps':
            return (
                (state.max_steps_in_episode - state.step_counter <= 1)  # last step
                |  (state.task_to_execute - state.quant_executed <= 0)  # task done
            )
        else:
            raise ValueError(f"Unknown episode type: {self.ep_type}")

    # def _get_pass_price_quant(self, orders, best_ask_p, best_bid_p, is_sell_task):
    #     price_passive_2 = jax.lax.cond(
    #         is_sell_task,
    #         lambda: best_ask_p + self.tick_size*self.n_ticks_in_book,
    #         lambda: best_bid_p - self.tick_size*self.n_ticks_in_book
    #     )
    #     # quantity at second passive price level in the book
    #     quant_passive_2 = job.get_volume_at_price(orders, price_passive_2)
    #     return price_passive_2, quant_passive_2
    
    def _get_pass_price_quant(self, state):
        price_passive_2 = jax.lax.cond(
            state.is_sell_task,
            lambda: state.best_asks[-1, 0] + self.tick_size * self.n_ticks_in_book,
            lambda: state.best_bids[-1, 0] - self.tick_size * self.n_ticks_in_book
        )
        orders = jax.lax.cond(
            state.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        # quantity at second passive price level in the book
        quant_passive_2 = job.get_volume_at_price(orders, price_passive_2)
        return price_passive_2, quant_passive_2
    
    def _get_state_from_data(self,first_message,book_data,max_steps_in_episode,window_index,start_index):
        #(self,message_data,book_data,max_steps_in_episode)
        base_state = super()._get_state_from_data(first_message, book_data, max_steps_in_episode, window_index, start_index)
        base_vals = jtu.tree_flatten(base_state)[0]
        best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(base_state.ask_raw_orders,base_state.bid_raw_orders)
        M = (best_bid[0] + best_ask[0]) // 2 // self.tick_size * self.tick_size 
        # if task is 'random', this will be randomly picked at env reset
        is_sell_task = 0 if self.task == 'buy' else 1 # if self.task == 'random', set defualt as 0
        # HERE...

        return EnvState(
            *base_vals,
            prev_action=jnp.zeros((self.n_actions, 2), jnp.int32),
            prev_executed=jnp.zeros((self.n_actions, ), jnp.int32),
            best_asks=jnp.resize(best_ask,(self.stepLines,2)),
            best_bids=jnp.resize(best_bid,(self.stepLines,2)),
            init_price=M,
            task_to_execute=self.max_task_size,
            quant_executed=0,
            total_revenue=0.,
            drift_return=0.,
            advantage_return=0.,
            slippage_rm=0.,
            price_adv_rm=0.,
            price_drift_rm=0.,
            vwap_rm=0.,
            is_sell_task=is_sell_task, # updated on reset
            trade_duration=0.,
            # updated on reset:
            quant_passive_2=0,
            price_passive_2=0,
            delta_time=0.,
        )

    def _reshape_action(self, action : jax.Array, state: EnvState, params : EnvParams, key:chex.PRNGKey) -> jax.Array:
        def twapV3(state, env_params):
            # ---------- ifMarketOrder ----------
            remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
            marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
            ifMarketOrder = (remainingTime <= marketOrderTime)
            # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
            # ---------- ifMarketOrder ----------
            # ---------- quants ----------
            remainedQuant = state.task_to_execute - state.quant_executed
            remainedStep = state.max_steps_in_episode - state.step_counter
            stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
            limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True)
            market_quants = jnp.array([stepQuant,stepQuant])
            quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
            # ---------- quants ----------
            return jnp.array(quants) 

        def truncate_action(action, remainQuant):
            action = jnp.round(action).clip(0, remainQuant).astype(jnp.int32)
            # scaledAction = utils.clip_by_sum_int(action, remainQuant)
            scaledAction = jnp.where(
                action.sum() <= remainQuant,
                action,
                utils.hamilton_apportionment_permuted_jax(action, remainQuant, key)
            ).astype(jnp.int32)
            return scaledAction

        if self.action_type == 'delta':
            action = twapV3(state, params) + action

        action = truncate_action(action, state.task_to_execute - state.quant_executed)
        # jax.debug.print("base_ {}, delta_ {}, action_ {}; action {}",base_, delta_,action_,action)
        # jax.debug.print("action {}", action)
        return action
      
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

    def _get_executed_by_price(self, agent_trades: jax.Array) -> jax.Array:
        """ 
        Get executed quantity by price from trades. Results are sorted by increasing price. 
        NOTE: this will not work for aggressive orders eating through the book (size limited by actions)
        TODO: make this more general for aggressive actions?
        """
        price_levels, r_idx = jnp.unique(
            agent_trades[:, 0], return_inverse=True, size=self.n_actions+1, fill_value=0)
        quant_by_price = jax.ops.segment_sum(agent_trades[:, 1], r_idx, num_segments=self.n_actions+1)
        price_quants = jnp.vstack((price_levels[1:], quant_by_price[1:])).T
        # jax.debug.print("_get_executed_by_level\n {}", price_quants)
        return price_quants
    
    def _get_executed_by_level(self, agent_trades: jax.Array, actions: jax.Array, state: EnvState) -> jax.Array:
        """ Get executed quantity by level from trades. Results are sorted from aggressive to passive
            using previous actions. (0 actions are skipped)
            NOTE: this will not work for aggressive orders eating through the book (size limited by actions)
            TODO: make this more general for aggressive actions?
        """
        is_sell_task = state.is_sell_task
        price_quants = self._get_executed_by_price(agent_trades)
        # sort from aggr to passive
        price_quants = jax.lax.cond(
            is_sell_task,
            lambda: price_quants,
            lambda: price_quants[::-1],  # for buy task, most aggressive is highest price
        )
        # put executions in non-zero action places (keeping the order)
        price_quants = price_quants[jnp.argsort(jnp.argsort(actions <= 0))]
        return price_quants
    
    def _get_executed_by_action(self, agent_trades: jax.Array, actions: jax.Array, state: EnvState) -> jax.Array:
        """ Get executed quantity by level from trades. Results are sorted from aggressive to passive
            using previous actions. (0 actions are skipped)
            Aggressive quantities at FT and more passive are summed as the first quantity.
        """
        best_price = jax.lax.cond(
            state.is_sell_task,
            lambda: state.best_bids[-1, 0],
            lambda: state.best_asks[-1, 0]
        )
        aggr_trades_mask = jax.lax.cond(
            state.is_sell_task,
            lambda: agent_trades[:, 0] <= best_price,
            lambda: agent_trades[:, 0] >= best_price
        )
        exec_quant_aggr = jnp.where(
            aggr_trades_mask,
            agent_trades[:, 1],
            0
        ).sum()
        # jax.debug.print('best_price\n {}', best_price)
        # jax.debug.print('exec_quant_aggr\n {}', exec_quant_aggr)
        
        price_quants_pass = self._get_executed_by_price(
            # agent_trades[~aggr_trades_mask]
            jnp.where(
                jnp.expand_dims(aggr_trades_mask, axis=1),
                0,
                agent_trades
            )
        )
        # jax.debug.print('price_quants_pass\n {}', price_quants_pass)
        # sort from aggr to passive
        price_quants = jax.lax.cond(
            state.is_sell_task,
            lambda: price_quants_pass,
            lambda: price_quants_pass[::-1],  # for buy task, most aggressive is highest price
        )
        # put executions in non-zero action places (keeping the order)
        price_quants = price_quants[jnp.argsort(jnp.argsort(actions[1:] <= 0))]
        price_quants = jnp.concatenate(
            (jnp.array([[best_price, exec_quant_aggr]]), price_quants),
        )
        # jax.debug.print("actions {} \n price_quants {} \n", actions, price_quants)
        # return quants only (aggressive prices could be multiple)
        return price_quants[:, 1]
    
    def _getActionMsgs(self, action: jax.Array, state: EnvState, params: EnvParams):

        def normal_quant_price(price_levels: jax.Array, action: jax.Array):
            def combine_mid_nt(quants, prices):
                quants = quants \
                    .at[2].set(quants[2] + quants[1]) \
                    .at[1].set(0)
                prices = prices.at[1].set(-1)
                return quants, prices

            quants = action.astype(jnp.int32)
            prices = jnp.array(price_levels[:-1])
            if self.n_actions == 4:
                # if mid_price == near_touch_price: combine orders into one
                return jax.lax.cond(
                    price_levels[1] == price_levels[2],
                    combine_mid_nt,
                    lambda q, p: (q, p),
                    quants, prices
                )
            else:
                return quants, prices
        
        # def market_quant_price(price_levels: jax.Array, state: EnvState, action: jax.Array):
        #     mkt_quant = state.task_to_execute - state.quant_executed
        #     quants = jnp.asarray((mkt_quant, 0, 0, 0), jnp.int32) 
        #     return quants, jnp.asarray((price_levels[-1], -1, -1, -1), jnp.int32)
        
        def buy_task_prices(best_ask, best_bid):
            # FT = best_ask
            # essentially convert to market order (20% higher price than best ask)
            FT = ((best_ask * 1.2) // self.tick_size * self.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size
            NT = best_bid
            PP = best_bid - self.tick_size*self.n_ticks_in_book
            MKT = job.MAX_INT
            if action.shape[0] == 4:
                return FT, M, NT, PP, MKT
            elif action.shape[0] == 3:
                return FT, NT, PP, MKT
            elif action.shape[0] == 2:
                return FT, NT, MKT
            elif action.shape[0] == 1:
                return FT, MKT

        def sell_task_prices(best_ask, best_bid):
            # FT = best_bid
            # essentially convert to market order (20% lower price than best bid)
            FT = ((best_bid * 0.8) // self.tick_size * self.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            NT = best_ask
            PP = best_ask + self.tick_size*self.n_ticks_in_book
            MKT = 0
            if action.shape[0] == 4:
                return FT, M, NT, PP, MKT
            elif action.shape[0] == 3:
                return FT, NT, PP, MKT
            elif action.shape[0] == 2:
                return FT, NT, MKT
            elif action.shape[0] == 1:
                return FT, MKT

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types = jnp.ones((self.n_actions,), jnp.int32)
        sides = (1 - state.is_sell_task*2) * jnp.ones((self.n_actions,), jnp.int32)
        trader_ids = jnp.ones((self.n_actions,), jnp.int32) * self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids = (jnp.ones((self.n_actions,), jnp.int32) *
                    (self.trader_unique_id + state.customIDcounter)) \
                    + jnp.arange(0, self.n_actions) #Each message has a unique ID
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (self.n_actions, 2)
        )
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        best_ask, best_bid = state.best_asks[-1, 0], state.best_bids[-1, 0]

        price_levels = jax.lax.cond(
            state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        # if self.ep_type == 'fixed_time':
        #     remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        #     ep_is_over = lambda: remainingTime <= 1
        # else:
        #     ep_is_over = lambda: state.max_steps_in_episode - state.step_counter <= 1

        # quants, prices = jax.lax.cond(
        #     ep_is_over,
        #     market_quant_price,
        #     normal_quant_price,
        #     price_levels, state, action
        # )
        quants, prices = normal_quant_price(price_levels, action)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        # jax.debug.print('action_msgs\n {}', action_msgs)
        return action_msgs
        # ============================== Get Action_msgs ==============================

    def _force_market_order_if_done(
            self,
            quant_left: jax.Array,
            bestask: jax.Array,
            bestbid: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: EnvState,
            params: EnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:
        """ Force a market order if episode is over (either in terms of time or steps). """
        
        def create_mkt_order():
            mkt_p = (1 - state.is_sell_task) * job.MAX_INT // self.tick_size * self.tick_size
            side = (1 - state.is_sell_task*2)
            # TODO: this addition wouldn't work if the ns time at index 1 increases to more than 1 sec
            new_time = time + params.time_delay_obs_act
            mkt_msg = jnp.array([
                # type, side, quant, price
                1, side, quant_left, mkt_p,
                self.trader_unique_id,
                self.trader_unique_id + state.customIDcounter + self.n_actions,  # unique order ID for market order
                *new_time,  # time of message
            ])
            next_id = state.customIDcounter + self.n_actions + 1
            return mkt_msg, next_id, new_time

        def create_dummy_order():
            next_id = state.customIDcounter + self.n_actions
            return jnp.zeros((8,), dtype=jnp.int32), next_id, time 
        
        def place_doom_trade(trades, price, quant, time):
            doom_trade = job.create_trade(
                price, quant, self.trader_unique_id + self.n_actions + 1, -666666, *time)
            # jax.debug.print('doom_trade\n {}', doom_trade)
            trades = job.add_trade(trades, doom_trade)
            return trades

        if self.ep_type == 'fixed_time':
            remainingTime = params.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= 5  # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1

        order_msg, id_counter, time = jax.lax.cond(
            ep_is_over,
            create_mkt_order,
            create_dummy_order
        )
        # jax.debug.print('market order msg: {}', order_msg)
        # jax.debug.print('remainingTime: {}, ep_is_over: {}, order_msg: {}, time: {}', remainingTime, ep_is_over, order_msg, time)

        # jax.debug.print("trades before mkt\n {}", trades[:20])

        (asks, bids, trades), (new_bestask, new_bestbid) = job.cond_type_side_save_bidask(
            (asks, bids, trades),
            order_msg
        )
        # jax.debug.print("trades after mkt\n {}", trades[:20])

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
        # jax.debug.print('best_ask: {}; best_bid {}', bestask, bestbid)

        # how much of the market order could be executed
        mkt_exec_quant = jnp.where(
            trades[:, 3] == order_msg[5],
            trades[:, 1],  # executed quantity
            0
        ).sum()
        # jax.debug.print('mkt_exec_quant: {}', mkt_exec_quant)
        
        # assume execution at really unfavorable price if market order doesn't execute (worst case)
        # create artificial trades for this
        quant_still_left = quant_left - mkt_exec_quant
        # jax.debug.print('quant_still_left: {}', quant_still_left)
        # assume doom price with 25% extra cost
        doom_price = jax.lax.cond(
            state.is_sell_task,
            lambda: ((0.75 * bestbid[0]) // self.tick_size * self.tick_size).astype(jnp.int32),
            lambda: ((1.25 * bestask[0]) // self.tick_size * self.tick_size).astype(jnp.int32),
        )
        # jax.debug.print('doom_price: {}', doom_price)
        # jax.debug.print('best_ask: {}; best_bid {}', bestask, bestbid)
        # jax.debug.print('ep_is_over: {}; quant_still_left: {}; remainingTime: {}', ep_is_over, quant_still_left, remainingTime)
        trades = jax.lax.cond(
            ep_is_over & (quant_still_left > 0),
            place_doom_trade,
            lambda trades, b, c, d: trades,
            trades, doom_price, quant_still_left, time
        )
        # jax.debug.print('trades after doom\n {}', trades[:20])
        # agent_trades = job.get_agent_trades(trades, self.trader_unique_id)
        # jax.debug.print('agent_trades\n {}', agent_trades[:20])
        # price_quants = self._get_executed_by_price(agent_trades)
        # jax.debug.print('price_quants\n {}', price_quants)
        doom_quant = ep_is_over * quant_still_left

        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant

    def _get_reward(self, state: EnvState, params: EnvParams, trades: chex.Array) -> jnp.int32:
        # ========== get reward and revenue ==========
        # Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        # mask2 = ((job.INITID < executed[:, 2]) & (executed[:, 2] < 0)) | ((job.INITID < executed[:, 3]) & (executed[:, 3] < 0))
        mask2 = ((self.trader_unique_id <= executed[:, 2]) & (executed[:, 2] < 0)) \
              | ((self.trader_unique_id <= executed[:, 3]) & (executed[:, 3] < 0))
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        otherTrades = jnp.where(mask2[:, jnp.newaxis], 0, executed)
        # jax.debug.print('agentTrades\n {}', agentTrades[:30])
        agentQuant = agentTrades[:,1].sum() # new_execution quants
        
        # ---------- used for vwap, revenue ----------
        # vwapFunc = lambda tr: jnp.nan_to_num(
        #     (tr[:,0] // self.tick_size * tr[:,1]).sum() / (tr[:,1]).sum(),
        #     state.init_price  # if no trades happened, use init price
        # ) # caution: this value can be zero (executed[:,1]).sum()
        # only use other traders' trades for value weighted price
        # vwap = vwapFunc(otherTrades) # average_price of all other trades

        other_exec_quants = otherTrades[:, 1].sum()
        vwap = jax.lax.cond(
            other_exec_quants == 0,
            lambda: state.init_price / self.tick_size,
            lambda: (otherTrades[:, 0] // self.tick_size * otherTrades[:, 1]).sum() / other_exec_quants
        )
        
        revenue = (agentTrades[:,0] // self.tick_size * agentTrades[:,1]).sum()
        
        # ---------- used for slippage, price_drift, and RM(rolling mean) ----------
        rollingMeanValueFunc_FLOAT = lambda average_val,new_val:(average_val*state.step_counter+new_val)/(state.step_counter+1)
        vwap_rm = rollingMeanValueFunc_FLOAT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)
        price_adv_rm = rollingMeanValueFunc_FLOAT(state.price_adv_rm,revenue/agentQuant - vwap) # slippage=revenue/agentQuant-vwap, where revenue/agentQuant means agentPrice 
        slippage_rm = rollingMeanValueFunc_FLOAT(state.slippage_rm,revenue - state.init_price//self.tick_size*agentQuant)
        price_drift_rm = rollingMeanValueFunc_FLOAT(state.price_drift_rm,(vwap - state.init_price//self.tick_size)) #price_drift = (vwap - state.init_price//self.tick_size)
        
        # ---------- used for advantage and drift ----------
        # switch sign for buy task
        direction_switch = jnp.sign(state.is_sell_task * 2 - 1)
        advantage = direction_switch * (revenue - vwap * agentQuant) # advantage_vwap
        drift = direction_switch * agentQuant * (vwap - state.init_price//self.tick_size)
        
        # ---------- compute the final reward ----------
        # rewardValue = revenue 
        # rewardValue =  advantage
        # rewardValue1 = advantage + params.reward_lambda * drift
        # rewardValue1 = advantage + 1.0 * drift
        # rewardValue2 = revenue - (state.init_price // self.tick_size) * agentQuant
        # rewardValue = rewardValue1 - rewardValue2
        # rewardValue = revenue - vwap_rm * agentQuant # advantage_vwap_rm

        # rewardValue = revenue - (state.init_price // self.tick_size) * agentQuant
        reward = advantage + params.reward_lambda * drift
        reward_lam1 = direction_switch * (
            revenue - (state.init_price // self.tick_size) * agentQuant
        )
        
        # jax.debug.print('reward: {}. reward_lam1: {}. is_sell_task {}. advantage {} drift {} vwap {} init_price {}', 
        #                 reward, reward_lam1, state.is_sell_task, advantage, drift, vwap, state.init_price)
        
        # ---------- normalize the reward ----------
        # reward /= 10_000
        reward_scaled = reward / 100_000
        # reward /= params.avg_twap_list[state.window_index]
        return reward_scaled, {
            "agentQuant": agentQuant,
            "revenue": reward_lam1 / 100_000,  # pure revenue is not informative if direction is random (-> flip and normalise)
            "slippage_rm": slippage_rm,
            "price_adv_rm": price_adv_rm,
            "price_drift_rm": price_drift_rm,
            "vwap_rm": vwap_rm,
            "advantage": advantage,
            "drift": drift
        }

    def _get_obs(
            self,
            state: EnvState,
            params: EnvParams,
            normalize: bool = True,
            flatten: bool = True,
        ) -> chex.Array:
        """ Return observation from raw state trafo. """
        # NOTE: only uses most recent observation from state
        quote_aggr, quote_pass = jax.lax.cond(
            state.is_sell_task,
            lambda: (state.best_bids[-1], state.best_asks[-1]),
            lambda: (state.best_asks[-1], state.best_bids[-1]),
        )
        time = state.time[0] + state.time[1]/1e9
        time_elapsed = time - (state.init_time[0] + state.init_time[1]/1e9)
        # print('prev_action_shape', state.prev_action.shape)
        sign_switch = 2 * state.is_sell_task - 1
        obs = {
            "is_sell_task": state.is_sell_task,
            "p_aggr": quote_aggr[0] * sign_switch,  # switch sign for buy task
            "p_pass": quote_pass[0] * sign_switch,  # switch sign for buy task
            "spread": jnp.abs(quote_aggr[0] - quote_pass[0]),
            "q_aggr": quote_aggr[1],
            "q_pass": quote_pass[1],
            "q_pass2": state.quant_passive_2,
            # "q_before2": None, # how much quantity lies above this price level
            "time": time,
            "delta_time": state.delta_time,
            # "episode_time": state.time - state.init_time,
            "time_remaining": params.episode_time - time_elapsed,
            "init_price": state.init_price,
            "task_size": state.task_to_execute,
            "executed_quant": state.quant_executed,
            "remaining_quant": state.task_to_execute - state.quant_executed,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
            # "remaining_ratio": 1. - jnp.nan_to_num(state.step_counter / state.max_steps_in_episode, nan=1.),
            "remaining_ratio": jnp.where(state.max_steps_in_episode==0, 0., 1. - state.step_counter / state.max_steps_in_episode),
            "prev_action": state.prev_action[:, 1],  # use quants only
            "prev_executed": state.prev_executed,  # use quants only
            "prev_executed_ratio": jnp.where(state.prev_action[:, 1]==0., 0., state.prev_executed / state.prev_action[:, 1]),
        }
        # jax.debug.print('prev_action {}', state.prev_action)
        # jax.debug.print('prev_executed {}', state.prev_executed)
        # jax.debug.print('obs:\n {}', obs)
        # TODO: put this into config somewhere?
        #       also check if we can get rid of manual normalization
        #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
        p_mean = 3.5e7
        p_std = 1e6
        means = {
            "is_sell_task": 0,
            "p_aggr": state.init_price * sign_switch, #p_mean,
            "p_pass": state.init_price * sign_switch, #p_mean,
            "spread": 0,
            "q_aggr": 0,
            "q_pass": 0,
            "q_pass2": 0,
            "time": 0,
            "delta_time": 0,
            # "episode_time": jnp.array([0, 0]),
            "time_remaining": 0,
            "init_price": 0, #p_mean,
            "task_size": 0,
            "executed_quant": 0,
            "remaining_quant": 0,
            "step_counter": 0,
            "max_steps": 0,
            "remaining_ratio": 0,
            "prev_action": 0,
            "prev_executed": 0,
            "prev_executed_ratio": 0,
        }
        stds = {
            "is_sell_task": 1,
            "p_aggr": 1e5, #p_std,
            "p_pass": 1e5, #p_std,
            "spread": 1e4,
            "q_aggr": 100,
            "q_pass": 100,
            "q_pass2": 100,
            "time": 1e5,
            "delta_time": 10,
            # "episode_time": jnp.array([1e3, 1e9]),
            "time_remaining": self.sliceTimeWindow, # 10 minutes = 600 seconds
            "init_price": 1e7, #p_std,
            "task_size": self.max_task_size,
            "executed_quant": self.max_task_size,
            "remaining_quant": self.max_task_size,
            "step_counter": 30,  # TODO: find way to make this dependent on episode length
            "max_steps": 30,
            "remaining_ratio": 1,
            "prev_action": 10,
            "prev_executed": 10,
            "prev_executed_ratio": 1,
        }
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)
        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        return obs

    def _get_obs_full(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # Note: uses entire observation history between steps
        # TODO: if we want to use this, we need to roll forward the RNN state with every step

        best_asks, best_bids = state.best_asks[:,0], state.best_bids[:,0]
        best_ask_qtys, best_bid_qtys = state.best_asks[:,1], state.best_bids[:,1]
        
        obs = {
            "is_sell_task": state.is_sell_task,
            "p_aggr": jnp.where(state.is_sell_task, best_bids, best_asks),
            "q_aggr": jnp.where(state.is_sell_task, best_bid_qtys, best_ask_qtys), 
            "p_pass": jnp.where(state.is_sell_task, best_asks, best_bids),
            "q_pass": jnp.where(state.is_sell_task, best_ask_qtys, best_bid_qtys), 
            "p_mid": (best_asks+best_bids)//2//self.tick_size*self.tick_size, 
            "p_pass2": jnp.where(state.is_sell_task, best_asks+self.tick_size*self.n_ticks_in_book, best_bids-self.tick_size*self.n_ticks_in_book), # second_passives
            "spread": best_asks - best_bids,
            "shallow_imbalance": state.best_asks[:,1]- state.best_bids[:,1],
            "time": state.time,
            "episode_time": state.time - state.init_time,
            "init_price": state.init_price,
            "task_size": state.task_to_execute,
            "executed_quant": state.quant_executed,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
        }
        p_mean = 3.5e7
        p_std = 1e6
        means = {
            "is_sell_task": 0,
            "p_aggr": p_mean,
            "q_aggr": 0,
            "p_pass": p_mean,
            "q_pass": 0,
            "p_mid": p_mean,
            "p_pass2":p_mean,
            "spread": 0,
            "shallow_imbalance":0,
            "time": jnp.array([0, 0]),
            "episode_time": jnp.array([0, 0]),
            "init_price": p_mean,
            "task_size": 0,
            "executed_quant": 0,
            "step_counter": 0,
            "max_steps": 0,
        }
        stds = {
            "is_sell_task": 1,
            "p_aggr": p_std,
            "q_aggr": 100,
            "p_pass": p_std,
            "q_pass": 100,
            "p_mid": p_std,
            "p_pass2": p_std,   
            "spread": 1e4,
            "shallow_imbalance": 10,
            "time": jnp.array([1e5, 1e9]),
            "episode_time": jnp.array([1e3, 1e9]),
            "init_price": p_std,
            "task_size": 500,
            "executed_quant": 500,
            "step_counter": 300,
            "max_steps": 300,
        }
        obs = self.normalize_obs(obs, means, stds)
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
        if self.action_type == 'delta':
            # return spaces.Box(-5, 5, (self.n_actions,), dtype=jnp.int32)
            return spaces.Box(-100, 100, (self.n_actions,), dtype=jnp.int32)
        else:
            # return spaces.Box(0, 100, (self.n_actions,), dtype=jnp.int32)
            return spaces.Box(0, self.max_task_size, (self.n_actions,), dtype=jnp.int32)
    
       

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        #space = spaces.Box(-10,10,(809,),dtype=jnp.float32) 
        # space = spaces.Box(-10, 10, (21,), dtype=jnp.float32) 
        space = spaces.Box(-10, 10, (23,), dtype=jnp.float32) 
        return space

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
        ATFolder = "./training_oneDay/"
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        # ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "buy", # "random", # "buy",
        "MAX_TASK_SIZE": 100, # 500,
        "WINDOW_INDEX": 1,
        "ACTION_TYPE": "pure", # "pure",
        "REWARD_LAMBDA": 1.0,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 5, # 60 seconds
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=ExecutionEnv(ATFolder,"sell",1)
    env = ExecutionEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=1,
        task_size=config["MAX_TASK_SIZE"],
        episode_time=config["EPISODE_TIME"],  # in seconds
    )
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset, env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
    print(state)

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        # test_action=env.action_space().sample(key_policy)
        test_action = env.action_space().sample(key_policy) // 10
        # test_action = jnp.array([100, 10])
        print(f"Sampled {i}th actions are: ", test_action)
        start=time.time()
        obs, state, reward, done, info = env.step(
            key_step, state, test_action, env_params)
        for key, value in info.items():
            print(key, value)
            print('is_sell_task', state.is_sell_task)
        # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        # print(f"Time for {i} step: \n",time.time()-start)
        if done:
            print("==="*20)
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
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
