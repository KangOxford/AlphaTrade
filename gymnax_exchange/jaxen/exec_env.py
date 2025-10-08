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
ExecEnvState:   Dataclass to encapsulate the current state of the environment, 
            including the raw order book, trades, and time information.
ExecEnvParams:  Configuration class for environment-specific parameters, 
            such as task details, message and book data, and episode timing.
ExecutionAgent: Environment class inheriting from BaseLOBEnv, 
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
sys.path.append('.')
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
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
from gymnax_exchange.utils import utils
import dataclasses
from gymnax_exchange.jaxob.jaxob_config import Execution_EnvironmentConfig,World_EnvironmentConfig
from gymnax_exchange.jaxen.StatesandParams import ExecEnvState, ExecEnvParams, WorldState
from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig
from gymnax_exchange.jaxen.StatesandParams import MultiAgentState, WorldState


#from gymnax_exchange.jaxen.from_JAXMARL import spaces
import jax.tree_util as jtu



class ExecutionAgent():
    def __init__(
            self, 
            cfg:Execution_EnvironmentConfig,
            world_config: World_EnvironmentConfig):
        
        #Define the config
        self.cfg=cfg
        self.world_config = world_config

         #----------------- Set the end function -----------------#
        # if self.cfg.end_fn=="force_market_order":
        #     self.end_fn =self._force_market_order_if_done
        # elif self.cfg.end_fn=="unwind_FT":
        #     self.end_fn=self.unwind_FT
        #else:
        #     raise ValueError(f"Unknown end function: {self.cfg.end_fn}")
        #---------------------------------------------------------#

        #----------------- Set the action function -----------------#
        if self.cfg.action_space == "fixed_quants":
            self.action_fn = self._getActionMsgs_fixedQuant
        elif self.cfg.action_space == "fixed_prices":
            self.action_fn = self._getActionMsgs_fixedPrice
        elif self.cfg.action_space == "fixed_quants_complex":
            self.action_fn = self._getActionMsgs_fixedQuant_complex
        elif self.cfg.action_space == "simplest_case":
            self.action_fn = self._getActionMsgs_simpleCase
        elif self.cfg.action_space == "fixed_quants_1msg":
            self.action_fn = self._getActionMsgs_fixedQuant_1msg
        elif self.cfg.action_space == "twap":
            self.action_fn = self._getActionMsgs_twap
        else:
            raise ValueError("Invalid action_space specified.")


        #Choose observation space based on config.
        if self.cfg.observation_space == "engineered":
            self.observation_fn = self._get_obs
        elif self.cfg.observation_space == "basic":
            self.observation_fn = self._get_obs_basic
        elif self.cfg.observation_space == "simplest_case":
            self.observation_fn = self._get_obs_simplest_case
        else:
            raise ValueError("Invalid observation_space specified.")

    def default_params(self,
                       agent_config:Execution_EnvironmentConfig,
                       trader_id_range_start:int,
                        number_of_agents_per_type:int) -> ExecEnvParams:
        next_trader_id_range_start = trader_id_range_start - number_of_agents_per_type
        trader_id = jnp.arange(trader_id_range_start, next_trader_id_range_start, -1)
        task_size = jnp.full((number_of_agents_per_type,), agent_config.task_size)
        reward_lambda = jnp.full((number_of_agents_per_type,), agent_config.reward_lambda)
        time_delay_obs_act = jnp.full((number_of_agents_per_type,), agent_config.time_delay_obs_act)
        normalize = jnp.full((number_of_agents_per_type,), agent_config.normalize)
        
        #print(f"task_size: {task_size}")
        #print(f"trader_id: {trader_id}")
        return ExecEnvParams(trader_id=trader_id, task_size=task_size, reward_lambda=reward_lambda, time_delay_obs_act=time_delay_obs_act, normalize=normalize), next_trader_id_range_start



    def step_env(
        self, key: chex.PRNGKey, state: ExecEnvState, input_action: jax.Array, params: ExecEnvParams
    ) -> Tuple[chex.Array, ExecEnvState, float, bool, dict]:

        data_messages = self._get_data_messages(
            params.message_data,
            state.start_index,
            state.step_counter,
            state.init_time[0] + self.world_config.episode_time
        )
    
        action = self._reshape_action(input_action, state, params,key)
        action_msgs = self.get_action(action, state, params)
        action_prices = action_msgs[:, 3]
        action_quants=action_msgs[:,2]
        #jax.debug.print('action_msgs\n {}', action_msgs)
        #jax.debug.print("bids:{}",state.bid_raw_orders)



        raw_order_side = jax.lax.cond(
            state.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        cnl_msgs = job.getCancelMsgs(
            raw_order_side,
            agent_params.trader_id,
            self.cfg.num_action_messages_by_agent,
            1 - state.is_sell_task * 2,
            state.time[0],
            state.time[1]
        )
        
        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self._filter_messages(action_msgs, cnl_msgs)
        #jax.debug.print('filtered action_msgs\n {}', action_msgs)
        #jax.debug.print("cln msgs:{}",cnl_msgs)
        
        # Add to the top of the data messages
        total_messages = jnp.concatenate([cnl_msgs, action_msgs, data_messages], axis=0)
        # Save time of final message to add to state
        time = total_messages[-1, -2:]
        # To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit = (jnp.ones((self.nTradesLogged, 8)) * -1).astype(jnp.int32)
        # Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestbids, bestasks) = job.scan_through_entire_array_save_bidask(self.cfg, key,
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            # TODO: this returns bid/ask for last n_data_msg_per_step only, could miss the direct impact of actions
            self.n_data_msg_per_step + self.cfg.num_messages_by_agent
        )
        #
        jax.debug.print("new best bids before:{}",bestbids.shape)

        # If best price is not available in the current step, use the last available price
        # TODO: check if we really only want the most recent n_data_msg_per_step prices (+1 for the additional market order)
        bestasks, bestbids = (
            self._ffill_best_prices(
                bestasks[-self.n_data_msg_per_step- self.cfg.num_messages_by_agent:],
                state.best_asks[-1, 0]
            ),
            self._ffill_best_prices(
                bestbids[-self.n_data_msg_per_step- self.cfg.num_messages_by_agent:],
                state.best_bids[-1, 0]
            )
        )
        jax.debug.print("new best bids after:{}",bestbids.shape)
        # jax.debug.print('agent_id {}, trades {}', agent_params.trader_id, trades)
        # filter to trades by our agent (rest are 0s)

        agent_trades = job.get_agent_trades(trades, agent_params.trader_id)

        #jax.debug.print("Agent trades:{}", agent_trades)
       # jax.debug.print(" trades :{}", trades)
        


        # executions = self._get_executed_by_level(agent_trades, action, state)
        executions = self._get_executed_by_action(agent_trades, action, state,action_prices)
        executions=jnp.abs(executions) #deal with the negative quants from trades
    
        quant_executed_this_step = executions[:,1].sum()#sum the quants..
        #jax.debug.print("executions:{}",executions)

        quant_left = state.task_to_execute - (state.quant_executed + quant_executed_this_step)
        
        # jax.debug.print('agent_trades\n {}', agent_trades[:30])
        #jax.debug.print('executions: {}', executions)
        #jax.debug.print(
        #     "quant_executed_this_step: {}, quant_left: {}, quant_executed_this_step {}",
        #     quant_executed_this_step, quant_left, quant_executed_this_step)

        # TODO: check if episode time is over and force market order if necessary
        (asks, bids, trades), (new_bestask, new_bestbid), new_id_counter, new_time, mkt_exec_quant, doom_quant = \
            self.get_episode_end_fn(key,
                quant_left, bestasks[-1], bestbids[-1], time, asks, bids, trades, state, params)

        #bestasks = jnp.concatenate([bestasks, jnp.resize(new_bestask, (1, 2))], axis=0, dtype=jnp.int32)
        #bestbids = jnp.concatenate([bestbids, jnp.resize(new_bestbid, (1, 2))], axis=0, dtype=jnp.int32)

        jax.debug.print("bestasks after2:{}",bestasks.shape)
        #jax.debug.print("bestasks\n {}", bestasks)
        


        # TODO: consider adding quantity before (in priority) to each price / level

        # TODO: use the agent quant identification from the separate function _get_executed_by_level instead of _get_reward
        reward, extras = self._get_reward(state, params, trades)
        quant_executed = state.quant_executed + extras["agentQuant"]
        # CAVE: uses seconds only (not ns)
        trade_duration_step = (jnp.abs(agent_trades[:, 1]) / state.task_to_execute * (agent_trades[:, -2] - state.init_time[0])).sum()
        trade_duration = state.trade_duration + trade_duration_step
        # jax.debug.print('trade_duration_step: {}, trade_duration: {}', trade_duration_step, trade_duration)
        # jax.debug.print('left before mkt: {}, left after mkt {}', quant_left, state.task_to_execute - state.quant_executed - extras["agentQuant"])
        state = ExecEnvState(
            prev_action = jnp.vstack([action_prices, action_quants]).T,  # includes prices and quantitites  
            prev_executed = executions[:,1],#just the quants to keep same setup  
            ask_raw_orders = asks,
            bid_raw_orders = bids,
            trades = trades,
            init_time = state.init_time,
            # time = time,
            time = new_time,
            # customIDcounter = state.customIDcounter + self.cfg.n_actions + 1,
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
            delta_time = new_time[0] + new_time[1]/1e9 - state.time[0] - state.time[1]/1e9,
        )
        done = self.is_terminal(state, params)
        if self.cfg.debug_mode==False:
            info = {
            "window_index": state.window_index,
            "total_revenue": state.total_revenue,
            "quant_executed": state.quant_executed,
            "task_to_execute": state.task_to_execute,
            "average_price": jnp.nan_to_num(state.total_revenue 
                                            / state.quant_executed, 0.0),
            "mid_price":((state.best_bids[:, 0] + state.best_asks[:, 0]) // 2).mean(),
            "current_step": state.step_counter,
            "done": done,
            "window_index": state.window_index,
            "reward_lam1":extras["reward_lam1"],
            "slippage_rm": state.slippage_rm,
            "price_adv_rm": state.price_adv_rm,
            "price_drift_rm": state.price_drift_rm,
            "vwap_rm": state.vwap_rm,
            "advantage_reward": state.advantage_return,
            "drift_reward": state.drift_return,
            "trade_duration": state.trade_duration,
            "mkt_forced_quant": mkt_exec_quant + doom_quant,
            "doom_quant": doom_quant,
            "drift":extras["drift"],
            "is_sell_task": state.is_sell_task,
            }
        if self.cfg.debug_mode==True:

            lob_state = job.get_L2_state(
                                state.ask_raw_orders,  # Current ask orders
                                state.bid_raw_orders,  # Current bid orders
                                10,  # Number of levels
                                self.cfg  
                                )
            info={
                "trades":trades,
                "total_msgs":total_messages,
                "lob_state":lob_state,
            "window_index": state.window_index,
            "total_revenue": state.total_revenue,
            "quant_executed": state.quant_executed,
            "task_to_execute": state.task_to_execute,
            "average_price": jnp.nan_to_num(state.total_revenue 
                                            / state.quant_executed, 0.0),
            "mid_price":((state.best_bids[:, 0] + state.best_asks[:, 0]) // 2).mean(),
            "current_step": state.step_counter,
            "done": done,
            "window_index": state.window_index,
            "reward_lam1":extras["reward_lam1"],
            "slippage_rm": state.slippage_rm,
            "price_adv_rm": state.price_adv_rm,
            "price_drift_rm": state.price_drift_rm,
            "vwap_rm": state.vwap_rm,
            "advantage_reward": state.advantage_return,
            "drift_reward": state.drift_return,
            "trade_duration": state.trade_duration,
            "mkt_forced_quant": mkt_exec_quant + doom_quant,
            "doom_quant": doom_quant,
            "drift":extras["drift"],
            "is_sell_task": state.is_sell_task,
            }

        return self.get_observation(state, params), state, reward, done, info
    


    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
            self,
            agent_param: ExecEnvParams,
            key : chex.PRNGKey,
            world_state: WorldState,
            num_msgs_per_step: int # Useful for message based obs space if we will implement that for exec aswell
        ) -> Tuple[chex.Array, ExecEnvState]:
        """ Reset the agent specific environment state"""


        if self.cfg.task == 'random':
            is_sell_task = jax.random.randint(key, minval=0, maxval=2, shape=())
        else:
            is_sell_task = 0 if self.cfg.task == 'buy' else 1
        n_trades=self.cfg.num_action_messages_by_agent

        agent_state = ExecEnvState(
            # Execution specific stuff
            init_price = world_state.mid_price,
            task_to_execute = self.cfg.task_size,
            quant_executed = 0,
            # Execution specific rewards. 
            total_revenue = 0.,
            drift_return = 0.,
            advantage_return = 0.,
            slippage_rm = 0.,
            price_adv_rm = 0.,
            price_drift_rm = 0.,
            vwap_rm = 0.,
            is_sell_task = is_sell_task,
            trade_duration = 0.,
        )

        # Calculate things for the message obs space
        if self.cfg.observation_space == "messages_new_tokenizer":
            lob_state_before = job.get_L2_state(
                world_state.ask_raw_orders,  # Current ask orders
                world_state.bid_raw_orders,  # Current bid orders
                10,  # Number of levels
                self.cfg  
            )
            blank_messages = jnp.zeros((num_msgs_per_step, 8), dtype=jnp.int32) # Reset for the message based obs space.
        else:
            lob_state_before = None
            blank_messages = None

        obs = self.get_observation(agent_state = agent_state, 
                            world_state = world_state, 
                            agent_param = agent_param,
                            total_messages = blank_messages,
                            old_time = world_state.time,
                            old_mid_price = world_state.mid_price,
                            lob_state_before = lob_state_before,
                            normalize = self.cfg.normalize,
                            flatten=True)

        return obs, agent_state



    def is_terminal(self, world_state: WorldState, agent_state: ExecEnvState) -> bool:
        """ Check whether state is terminal. """
        if self.world_config.ep_type == 'fixed_time':
            #jax.debug.print("params_episode_time:{}",self.world_config.episode_time)
            #jax.debug.print("time:{}",state.time)
            
            #jax.debug.print("init_time:{}",state.init_time)
            # TODO: make the 5 sec a function of the step size
            time_done = (self.world_config.episode_time - (world_state.time - world_state.init_time)[0] <= self.cfg.seconds_before_episode_end)  # time over (last 5 seconds)
            task_done = (agent_state.task_to_execute - agent_state.quant_executed <= 0)
            done = time_done | task_done
            def done_callback(world_state, agent_state, done,time_done, task_done):
                if done:
                    print(f"Episode done: time_done {time_done}, task_done {task_done}")
                    print(f"Window Index: {world_state.window_index}")
                    if time_done:
                        print("Episode done: time over, last 5 seconds")
                        print(f"Seconds before episode end: {self.cfg.seconds_before_episode_end}")
                        print(f"Episode time: {self.world_config.episode_time}")
                        print(f"Time Elapsed: {(world_state.time - world_state.init_time)[0]}")
                        print(f"Time : {world_state.time}")
                        print(f"Init Time : {world_state.init_time}")
                    if task_done:
                        print("Episode done: task done")
                        print(f"Task to execute: {agent_state.task_to_execute}")
                        print(f"Quant executed: {agent_state.quant_executed}")

            # jax.debug.callback(done_callback, world_state, agent_state, done, time_done, task_done)
            return done
        
        elif self.world_config.ep_type == 'fixed_steps':
            #jax.debug.print(f"done exec step: {world_state.max_steps_in_episode - world_state.step_counter <= 1}")
            #jax.debug.print(f"done exec task: {agent_state.task_to_execute - agent_state.quant_executed <= 0}")
            return (
                (world_state.max_steps_in_episode - world_state.step_counter <= 1)  # last step
                |  (agent_state.task_to_execute - agent_state.quant_executed <= 0)  # task done
            )
        else:
            raise ValueError(f"Unknown episode type: {self.world_config.ep_type}")

    # def _get_pass_price_quant(self, orders, best_ask_p, best_bid_p, is_sell_task):
    #     price_passive_2 = jax.lax.cond(
    #         is_sell_task,
    #         lambda: best_ask_p + self.world_config.tick_size*self.n_ticks_in_book,
    #         lambda: best_bid_p - self.world_config.tick_size*self.n_ticks_in_book
    #     )
    #     # quantity at second passive price level in the book
    #     quant_passive_2 = job.get_volume_at_price(orders, price_passive_2)
    #     return price_passive_2, quant_passive_2
    
    def _get_pass_price_quant(self, state):
        price_passive_2 = jax.lax.cond(
            state.is_sell_task,
            lambda: state.best_asks[-1, 0] + self.world_config.tick_size * self.cfg.n_ticks_in_book,
            lambda: state.best_bids[-1, 0] - self.world_config.tick_size * self.cfg.n_ticks_in_book
        )
        orders = jax.lax.cond(
            state.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        # quantity at second passive price level in the book
        quant_passive_2 = job.get_volume_at_price(orders, price_passive_2)
        return price_passive_2, quant_passive_2
    
    def _get_state_from_data(self,key,first_message,book_data,max_steps_in_episode,window_index,start_index):
        #(self,message_data,book_data,max_steps_in_episode)
        base_state = super()._get_state_from_data(key,first_message, book_data, max_steps_in_episode, window_index, start_index)
        base_vals = jtu.tree_flatten(base_state)[0]
        best_bid, best_ask = job.get_best_bid_and_ask_inclQuants(self.cfg,base_state.ask_raw_orders,base_state.bid_raw_orders)
        M = (best_bid[0] + best_ask[0]) // 2 // self.world_config.tick_size * self.world_config.tick_size 
        # if task is 'random', this will be randomly picked at env reset
        is_sell_task = 0 if self.cfg.task == 'buy' else 1 # if self.cfg.task == 'random', set defualt as 0
        # HERE...
        n_trades=self.cfg.num_action_messages_by_agent
        return ExecEnvState(
            *base_vals,
            prev_action=jnp.zeros((n_trades, 2), jnp.int32),
            prev_executed=jnp.zeros((n_trades, ), jnp.int32),
            best_asks=jnp.resize(best_ask,(self.n_data_msg_per_step,2)),
            best_bids=jnp.resize(best_bid,(self.n_data_msg_per_step,2)),
            init_price=M,
            task_to_execute=self.cfg.task_size,
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
            delta_time=0.,
        )

    def _reshape_action(self, action : jax.Array, state: ExecEnvState, params : ExecEnvParams, key:chex.PRNGKey) -> jax.Array:
        def twapV3(state, env_params):
            # ---------- ifMarketOrder ----------
            remainingTime = self.world_config.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
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

        ##----Only truncate for the non fixed quants, this is handled in action space for fixed quants
        if self.cfg.action_space=="fixed_prices":
            #Only do delta for non fixed quants
            if self.cfg.action_type == 'delta':
                action = twapV3(state, params) + action
            action = truncate_action(action, state.task_to_execute - state.quant_executed)
        elif self.cfg.action_space=="fixed_quants":
            action=action
        elif self.cfg.action_space=="fixed_quants_complex":
            action=action
        else:
            raise ValueError("Invalid Action Space")
        
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
            agent_trades[:, 0], return_inverse=True, size=self.cfg.n_actions+1, fill_value=0)
        quant_by_price = jax.ops.segment_sum(jnp.abs(agent_trades[:, 1]), r_idx, num_segments=self.cfg.n_actions+1)
        price_quants = jnp.vstack((price_levels[1:], quant_by_price[1:])).T
        # jax.debug.print("_get_executed_by_level\n {}", price_quants)
        return price_quants
    
    def _get_executed_by_level(self, agent_trades: jax.Array, actions: jax.Array, state: ExecEnvState) -> jax.Array:
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
    
    def _get_executed_by_action(self, agent_trades: jax.Array, actions: jax.Array, state: ExecEnvState,action_prices:jax.Array) -> jax.Array:
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

        # Mask trades and indices instead of boolean indexing
        valid_trades = jnp.where(valid_indices, agent_trades[:, 1], 0)
        #jax.debug.print("valid_trades:{}",valid_trades)
        valid_price_to_index = jnp.where(valid_indices, price_to_index, 0)

        # Sum trades by price level
        executions = jax.ops.segment_sum(valid_trades, valid_price_to_index, num_segments=num_prices)
       # Create a 2D array with price levels and corresponding trade quantities
        price_quantity_pairs = jnp.stack([action_prices, executions], axis=-1)

        return price_quantity_pairs
    
    def _get_executed_by_action_old(self, agent_trades: jax.Array, actions: jax.Array, state: ExecEnvState) -> jax.Array:
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
            jnp.abs(agent_trades[:, 1]),
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


    #-------Action Functions-------#
    def _getActionMsgs_fixedQuant(self, action: jax.Array, world_state: WorldState, agent_state: ExecEnvState, agent_params: ExecEnvParams):
        """Action function for the fixed Quant Action space
        Pick for a ladder of quant execution options
        Always send 4 messages
        0 = No trade
        1=      # FT
        2=     # M
        3=    # NT
        4=    # PP
       """

        #----01 get price levels----#
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        #jax.debug.print('best_ask: {}, best_bid: {}', best_ask, best_bid)

        def buy_task_prices(best_ask, best_bid):
            FT = best_ask
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.world_config.tick_size) * self.world_config.tick_size
            NT = best_bid
            PP = best_bid - self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, M, NT, PP
        def sell_task_prices(best_ask, best_bid):
            FT = best_bid
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                 * self.world_config.tick_size).astype(jnp.int32)
            NT = best_ask
            PP = best_ask + self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, M, NT, PP
        
        price_levels = jax.lax.cond(
            agent_state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )

        #----02 get quants----#
        #jax.debug.print("action:{}",action)
        
        quant_array = jnp.array([
            [0, 0, 0, 0],  # No trade
            [1, 0, 0, 0],  # FT
            [0, 1, 0, 0],  # M
            [0, 0, 1, 0],  # NT
            [0, 0, 0, 1],  # PP
        ])

        if self.cfg.larger_far_touch_quant:
            quant_array = jnp.array([
                [0, 0, 0, 0],  # No trade
                [10, 0, 0, 0],  # FT
                [0, 1, 0, 0],  # M
                [0, 0, 1, 0],  # NT
                [0, 0, 0, 1],  # PP
            ])



        quants=quant_array[action,:]*self.cfg.fixed_quant_value #Get the quant array based on the action
        quants = quants.flatten() #Flatten the array to 1D
        #----03 get the rest of the message----#
        types = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        sides = (1 - agent_state.is_sell_task*2) * jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        trader_ids = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32) * agent_params.trader_id #This agent will always have the same (unique) trader ID
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (self.cfg.num_action_messages_by_agent, 2)#4 trades, 2 times
        )
        #------Check quants dont exceed inv----#
        quant_left=agent_state.task_to_execute-agent_state.quant_executed
        total_quant=quants.sum()
        quants = jnp.where(
                total_quant <= quant_left,
                quants,
                jnp.floor(quant_array[1]*quant_left)##spread evely across choices
            ).astype(jnp.int32)

        #jax.debug.print("quants:{}",quants)
        #jax.debug.print("quants left:{}",quant_left)
        #jax.debug.print("total quant:{}",total_quant)
        #jax.debug.print("quant_array of 0:{}",quant_array[0,:])
        #jax.debug.print("quant_array of 0:{}",quant_array[1,:])


        #--make arrays--#
        quants=jnp.array(quants)
        #jax.debug.print("quants:{}",quants)
        price_levels=jnp.array(price_levels)

        #---form messages---#

        # print([types, sides, quants, price_levels, order_ids,trader_ids])
        action_msgs = jnp.stack([types, sides, quants, price_levels, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)

        #jax.debug.print("action_msgs exec: {}", action_msgs)
        return action_msgs 






    #-------Action Functions-------#
    def _getActionMsgs_fixedQuant_1msg(self, action: jax.Array, world_state: WorldState, agent_state: ExecEnvState, agent_params: ExecEnvParams):
        """Action function for the fixed Quant Action space
        Pick for a ladder of quant execution options
        Always send 4 messages
        0 = No trade
        1=      # FT
        2=     # M
        3=    # NT
        4=    # PP
       """


        #######################################################
        # new way of implementing it with just 1 message
        #######################################################


        #----01 get price levels----#
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)

        def buy_task_prices(best_ask, best_bid):
            FT = best_ask
            M = ((best_bid + best_ask) // 2 // self.world_config.tick_size) * self.world_config.tick_size
            NT = best_bid
            PP = best_bid - self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, M, NT, PP
        
        def sell_task_prices(best_ask, best_bid):
            FT = best_bid
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                * self.world_config.tick_size).astype(jnp.int32)
            NT = best_ask
            PP = best_ask + self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, M, NT, PP
        
        price_levels = jax.lax.cond(
            agent_state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )

        #jax.debug.print("price_levels 1msg: {}", price_levels)

        #----02 get price and quantity based on action----#
        # Map action to specific price level and quantity
        # Action 0: No trade (quantity = 0, price = 0)
        # Action 1-4: Trade at specific price level with fixed quantity
        
        # Get the price for the selected action
        prices_array = jnp.array([0, price_levels[0], price_levels[1], price_levels[2], price_levels[3]])
        selected_price = prices_array[action]
        
        # Get the quantity for the selected action
        base_quant = self.cfg.fixed_quant_value
        if self.cfg.larger_far_touch_quant and action == 1:  # FT action
            base_quant = base_quant * 10
        
        quant_array = jnp.array([0, base_quant, base_quant, base_quant, base_quant])

        selected_quant = quant_array[action]
        
        #----03 check if quantity exceeds remaining inventory----#
        quant_left = agent_state.task_to_execute - agent_state.quant_executed
        selected_quant = jnp.where(
            selected_quant <= quant_left,
            selected_quant,
            0  # If exceeds inventory, set to 0 (no trade)
        ).astype(jnp.int32)

        #----04 construct single message----#
        # Message components for single message
        types = jnp.array([1], dtype=jnp.int32)  # 1 = limit order
        sides = jnp.array([(1 - agent_state.is_sell_task*2)], dtype=jnp.int32)  # 1 for buy, -1 for sell
        quants = jnp.array([selected_quant], dtype=jnp.int32).flatten()
        prices = jnp.array([selected_price], dtype=jnp.int32).flatten()
        trader_ids = jnp.array([agent_params.trader_id], dtype=jnp.int32)
        
        # Placeholder for order ids
        order_ids = jnp.array([self.world_config.placeholder_order_id], dtype=jnp.int32)
        
        # Time fields
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (1, 2)  # Shape (1 message, 2 time fields)
        )


        #jax.debug.print("task to execute: {}", agent_state.task_to_execute)
        #jax.debug.print("quant executed: {}", agent_state.quant_executed)

        #----05 form message----#
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids, trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times], axis=1)


        #jax.debug.print("action_msgs exec 1msg: {}", action_msgs)


        #jax.debug.print("action_msgs exec: {}", action_msgs)
        return action_msgs 



    def _getActionMsgs_fixedQuant_complex(self, action: jax.Array, world_state: WorldState, agent_state: ExecEnvState, agent_params: ExecEnvParams):
        """Action function for the fixed Quant Action space
        Pick for a ladder of quant execution options
        Always send 4 messages
        0 = No trade
        1=      # FT
        2=     # M
        3=    # NT
        4=    # PP
        5=    # FT*2 quant
        6=    # M*2 quant
        7=    # NT*2 quant
        8=    # PP*2 quant
        9=    # FT*5 quant
        10=   # M*5 quant
        11=   # NT*5 quant
        12=   # PP*5 quant

        
       """

        #----01 get price levels----#
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        #jax.debug.print('best_ask: {}, best_bid: {}', best_ask, best_bid)

        def buy_task_prices(best_ask, best_bid):
            FT = best_ask
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.world_config.tick_size) * self.world_config.tick_size
            NT = best_bid
            PP = best_bid - self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, M, NT, PP
        def sell_task_prices(best_ask, best_bid):
            FT = best_bid
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                 * self.world_config.tick_size).astype(jnp.int32)
            NT = best_ask
            PP = best_ask + self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, M, NT, PP
        
        price_levels = jax.lax.cond(
            agent_state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )

        #----02 get quants----#
        #jax.debug.print("action:{}",action)
        
        quant_array = jnp.array([
            [0, 0, 0, 0],  # No trade
            [1, 0, 0, 0],  # FT
            [0, 1, 0, 0],  # M
            [0, 0, 1, 0],  # NT
            [0, 0, 0, 1],  # PP
            [2, 0, 0, 0],  # FT*2 quant
            [0, 2, 0, 0],  # M*2 quant
            [0, 0, 2, 0],  # NT*2 quant
            [0, 0, 0, 2],  # PP*2 quant
            [5, 0, 0, 0],  # FT*3 quant
            [0, 5, 0, 0],  # M*3 quant
            [0, 0, 5, 0],  # NT*3 quant
            [0, 0, 0, 5],  # PP*3 quant
        ])
        quants=quant_array[action,:]*self.cfg.fixed_quant_value #Get the quant array based on the action
        quants = quants.flatten() #Flatten the array to 1D
        #----03 get the rest of the message----#
        types = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        sides = (1 - agent_state.is_sell_task*2) * jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        trader_ids = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32) * agent_params.trader_id #This agent will always have the same (unique) trader ID
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (self.cfg.num_action_messages_by_agent, 2)#4 trades, 2 times
        )
        #------Check quants dont exceed inv----#
        quant_left=agent_state.task_to_execute-agent_state.quant_executed
        total_quant=quants.sum()
        quants = jnp.where(
                total_quant <= quant_left,
                quants,
                jnp.floor(quant_array[1]*quant_left)##spread evely across choices
            ).astype(jnp.int32)
        #--make arrays--#
        quants=jnp.array(quants)
        #jax.debug.print("quants:{}",quants)
        price_levels=jnp.array(price_levels)
        #---form messages---#
        action_msgs = jnp.stack([types, sides, quants, price_levels, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        #jax.debug.print("action_msgs exec complex: {}", action_msgs)
        #jax.debug.print("quant_left: {}", quant_left)
        return action_msgs         


    def _getActionMsgs_simpleCase(self, action: jax.Array, world_state: WorldState, agent_state: ExecEnvState, agent_params: ExecEnvParams):
        """Action function for the simplest execution case
        Always send 1 message
        0 = No trade
        1 = Submit order at mkt price (FT)
        2 = Submit order at a passive price (limit order at near toucch)
       """

        #----01 get price levels----#
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        #jax.debug.print('best_ask: {}, best_bid: {}', best_ask, best_bid)

        def buy_task_prices(best_ask, best_bid):
            FT = best_ask
            NT = best_bid
            return FT, NT
        def sell_task_prices(best_ask, best_bid):
            FT = best_bid
            NT = best_ask
            return FT, NT, 
        
        price_levels = jax.lax.cond(
            agent_state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )

        #----02 get quants----#
        #jax.debug.print("action:{}",action)
        
        quant_array = jnp.array([
            [0, 0],  # No trade
            [self.cfg.fixed_quant_value, 0],  # FT-Aggressive
            [0, self.cfg.fixed_quant_value],  # NT-Passive
        ])
        quants=quant_array[action,:] #Get the quant array based on the action
        #----03 get the rest of the message----#
        types = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        sides = (1 - agent_state.is_sell_task*2) * jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        trader_ids = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32) * agent_params.trader_id #This agent will always have the same (unique) trader ID
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (self.cfg.num_action_messages_by_agent, 2)#4 trades, 2 times
        )
        #------Check quants dont exceed inv----#
        quant_left=agent_state.task_to_execute-agent_state.quant_executed
        total_quant=quants.sum()
        quants = jnp.where(
                total_quant <= quant_left,
                quants,
                jnp.floor(quant_array[1]*quant_left)##spread evely across choices
            ).astype(jnp.int32)
        #--make arrays--#
        quants=jnp.array(quants)
        #jax.debug.print("quants:{}",quants)
        price_levels=jnp.array(price_levels)
        #---form messages---#
        action_msgs = jnp.stack([types, sides, quants, price_levels, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        return action_msgs

    
    def _getActionMsgs_fixedPrice(self, action: jax.Array, world_state: WorldState, agent_state: ExecEnvState, agent_params: ExecEnvParams):
        """get messages for action space where input is quantity at each price level"""
        

        def normal_quant_price(price_levels: jax.Array, action: jax.Array):
            def combine_mid_nt(quants, prices):
                quants = quants \
                    .at[2].set(quants[2] + quants[1]) \
                    .at[1].set(0)
                prices = prices.at[1].set(-1)
                return quants, prices

            quants = action.astype(jnp.int32)
            prices = jnp.array(price_levels[:-1])
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
        
        # def market_quant_price(price_levels: jax.Array, state: ExecEnvState, action: jax.Array):
        #     mkt_quant = state.task_to_execute - state.quant_executed
        #     quants = jnp.asarray((mkt_quant, 0, 0, 0), jnp.int32) 
        #     return quants, jnp.asarray((price_levels[-1], -1, -1, -1), jnp.int32)
        
        def buy_task_prices(best_ask, best_bid):
            # FT = best_ask
            # essentially convert to market order (20% higher price than best ask)
            FT = ((best_ask) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.world_config.tick_size) * self.world_config.tick_size
            NT = best_bid
            PP = best_bid - self.world_config.tick_size*self.cfg.n_ticks_in_book
            MKT = self.world_config.maxint
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
            FT = ((best_bid) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32)
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                 * self.world_config.tick_size).astype(jnp.int32)
            NT = best_ask
            PP = best_ask + self.world_config.tick_size*self.cfg.n_ticks_in_book
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
        types = jnp.ones((self.cfg.n_actions,), jnp.int32)
        sides = (1 - agent_state.is_sell_task*2) * jnp.ones((self.cfg.n_actions,), jnp.int32)
        trader_ids = jnp.ones((self.cfg.n_actions,), jnp.int32) * agent_params.trader_id #This agent will always have the same (unique) trader ID
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (self.cfg.n_actions, 2)
        )
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        best_ask = jnp.int32((world_state.best_asks[-10:].mean(axis=0)[0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-10:].mean(axis=0)[0] // self.world_config.tick_size) * self.world_config.tick_size)
        jax.debug.print('best_ask: {}, best_bid: {}', best_ask, best_bid)

        price_levels = jax.lax.cond(
            agent_state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )
        jax.debug.print('price_levels\n {}', price_levels)
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        # if self.ep_type == 'fixed_time':
        #     remainingTime = self.world_config.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
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
        action_msgs = jnp.stack([types, sides, quants, prices, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)
        # jax.debug.print('action_msgs\n {}', action_mgs)
        return action_msgs
        # ============================== Get Action_msgs ==============================


    def _getActionMsgs_twap(self, action: jax.Array, world_state : WorldState, agent_state: ExecEnvState, agent_params: ExecEnvParams):
        """Action function for the twap baseline action space
            For now, assume only one possible action. 
            Can expand later to allow for flavours of twap, or parametrised twap. 
        0 = Execute TWAP Strategy with Aggressive Price (FT)
        1 = Execute TWAP Strategy with Passive Price (NT)
       """



        #calculate % time (steps) remaining in the episode 
        # Calculate % time or steps remaining in the episode 
        if self.world_config.ep_type == 'fixed_time':
            raise NotImplementedError("TWAP not implemented for fixed time episodes, need to have some notion of delta_time per step.")
        elif self.world_config.ep_type == 'fixed_steps':
            # Calculate remaining steps as a percentage
            steps_left=world_state.max_steps_in_episode - world_state.step_counter-1
            quant_left = agent_state.task_to_execute - agent_state.quant_executed
            quant_this_step= jnp.ceil(quant_left / steps_left).astype(jnp.int32)  # quant to execute this step
        # Get the quants based on the action


        #----01 get price levels----#
        best_ask = jnp.int32((world_state.best_asks[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        best_bid = jnp.int32((world_state.best_bids[-1][0] // self.world_config.tick_size) * self.world_config.tick_size)
        #jax.debug.print('best_ask: {}, best_bid: {}', best_ask, best_bid)

        def buy_task_prices(best_ask, best_bid):
            FT = best_ask
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.world_config.tick_size) * self.world_config.tick_size
            NT = best_bid
            PP = best_bid - self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, NT
        def sell_task_prices(best_ask, best_bid):
            FT = best_bid
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 // self.world_config.tick_size)
                 * self.world_config.tick_size).astype(jnp.int32)
            NT = best_ask
            PP = best_ask + self.world_config.tick_size*self.cfg.n_ticks_in_book
            return FT, NT
        
        price_levels = jax.lax.cond(
            agent_state.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_ask, best_bid
        )

        quant_array = jnp.array([
                [1, 0],  # FT
                [0, 1],  # NT
            ])



        quants=quant_array[action,:]*quant_this_step



        quants = quants.flatten() #Flatten the array to 1D
        #----03 get the rest of the message----#
        types = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        sides = (1 - agent_state.is_sell_task*2) * jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32)
        trader_ids = jnp.ones((self.cfg.num_action_messages_by_agent,), jnp.int32) * agent_params.trader_id #This agent will always have the same (unique) trader ID
        # Placeholder for order ids
        order_ids = jnp.full((self.cfg.num_action_messages_by_agent,), self.world_config.placeholder_order_id, dtype=jnp.int32)
        times = jnp.resize(
            world_state.time + self.cfg.time_delay_obs_act,
            (self.cfg.num_action_messages_by_agent, 2)#4 trades, 2 times
        )


        #jax.debug.print("quants:{}",quants)
        #jax.debug.print("quants left:{}",quant_left)
        #jax.debug.print("total quant:{}",total_quant)
        #jax.debug.print("quant_array of 0:{}",quant_array[0,:])
        #jax.debug.print("quant_array of 0:{}",quant_array[1,:])


        #--make arrays--#
        quants=jnp.array(quants)
        #jax.debug.print("quants:{}",quants)
        price_levels=jnp.array(price_levels,ndmin=1)

        #---form messages---#

        # print([types, sides, quants, price_levels, order_ids,trader_ids])
        action_msgs = jnp.stack([types, sides, quants, price_levels, order_ids,trader_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs, times],axis=1)

        # jax.debug.print("action_msgs exec twap: \n  {}", action_msgs)

        #jax.debug.print("action_msgs exec: {}", action_msgs)
        return action_msgs 



    def _get_messages(
        self,
        action: jax.Array,
        world_state: MultiAgentState,
        agent_state: ExecEnvState,
        agent_params: ExecEnvParams,
    ) -> Tuple[jax.Array, jax.Array]:
        """Get the action and cancel messages for the execution agent."""

        # 1. Get action messages
        action_msgs = self.get_action(
            action=action,
            world_state=world_state,
            agent_state=agent_state,
            agent_params=agent_params
        )

        # 2. Determine which side to cancel (buy or sell task)
        side_for_exe = 1 - agent_state.is_sell_task * 2  # 1 for buy, -1 for sell

        # 3. Select the correct book side
        raw_order_side = jax.lax.cond(
            agent_state.is_sell_task,
            lambda: world_state.ask_raw_orders,
            lambda: world_state.bid_raw_orders
        )

        # 4. Get cancel messages
        cancel_msgs = job.getCancelMsgs(
            bookside=raw_order_side,
            agentID=agent_params.trader_id,
            size=self.cfg.num_messages_by_agent // 2,  # adjust if needed
            side=side_for_exe,
            cancel_time=world_state.time[0],
            cancel_time_ns=world_state.time[1]
        )

        # 5. Filter messages
        action_msgs, cancel_msgs = self._filter_messages(action_msgs, cancel_msgs)

        #jax.debug.print("action messages order exec: {}", action_msgs)
        #jax.debug.print("cancel messages order exec: {}", cancel_msgs)

        # 6. Return
        return action_msgs, cancel_msgs






    #======================Wrappers to choose funcitons=========================================#    
    def get_episode_end_fn(self,key,quant_left,bestasks, bestbids, time, asks, bids, trades, state, params):
        """
        Wrapper function to call the appropriate episode end function.
        """
        if self.cfg.end_fn == "force_market_order":
            return self.end_fn(key,quant_left,bestasks, bestbids, time, asks, bids, trades, state, params)
        elif self.cfg.end_fn =="unwind_FT":
            return self.end_fn(quant_left,bestasks, bestbids, time, asks, bids, trades, state, params)
        else:
            raise ValueError("Invalid end_fn specified.")
        
    def get_action(self,action, world_state, agent_state, agent_params):
        """
        Wrapper function to call the appropriate action function.
        """
        if self.cfg.action_space == "fixed_quants":
            return self.action_fn(action = action, world_state = world_state, agent_state = agent_state, agent_params = agent_params)
        elif self.cfg.action_space == "fixed_prices":
            return self.action_fn(action = action, world_state = world_state, agent_state = agent_state, agent_params = agent_params)
        elif self.cfg.action_space == "fixed_quants_complex":
            return self.action_fn(action = action, world_state = world_state, agent_state = agent_state, agent_params = agent_params)
        elif self.cfg.action_space == "simplest_case":
            return self.action_fn(action = action, world_state = world_state, agent_state = agent_state, agent_params = agent_params)
        elif self.cfg.action_space == "fixed_quants_1msg":
            return self.action_fn(action = action, world_state = world_state, agent_state = agent_state, agent_params = agent_params)
        elif self.cfg.action_space == "twap":
            return self.action_fn(action = action, world_state = world_state, agent_state = agent_state, agent_params = agent_params)
        else:
            raise ValueError("Invalid action space specified.")    
    

    def get_observation(self, world_state, agent_state, agent_param, total_messages, old_time, old_mid_price, lob_state_before, normalize,flatten):
        """
        Wrapper function to call the appropriate observation function.
        """
        if self.cfg.observation_space == "engineered":
            return self.observation_fn(world_state=world_state, 
                                       agent_state=agent_state, 
                                       normalize=normalize,
                                       flatten=flatten)
        elif self.cfg.observation_space == "basic":
            return self.observation_fn(world_state=world_state, 
                                       agent_state=agent_state, 
                                       normalize=normalize,
                                       flatten=flatten)
        elif self.cfg.observation_space == "simplest_case":
            return self.observation_fn(world_state=world_state, 
                                       agent_state=agent_state, 
                                       normalize=normalize,
                                       flatten=flatten)
        else:
            raise ValueError("Invalid observation_space specified.")
        



    #--------unwind at mid FT-good for MARL------#
    def unwind_FT(
            self,
            quant_left: jax.Array,
            bestask: jax.Array,
            bestbid: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: ExecEnvState,
            params: ExecEnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:   
        
        #-----check if ep over-----#
        if self.ep_type == 'fixed_time':
            remainingTime = self.world_config.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= self.world_config.last_step_seconds   # 5 seconds
        else:
            ep_is_over = state.max_steps_in_episode - state.step_counter <= 1
        averageMidprice = ((bestask[0] + bestbid[0]) // 2).mean() // self.world_config.tick_size * self.world_config.tick_size
        #jax.debug.print("mid_price:{}",mid_price)
        
        new_time = time + self.cfg.time_delay_obs_act
        next_id = state.customIDcounter + self.cfg.n_actions + 1

        doom_price = jax.lax.cond(
            state.is_sell_task,
            lambda: ((bestbid[0])// self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            lambda: ((bestask[0])// self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
        )

        def place_midprice_trade(trades, price, quant, time):
            '''Place a doom trade at a trade at mid price to close out our mm agent at the end of the episode.'''
            mid_trade = job.create_trade(
                price, quant, -666666,  agent_params.trader_id + state.customIDcounter+ 1 +self.cfg.n_actions, *time, -666666, agent_params.trader_id)
            trades = job.add_trade(trades, mid_trade)
            #jax.debug.print("called?")
            return trades
        
        #Get side to place trade. +ve quant means we (aggresive) sold.
        side_sign=(state.is_sell_task*2-1) # 1 if sell, -1 if buy
        
        trades = jax.lax.cond(
            ep_is_over & (jnp.abs(quant_left) > 0),  # Check if episode is over and we still have remaining quantity
            place_midprice_trade,  # Place a midprice trade
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, doom_price, side_sign*jnp.abs(quant_left), new_time  # Inv +ve means incoming is sell so standing buy.
        )
        #Return traded amounts
        doom_quant = ep_is_over * quant_left
        mkt_exec_quant=0 #return for consitency
        id_counter=next_id

        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant
    


    
    #--------Force market if done-------------#
    def _force_market_order_if_done(
            self,
            key: chex.PRNGKey,
            quant_left: jax.Array,
            bestask: jax.Array,
            bestbid: jax.Array,
            time: jax.Array,
            asks: jax.Array,
            bids: jax.Array,
            trades: jax.Array,
            state: ExecEnvState,
            params: ExecEnvParams,
        ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array], int, int, int, int]:
        """ Force a market order if episode is over (either in terms of time or steps). """
        
        def create_mkt_order():
            mkt_p = (1 - state.is_sell_task) * self.cfg.maxint // self.world_config.tick_size * self.world_config.tick_size
            side = (1 - state.is_sell_task*2)
            # TODO: this addition wouldn't work if the ns time at index 1 increases to more than 1 sec
            new_time = time + self.cfg.time_delay_obs_act
            mkt_msg = jnp.array([
                # type, side, quant, price
                1, side, quant_left, mkt_p,
                agent_params.trader_id,
                agent_params.trader_id + state.customIDcounter + self.cfg.n_actions,  # unique order ID for market order
                *new_time,  # time of message
            ])
            next_id = state.customIDcounter + self.cfg.n_actions + 1
            return mkt_msg, next_id, new_time

        def create_dummy_order():
            next_id = state.customIDcounter + self.cfg.n_actions
            return jnp.zeros((8,), dtype=jnp.int32), next_id, time 
        

        def place_doom_trade(trades, price, quant, time):
            doom_trade = job.create_trade(
                price, quant, agent_params.trader_id + self.cfg.n_actions + 1, -666666, *time, agent_params.trader_id, -666666)
            # jax.debug.print('doom_trade\n {}', doom_trade)
            trades = job.add_trade(trades, doom_trade)
            return trades

        if self.ep_type == 'fixed_time':
            remainingTime = self.world_config.episode_time - jnp.array((time - state.init_time)[0], dtype=jnp.int32)
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

        (asks, bids, trades), (new_bestask, new_bestbid) = job.cond_type_side_save_bidask(self.cfg,
            (asks, bids, trades),
            (key,order_msg)
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
            jnp.abs(trades[:, 1]),  # executed quantity
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
            lambda: ((0.75 * bestbid[0]) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            lambda: ((1.25 * bestask[0]) // self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
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
        # agent_trades = job.get_agent_trades(trades, agent_params.trader_id)
        # jax.debug.print('agent_trades\n {}', agent_trades[:20])
        # price_quants = self._get_executed_by_price(agent_trades)
        # jax.debug.print('price_quants\n {}', price_quants)
        doom_quant = ep_is_over * quant_still_left

        return (asks, bids, trades), (bestask, bestbid), id_counter, time, mkt_exec_quant, doom_quant

    def _get_reward(self, 
                    world_state: WorldState, 
                    agent_state: ExecEnvState, 
                    agent_params: ExecEnvParams, 
                    trades: chex.Array, 
                    bestasks: chex.Array, 
                    bestbids: chex.Array, 
                    time: jax.Array) -> jnp.int32:

        #########################################################################################
        # Add artificial trade if episode is done
        # Important: this artificial trade is not saved, its just used to calculate the reward
        #########################################################################################

        agent_trades_before_unwind = job.get_agent_trades(trades, agent_params.trader_id)
        quant_executed_this_step = jnp.abs(agent_trades_before_unwind[:,1].sum()) # QUants can be negative, therefore take absolute value
        quant_left = agent_state.task_to_execute - (agent_state.quant_executed + quant_executed_this_step)

        #jax.debug.print(f"quant_left: {quant_left}")

        # print("trader id: ", agent_params.trader_id)
        # print("trades before unwind: ", agent_trades_before_unwind)
        #jax.debug.print(f"quant_executed_this_step: {quant_executed_this_step}")
        # print(f"agent_state.task_to_execute: {agent_state.task_to_execute}")
        # print(f"quant_left: {quant_left}")

        #-----check if ep over-----#
        if self.world_config.ep_type == 'fixed_time':
            remainingTime = self.world_config.episode_time - jnp.array((time - world_state.init_time)[0], dtype=jnp.int32)
            ep_is_over = remainingTime <= self.world_config.last_step_seconds   # 5 seconds
            #jax.debug.print("ep_is_over fixed time: {}", ep_is_over)
            #jax.debug.print("remainingTime: {}", remainingTime)
            #jax.debug.print("world_state.init_time: {}", world_state.init_time)
            #jax.debug.print("time: {}", time)
        else:
            ep_is_over = world_state.max_steps_in_episode - world_state.step_counter - 1 <= 1
            #jax.debug.print("ep_is_over: {}", ep_is_over)
            #jax.debug.print("world_state.max_steps_in_episode: {}", world_state.max_steps_in_episode)
            #jax.debug.print("world_state.step_counter: {}", world_state.step_counter)
        averageMidprice = ((bestbids[:, 0] + bestasks[:, 0]) / 2).mean() // self.world_config.tick_size * self.world_config.tick_size
        #jax.debug.print("mid_price:{}",mid_price)


        #jax.debug.print(f"bestbid 0: {bestbids[-1,0]}")
        #jax.debug.print(f"bestask 0: {bestasks[-1,0]}")
        # print(bestasks[-10,0])

        penalty = self.cfg.doom_price_penalty

        doom_price = jax.lax.cond(
            agent_state.is_sell_task,
            lambda: (((bestbids[-1,0]) * (1-penalty))// self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
            lambda: (((bestasks[-1,0]) * (1+penalty))// self.world_config.tick_size * self.world_config.tick_size).astype(jnp.int32),
        )

        #jax.debug.print("doom_price: {}", doom_price)

        def place_midprice_trade(trades, price, quant, time):
            '''Place a doom trade at a trade at mid price to close out our mm agent at the end of the episode.'''
            
            # print("price shape: {}", price.shape)
            # print("quant shape: {}", quant.shape)
            # print("time shape: {}", time.shape)
            # print("trader id shape: {}", agent_params.trader_id.shape)
            #jax.debug.print("quant_left: {}", jnp.abs(quant_left))
            
            
            mid_trade = job.create_trade(
                price, quant, self.world_config.artificial_order_id_end_episode,  self.world_config.placeholder_order_id, *time, self.world_config.artificial_trader_id_end_episode, agent_params.trader_id)
            trades = job.add_trade(trades, mid_trade)
            #jax.debug.print("called?")
            return trades
        
        #Get side to place trade. +ve quant means we (aggresive) sold.
        side_sign=(agent_state.is_sell_task*2-1) # 1 if sell, -1 if buy
        
        # Add artificial trade to trades object if episode is over and we still have remaining quantity
        trades = jax.lax.cond(
            ep_is_over & (jnp.abs(quant_left) > 0),  # Check if episode is over and we still have remaining quantity
            place_midprice_trade,  # Place a midprice trade
            lambda trades, b, c, d: trades,  # If not, return the existing trades
            trades, doom_price, side_sign*jnp.abs(quant_left), time  # Inv +ve means incoming is sell so standing buy.
        )
        #Return traded amounts
        doom_quant = ep_is_over * quant_left

        #jax.debug.print("trades exec env: {}", trades)

        #################################
        # Get reward
        #################################

        # ========== get reward and revenue ==========
        # Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        # mask2 = ((job.INITID < executed[:, 2]) & (executed[:, 2] < 0)) | ((job.INITID < executed[:, 3]) & (executed[:, 3] < 0))
        mask2 = (agent_params.trader_id == executed[:, 6])  | (agent_params.trader_id == executed[:, 7]) #Mask to find trader ID
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        otherTrades = jnp.where(mask2[:, jnp.newaxis], 0, executed)
        # jax.debug.print('agentTrades\n {}', agentTrades[:30])
        agentQuant = jnp.abs(agentTrades[:,1]).sum() # new_execution quants


        def check_final_quant(ep_is_over,quant_left,doom_quant,doom_price,trades,agentTrades,otherTrades):
            if ep_is_over and quant_left!=0:
                print("DOOM TRADE HAPPENED")
                print(doom_quant,doom_price)
                print("The trades post doom are.")
                print(trades)
                print("The agent trades post doom are.")
                print(agentTrades)
                print("The other trades post doom are.")
                print(otherTrades)

        # jax.debug.callback(check_final_quant,ep_is_over,quant_left,doom_quant,doom_price,trades,agentTrades,otherTrades)
        
        #jax.debug.print("agentTrades:{}",agentTrades)

        # ---------- used for vwap, revenue ----------
        # vwapFunc = lambda tr: jnp.nan_to_num(
        #     (tr[:,0] // self.world_config.tick_size * tr[:,1]).sum() / (tr[:,1]).sum(),
        #     state.init_price  # if no trades happened, use init price
        # ) # caution: this value can be zero (executed[:,1]).sum()
        # only use other traders' trades for value weighted price
        # vwap = vwapFunc(otherTrades) # average_price of all other trades

        other_exec_quants = jnp.abs(otherTrades[:, 1]).sum()
        vwap = jax.lax.cond(
            other_exec_quants == 0,
            lambda: agent_state.init_price / self.world_config.tick_size,
            lambda: (otherTrades[:, 0] // self.world_config.tick_size * jnp.abs(otherTrades[:, 1])).sum() / other_exec_quants
        )
        
        #jax.debug.print("vwap: {} ", vwap) 

        revenue = (agentTrades[:,0] // self.world_config.tick_size * jnp.abs(agentTrades[:,1])).sum()
        
        # ---------- used for slippage, price_drift, and RM(rolling mean) ----------
        rollingMeanValueFunc_FLOAT = lambda average_val,new_val:(average_val*world_state.step_counter+new_val)/(world_state.step_counter+1)
        vwap_rm = rollingMeanValueFunc_FLOAT(agent_state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)
        price_adv_rm = rollingMeanValueFunc_FLOAT(agent_state.price_adv_rm,revenue/(agentQuant+0.001) - vwap) # slippage=revenue/agentQuant-vwap, where revenue/agentQuant means agentPrice 
        slippage_rm = rollingMeanValueFunc_FLOAT(agent_state.slippage_rm,revenue - agent_state.init_price//self.world_config.tick_size*agentQuant)
        price_drift_rm = rollingMeanValueFunc_FLOAT(agent_state.price_drift_rm,(vwap - agent_state.init_price//self.world_config.tick_size)) #price_drift = (vwap - state.init_price//self.world_config.tick_size)
        
        # ---------- used for advantage and drift ----------
        # switch sign for buy task
        direction_switch = jnp.sign(agent_state.is_sell_task * 2 - 1)
        advantage = direction_switch * (revenue - vwap * agentQuant) # advantage_vwap

        #jax.debug.print("init price: {}", agent_state.init_price)
        #jax.debug.print("advantage: {}", advantage)
        #jax.debug.print("vwap: {}", vwap)
       # jax.debug.print("exec quant left: {}", quant_left)


        drift = direction_switch * agentQuant * (vwap - agent_state.init_price//self.world_config.tick_size)
        
        # ---------- compute the final reward ----------
        # rewardValue = revenue 
        # rewardValue =  advantage
        # rewardValue1 = advantage + params.reward_lambda * drift
        # rewardValue1 = advantage + 1.0 * drift
        # rewardValue2 = revenue - (state.init_price // self.world_config.tick_size) * agentQuant
        # rewardValue = rewardValue1 - rewardValue2
        # rewardValue = revenue - vwap_rm * agentQuant # advantage_vwap_rm

        # rewardValue = revenue - (state.init_price // self.world_config.tick_size) * agentQuant
        reward = advantage + self.cfg.reward_lambda * drift
        reward_lam1 = direction_switch * (
            revenue - (agent_state.init_price // self.world_config.tick_size) * agentQuant
        )
        
        #jax.debug.print("reward exec: {}", reward)

        # Add other extras

        trade_duration_step = (jnp.abs(agentTrades[:, 1]) / agent_state.task_to_execute * (agentTrades[:, -2] - world_state.init_time[0])).sum()
        trade_duration = agent_state.trade_duration + trade_duration_step
        quant_left = agent_state.task_to_execute - agent_state.quant_executed - agentQuant

        reward_info={
        "reward":reward,
        "agentQuant": agentQuant,
        "revenue": revenue,
        "reward_lam1":reward_lam1 ,  # pure revenue is not informative if direction is random (-> flip and normalise)
        "slippage_rm": slippage_rm,
        "price_adv_rm": price_adv_rm,
        "price_drift_rm": price_drift_rm,
        "vwap_rm": vwap_rm,
        "advantage": advantage,
        "drift": drift,
        "doom_quant": doom_quant,
        "quant_left": quant_left,
        "trade_duration": trade_duration,
        }
        reward_scaled = reward


        if self.cfg.reward_space == "finish_fast":
            reward = -jnp.abs(quant_left) #/ agent_state.task_to_execute
            #jax.debug.print("reward:{}",reward)
            #jax.debug.print("agentQuant:{}",agentQuant)
            reward_scaled = reward / 10


        if self.cfg.reward_space == "simplest_case":
            entry_price=agent_state.init_price
            price_slip=agentTrades[:,0]-jnp.ones_like(agentTrades[:,0])*entry_price #Trade price - 1st price.
            price_slip=jnp.where(agent_state.is_sell_task,price_slip,-price_slip)
            reward=jnp.dot(price_slip,jnp.abs(agentTrades[:,1]))

            # jax.debug.print("entry_price: {}", entry_price)
            # jax.debug.print("agentTrades[:,0]: {}", agentTrades[:,0])
            # jax.debug.print("price_slip: {}", price_slip)
            # jax.debug.print("agentTrades[:,1]: {}", agentTrades[:,1])
            # jax.debug.print("Reward: {}", reward)

            reward_scaled=reward/self.cfg.task_size
            # jax.debug.print("dot(price_slip, agentTrades[:,1]): {}", jnp.dot(price_slip, agentTrades[:,1]))

            # price_slip=jax.lax.cond(
            #     agent_state.is_sell_task,
            #     ,
            #     agentTrades[:,0]-jnp.ones_like(agentTrades[:,0]*entry_price)  
            # )
        
        # jax.debug.print('reward: {}. reward_lam1: {}. is_sell_task {}. advantage {} drift {} vwap {} init_price {}', 
        #                 reward, reward_lam1, state.is_sell_task, advantage, drift, vwap, state.init_price)
        
        # ---------- normalize the reward ----------
        # reward /= 10_000
        # reward /= params.avg_twap_list[state.window_index]



        return reward_scaled, reward_info


    def update_state_and_get_done_and_info(self, world_state:WorldState, agent_state_old: ExecEnvState, extras) -> Tuple[ExecEnvState, Dict]:
        # Get new state
        new_quant_executed = agent_state_old.quant_executed + extras["agentQuant"]
        new_total_revenue = agent_state_old.total_revenue + extras["revenue"]
        new_drift_return = agent_state_old.drift_return + extras["drift"]
        new_advantage_return = agent_state_old.advantage_return + extras["advantage"]
        new_slippage_rm = extras["slippage_rm"]
        new_price_adv_rm = extras["price_adv_rm"]
        new_price_drift_rm = extras["price_drift_rm"]
        new_vwap_rm = extras["vwap_rm"]
        new_trade_duration = extras["trade_duration"]
        new_quant_left = extras["quant_left"]
        new_reward = extras["reward"]

        # Note: we use replace because init_price, task_to_execute, is_sell_task do not change
        agent_state = agent_state_old.replace(
            quant_executed = new_quant_executed,
            total_revenue = new_total_revenue,
            drift_return = new_drift_return,
            advantage_return = new_advantage_return,
            slippage_rm = new_slippage_rm,
            price_adv_rm = new_price_adv_rm,
            price_drift_rm = new_price_drift_rm,
            vwap_rm = new_vwap_rm,
            trade_duration = new_trade_duration)
        
        # Get done
        done = self.is_terminal(world_state, agent_state)

        # Get info
        average_price = jnp.nan_to_num(agent_state.total_revenue 
                                            / agent_state.quant_executed, 0.0)
        drift = extras["drift"]
        advantage= extras["advantage"]
        doom_quant = extras["doom_quant"]

        info = {
            # "total_revenue": agent_state.total_revenue,
            # "quant_executed": agent_state.quant_executed,
            # "task_to_execute": agent_state.task_to_execute,
            "quant_left": new_quant_left,
            # "average_price": average_price,
            "done": done,
            "revenue_direction_normalised": extras["reward_lam1"],  # pure revenue is not informative if direction is random (-> flip and normalise)
            # "slippage_rm": agent_state.slippage_rm,
            # "price_adv_rm": agent_state.price_adv_rm,
            # "price_drift_rm": agent_state.price_drift_rm,
            # "vwap_rm": agent_state.vwap_rm,
            #"advantage_reward": agent_state.advantage_return,
            #"drift_reward": agent_state.drift_return,
            "drift" : drift,
            "advantage": advantage,
            # "trade_duration": agent_state.trade_duration,
            "doom_quant": doom_quant,
            "is_sell_task": agent_state.is_sell_task,
            "reward": new_reward,
        }

        #jax.debug.print("info exec env: {}", info)


        return agent_state, done, info

    def _get_obs_simplest_case(self, world_state: WorldState, agent_state: ExecEnvState, normalize: bool, flatten: bool = True) -> chex.Array:
        """ Return very basic obs space"""
        time_used= world_state.time - world_state.init_time
        # jax.debug.print('Time used:\n {}', time_used)
        # jax.debug.print('Task to exec executed\n {}', agent_state.task_to_execute)
        # jax.debug.print('Quant executed\n {}', agent_state.quant_executed)
        obs = {
            "percent_time_remaining": (self.world_config.episode_time-(time_used[0]+time_used[1]/1e9))/ self.world_config.episode_time,# time is [s,ns] # ep time is in seconds
            "percent_remaining_quant": (agent_state.task_to_execute - agent_state.quant_executed)/agent_state.task_to_execute,
            "mid_price": world_state.mid_price,
        }

        # jax.debug.print('obs:\n {}', obs)

        # FIXME: These are hardcoded values which are extremely stock-specific and should be replaced with dynamic values
        means = {
            "percent_time_remaining": 0.5,
            "percent_remaining_quant": 0.5,
            "mid_price": 7560000,
        }
        
        stds = {
            "percent_time_remaining": 1,
            "percent_remaining_quant": 1,
            "mid_price": 1e3,
        }
        
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)

        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        
        return obs



    def _get_obs_basic(self, world_state: WorldState, agent_state: ExecEnvState, normalize: bool, flatten: bool = True) -> chex.Array:
        """ Return very basic obs space"""
        obs = {
            "best_ask_price": world_state.best_asks[-1][0],
            "best_bid_price": world_state.best_bids[-1][0],
            "remaining_quant": agent_state.task_to_execute - agent_state.quant_executed,
        }

        means = {
            "best_ask_price": 1550000,
            "best_bid_price": 1550000,
            "remaining_quant": 0,
        }
        
        stds = {
            "best_ask_price": 1e3,
            "best_bid_price": 1e3,
            "remaining_quant": self.cfg.task_size,
        }
        
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)

        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs)
        
        return obs






    def _get_obs(
            self,
            agent_state: ExecEnvState,
            world_state: WorldState,
            normalize: bool,
            flatten: bool = True,
        ) -> chex.Array:
        """ Return observation from raw state trafo. """

        # NOTE: only uses most recent observation from state
        quote_aggr, quote_pass = jax.lax.cond( # Quote includes price and quantity
            agent_state.is_sell_task,
            lambda: (world_state.best_bids[-1], world_state.best_asks[-1]),
            lambda: (world_state.best_asks[-1], world_state.best_bids[-1]),
        )

        # print("agent_state:", agent_state.is_sell_task)
        # print(f"quite aggr: {quote_aggr}, quote pass: {quote_pass}")


        time = world_state.time[0] + world_state.time[1]/1e9
        time_elapsed = time - (world_state.init_time[0] + world_state.init_time[1]/1e9)
        # print('prev_action_shape', world_state.prev_action.shape)
        sign_switch = 2 * agent_state.is_sell_task - 1
        if self.world_config.ep_type == "fixed_time":
            obs = {
                "is_sell_task": agent_state.is_sell_task,
                "p_aggr": quote_aggr[0] * sign_switch,  # switch sign for buy task TODO why do we have a sign switch here?
                "p_pass": quote_pass[0] * sign_switch,  # switch sign for buy task
                "spread": jnp.abs(quote_aggr[0] - quote_pass[0]),
                "q_aggr": quote_aggr[1],
                "q_pass": quote_pass[1],
                #"q_pass2": state.quant_passive_2, # TODO add price here, calculate it correctly
                # "q_before2": None, # how much quantity lies above this price level
                "time": time,
                "delta_time": world_state.delta_time,
                # "episode_time": state.time - state.init_time,
                "time_remaining": self.world_config.episode_time - time_elapsed,
                "init_price": agent_state.init_price,
                "current_task_size": agent_state.task_to_execute,
                "executed_quant": agent_state.quant_executed,
                "remaining_quant": agent_state.task_to_execute - agent_state.quant_executed,
                "step_counter": world_state.step_counter,
                # "remaining_ratio": 1. - jnp.nan_to_num(state.step_counter / state.max_steps_in_episode, nan=1.),
                "remaining_ratio": jnp.where(world_state.max_steps_in_episode==0, 0., 1. - world_state.step_counter / world_state.max_steps_in_episode),#17
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
                "p_aggr": agent_state.init_price * sign_switch, #p_mean,
                "p_pass": agent_state.init_price * sign_switch, #p_mean,
                "spread": 0,
                "q_aggr": 0,
                "q_pass": 0,
                #"q_pass2": 0,
                "time": 0,
                "delta_time": 0,
                # "episode_time": jnp.array([0, 0]),
                "time_remaining": 0,
                "init_price": 0, #p_mean,
                "current_task_size": 0,
                "executed_quant": 0,
                "remaining_quant": 0,
                "step_counter": 0,
                "remaining_ratio": 0,
            }
            stds = {
                "is_sell_task": 1,
                "p_aggr": 1e5, #p_std,
                "p_pass": 1e5, #p_std,
                "spread": 1e4,
                "q_aggr": 100,
                "q_pass": 100,
            #"q_pass2": 100,
                "time": 1e5,
                "delta_time": 10,
                # "episode_time": jnp.array([1e3, 1e9]),
                "time_remaining": self.world_config.episode_time, # 10 minutes = 600 seconds
                "init_price": 1e7, #p_std,
                "current_task_size": self.cfg.task_size,
                "executed_quant": self.cfg.task_size,
                "remaining_quant": self.cfg.task_size,
                "step_counter": 30,  # TODO: find way to make this dependent on episode length
                "remaining_ratio": 1,
            }
        elif self.world_config.ep_type == "fixed_steps": # leave away time related stuff
            obs = {
                "is_sell_task": agent_state.is_sell_task,
                "p_aggr": quote_aggr[0] * sign_switch,  # switch sign for buy task TODO why do we have a sign switch here?
                "p_pass": quote_pass[0] * sign_switch,  # switch sign for buy task
                "spread": jnp.abs(quote_aggr[0] - quote_pass[0]),
                "q_aggr": quote_aggr[1],
                "q_pass": quote_pass[1],
                #"q_pass2": state.quant_passive_2, # TODO add price here, calculate it correctly
                # "q_before2": None, # how much quantity lies above this price level
                "init_price": agent_state.init_price,
                "current_task_size": agent_state.task_to_execute,
                "executed_quant": agent_state.quant_executed,
                "remaining_quant": agent_state.task_to_execute - agent_state.quant_executed,
                "step_counter": world_state.step_counter,
                # "remaining_ratio": 1. - jnp.nan_to_num(state.step_counter / state.max_steps_in_episode, nan=1.),
                "remaining_ratio": jnp.where(world_state.max_steps_in_episode==0, 0., 1. - world_state.step_counter / world_state.max_steps_in_episode),#17
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
                "p_aggr": agent_state.init_price * sign_switch, #p_mean,
                "p_pass": agent_state.init_price * sign_switch, #p_mean,
                "spread": 0,
                "q_aggr": 0,
                "q_pass": 0,
                #"q_pass2": 0,
                "init_price": 0, #p_mean,
                "current_task_size": 0,
                "executed_quant": 0,
                "remaining_quant": 0,
                "step_counter": 0,
                "remaining_ratio": 0,
            }
            stds = {
                "is_sell_task": 1,
                "p_aggr": 1e5, #p_std,
                "p_pass": 1e5, #p_std,
                "spread": 1e4,
                "q_aggr": 100,
                "q_pass": 100,
            #"q_pass2": 100,
                "init_price": 1e7, #p_std,
                "current_task_size": self.cfg.task_size,
                "executed_quant": self.cfg.task_size,
                "remaining_quant": self.cfg.task_size,
                "step_counter": 30,  # TODO: find way to make this dependent on episode length
                "remaining_ratio": 1,
            }
        # print("obs:", obs)


        if normalize:
            obs = self.normalize_obs(obs, means, stds)
            # jax.debug.print('normalized obs:\n {}', obs)

        # print("normalized obs:", obs)

        if flatten:
            obs, _ = jax.flatten_util.ravel_pytree(obs) # Important: this can change the order of the values

        return obs




    def _get_obs_full(self, state: ExecEnvState, params:ExecEnvParams) -> chex.Array:
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
            "p_mid": (best_asks+best_bids)//2//self.world_config.tick_size*self.world_config.tick_size, 
            "p_pass2": jnp.where(state.is_sell_task, best_asks+self.world_config.tick_size*self.cfg.n_ticks_in_book, best_bids-self.world_config.tick_size*self.cfg.n_ticks_in_book), # second_passives
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
        obs = jax.tree.map(lambda x, m, s: (x - m) / s, obs, means, stds)
        return obs

    def action_space(
        self) -> spaces.Box:
        """ Action space of the environment. """
        if self.cfg.action_space=="fixed_prices":
            if self.cfg.action_type == 'delta':
                return spaces.Box(-100, 100, (self.cfg.n_actions,), dtype=jnp.int32)
            elif self.cfg.action_type == 'pure':
                return spaces.Box(0, 100, (self.cfg.n_actions,), dtype=jnp.int32)
            else:
                raise ValueError("Invalid action_type specified.")
        elif self.cfg.action_space=="fixed_quants":
            return spaces.Discrete(self.cfg.n_actions)
        elif self.cfg.action_space=="fixed_quants_1msg":
            return spaces.Discrete(self.cfg.n_actions)
        elif self.cfg.action_space=="fixed_quants_complex":
            return spaces.Discrete(self.cfg.n_actions)
        elif self.cfg.action_space=="simplest_case":
            return spaces.Discrete(self.cfg.n_actions)
        elif self.cfg.action_space=="twap":
            return spaces.Discrete(self.cfg.n_actions)
        else:    
            raise ValueError("Invalid action_space specified.")

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self):
        """Observation space of the environment."""
        if self.cfg.observation_space == "basic":
            return spaces.Box(low=-10000, high=10000, shape=(3,), dtype=jnp.float32)
        elif self.cfg.observation_space == "engineered":
            if self.world_config.ep_type == "fixed_time":
                space = spaces.Box(-10000, 10000, (15,), dtype=jnp.float32) 
            elif self.world_config.ep_type == "fixed_steps":
                space = spaces.Box(-10000, 10000, (12,), dtype=jnp.float32) 
            return space
        elif self.cfg.observation_space == "simplest_case":
            space = spaces.Box(-10000, 10000, (3,), dtype=jnp.float32) 
            return space
        else:
            raise ValueError("Invalid observation_space specified.")

    def state_space(self, params: ExecEnvParams) -> spaces.Dict:
        """State space of the environment."""
        return NotImplementedError


    
    

# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #


if __name__ == "__main__":
    
    print("This main script aims only to test the functionality of the ExecutionAgent class.\n" \
    " Since introduction of JAXMARL as a framework, we need to run these agent classes using the world class, but this can be configured to test just a single class.")
    
    enable_vmap=False
    enable_single_env=True

    print(f"VMAP enabled: {enable_vmap} \n Single environment enabled: {enable_single_env}")

    # Add a sleep step to simulate latency or processing delay
    time.sleep(1)

    from gymnax_exchange.jaxen.marl_env import MARLEnv
    from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig

    multi_agent_config = MultiAgentConfig(list_of_agents_configs=[
                                Execution_EnvironmentConfig(action_space="simplest_case",
                                                            observation_space="simplest_case",
                                                            reward_space="simplest_case"
                                                            )],
                                        number_of_agents_per_type=[1],)

    rng = jax.random.PRNGKey(30) # TODO i think this should be changed to the new key function in JAX .key()
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Instantiate the MARL environment.
    env = MARLEnv(
        key=key_reset,
        multi_agent_config=multi_agent_config,
    )
    # Get the default combined parameters.
    print("starting default parameters")
    env_params = env.default_params


    print(f"The configuration for {Execution_EnvironmentConfig.__name__} is:")
    for attr, value in vars(multi_agent_config.list_of_agents_configs[0]).items():
        print(f"    {attr}: {value}")
    print(f"The configuration for {World_EnvironmentConfig.__name__} is:")
    for attr, value in vars(multi_agent_config.world_config).items():
        print(f"    {attr}: {value}")
    time.sleep(1)



    start=time.time()
    obs, state = env.reset(key_reset, env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
    print("observations_per_type post reset:")
    for i, obs in enumerate(obs):
        print(f"    Agent type {env.instance_list[i].__class__.__name__}: {obs}")
    

    num_steps = 30
    fixed_actions = False

    for i in range(1, num_steps+1):
        print("=" * 40)
        
        print(f"Step {i}")
        # if i > 3 and i < 5:    
        #     jax.profiler.start_trace("tensorboard_logs")


        key_step, _ = jax.random.split(key_step, 2)

        
        # Get random actions from each agent's action space.
        actions_per_type = []
        key, *subkeys = jax.random.split(key_step, len(multi_agent_config.list_of_agents_configs) + 1)
        subkeys = jnp.array(subkeys)
        for i, (space, num_agents) in enumerate(zip(env.action_spaces, multi_agent_config.number_of_agents_per_type)):
            # Split keys for this agent type
            keys = jax.random.split(subkeys[i], num_agents)
            # Sample actions for all agents of this type
            actions = jax.vmap(space.sample)(keys)
            actions_per_type.append(actions)



        if fixed_actions:
            actions_per_type = [jnp.array([3]),jnp.array([1])]
            #print("actions_per_type fixed: ", actions_per_type)

        print("actions_per_type:")
        for i, actions in enumerate(actions_per_type):
            print(f"    Agent type {env.instance_list[i].__class__.__name__}: {actions}")

        obs, state, rewards, done, info = env.step(key=key_step, state=state, actions=actions_per_type, params=env_params)
        
        print("observations_per_type:")
        for i, obs in enumerate(obs):
            print(f"    Agent type {env.instance_list[i].__class__.__name__}: {obs}")
        print("Rewards:")
        for i, r in enumerate(rewards):
            print(f"    Agent type {env.instance_list[i].__class__.__name__}: {r}")

        
        #DEBUG PRINTS
        #print("obs main function: ", obs)
        
        #print(f"Actions: {actions}")
        #print("Step rewards:", rewards)
        #print("Step info:", info)
        #print("Market Maker Raw Action:", action_mm.tolist())
        #print("Execution Raw Action:", action_exe.tolist())
        #print("Done:", done)
        if done["__all__"]:
            print("Episode finished!")
            break
    # jax.profiler.stop_trace()

    
    # Set number of environments to batch




    # # ####### Testing the vmap abilities ########
    
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

            #actions = vmap_sample_action(keys_action)
            fixed_action=0
            actions = jnp.full((num_envs,), fixed_action)

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
