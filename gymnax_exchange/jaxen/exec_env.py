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
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, flatten_util
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
import pickle
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
    # Potentially could be moved to base,
    # so long as saving of best ask/bids is base behaviour. 
    best_asks: chex.Array
    best_bids: chex.Array
    # Execution specific stuff
    init_price:int
    task_to_execute:int
    quant_executed:int
    # Execution specific rewards. 
    total_revenue:float
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float
    is_sell_task: int = 1



@struct.dataclass
class EnvParams:
    task_size: int 
    reward_lambda: float = 1.0
    


class ExecutionEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, task, window_index, action_type,
            max_task_size = 500, rewardLambda=0.0,ep_type="fixed_time"):
        #Define Execution-specific attributes.
        self.task = task # "random", "buy", "sell"
        self.n_actions=4 #(FT, M, NT, PP)
        self.n_ticks_in_book = 2 # Depth of PP actions
        self.action_type = action_type # 'delta' or 'pure'
        self.max_task_size = max_task_size
        self.rewardLambda = rewardLambda
        #Call base-class init function
        super().__init__(alphatradePath,window_index,ep_type)
        

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        base_params=super().default_params
        flat_tree=jtu.tree_flatten(base_params)[0]
        base_vals=flat_tree[0:4] #Considers the base parameter values other than init state.
        state_vals=flat_tree[4:] #Considers the state values
        return EnvParams(*base_vals,EnvState(*state_vals),self.max_task_size,episode_time=60*10)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, input_action: jax.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        data_messages = self._get_data_messages(params.message_data,
                                                state.start_index,
                                                state.step_counter,
                                                state.init_time[0]+params.episode_time)
        
        action = self._reshape_action(action, state, params,key)
        action_msgs = self._getActionMsgs(action, state, params)
       
        raw_order_side = jax.lax.cond(state.is_sell_task,
                                      lambda: state.ask_raw_orders,
                                      lambda: state.bid_raw_orders)
        #TODO avoid being sent to the back of the queue every time. 
        cnl_msgs = job.getCancelMsgs(raw_order_side,
                                     job.INITID + 1,
                                     self.n_actions,
                                     1 - state.is_sell_task * 2)
        
        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self.filter_messages(action_msgs, cnl_msgs)
        
        #Add to the top of the data messages
        total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0)
        #Save time of final message to add to state
        time=total_messages[-1, -2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines) 
        
        #Error correction on the best prices.
        (bestasks,
        bestbids) = (self._best_prices_impute(bestasks[-self.stepLines:],
                                                       state.best_asks[-1,0]),
                              self._best_prices_impute(bestbids[-self.stepLines:],
                                                       state.best_bids[-1,0]))

        reward,extras=self._get_reward(state,trades)
        state = EnvState(ask_raw_orders=asks,
                         bid_raw_orders=bids,
                         trades=trades,
                         init_time=state.init_time,
                         time=time,
                         customIDcounter=state.customIDcounter+self.n_actions,
                         window_index=state.window_index,
                         step_counter=state.step_counter+1,
                         max_steps_in_episode=state.max_steps_in_episode,
                         start_index=state.start_index,
                         best_asks=bestasks,
                         best_bids=bestbids,
                         init_price=state.init_price,
                         task_to_execute=state.task_to_execute,
                         quant_executed=state.quant_executed+extras["agentQuant"],
                         total_revenue=state.total_revenue+extras["revenue"],
                         slippage_rm=extras["slippage_rm"],
                         price_adv_rm=extras["price_adv_rm"],
                         price_drift_rm=extras["price_drift_rm"],
                         vwap_rm=extras["vwap_rm"])
        done = self.is_terminal(state, params)
        info={"window_index": state.window_index,
                "total_revenue": state.total_revenue,
                "quant_executed": state.quant_executed,
                "task_to_execute": state.task_to_execute,
                "average_price": jnp.nan_to_num(state.total_revenue 
                                                / state.quant_executed, 0.0),
                "current_step": state.step_counter,
                'done': done,
                'slippage_rm': state.slippage_rm,
                "price_adv_rm": state.price_adv_rm,
                "price_drift_rm": state.price_drift_rm,
                "vwap_rm": state.vwap_rm,
                "advantage_reward": extras["advantage"],}
        return self._get_obs(state, params), state, reward, done,info
    

    def reset_env(self,
                  key : chex.PRNGKey,
                  params: EnvParams) -> Tuple[chex.Array, EnvState]:
        key_, key = jax.random.split(key)

        obs,state=super().reset_env(key,params)
        if self.task == 'random':
            direction = jax.random.randint(key_, minval=0, maxval=2, shape=())
            state = dataclasses.replace(state, is_sell_task=direction)
        return obs,state
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (
            # (params.episode_time - (state.time - state.init_time)[0] <= 0) 
            (state.max_steps_in_episode - state.step_counter <= 0)
            |  (state.task_to_execute - state.quant_executed <= 0)
        )
    
    def _get_state_from_data(self,first_message,book_data,max_steps_in_episode,window_index,start_index):
        #(self,message_data,book_data,max_steps_in_episode)
        base_state=super()._get_state_from_data(first_message,book_data,max_steps_in_episode,window_index,start_index)
        base_vals=jtu.tree_flatten(base_state)[0]
        best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(base_state.ask_raw_orders,base_state.bid_raw_orders)
        M = (best_bid[0] + best_ask[0])//2//self.tick_size*self.tick_size 
        is_sell_task = 0 if self.task == 'buy' else 1 # if self.task == 'random', set defualt as 0

        return EnvState(*base_vals,
                        best_asks=jnp.resize(best_ask,(self.stepLines,2)),
                        best_bids=jnp.resize(best_bid,(self.stepLines,2)),
                        init_price=M,
                        task_to_execute=self.max_task_size,
                        quant_executed=0,
                        total_revenue=0,
                        slippage_rm=0.,
                        price_adv_rm=0.,
                        price_drift_rm=0.,
                        vwap_rm=0.,
                        is_sell_task=is_sell_task)

    def _get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
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
            "p_mid":p_mean,
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


        obs = normalize_obs(obs)
        # jax.debug.print("obs {}", obs)
        obs, _ = jax.flatten_util.ravel_pytree(obs)
        # jax.debug.breakpoint()
        return obs
    
    def _getActionMsgs(self, action: Dict, state: EnvState, params: EnvParams):
        def normal_order_logic(action: jnp.ndarray):
            quants = action.astype(jnp.int32) # from action space
            return quants

        def market_order_logic(state: EnvState):
            quant = state.task_to_execute - state.quant_executed
            quants = jnp.asarray((quant, 0, 0, 0), jnp.int32) 
            return quants

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        # sides = (params.is_buy_task*2 - 1) * jnp.ones((self.n_actions,),jnp.int32)
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
        NT, FT, PP, MKT = jax.lax.cond(
            params.is_sell_task,
            lambda: (best_ask, best_bid, best_ask + self.tick_size*self.n_ticks_in_book, 0),
            lambda: (best_bid, best_ask, best_bid - self.tick_size*self.n_ticks_in_book, job.MAX_INT)
        )
        M = ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size # Mid price
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        # ---------- ifMarketOrder BGN ----------
        # ·········· ifMarketOrder determined by time ··········
        # remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        # marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
        # ifMarketOrder = (remainingTime <= marketOrderTime)
        # ·········· ifMarketOrder determined by steps ··········
        remainingSteps = state.max_steps_in_episode - state.step_counter 
        marketOrderSteps = jnp.array(1, dtype=jnp.int32) # in steps, means the last step was left for market order
        ifMarketOrder = (remainingSteps <= marketOrderSteps)
        # ---------- ifMarketOrder END ----------
        def normal_order_logic(state: EnvState, action: jnp.ndarray):
            quants = action.astype(jnp.int32) # from action space
            prices = jnp.asarray((FT,NT), jnp.int32) if self.n_actions == 2 else jnp.asarray((FT,M,NT,PP), jnp.int32) 
            return quants, prices
        def market_order_logic(state: EnvState):
            quant = state.task_to_execute - state.quant_executed
            quants =  jnp.asarray((quant,0),jnp.int32) if self.n_actions == 2 else jnp.asarray((quant,0,0,0),jnp.int32) 
            prices =  jnp.asarray((MKT,  M),jnp.int32) if self.n_actions == 2 else jnp.asarray((MKT, M,M,M),jnp.int32)
            return quants, prices
        market_quants, market_prices = market_order_logic(state)
        normal_quants, normal_prices = normal_order_logic(state, action)
        quants = jnp.where(ifMarketOrder, market_quants, normal_quants)
        prices = jnp.where(ifMarketOrder, market_prices, normal_prices)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================


    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # Note: uses entire observation history between steps
        # TODO: if we want to use this, we need to roll forward the RNN state with every step

        best_asks, best_bids = state.best_asks[:,0], state.best_bids[:,0]
        best_ask_qtys, best_bid_qtys = state.best_asks[:,1], state.best_bids[:,1]
        obs = {
            "is_sell_task": params.is_sell_task,
            "p_aggr": jnp.where(params.is_sell_task, best_bids, best_asks),
            "q_aggr": jnp.where(params.is_sell_task, best_bid_qtys, best_ask_qtys), 
            "p_pass": jnp.where(params.is_sell_task, best_asks, best_bids),
            "q_pass": jnp.where(params.is_sell_task, best_ask_qtys, best_bid_qtys), 
            "p_mid": (best_asks+best_bids)//2//self.tick_size*self.tick_size, 
            "p_pass2": jnp.where(params.is_sell_task, best_asks+self.tick_size*self.n_ticks_in_book, best_bids-self.tick_size*self.n_ticks_in_book), # second_passives
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

        def normalize_obs(obs: Dict[str, jax.Array]):
            """ normalized observation by substracting 'mean' and dividing by 'std'
                (config values don't need to be actual mean and std)
            """
            # TODO: put this into config somewhere?
            #       also check if we can get rid of manual normalization
            #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
            p_mean = 3.5e7
            p_std = 1e6
            means = {
                "is_sell_task": 0,
                "p_aggr": p_mean,
                "q_aggr": 0,
                "p_pass": p_mean,
                "q_pass": 0,
                "p_mid":p_mean,
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
            obs = jax.tree_map(lambda x, m, s: (x - m) / s, obs, means, stds)
            return obs

    def get_obs(
            self,
            state: EnvState,
            params:EnvParams,
            normalize:bool = True,
            flatten:bool = True,
        ) -> chex.Array:
        """Return observation from raw state trafo."""
        # NOTE: only uses most recent observation from state
        quote_aggr, quote_pass = jax.lax.cond(
            state.is_sell_task,
            lambda: (state.best_bids[-1], state.best_asks[-1]),
            lambda: (state.best_asks[-1], state.best_bids[-1]),
        )
        time = state.time[0] + state.time[1]/1e9
        time_elapsed = time - (state.init_time[0] + state.init_time[1]/1e9)
        obs = {
            "is_sell_task": state.is_sell_task,
            "p_aggr": quote_aggr[0],
            "p_pass": quote_pass[0],
            "spread": jnp.abs(quote_aggr[0] - quote_pass[0]),
            "q_aggr": quote_aggr[1],
            "q_pass": quote_pass[1],
            # TODO: add "q_pass2" as passive quantity to state in step_env and here
            "time": time,
            # "episode_time": state.time - state.init_time,
            "time_remaining": params.episode_time - time_elapsed,
            "init_price": state.init_price,
            "task_size": state.task_to_execute,
            "executed_quant": state.quant_executed,
            "step_counter": state.step_counter,
            "max_steps": state.max_steps_in_episode,
        }
        # TODO: put this into config somewhere?
        #       also check if we can get rid of manual normalization
        #       by e.g. functional transformations or maybe gymnax obs norm wrapper suffices?
        p_mean = 3.5e7
        p_std = 1e6
        means = {
            "is_sell_task": 0,
            "p_aggr": p_mean,
            "p_pass": p_mean,
            "spread": 0,
            "q_aggr": 0,
            "q_pass": 0,
            "time": 0,
            # "episode_time": jnp.array([0, 0]),
            "time_remaining": 0,
            "init_price": p_mean,
            "task_size": 0,
            "executed_quant": 0,
            "step_counter": 0,
            "max_steps": 0,
        }
        stds = {
            "is_sell_task": 1,
            "p_aggr": p_std,
            "p_pass": p_std,
            "spread": 1e4,
            "q_aggr": 100,
            "q_pass": 100,
            "time": 1e5,
            # "episode_time": jnp.array([1e3, 1e9]),
            "time_remaining": 600, # 10 minutes = 600 seconds
            "init_price": p_std,
            "task_size": 500,
            "executed_quant": 500,
            "step_counter": 300,
            "max_steps": 300,
        }
        if normalize:
            obs = self.normalize_obs(obs, means, stds)
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
        space = spaces.Box(-10, 10, (13,), dtype=jnp.float32) 
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
        ATFolder = "./testing_oneDay"
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "buy", # "random", # "buy",
        "MAX_TASK_SIZE": 500, # 500,
        "WINDOW_INDEX": -1,
        "ACTION_TYPE": "pure", # "pure",
        "REWARD_LAMBDA": 1.0,
        "EP_TYPE": "fixed_time",
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)




    

    # env=ExecutionEnv(ATFolder,"sell",1)
    env = ExecutionEnv(
        config["ATFOLDER"],
        config["TASKSIDE"],
        config["WINDOW_INDEX"],
        config["ACTION_TYPE"],
        config["EP_TYPE"],
        config["MAX_TASK_SIZE"],
    )
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset, env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
    print(state)

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        # test_action=env.action_space().sample(key_policy)
        test_action = env.action_space().sample(key_policy) // 10
        print(f"Sampled {i}th actions are: ", test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state, test_action, env_params)
        for key, value in info.items():
            print(key, value)
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

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
