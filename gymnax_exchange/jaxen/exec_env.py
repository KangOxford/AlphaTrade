"""
Execution Environment for Limit Order Book

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
is_terminal:        Checks if the current state meets the terminal condition, 
                    specific to execution tasks, such as completion of the 
                    execution order or time constraints.
get_obs:            Constructs and returns the current observation for the 
                    execution environment, derived from the state.
name, num_actions:  Inherited methods providing the name of the environment 
                    and the number of possible actions.
action_space:       Defines the action space for execution tasks, including 
                    order types and quantities.
observation_space:  Define the observation space for execution tasks.
state_space:        Describes the state space of the environment, tailored 
                    for execution tasks with components 
                    like bids, asks, and trades.
reset_env:          Resets the environment to a specific state for execution. 
                    It selects a new data window, initializes the order book, 
                    and sets the initial state for execution tasks.
is_terminal:        Checks whether the current state is terminal, based on 
                    the number of steps executed or tasks completed.
getActionMsgs:      Generates action messages based on 
                    the current state and action. 
                    It determines the type, side, quantity, 
                    and price of orders to be executed.
                    including detailed order book information and trade history
hamilton_apportionment_permuted_jax: A utility function using JAX, 
                                     implementing a Hamilton apportionment 
                                     method with randomized seat allocation.
_get_initial_time:  Inherited method to retrieve the 
                    initial time of a data window.
_get_data_messages: Inherited method to fetch market messages for a given 
                    step within a data window.
"""

# NOTE line length maximum for comments is 79, 
#      and for codes is 99, 
#      as we do functional computation and nested functions
#      more spaces is in need,
#      according to the PEP8, 99 is also allowed.
#      I would suggest to follow the code style of purejaxrl,
#      and in his blog he mentioned to be inspired by cleanrl
#      and have all things of a functionality inside one file.
#      In the meanwhile, keep one single file not too long.


# ================= imports ==================
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
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
import dataclasses
# import utils

@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index:int
    init_price:int
    task_to_execute:int
    quant_executed:int
    step_counter: int
    max_steps_in_episode: int
    
    total_revenue:float
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float


@struct.dataclass
class EnvParams:
    is_sell_task: int
    message_data: chex.Array
    book_data: chex.Array
    # stateArray_list: chex.Array
    episode_time: int = 60*10  # 60*10, 10 mins
    # max_steps_in_episode: int = 100 # TODO should be a variable, decied by the data_window
    # messages_per_step: int=1 # TODO never used, should be removed?
    time_per_step: int = 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    


class ExecutionEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, task, window_index, action_type, 
            task_size = 500, rewardLambda=0.0, data_type="fixed_steps"
        ):
        super().__init__(alphatradePath, data_type)
        #self.n_actions = 2 # [A, MASKED, P, MASKED] Agressive, MidPrice, Passive, Second Passive
        # self.n_actions = 2 # [MASKED, MASKED, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.n_actions = 4 # [FT, M, NT, PP] Agressive, MidPrice, Passive, Second Passive
        self.task = task # "random", "buy", "sell"
        # self.randomize_direction = randomize_direction
        self.window_index = window_index
        self.action_type = action_type
        self.data_type = data_type # fixed_steps, fixed_time
        self.rewardLambda = rewardLambda
        # self.task_size = 5000 # num to sell or buy for the task
        # self.task_size = 2000 # num to sell or buy for the task
        self.task_size = task_size # num to sell or buy for the task
        # self.task_size = 200 # num to sell or buy for the task
        self.n_fragment_max = 2
        self.n_ticks_in_book = 2 #TODO: Used to be 20, too large for stocks with dense LOBs
        # self.debug : bool = False
        
        # ==================================================================
        # =================    MOVE THE PRE_RESET HERE     =================
        # ================= CAUTION NOT BELONG TO BASE ENV =================
        # ================= EPECIALLY SUPPORT FOR EXEC ENV =================
        print("START:  pre-reset in the initialization")
        pkl_file_name_state = alphatradePath+'/state_arrays_'+alphatradePath.split("/")[-2]+'.pkl'
        pkl_file_name_obs = alphatradePath+'/obs_arrays_'+alphatradePath.split("/")[-2]+'.pkl'
        print("pre-reset will be saved to ",pkl_file_name_state)
        print("pre-reset will be saved to ",pkl_file_name_obs)
        try:
            # import pickle
            # # Restore the list
            # with open(pkl_file_name_state, 'rb') as f:
            #     self.stateArray_list = pickle.load(f)
            # with open(pkl_file_name_obs, 'rb') as f:
            #     self.obs_list = pickle.load(f)
            # print("LOAD FROM PKL")
            raise NotImplementedError
        except:
            print("DO COMPUTATION")
            def get_state(message_data, book_data,max_steps_in_episode):
                time=jnp.array(message_data[0,0,-2:])
                #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
                def get_initial_orders(book_data,time):
                    orderbookLevels=10
                    initid=job.INITID
                    data=jnp.array(book_data).reshape(int(10*2),2)
                    newarr = jnp.zeros((int(orderbookLevels*2),8),dtype=jnp.int32)
                    initOB = newarr \
                        .at[:,3].set(data[:,0]) \
                        .at[:,2].set(data[:,1]) \
                        .at[:,0].set(1) \
                        .at[0:orderbookLevels*4:2,1].set(-1) \
                        .at[1:orderbookLevels*4:2,1].set(1) \
                        .at[:,4].set(initid) \
                        .at[:,5].set(initid-jnp.arange(0,orderbookLevels*2)) \
                        .at[:,6].set(time[0]) \
                        .at[:,7].set(time[1])
                    return initOB
                init_orders=get_initial_orders(book_data,time)
                #Initialise both sides of the book as being empty
                asks_raw=job.init_orderside(self.nOrdersPerSide)
                bids_raw=job.init_orderside(self.nOrdersPerSide)
                trades_init=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
                #Process the initial messages through the orderbook
                ordersides=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw,trades_init))
                # Mid Price after init added to env state as the initial price --> Do not at to self as this applies to all environments.
                best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(ordersides[0],ordersides[1])
                M = (best_bid[0] + best_ask[0])//2//self.tick_size*self.tick_size 
                state = (ordersides[0],ordersides[1],ordersides[2],
                         jnp.resize(best_ask,(self.stepLines,2)),
                         jnp.resize(best_bid,(self.stepLines,2)),
                         time,time,0,-1,M,self.task_size,0,0,0,max_steps_in_episode,0,0,0,0,0)
                return state
            states_ = [get_state(self.messages[i], self.books[i], self.max_steps_in_episode_arr[i])
                      for i in range(len(self.max_steps_in_episode_arr))]
            states = [EnvState(*[*state[:-6], *[jnp.zeros(1)]*5])
                      for state in states_]
            # jax.debug.breakpoint()
            self.obs_list = jnp.array([self.get_obs(state, self.default_params) 
                            for state in states])
            def state2stateArray(state):
                state_5 = jnp.hstack((state[5],state[6],state[9],state[15]))
                padded_state = jnp.pad(state_5, (0, 100 - state_5.shape[0]), constant_values=-1)[:,jnp.newaxis]
                stateArray = jnp.hstack((state[0],state[1],state[2],state[3],state[4],padded_state))
                return stateArray
            self.stateArray_list = jnp.array([state2stateArray(state) for state in states_])
            import pickle
            # Save the list
            with open(pkl_file_name_state, 'wb') as f:
                pickle.dump(self.stateArray_list, f) 
            with open(pkl_file_name_obs, 'wb') as f:
                pickle.dump(self.obs_list, f) 
        print("FINISH: pre-reset in the initialization")

        #TODO Most of the state space should be exactly the same for the base and exec env, 
        # can we think about keeping the base part seperate from the exec part? 
        # ================= CAUTION NOT BELONG TO BASE ENV =================
        # ================= EPECIALLY SUPPORT FOR EXEC ENV =================
        # =================    MOVE THE PRE_RESET HERE     =================
        # ==================================================================


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        is_sell_task = 0 if self.task == 'buy' else 1 # {("random","sell"): sell_task, ("buy",): buy_task}
        return EnvParams(is_sell_task, self.messages,self.books)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, input_action: jax.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        # '''
        # action = jnp.array([delta,0,0,0],dtype=jnp.int32)
        def reshape_action(action : Dict, state: EnvState, params : EnvParams):
            if self.action_type == 'delta':
                def twapV3(state, env_params):
                    # ---------- ifMarketOrder BGN ----------
                    # ·········· ifMarketOrder determined by time ··········
                    # remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                    # marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                    # ifMarketOrder = (remainingTime <= marketOrderTime)
                    # ·········· ifMarketOrder determined by steps ··········
                    remainingSteps = state.max_steps_in_episode - state.step_counter 
                    marketOrderSteps = jnp.array(1, dtype=jnp.int32) 
                    ifMarketOrder = (remainingSteps <= marketOrderSteps)
                    # ---------- ifMarketOrder END ----------
                    # ---------- quants ----------
                    remainedQuant = state.task_to_execute - state.quant_executed
                    remainedStep = state.max_steps_in_episode - state.step_counter
                    stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                    limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True) if self.n_actions == 2 \
                        else jax.random.permutation(key, jnp.array([stepQuant-3*stepQuant//4,stepQuant//4,stepQuant//4,stepQuant//4]), independent=True)
                    market_quants = jnp.array([stepQuant,stepQuant]) if self.n_actions == 2 else jnp.array([stepQuant,stepQuant,stepQuant,stepQuant])
                    quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                    # ---------- quants ----------
                    # jax.debug.breakpoint()
                    return jnp.array(quants) 
                action_space_clipping = lambda action, task_size: jnp.round(action).astype(jnp.int32).clip(-1*task_size//100,task_size//100) 
                action_ = twapV3(state, params) + action_space_clipping(input_action, state.task_to_execute)
            else:
                action_space_clipping = lambda action, task_size: jnp.round(action).astype(jnp.int32).clip(0,task_size//5)# clippedAction, CAUTION not clipped by task_size, but task_size//5
                action_ = action_space_clipping(input_action, state.task_to_execute)
            
            def truncate_action(action, remainQuant):
                action = jnp.round(action).astype(jnp.int32).clip(0,remainQuant)
                # NOTE: didn't know this was already implemented? <= posted in channel 2 days(8th Nov) before your commit(10th Nov)
                # NOTE: already add comments and variable names to make it readable
                # NOTE: same thing with Peer's commit, and the logic from our discussion proposed by Sascha
                #       but hamilton_apportionment_permuted_jax add permutation if two poistions shares the same probability
                #       such as [2,1,0,1] and remain 2, in the utils.clip_by_sum_int 
                #       implementation, the second position will always be added by 1
                #       it will turns out to be [3,2,0,1]
                #       but in reality we should permutate between the second and fourth position
                #       it will turns out to be [3,2,0,1] or [3,1,0,2]
                #       the name hamilton_apportionment means our interger split problem, a classical math problem
                #       permuted in the name means randomly choose a position if two or more positions 
                #       shares the same probability.
                #       jax in the name means it is jittable
                # TODO  delete this comment after read
                scaledAction = jnp.where(action.sum() <= remainQuant, action, self.hamilton_apportionment_permuted_jax(action, remainQuant, key)) 
                return scaledAction
            action = truncate_action(action_, state.task_to_execute - state.quant_executed)
            return action.astype(jnp.int32)
        action = reshape_action(input_action, state, params)        
        data_messages = self._get_data_messages(params.message_data,state.window_index,state.step_counter)
        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        
        action_msgs = self.getActionMsgs(action, state, params)
        #Currently just naive cancellation of all agent orders in the book. #TODO avoid being sent to the back of the queue every time. 

        raw_orders = jax.lax.cond(
            params.is_sell_task,
            lambda: state.ask_raw_orders,
            lambda: state.bid_raw_orders
        )
        cnl_msgs = job.getCancelMsgs(
            raw_orders,
            job.INITID + 1,
            self.n_actions,
            1 - params.is_sell_task * 2 
        )
        
        # net actions and cancellations at same price if new action is not bigger than cancellation
        action_msgs, cnl_msgs = self.filter_messages(action_msgs, cnl_msgs)
        
        #Add to the top of the data messages
        total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0) # TODO DO NOT FORGET TO ENABLE CANCEL MSG
        #Save time of final message to add to state
        time=total_messages[-1, -2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process messages of step (action+data) through the orderbook
        (asks, bids, trades), (bestasks, bestbids) = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        ) 
        
        # ========== get reward and revenue ==========
        # Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = ((job.INITID < executed[:, 2]) & (executed[:, 2] < 0)) | ((job.INITID < executed[:, 3]) & (executed[:, 3] < 0))
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)

        agentQuant = agentTrades[:,1].sum() # new_execution quants
        # ---------- used for vwap, revenue ----------
        vwapFunc = lambda executed: jnp.nan_to_num((executed[:,0]//self.tick_size* executed[:,1]).sum()/(executed[:,1]).sum(),0.0) # caution: this value can be zero (executed[:,1]).sum()
        vwap = vwapFunc(executed) # average_price of all the tradings, from the varaible executed
        revenue = (agentTrades[:,0]//self.tick_size * agentTrades[:,1]).sum()
        # ---------- used for slippage, price_drift, and RM(rolling mean) ----------
        rollingMeanValueFunc_FLOAT = lambda average_price,new_price:(average_price*state.step_counter+new_price)/(state.step_counter+1)
        vwap_rm = rollingMeanValueFunc_FLOAT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)
        price_adv_rm = rollingMeanValueFunc_FLOAT(state.price_adv_rm,revenue/agentQuant - vwap) # slippage=revenue/agentQuant-vwap, where revenue/agentQuant means agentPrice 
        slippage_rm = rollingMeanValueFunc_FLOAT(state.slippage_rm,revenue - state.init_price//self.tick_size*agentQuant)
        price_drift_rm = rollingMeanValueFunc_FLOAT(state.price_drift_rm,(vwap - state.init_price//self.tick_size)) #price_drift = (vwap - state.init_price//self.tick_size)
        # ---------- used for advantage and drift ----------
        advantage = revenue - vwap * agentQuant # advantage_vwap
        drift = agentQuant * (vwap - state.init_price//self.tick_size)
        # drift = agentQuant * (vwap_rm - state.init_price//self.tick_size)
        # ---------- compute the final reward ----------
        rewardValue = revenue 
        # rewardValue =  advantage
        # rewardValue1 = advantage + self.rewardLambda * drift
        # rewardValue1 = advantage + 1.0 * drift
        # rewardValue2 = revenue - (state.init_price // self.tick_size) * agentQuant
        # rewardValue = rewardValue1 - rewardValue2
        # rewardValue = revenue - vwap_rm * agentQuant # advantage_vwap_rm
        reward = jnp.sign(agentQuant) * rewardValue # if no value agentTrades then the reward is set to be zero
        jax.debug.print("reward {}", reward)
        # ---------- normalize the reward ----------
        reward /= 10000
        # reward /= params.avg_twap_list[state.window_index]
        # ========== get reward and revenue END ==========
            
        
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        def bestPricesImpute(bestprices, lastBestPrice):
            def replace_values(prev, curr):
                last_non_999999999_values = jnp.where(curr != 999999999, curr, prev) #non_999999999_mask
                replaced_curr = jnp.where(curr == 999999999, last_non_999999999_values, curr)
                return last_non_999999999_values, replaced_curr
            def forward_fill_999999999_int(arr):
                last_non_999999999_values, replaced = jax.lax.scan(replace_values, arr[0], arr[1:])
                return jnp.concatenate([arr[:1], replaced])
            def forward_fill(arr):
                index = jnp.argmax(arr[:, 0] != 999999999)
                return forward_fill_999999999_int(arr.at[0, 0].set(jnp.where(index == 0, arr[0, 0], arr[index][0])))
            back_fill = lambda arr: jnp.flip(forward_fill(jnp.flip(arr, axis=0)), axis=0)
            mean_forward_back_fill = lambda arr: (forward_fill(arr)+back_fill(arr))//2
            return jnp.where(
                (bestprices[:,0] == 999999999).all(),
                jnp.tile(
                    jnp.array([lastBestPrice, 0]),
                    (bestprices.shape[0], 1)
                ),
                mean_forward_back_fill(bestprices)
            )

        bestasks = bestPricesImpute(bestasks[-self.stepLines:], state.best_asks[-1,0])
        bestbids = bestPricesImpute(bestbids[-self.stepLines:], state.best_bids[-1,0])
        state = EnvState(
            asks, bids, trades, bestasks, bestbids,
            state.init_time, time, state.customIDcounter + self.n_actions, state.window_index,
            state.init_price, state.task_to_execute, state.quant_executed + agentQuant,
            state.step_counter + 1,
            state.max_steps_in_episode,
            state.total_revenue + revenue,
            slippage_rm, price_adv_rm, price_drift_rm, vwap_rm)
            # state.max_steps_in_episode,state.twap_total_revenue+twapRevenue,state.twap_quant_arr)

        done = self.is_terminal(state, params)
        return self.get_obs(state, params), state, reward, done, {
            "window_index": state.window_index,
            "total_revenue": state.total_revenue,
            "quant_executed": state.quant_executed,
            "task_to_execute": state.task_to_execute,
            "average_price": jnp.nan_to_num(state.total_revenue / state.quant_executed, 0.0),
            "current_step": state.step_counter,
            'done': done,
            'slippage_rm': state.slippage_rm,
            "price_adv_rm": state.price_adv_rm,
            "price_drift_rm": state.price_drift_rm,
            "vwap_rm": state.vwap_rm,
            "advantage_reward": advantage,
        }
    
    def filter_messages(
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




    def reset_env(
        self, key : chex.PRNGKey, params: EnvParams, reset_window_index = -999
        ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        # all windows can be reached

        window_index = jnp.where(reset_window_index == -999, self.window_index, reset_window_index)
        '''if -999 use default static index [self.window_index], else use provided dynamic index [reset_window_index]'''
        

        idx_data_window = jnp.where(
            window_index == -1,
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()),  
            jnp.array(window_index, dtype=jnp.int32)
        )

        def stateArray2state(stateArray):
            state0 = stateArray[:,0:6]
            state1 = stateArray[:,6:12]
            state2 = stateArray[:,12:18]
            state3 = stateArray[:,18:20]
            state4 = stateArray[:,20:22]
            state5 = stateArray[0:2,22:23].squeeze(axis=-1)
            state6 = stateArray[2:4,22:23].squeeze(axis=-1)
            state9= stateArray[4:5,22:23][0].squeeze(axis=-1)
            return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,
                    self.task_size,0,0,
                    self.max_steps_in_episode_arr[idx_data_window],
                    jnp.array(0.0,dtype=jnp.float32),
                    jnp.array(0.0,dtype=jnp.float32), jnp.array(0.0,dtype=jnp.float32), 
                    jnp.array(0.0,dtype=jnp.float32), jnp.array(0.0,dtype=jnp.float32))
        
        
        stateArray = self.stateArray_list[idx_data_window]
        # stateArray = params.stateArray_list[idx_data_window]
        state_ = stateArray2state(stateArray)
        state = EnvState(*state_)
        
        key_, key = jax.random.split(key)
        if self.task == 'random':
            direction = jax.random.randint(key_, minval=0, maxval=2, shape=())
            params = dataclasses.replace(params, is_sell_task=direction)
        
        obs = self.get_obs(state, params)
        return obs,state
    

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (
            # (params.episode_time - (state.time - state.init_time)[0] <= 0) 
            (state.max_steps_in_episode - state.step_counter <= 0)
            |  (state.task_to_execute - state.quant_executed <= 0)
        )
    
    def getActionMsgs(self, action: jax.Array, state: EnvState, params: EnvParams):
        # def normal_order_logic(action: jnp.ndarray):
        #     quants = action.astype(jnp.int32) # from action space
        #     return quants

        # def market_order_logic(state: EnvState):
        #     quant = state.task_to_execute - state.quant_executed
        #     quants = jnp.asarray((quant, 0, 0, 0), jnp.int32) 
        #     return quants
        
        def normal_quant_price(price_levels: jax.Array, state: EnvState, action: jax.Array):
            def combine_mid_nt(quants, prices):
                quants = quants \
                    .at[2].set(quants[2] + quants[1]) \
                    .at[1].set(0)
                prices = prices.at[1].set(-1)
                return quants, prices

            quants = action.astype(jnp.int32)
            prices = jnp.array(price_levels[:-1])
            # if mid_price == near_touch_price: combine orders into one
            return jax.lax.cond(
                price_levels[1] != price_levels[2],
                lambda q, p: (q, p),
                combine_mid_nt,
                quants, prices
            )
        
        def market_quant_price(price_levels: jax.Array, state: EnvState, action: jax.Array):
            mkt_quant = state.task_to_execute - state.quant_executed
            quants = jnp.asarray((mkt_quant, 0, 0, 0), jnp.int32) 
            return quants, jnp.asarray((price_levels[-1], -1, -1, -1), jnp.int32)
        
        def buy_task_prices(best_ask, best_bid):
            NT = best_bid
            # mid defaults to one tick more passive if between ticks
            M = ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size
            FT = best_ask
            PP = best_bid - self.tick_size*self.n_ticks_in_book
            MKT = job.MAX_INT
            return NT, M, FT, PP, MKT

        def sell_task_prices(best_ask, best_bid):
            NT = best_ask
            # mid defaults to one tick more passive if between ticks
            M = (jnp.ceil((best_bid + best_ask) / 2 / self.tick_size)
                 * self.tick_size).astype(jnp.int32)
            FT = best_bid
            PP = best_ask + self.tick_size*self.n_ticks_in_book
            MKT = 0
            return NT, M, FT, PP, MKT

        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types = jnp.ones((self.n_actions,), jnp.int32)
        # sides=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        sides = (1 - params.is_sell_task*2) * jnp.ones((self.n_actions,), jnp.int32)
        trader_ids = jnp.ones((self.n_actions,), jnp.int32) * self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids = (jnp.ones((self.n_actions,), jnp.int32) *
                    (self.trader_unique_id + state.customIDcounter)) \
                    + jnp.arange(0, self.n_actions) #Each message has a unique ID
        times = jnp.resize(
            state.time + params.time_delay_obs_act,
            (self.n_actions, 2)
        ) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        best_ask, best_bid = state.best_asks[-1, 0], state.best_bids[-1, 0]
        # jax.debug.print("ask - bid {}", best_ask - best_bid)

        NT, M, FT, PP, MKT = jax.lax.cond(
            params.is_sell_task,
            sell_task_prices,
            buy_task_prices,
            best_bid, best_ask
        )
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        marketOrderTime = jnp.array(1, dtype=jnp.int32) 

        price_levels = (FT, M, NT, PP, MKT)
        quants, prices = jax.lax.cond(
            (remainingTime <= marketOrderTime),
            market_quant_price,
            normal_quant_price,
            price_levels, state, action
        )
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs = jnp.stack([types, sides, quants, prices, trader_ids, order_ids], axis=1)
        action_msgs = jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================

    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        best_asks, best_bids=state.best_asks[:,0], state.best_bids[:,0]
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
        obs = self.normalize_obs(obs)
        # jax.debug.print("obs {}", obs)
        obs, _ = jax.flatten_util.ravel_pytree(obs)
        # jax.debug.breakpoint()
        return obs

    def normalize_obs(self, obs: Dict[str, jax.Array]):
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

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """ Action space of the environment. """
        # return spaces.Box(-100,100,(self.n_actions,),dtype=jnp.int32) if self.action_type=='delta' else spaces.Box(0,500,(self.n_actions,),dtype=jnp.int32)
        if self.action_type == 'delta':
            return spaces.Box(-5,5,(self.n_actions,),dtype=jnp.int32)
            # return spaces.Box(-100, 100, (self.n_actions,), dtype=jnp.int32)
        else:
            return spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32)
            # return spaces.Box(0, self.task_size, (self.n_actions,), dtype=jnp.int32)
          

    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        space = spaces.Box(-10,10,(809,),dtype=jnp.float32) 
        # space = spaces.Box(-10,10,(15,),dtype=jnp.float32) 
        return space

    #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions
    
    def hamilton_apportionment_permuted_jax(self, votes, seats, key):
        """
        Compute the Hamilton apportionment method with permutation using JAX.

        Args:
            votes (jax.Array): Array of votes for each party/entity.
            seats (int): Total number of seats to be apportioned.
            key (chex.PRNGKey): JAX key for random number generation.

        Returns:
            jax.Array: Array of allocated seats to each party/entity.
        """
        std_divisor = jnp.sum(votes) / seats # Calculate the standard divisor.
        # Initial allocation of seats based on the standard divisor and compute remainders.
        init_seats, remainders = jnp.divmod(votes, std_divisor)
        # Compute the number of remaining seats to be allocated.
        remaining_seats = jnp.array(seats - init_seats.sum(), dtype=jnp.int32) 
        # Define the scanning function for iterative seat allocation.
        def allocate_remaining_seats(carry,x): # only iterate 4 times, as remaining_seats in {0,1,2,3}
            key,init_seats,remainders = carry
            key, subkey = jax.random.split(key)
            # Create a probability distribution based on the maximum remainder.
            distribution = (remainders == remainders.max())/(remainders == remainders.max()).sum()
            # Randomly choose a party/entity to allocate a seat based on the distribution.
            chosen_index = jax.random.choice(subkey, remainders.size, p=distribution)
            # Update the initial seats and remainders for the chosen party/entity.
            updated_init_seats = init_seats.at[chosen_index].add(jnp.where(x < remaining_seats, 1, 0))
            updated_remainders = remainders.at[chosen_index].set(0)
            return (key, updated_init_seats, updated_remainders), x
            # Iterate over parties/entities to allocate the remaining seats.
        (key, init_seats, remainders), _ = jax.lax.scan(
                                                        allocate_remaining_seats,
                                                        (key, init_seats, remainders), 
                                                        xs=jnp.arange(votes.shape[0])
                                                        )
        return init_seats

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
        # ATFolder = '/home/duser/AlphaTrade'
        # ATFolder = '/homes/80/kang/AlphaTrade'
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell", # "random", # "buy",
        "TASK_SIZE": 100, # 500,
        "WINDOW_INDEX": -1,
        "ACTION_TYPE": "delta", # "pure",
        "REWARD_LAMBDA": 1.0,
        "DTAT_TYPE":"fixed_time",
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=ExecutionEnv(ATFolder,"sell",1)
    env= ExecutionEnv(
      config["ATFOLDER"],
      config["TASKSIDE"],
      config["WINDOW_INDEX"],
      config["ACTION_TYPE"],
      config["TASK_SIZE"],
      config["REWARD_LAMBDA"],
      config["DTAT_TYPE"]
    )
    env_params=env.default_params
    # print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("Time for reset: \n",time.time()-start)
    # print("State after reset: \n",state)
   

    # print(env_params.message_data.shape, env_params.book_data.shape)
    for i in range(1,100000):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        print("-"*20)
        key_policy, _ =  jax.random.split(key_policy, 2)
        key_step, _ =  jax.random.split(key_step, 2)
        # test_action=env.action_space().sample(key_policy)
        test_action=env.action_space().sample(key_policy)
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
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
