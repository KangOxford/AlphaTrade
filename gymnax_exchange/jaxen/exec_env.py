# from jax import config
# config.update("jax_enable_x64",True)
# ============== testing scripts ===============
import os
import sys
import time 
import timeit
import random
# sys.path.append('/Users/sasrey/AlphaTrade')
# sys.path.append('/homes/80/kang/AlphaTrade')
sys.path.append('.')
import jax
import jax.numpy as jnp


import gymnax
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
import chex

import faulthandler
faulthandler.enable()

# from jax import config
# config.update('jax_platform_name', 'cpu')
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())
chex.assert_gpu_available(backend=None)

# #Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) # use this during training
# config.update("jax_disable_jit", True)
# ============== testing scripts ===============
jax.numpy.set_printoptions(linewidth=183)


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
    total_revenue:float
    step_counter: int
    max_steps_in_episode: int
    
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    stateArray_list: chex.Array
    episode_time: int = 60*10  # 60*10, 10 mins
    # max_steps_in_episode: int = 100 # TODO should be a variable, decied by the data_window
    # messages_per_step: int=1 # TODO never used, should be removed?
    time_per_step: int = 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    


class ExecutionEnv(BaseLOBEnv):
    def __init__(
            self, alphatradePath, task, window_index, action_type,
            task_size = 500, rewardLambda=0.0, Gamma=0.00
        ):
        super().__init__(alphatradePath)
        #self.n_actions = 2 # [A, MASKED, P, MASKED] Agressive, MidPrice, Passive, Second Passive
        # self.n_actions = 2 # [MASKED, MASKED, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.n_actions = 4 # [FT, M, NT, PP] Agressive, MidPrice, Passive, Second Passive
        self.task = task
        # self.randomize_direction = randomize_direction
        self.window_index = window_index
        self.action_type = action_type
        self.rewardLambda = rewardLambda
        self.Gamma = Gamma
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
        pkl_file_name = alphatradePath+'state_arrays_'+alphatradePath.split("/")[-2]+'.pkl'
        print("pre-reset will be saved to ",pkl_file_name)
        try:
            import pickle
            # Restore the list
            with open(pkl_file_name, 'rb') as f:
                self.stateArray_list = pickle.load(f)
            print("LOAD FROM PKL")
        except:
            print("DO COMPUTATION")
            def get_state(message_data, book_data,max_steps_in_episode):
                time=jnp.array(message_data[0,0,-2:])
                #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
                def get_initial_orders(book_data,time):
                    orderbookLevels=10
                    initid=-9000
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
                state = (ordersides[0],ordersides[1],ordersides[2],jnp.resize(best_ask,(self.stepLines,2)),jnp.resize(best_bid,(self.stepLines,2)),\
                    time,time,0,-1,M,self.task_size,0,0,0,0,max_steps_in_episode)
                return state
            states = [get_state(self.messages[i], self.books[i], self.max_steps_in_episode_arr[i]) for i in range(len(self.max_steps_in_episode_arr))]
            
            def state2stateArray(state):
                state_5 = jnp.hstack((state[5],state[6],state[9],state[15]))
                padded_state = jnp.pad(state_5, (0, 100 - state_5.shape[0]), constant_values=-1)[:,jnp.newaxis]
                stateArray = jnp.hstack((state[0],state[1],state[2],state[3],state[4],padded_state))
                return stateArray
            self.stateArray_list = jnp.array([state2stateArray(state) for state in states])
            import pickle
            # Save the list
            with open(pkl_file_name, 'wb') as f:
                pickle.dump(self.stateArray_list, f) 
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
        # return EnvParams(self.messages,self.books)
        return EnvParams(self.messages,self.books,self.stateArray_list)
        # return EnvParams(0 if self.task =='buy' else 1 if self.task=='sell' else -1, self.messages,self.books,self.stateArray_list,self.obs_sell_list,self.obs_buy_list)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, delta: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        # '''
        
        action = jnp.array([delta,0,0,0],dtype=jnp.int32)
        jax.debug.print("action {}",action)
        # TODO remains bugs in action and it wasn't caused by merging
        
        
        data_messages = job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        
        action_msgs = self.getActionMsgs(action, state, params)
        #Currently just naive cancellation of all agent orders in the book. #TODO avoid being sent to the back of the queue every time. 
        cnl_msgs=job.getCancelMsgs(state.ask_raw_orders if self.task=='sell' else state.bid_raw_orders,-8999,self.n_actions,-1 if self.task=='sell' else 1)
        #Add to the top of the data messages
        total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0) # TODO DO NOT FORGET TO ENABLE CANCEL MSG
        #Save time of final message to add to state
        # time=total_messages[-1:][0][-2:]
        time=total_messages[-1, -2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process messages of step (action+data) through the orderbook

        # jax.debug.breakpoint()
        asks, bids, trades, bestasks, bestbids = job.scan_through_entire_array_save_bidask(
            total_messages,
            (state.ask_raw_orders, state.bid_raw_orders, trades_reinit),
            self.stepLines
        ) 

        
        # ========== get reward and revenue ==========
        # Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)
        # Mask to keep only the trades where the RL agent is involved, apply mask.
        mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
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
        drift = agentQuant * (vwap_rm - state.init_price//self.tick_size)
        # ---------- compute the final reward ----------
        # rewardValue = revenue 
        rewardValue =  advantage
        # rewardValue = advantage + self.rewardLambda * drift
        # rewardValue = revenue - (state.init_price // self.tick_size) * agentQuant
        # rewardValue = revenue - vwap_rm * agentQuant # advantage_vwap_rm
        reward = jnp.sign(agentQuant) * rewardValue # if no value agentTrades then the reward is set to be zero
        # ---------- normalize the reward ----------
        reward /= 10000
        # reward /= params.avg_twap_list[state.window_index]
        # ========== get reward and revenue END ==========
        
        
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        def bestPircesImpute(bestprices,lastBestPrice):
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
            return jnp.where((bestprices[:,0] == 999999999).all(),jnp.tile(jnp.array([lastBestPrice, 0]), (bestprices.shape[0],1)),mean_forward_back_fill(bestprices))
        bestasks, bestbids = bestPircesImpute(bestasks[-self.stepLines:],state.best_asks[-1,0]),bestPircesImpute(bestbids[-self.stepLines:],state.best_bids[-1,0])
        state = EnvState(
            asks, bids, trades, bestasks, bestbids,
            state.init_time, time, state.customIDcounter + self.n_actions, state.window_index,
            state.init_price, state.task_to_execute, state.quant_executed + agentQuant,
            state.total_revenue + revenue, state.step_counter + 1,
            state.max_steps_in_episode,
            slippage_rm, price_adv_rm, price_drift_rm, vwap_rm)
            # state.max_steps_in_episode,state.twap_total_revenue+twapRevenue,state.twap_quant_arr)
        # jax.debug.breakpoint()

        done = self.is_terminal(state, params)
        # jax.debug.print("window_index {}, current_step {}, quant_executed {}, average_price {}", state.window_index, state.step_counter, state.quant_executed, state.total_revenue / state.quant_executed)
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
    

    def reset_env(
        self, key : chex.PRNGKey, params: EnvParams, reset_window_index = -999
        ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        # all windows can be reached

        window_index = jnp.where(reset_window_index == -999, self.window_index, reset_window_index)
        '''if -999 use default static index, else use provided dynamic index'''
        

        idx_data_window = jnp.where(
            window_index == -1,
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()),  
            jnp.array(window_index, dtype=jnp.int32)
        )

        
        def stateArray2state(stateArray):
            state0 = stateArray[:,0:6];state1 = stateArray[:,6:12];state2 = stateArray[:,12:18];state3 = stateArray[:,18:20];state4 = stateArray[:,20:22]
            state5 = stateArray[0:2,22:23].squeeze(axis=-1);state6 = stateArray[2:4,22:23].squeeze(axis=-1);state9= stateArray[4:5,22:23][0].squeeze(axis=-1)
            return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,jnp.array(0.0,dtype=jnp.float32),0,self.max_steps_in_episode_arr[idx_data_window],jnp.array(0.0,dtype=jnp.float32), jnp.array(0.0,dtype=jnp.float32), jnp.array(0.0,dtype=jnp.float32), jnp.array(0.0,dtype=jnp.float32))
            # return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window],0.0, 0.0, 0.0, 0.0)
            # return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window])
            # return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window],0,twap_quant_arr)
        stateArray = params.stateArray_list[idx_data_window]
        state_ = stateArray2state(stateArray)
        state = EnvState(*state_)
        obs = self.get_obs(state, params)
        return obs,state
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (
            (state.task_to_execute - state.quant_executed <= 0) | (state.max_steps_in_episode - state.step_counter<= 0)
        )
    
    def getActionMsgs(self, action: Dict, state: EnvState, params: EnvParams):
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
        # NT, FT, PP, MKT = jax.lax.cond(
        #     params.is_buy_task,
        #     lambda: (best_bid, best_ask, best_bid - self.tick_size*self.n_ticks_in_book, job.MAX_INT),
        #     lambda: (best_ask, best_bid, best_ask + self.tick_size*self.n_ticks_in_book, 0)
        # )
        FT = best_bid if self.task=='sell' else best_ask # aggressive: far touch
        M = ((best_bid + best_ask) // 2 // self.tick_size) * self.tick_size # Mid price
        NT = best_ask if self.task=='sell' else best_bid #Near touch: passive
        PP = best_ask+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bid-self.tick_size*self.n_ticks_in_book #Passive, N ticks deep in book
        MKT = 0 if self.task=='sell' else job.MAX_INT
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        # ---------- ifMarketOrder BGN ----------
        # ·········· ifMarketOrder determined by time ··········
        # remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        # marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
        # ifMarketOrder = (remainingTime <= marketOrderTime)
        # ·········· ifMarketOrder determined by steps ··········
        remainingSteps = state.max_steps_in_episode - state.step_counter 
        marketOrderSteps = jnp.array(5, dtype=jnp.int32) # in steps, means the last minute was left for market order
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
        best_asks, best_bids=state.best_asks[:,0], state.best_bids[:,0]
        best_ask_qtys, best_bid_qtys = state.best_asks[:,1], state.best_bids[:,1]
        
        obs = {
            # "is_buy_task": params.is_buy_task,
            "p_aggr": best_bids if self.task=='sell' else best_asks,
            "q_aggr": best_bid_qtys if self.task=='sell' else best_ask_qtys, 
            "p_pass": best_asks if self.task=='sell' else best_bids,
            "q_pass": best_ask_qtys if self.task=='sell' else best_bid_qtys, 
            "p_mid": (best_asks+best_bids)//2//self.tick_size*self.tick_size, 
            "p_pass2": best_asks+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bids-self.tick_size*self.n_ticks_in_book, # second_passives
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
                # "is_buy_task": 0,
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
                # "is_buy_task": 1,
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


        obs = normalize_obs(obs)
        # jax.debug.print("obs {}", obs)
        obs, _ = jax.flatten_util.ravel_pytree(obs)
        # jax.debug.breakpoint()
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""

        return spaces.Box(-5,5,(self.n_actions,),dtype=jnp.int32) if self.action_type=='delta' \
          else spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32)

    
    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        space = spaces.Box(-10,10,(809,),dtype=jnp.float32) 
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
        ATFolder = "/homes/80/kang/AlphaTrade/testing_oneDay/"
        # ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
        # ATFolder = "/homes/80/kang/AlphaTrade/testing"
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell",
        "TASK_SIZE": 100, # 500,
        "WINDOW_INDEX": -1,
        "ACTION_TYPE": "delta", # "pure",
        "REWARD_LAMBDA": 1.0,
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=ExecutionEnv(ATFolder,"sell",1)
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["WINDOW_INDEX"],config["ACTION_TYPE"],config["TASK_SIZE"],config["REWARD_LAMBDA"])
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
        # test_action=env.action_space().sample(key_policy)//10 # CAUTION not real action
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
    
    enable_vmap=True
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
