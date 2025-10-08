"""
Base Environment with variable start time for episodes. 

University of Oxford
Corresponding Author: 
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
Kang Li     (kang.li@keble.ox.ac.uk)
Peer Nagy   (peer.nagy@reuben.ox.ac.uk)
V1.0

Module Description
This module offers an advanced simulation environment for limit order books 
 using JAX for high-performance computations. It is designed for reinforcement
 learning applications in financial markets.

Key Components
EnvState:   Dataclass to manage the state of the environment, 
            including order book states, trade records, and timing information.
EnvParams:  Configuration class for environment parameters, 
            including message data, book data, and episode timing.
BaseLOBEnv: Main environment class inheriting from Gymnax's base environment, 
            providing methods for environment initialization, 
            stepping through time steps, and resetting the environment. 

Functionality Overview
__init__:           Sets up initial values and paths. Loads data from 
                    LOBSTER and pre-calculates all initial states for 
                    reset.
default_params:     Returns the default environment parameters, 
                    including the preprocessed message and book data.
step_env:           Advances the environment by one step. It processes both the
                    action messages and data messages through the order book, 
                    updates the state, and determines the reward 
                    and termination condition.
reset_env:          Resets the environment to an initial state. 
                    It selects a new data window, initializes the order book, 
                    and sets the initial state.
is_terminal:        Checks whether the current state is terminal, 
                    based on the elapsed time since the episode's start.
get_obs:            Returns the current observation from environment's state.
name:               Provides the name of the environment.
num_actions:        Returns the number of possible actions in the environment.
action_space:       Defines the action space of the environment, including 
                    sides, quantities, and prices of actions.
observation_space:  (Not implemented) Intended to define 
                    the observation space of the environment.
state_space:        Defines the state space of the environment, 
                    including bids, asks, trades, and time.
_get_data_messages: Fetches an array of messages for a given step 
                    within a data window.
"""

# from jax import config
# config.update("jax_enable_x64",True)
import sys
from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional,Union
import chex
from flax import struct
import itertools
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER_resample
#from gymnax_exchange.jaxlobster.gen_loader import GenLoader
from gymnax_exchange.utils.utils import *
import pickle
from jax.experimental import checkify
import os

#Config File:
from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig
from gymnax_exchange.jaxen.StatesandParams import LoadedEnvParams, LoadedEnvState, WorldState



class BaseLOBEnv(environment.Environment):

    """The basic RL environment for the limit order book (LOB) using
    JAX-LOB functions for manipulating the orderbook. 
    Inherits from gymnax base environment. 
    Attributes
    ----------
    window_selector : int
        -1 to randomly choose start times from all available.
        int in range(0,n_starts) to choose a specific window for debug. 
    data_type : str
        "fixed_steps" and "fixed_time" to defn episode end crit.
    episode_time : int
        Length of episode in steps or seconds based on above.
    n_data_msg_per_step : int
        number of messages to process per step. 
    day_start : int
        Beginning time of day in seconds
    day_end : int
        End time of day in seconds
    nOrdersPerSide : int
        Maximum capacity of orders for JAXLOB
    nTradesLogged : int
        Maximum number of trades logged (in a step)
    book_depth : int
        Depth considered for LOBSTER data retrieval
    n_actions : int
        Number of actions. Dimension of act space
    n_ticks_in_book : int
        Depth of passive order in act space in ticks.
    customIDCounter : int
        Ensures unique IDs for orders submitted by agent.
    trader_unique_id : int
        Offset of unique ID that can be used.
    tick_size : int
        Price tick size. Lobster counts in hundreths of cents.
    start_resolution: int
        Interval, in seconds, at which episodes may start.
    loader : LoadLOBSTER 
        Object that deals with data-loading.
    max_messages_in_episode_arr : jnp.Array 
        Total messages for each possible window.
    messages : jnp.Array  
        Loaded message data.
    books : jnp.Array  
        Loaded book data for start-points
    n_windows : int 
        Number of start points
    start_indeces : jnp.Array  
        Ineces for start points
    end_indeces : jnp.Array  
        Indeces for ep end for each start-point. 
    init_states_array : jnp.Array  
        Initial state for each start point: for reset func. 

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self,cfg:World_EnvironmentConfig, key):
        super().__init__()
        self.window_selector = cfg.window_selector
        self.ep_type = cfg.ep_type # fixed_steps, fixed_time
        self.episode_time = cfg.episode_time # counted by seconds, 1800s=0.5h or steps
        self.n_data_msg_per_step = cfg.n_data_msg_per_step
        self.day_start = cfg.day_start  # 09:30
        self.day_end = cfg.day_end  # 16:00
        self.nOrdersPerSide=cfg.nOrdersPerSide #100
        self.nTradesLogged=cfg.nTradesLogged
        self.book_depth=cfg.book_depth
        self.n_ticks_in_book = cfg.n_ticks_in_book 
        self.customIDCounter=cfg.customIDCounter
        self.tick_size=cfg.tick_size
        self.start_resolution = cfg.start_resolution  # Use value from config
        self.cfg = cfg

        loader=LoadLOBSTER_resample(self.cfg.dataPath,
                                    self.cfg.alphatradePath,
                                self.book_depth,
                                self.ep_type,
                                window_length=self.episode_time,
                                n_data_msg_per_step=self.n_data_msg_per_step,
                                window_resolution=self.start_resolution,
                                day_start=self.day_start,
                                day_end=self.day_end,
                                stock=self.cfg.stock,
                                time_period=self.cfg.timePeriod) 
        msgs,starts,ends,books,max_messages_arr=loader.run_loading()


        self.max_messages_in_episode_arr = max_messages_arr
        self.messages=msgs #Is different to trad. base: all msgs concat. TODO this should not be saved here
        self.books=books
        self.n_windows = starts.shape[0]
        self.start_indeces=starts
        self.end_indeces=ends
        self._init_states(key,self.cfg.alphatradePath,self.start_indeces)

    
    @property
    def default_params(self) -> LoadedEnvParams:
        # Default environment parameters
        return LoadedEnvParams(
            message_data=jnp.asarray(self.messages), 
            book_data=jnp.asarray(self.books),
            init_states_array=self.init_states_array
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: LoadedEnvState, action: Dict, params: LoadedEnvParams
    ) -> Tuple[chex.Array, LoadedEnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        data_messages=self._get_data_messages(params.message_data,
                                              state.start_index,
                                              state.step_counter,
                                              state.init_time[0]+self.cfg.episode_time)
        #data_messages=self._get_generative_messages(params.message_data,n_messages)
        
        #Note: Action of the base environment should consistently be "DO NOTHING"

        total_messages=data_messages

        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]
        #Process messages of step (action+data) through the orderbook
        
        ordersides=job.scan_through_entire_array(self.cfg,key,total_messages,(state.ask_raw_orders,state.bid_raw_orders,state.trades))

        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter)
        state = LoadedEnvState(ordersides[0],ordersides[1],ordersides[2],state.init_time,\
            state.window_index,state.step_counter+1,state.max_steps_in_episode,state.start_index)
        done = self._internal_terminal_debug(state,params,time)
        reward=0
        #jax.debug.print("Final state after step: \n {}", state)
        return self._get_obs(state,params),state,reward,done,{"info":0}

    def reset_env(
        self, key: chex.PRNGKey, params: LoadedEnvParams
    ) -> Tuple[int,LoadedEnvState]:
        """Reset environment state by sampling initial position in OB."""
        idx_data_window = jnp.where(
            self.cfg.window_selector == -1,
            jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()),  
            jnp.array(self.cfg.window_selector, dtype=jnp.int32))
        #jax.debug.print("idx_data_window: {}", idx_data_window)
        first_state = index_tree(params.init_states_array, idx_data_window)
        # def debug_callback(first_state,idx_data_window):
        #     if idx_data_window == 427:  # Debugging for specific window index
        #         print("Debugging reset for window index:", idx_data_window)
        #         print("Resetting environment to initial state for window index:", first_state.window_index)
        #         print("First state details:", first_state)
        # jax.debug.callback(debug_callback, first_state, idx_data_window)
        return 0,first_state
    
    def _internal_terminal_debug(self, state: LoadedEnvState, params: LoadedEnvParams,time : chex.Array) -> bool:
        return (time-state.init_time)[0]>=self.cfg.episode_time


    # def is_terminal(self, state: LoadedEnvState, params: LoadedEnvParams) -> bool:
    #     """Check whether state is terminal."""
    #     #jax.debug.print("Time: {} , Init time: {}, Difference: {}",state.time, state.init_time,(state.time-state.init_time)[0])
    #     return (state.time-state.init_time)[0]>=self.cfg.episode_time

    def _get_state_from_data(self,key,first_message,book_data,max_steps_in_episode,window_index,start_index)->LoadedEnvState:
        time=jnp.array(first_message[-2:])
        #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
        def get_initial_orders(book_data,time):
            orderbookLevels=10
            initid=self.cfg.init_id
            #jax.debug.print("\n=== Debug Order Book Initialization ===")
            #jax.debug.print("Raw book_data shape: {}", book_data.shape)
            #jax.debug.print("Raw book_data: {}", book_data)
            data=jnp.array(book_data).reshape(int(10*2),2)
            #jax.debug.print("\nReshaped data: {}", data)
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
            #jax.debug.print("\nFinal initOB array:")
            #jax.debug.print("Shape: {}", initOB.shape)
            #jax.debug.print("Content: {}", initOB)
            #jax.debug.print("Side assignments (column 1): {}", initOB[:,1])
            #jax.debug.print("Price assignments (column 3): {}", initOB[:,3])
            #jax.debug.print("Quantity assignments (column 2): {}", initOB[:,2])
            return initOB
        init_orders=get_initial_orders(book_data,time)
        #jax.debug.print("init_orders {}",init_orders)
        #Initialise both sides of the book as being empty
        asks_raw=job.init_orderside(self.cfg.nOrders)
        bids_raw=job.init_orderside(self.cfg.nOrders)
        trades_init=(jnp.ones((self.cfg.nTrades,8))*-1).astype(jnp.int32)
        #Process the initial messages through the orderbook
        ordersides=job.scan_through_entire_array(self.cfg,key,init_orders,(asks_raw,bids_raw,trades_init))
        
 
        
        return LoadedEnvState(ask_raw_orders=ordersides[0],
                        bid_raw_orders=ordersides[1],
                        trades=ordersides[2],
                        init_time=jnp.array([(window_index*self.start_resolution) 
                                                        %(self.day_end-self.day_start-self.episode_time+self.start_resolution)
                                                        +self.day_start,0])
                                    if self.ep_type=="fixed_time" else time,
                        window_index=window_index,
                        max_steps_in_episode=max_steps_in_episode,
                        start_index=start_index,
                        step_counter=0,
)

    def _init_states(self,key,alphatradePath,starts):
        print("START:  pre-reset in the initialization")
        os.makedirs(alphatradePath + '/pre_reset_states/', exist_ok=True)
        pkl_file_name = (alphatradePath + '/pre_reset_states/'
                         + 'ResetState_window_resolution_' + str(self.cfg.start_resolution)
                         + '_eptype_"' + str(self.cfg.ep_type)
                         + '"_depth_' + str(self.cfg.book_depth)
                         + "_stock_" + str(self.cfg.stock)
                         + "_windowidx_"+str(self.cfg.window_selector)
                         + "_nMsgPerStep_"+str(self.cfg.n_data_msg_per_step)
                         + "_episode_time_"+str(self.cfg.episode_time)
                         + "_TimePeriod_"+str(self.cfg.timePeriod)
                         + '.pkl')
        print("pre-reset will be saved to ", pkl_file_name)
        try:
            if self.cfg.use_pickles_for_init:
                with open(pkl_file_name, 'rb') as f:
                    self.init_states_array = pickle.load(f)
                    print("LOADING STATES FROM PKL...")
            else:
                raise ValueError("Throw error so re-computes")
        except:
            print("COMPUTING INIT STATES...")
            #for i in range(self.n_windows):
                #print("message starts",self.messages[starts[i]])
            get_state_jitted= jax.jit(self._get_state_from_data)

            states = [get_state_jitted(key,
                                                self.messages[starts[i]],
                                                self.books[i],
                                                self.max_messages_in_episode_arr[i]
                                                    //self.n_data_msg_per_step+1,
                                                    i,
                                                    starts[i]) 
                        for i in range(self.n_windows)]
            self.init_states_array=tree_stack(states)
            print("SAVING STATES TO PKL...")
            with open(pkl_file_name, 'wb') as f:
                pickle.dump(self.init_states_array, f)
        print("DONE: pre-reset in the initialization")

    def _get_obs(self, state: LoadedEnvState, params:LoadedEnvParams) -> chex.Array:
        """Return dummy observation."""
        return 0
    
    def _get_data_messages(self,messageData,start,step_counter,end_time_s):
        """Returns an array of messages for a given step. 
            Parameters:
                    messageData (Array): 2D array of all msgs with
                                        dimensions: messages, features.
                    start (int): Index of first message to in episode
                    step_counter (int): desired step to consider
                    end_time_s (int): End time of ep in seconds
            Returns:
                    Messages (Array): 2D array of messages for step 
        """
        index_offset=start+self.n_data_msg_per_step*step_counter
        
        messages=jax.lax.dynamic_slice_in_dim(messageData,index_offset,self.n_data_msg_per_step,axis=0)
        #jax.debug.print("{}",messages)
        #jax.debug.print("End time: {}",end_time_s)
        #messages=messageData[index_offset:(index_offset+self.n_data_msg_per_step),:]
        #Replace messages after the cutoff time with padded 0s (except time)
        #jax.debug.print("m_wout_time {}",jnp.transpose(jnp.resize(messages[:,-2]>=end_time_s,messages[:,:-2].shape[::-1])))
        if self.cfg.ep_type == "fixed_time":
            # If fixed time, we need to remove messages that are after the end time
            # We do this by replacing the messages with 0s if they are after the end time
            m_wout_time = jnp.where(jnp.transpose(jnp.resize(
                                messages[:,-2]>=end_time_s,
                                messages[:,:-2].shape[::-1])),
                              jnp.zeros_like(messages[:,:-2]),
                              messages[:,:-2])
        #jax.debug.print("m_wout_time {}",m_wout_time)

            messages=jnp.concatenate((m_wout_time,messages[:,-2:]),axis=1,dtype=jnp.int32)
        return messages
    
    def _get_generative_messages(self,previous_messages,n_messages):
        """Draft Generative Loader:
        Inputs:
        
        Outputs:
        
        
        """
        loader = GenLoader(
        model=GenLoader.dummy_model,
        initial_context=previous_messages,
        initial_ob_state=jnp.zeros((2, 10)),
        n_messages=100
        )
        messages, _ = loader.generate_step()
        

        return messages
    
    def _get_pass_price_quant(self, state):
            """Get price and quanitity n_ticks into books"""
            bid_passive_2=state.best_bids[-1, 0] - self.tick_size * self.cfg.n_ticks_in_book
            ask_passive_2=state.best_asks[-1, 0] + self.tick_size * self.cfg.n_ticks_in_book
            quant_bid_passive_2 = job.get_volume_at_price(state.bid_raw_orders, bid_passive_2)
            quant_ask_passive_2 = job.get_volume_at_price(state.ask_raw_orders, ask_passive_2)
            return bid_passive_2,quant_bid_passive_2,ask_passive_2,quant_ask_passive_2
        


    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeBase-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def action_space(
        self, params: Optional[LoadedEnvParams] = None
    ) -> Union[spaces.Discrete,spaces.Dict]:
        """Action space of the environment."""
        return spaces.Dict(
            {
                "sides":spaces.Box(0,2,(self.n_actions,),dtype=jnp.int32),
                "quantities":spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32),
                "prices":spaces.Box(0,99999999,(self.n_actions,),dtype=jnp.int32)
            }
        )

    #TODO: define obs space (4xnDepth) array of quants&prices. Not that important right now. 
    def observation_space(self, params: LoadedEnvParams):
        """Observation space of the environment."""
        return NotImplementedError

    def state_space(self, params: LoadedEnvParams) -> spaces.Dict:
        """State space of the environment. #FIXME Samples absolute
          nonsense, don't use.
        """
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,999999999,shape=(6,self.cfg.nOrders),dtype=jnp.int32),
                "asks": spaces.Box(-1,999999999,shape=(6,self.cfg.nOrders),dtype=jnp.int32),
                "trades": spaces.Box(-1,999999999,shape=(8,self.cfg.nTrades),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
    


    
if __name__ == "__main__":
    world_config = World_EnvironmentConfig()

    rng = jax.random.PRNGKey(0)
    rng,key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)


    env= BaseLOBEnv(cfg=world_config,
                    key=key_init)
    
    jax.profiler.start_trace("/tmp/profile-data")
    env_params=env.default_params

    obs,state=env.reset(key_reset,env_params)
    # done=False

    # while not done :
    obs,state,rewards,done,info=env.step_env(key_step,state,{},env_params)
    rng,key_step = jax.random.split(rng, 2)
    obs,state,rewards,done,info=env.step_env(key_step,state,{},env_params)

    rng,key_reset = jax.random.split(rng, 2)
    obs,state=env.reset(key_reset,env_params)

    #     print(done)

    jax.block_until_ready(state)
    jax.profiler.stop_trace()



    # print(state)
    # print(obs)
