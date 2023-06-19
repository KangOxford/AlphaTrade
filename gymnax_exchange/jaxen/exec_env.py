# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import time

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

#Code snippet to disable all jitting.
from jax import config
# config.update("jax_disable_jit", False)
config.update("jax_disable_jit", True)
# ============== testing scripts ===============



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
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv


@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index:int
    step_counter: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    max_steps_in_episode: int = 100
    messages_per_step: int=1
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    




class ExecutionEnv(BaseLOBEnv):
    def __init__(self,alphatradePath,task):
        super().__init__(alphatradePath)
        self.n_actions = 4 # [A, M, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.task = task
        self.n_fragment_max=2
        self.n_ticks_in_book=20
        assert task in ['buy','sell'], "\n{'='*20}\nCannot handle this task[{task}], must be chosen from ['buy','sell'].\n{'='*20}\n"

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(self.messages,self.books)


    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #jax.debug.print("Data Messages to process \n: {}",data_messages)

        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        types=jnp.ones((self.n_actions,),jnp.int32)
        # jax.debug.breakpoint()
        sides=jnp.zeros((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        quants=action #from action space
        
        #Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        def get_prices(state,task):
            # best_ask, best_bid = job.get_best_bid_and_ask(state.ask_raw_orders[-1],state.bid_raw_orders[-1]) # doesnt work
            best_ask, best_bid = job.get_best_bid_and_ask(state.ask_raw_orders,state.bid_raw_orders)
            A = best_bid if task=='sell' else best_ask # aggressive would be at bids
            M = (best_bid + best_ask)//2//self.tick_size*self.tick_size 
            P = best_ask if task=='sell' else best_bid
            PP= best_ask+self.tick_size*self.n_ticks_in_book if task=='sell' else best_bid-self.tick_size*self.n_ticks_in_book
            return (A,M,P,PP)

        prices=jnp.asarray(get_prices(state,self.task),jnp.int32)
        jax.debug.print("Prices: {}",prices)
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)

        #jax.debug.print("Input to cancel function: {}",state.bid_raw_orders[-1])
        # cnl_msgs=job.getCancelMsgs(state.ask_raw_orders[-1] if self.task=='sell' else state.bid_raw_orders[-1],-8999,self.n_fragment_max*self.n_actions,-1 if self.task=='sell' else 1)
        #jax.debug.print("Output from cancel function: {}",cnl_msgs)

        #Add to the top of the data messages 
        total_messages=jnp.concatenate([action_msgs,data_messages],axis=0)
        # total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0)
        jax.debug.print("Total messages: \n {}",total_messages)

        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]

        #Process messages of step (action+data) through the orderbook
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        # ordersides=job.scan_through_entire_array_save_states(total_messages,(state.ask_raw_orders[-1,:,:],state.bid_raw_orders[-1,:,:],state.trades),self.stepLines) # doesnt work: reset state and step state doesnt match
        ordersides=job.scan_through_entire_array_save_states(total_messages,(state.ask_raw_orders,state.bid_raw_orders,state.trades),self.stepLines)
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter)
        state = EnvState(*ordersides,state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,state.step_counter+1)
        
        # ========= self.get_obs(state,params) =============
        # get_best_bids = lambda x: x[np.argmax(np.max(x, axis=0), axis=0), 0]
        get_best_bids = lambda x: jnp.max(x[:, :, 0], axis=1)
        best_bids = get_best_bids(state.bid_raw_orders)
        get_best_asks = lambda x: jnp.min(jnp.where(x[:, :, 0] >= 0, x[:, :, 0], np.inf), axis=1).astype(jnp.int32)
        best_asks = get_best_asks(state.ask_raw_orders)
        # --------------------------------------------------
        mid_prices = (best_asks + best_bids)//2//self.tick_size*self.tick_size 
        second_passives = best_asks+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bids-self.tick_size*self.n_ticks_in_book
        # --------------------------------------------------
        
        # ========= self.get_obs(state,params) =============
        jax.debug.breakpoint()
        
        done = self.is_terminal(state,params)
        reward=self.get_reward(state, params)
        #jax.debug.print("Final state after step: \n {}", state)
        jax.debug.breakpoint()
        return self.get_obs(state,params),state,reward,done,{"info":0}



    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        
        """Reset environment state by sampling initial position in OB."""
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=())

        #Get the init time based on the first message to be processed in the first step. 
        time=job.get_initial_time(params.message_data,idx_data_window) 
        #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
        init_orders=job.get_initial_orders(params.book_data,idx_data_window,time)
        #Initialise both sides of the book as being empty
        asks_raw=job.init_orderside(self.nOrdersPerSide)
        bids_raw=job.init_orderside(self.nOrdersPerSide)
        trades_init=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process the initial messages through the orderbook
        ordersides=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw,trades_init))

        #Craft the first state
        state = EnvState(jnp.resize(ordersides[0],(self.stepLines,self.nOrdersPerSide,6)),jnp.resize(ordersides[1],(self.stepLines,self.nOrdersPerSide,6)),ordersides[2],time,time,0,idx_data_window,0)
        state = EnvState(*ordersides,time,time,0,idx_data_window,0)
        # jax.debug.breakpoint()
        
        # self.p_0
        self.p_0 = 10000000

        return self.get_obs(state,params),state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (state.time-state.init_time)[0]>params.episode_time
    
    def get_reward(self, state: EnvState, params: EnvParams) -> float:
        return 0.0

    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        return job.get_L2_state(self.book_depth,state.ask_raw_orders,state.bid_raw_orders)

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32)

    #TODO: define obs space (4xnDepth) array of quants&prices. Not that important right now. 
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return NotImplementedError

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


if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        ATFolder = '/homes/80/kang/AlphaTrade'


    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)


    # start=time.time()
    # obs,state=env.reset(key_reset,env_params)
    # print("State after reset: \n",state)
    # print("Time for 2nd reset: \n",time.time()-start)
    # print(env_params.message_data.shape, env_params.book_data.shape)

    #print(job.get_data_messages(env_params.message_data,state.window_index,state.step_counter+1))

    #print(env.action_space().sample(key_policy))
    #print(env.state_space(env_params).sample(key_policy))


    """test_action={"sides":jnp.array([1,1,1]),
                 "quantities":jnp.array([10,10,10]),
                 "prices":jnp.array([2154900,2154000,2153900]),
                 }"""
    
    test_action=env.action_space().sample(key_policy)
    print("Sampled actions are: ",test_action)

    start=time.time()
    obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    print("State after one step: \n",state,done)
    print("Time for one step: \n",time.time()-start)

    # test_action=env.action_space().sample(key_policy)
    # print("Sampled actions are: \n",test_action)
    
    # start=time.time()
    # obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    # print("State after 2 steps: \n",state,done)
    # print("Time for 2nd step: \n",time.time()-start)
    #comment


    


    ####### Testing the vmap abilities ########
    
    enable_vmap=False
    if enable_vmap:
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
