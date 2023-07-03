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
import timeit

import faulthandler

faulthandler.enable()
print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
# config.update("jax_disable_jit", True)
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

import time 

@struct.dataclass
class NEnvState:
    nask_raw_orders: chex.Array
    nbid_raw_orders: chex.Array
    ntrades: chex.Array
    nbest_asks: chex.Array
    nbest_bids: chex.Array
    ninit_time: chex.Array
    ntime: chex.Array
    # ----------------------------
    ncustomIDcounter: chex.Array
    nwindow_index: chex.Array
    nstep_counter: chex.Array
    ninit_price: chex.Array
    ntask_to_execute: chex.Array
    nquant_executed: chex.Array
    

@struct.dataclass
class NEnvParams:
    message_data: chex.Array
    book_data: chex.Array
    num_envs: int = 3
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    max_steps_in_episode: int = 100
    messages_per_step: int=1
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    




class ExecutionEnv(BaseLOBEnv):
    def __init__(self,alphatradePath,task,debug=False):
        super().__init__(alphatradePath)
        self.n_actions = 4 # [A, M, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.task = task
        self.task_size = 200 # num to sell or buy for the task
        self.n_fragment_max=2
        self.n_ticks_in_book=20
        self.debug : bool = False
        # self.vwap = 0.0 # vwap at current step
        # assert task in ['buy','sell'], "\n{'='*20}\nCannot handle this task[{task}], must be chosen from ['buy','sell'].\n{'='*20}\n"

    @property
    def default_params(self) -> NEnvParams:
        # Default environment parameters
        return NEnvParams(self.messages,self.books)
    
    def action_space(
            self, nparams: Optional[NEnvParams] = None
        ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(0,100,(self.n_actions,nparams.num_envs),dtype=jnp.int32)
    
    def getActionMsgs(self, naction: chex.Array, nstate: NEnvState, nparams: NEnvParams):
        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types=jnp.ones((4,nparams.num_envs),jnp.int32)
        # types=jnp.ones((self.n_actions,nparams.num_envs),jnp.int32)
        sides=-1*jnp.ones((4,nparams.num_envs),jnp.int32) if 'sell'=='sell' else jnp.ones((4,nparams.num_envs),jnp.int32) #if self.task=='buy'
        # sides=-1*jnp.ones((self.n_actions,nparams.num_envs),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions,nparams.num_envs),jnp.int32) #if self.task=='buy'
        trader_ids=jnp.ones((4,nparams.num_envs),jnp.int32)*(-8999) #This agent will always have the same (unique) trader ID
        # trader_ids=jnp.ones((self.n_actions,nparams.num_envs),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((4,nparams.num_envs),jnp.int32)*(-8999+nstate.ncustomIDcounter)+jnp.arange(0,4).reshape(-1, 1).repeat(nparams.num_envs, axis=1) #Each message has a unique ID
        # order_ids=jnp.ones((self.n_actions,nparams.num_envs),jnp.int32)*(self.trader_unique_id+nstate.customIDcounter)+jnp.arange(0,self.n_actions).reshape(-1, 1).repeat(nparams.num_envs, axis=1) #Each message has a unique ID
        times=jnp.resize(nstate.ntime+nparams.time_delay_obs_act,(nparams.num_envs,4,2)) #time from last (data) message of prev. step + some delay
        # times=jnp.resize(nstate.time+nparams.time_delay_obs_act,(nparams.num_envs,self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        # jax.debug.breakpoint()
        nbest_ask, nbest_bid = nstate.nbest_asks[-1,:], nstate.nbest_bids[-1,:]
        As = nbest_bid if "sell"=='sell' else nbest_ask # aggressive would be at bids
        # A = nbest_bid if self.task=='sell' else nbest_ask # aggressive would be at bids
        Ms = (nbest_bid + nbest_ask)//2//100*100
        # M = (nbest_bid + nbest_ask)//2//self.tick_size*self.tick_size 
        Ps = nbest_ask if "sell"=='sell' else nbest_bid
        # P = nbest_ask if self.task=='sell' else nbest_bid
        PPs= nbest_ask+100*20 if "sell"=='sell' else nbest_bid-100*20
        # PP= nbest_ask+self.tick_size*self.n_ticks_in_book if self.task=='sell' else nbest_bid-self.tick_size*self.n_ticks_in_book
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        remainingTime = nparams.episode_time - jnp.array((nstate.ntime-nstate.ninit_time)[:,0], dtype=jnp.int32) # CAUTION only consider the s, ignore ns
        marketOrderTime = jnp.array(60, dtype=jnp.int32).repeat(nparams.num_envs) # in seconds, means the last minute was left for market order
        ifMarketOrder = (remainingTime <= marketOrderTime)
        
        def market_order_logic(nstate: NEnvState,  As: jnp.ndarray):
            quant = nstate.ntask_to_execute - nstate.nquant_executed
            price = As + (-1 if "sell" == 'sell' else 1) * (100 * 100) * 100
            # price = As + (-1 if self.task == 'sell' else 1) * (self.tick_size * 100) * 100
            quants = jnp.array((quant//4,quant//4,quant//4,quant-3*quant//4),jnp.int32)
            prices = jnp.array((price, price , price, price),jnp.int32)
            # (self.tick_size * 100) : one dollar
            # (self.tick_size * 100) * 100: choose your own number here(the second 100)
            return quants, prices
        def normal_order_logic(nstate: NEnvState, naction: jnp.ndarray, As: jnp.ndarray, Ms: jnp.ndarray, Ps: jnp.ndarray, PPs: jnp.ndarray):
            quants = naction # from action space
            prices = jnp.vstack((As, Ms, Ps, PPs))
            return quants, prices
        market_quants, market_prices = market_order_logic(nstate, As)
        normal_quants, normal_prices = normal_order_logic(nstate, naction, As, Ms, Ps, PPs)
        quants = jnp.where(ifMarketOrder, market_quants, normal_quants) 
        prices = jnp.where(ifMarketOrder, market_prices, normal_prices) 
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        times_s, times_ns =  times[:,:,0].T, times[:,:,1].T
        naction_msgs_=jnp.stack([types,sides,quants,prices,trader_ids,order_ids,times_s, times_ns],axis=1) 
        naction_msgs = jnp.transpose(naction_msgs_, axes=(2, 0, 1))
        return naction_msgs
        # ============================== Get Action_msgs ==============================    

    def step_env(
        self, key: chex.PRNGKey, nstate: NEnvState, naction: chex.Array, nparams: NEnvParams
    ) -> Tuple[chex.Array, NEnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        ndata_messages=jnp.array([job.get_data_messages(nparams.message_data,nstate.nwindow_index[i],nstate.nstep_counter[i]) for i in range(nparams.num_envs)])
        #jax.debug.print("Data Messages to process \n: {}",data_messages)
        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        
        naction_msgs = self.getActionMsgs(naction, nstate, nparams)

        #jax.debug.print(f"action shape: {action_msgs.shape}")
        #jax.debug.print("Input to cancel function: {}",state.bid_raw_orders[-1])
        ncnl_msgs=jnp.array([job.getCancelMsgs(nstate.nask_raw_orders[i] if "sell"=='sell' else nstate.nbid_raw_orders[i],
                                   -8999,2*4,-1 if "sell"=='sell' else 1) for i in range(nparams.num_envs)])
        # ncnl_msgs=jnp.array([job.getCancelMsgs(nstate.nask_raw_orders[i] if self.task=='sell' else nstate.nbid_raw_orders[i],
        #                            -8999,self.n_fragment_max*self.n_actions,-1 if self.task=='sell' else 1) for i in range(nparams.num_envs)])
        #jax.debug.print("Output from cancel function: {}",cnl_msgs)


        #Add to the top of the data messages
        ntotal_messages=jnp.concatenate([naction_msgs,ndata_messages],axis=1)
        # total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0)
        # jax.debug.print("Total messages: \n {}",total_messages)


        #Process messages of step (action+data) through the orderbook
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((100,6))*-1).astype(jnp.int32)
        # trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)

        nscan_results=[job.scan_through_entire_array_save_bidask(ntotal_messages[i],(nstate.nask_raw_orders[i,:,:],nstate.nbid_raw_orders[i,:,:],trades_reinit),100) for i in range(nparams.num_envs)]
        # nscan_results=jnp.array([job.scan_through_entire_array_save_bidask(ntotal_messages[i],(nstate.nask_raw_orders[i,:,:],nstate.nbid_raw_orders[i,:,:],trades_reinit),self.stepLines) for i in range(nparams.num_envs)])
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        
        # =========ECEC QTY========
        executed = jnp.where((state.trades[:, 0] > 0)[:, jnp.newaxis], state.trades, 0)
        new_execution = executed[:,1].sum() # sumExecutedQty
        # CAUTION not same executed with the one in the reward
        # CAUTION the array executed here is calculated from the last state
        # CAUTION while the array executedin reward is calc from the update state in this step
        # =========ECEC QTY========
        
        scan_start = timeit.default_timer()
        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]
        state = EnvState(*scan_results,state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,state.step_counter+1,state.init_price,state.task_to_execute,state.quant_executed+new_execution)
        scan_end = timeit.default_timer()
        # jax.debug.print("Trades: \n {}",state.trades)
        
        
        done_start = timeit.default_timer()
        # .block_until_ready()
        done = self.is_terminal(state,params)
        done_end = timeit.default_timer()
        
        # jax.debug.breakpoint()
        #jax.debug.print("Final state after step: \n {}", state)
        return self.get_obs(state,params),state,reward,done,{"info":0}


    def reset_env(
        self, key: chex.PRNGKey, nparams: NEnvParams
    ) -> Tuple[chex.Array, NEnvState]:
        #jax.debug.breakpoint()
        """Reset environment nstate by sampling initial position in OB."""
        key, nparams = key_reset,env_params#$
        keys = jax.random.split(key, nparams.num_envs)
        nidx_data_window = jnp.array([jax.random.randint(keys[i], minval=0, maxval=12, shape=()) for i in range(nparams.num_envs)])
        # nidx_data_window = jnp.array([jax.random.randint(keys[i], minval=0, maxval=self.n_windows, shape=()) for i in range(nparams.num_envs)])
        #Get the init time based on the first message to be processed in the first step. 
        ntime=jnp.array([job.get_initial_time(nparams.message_data,idx_data_window) for idx_data_window in nidx_data_window])
        #Get initial orders (2xNdepth)x6 based on the initial L2 orderbook for this window 
        ninit_orders=jnp.array([job.get_initial_orders(nparams.book_data,nidx_data_window[i],ntime[i]) for i in range(nparams.num_envs)])
        #Initialise both sides of the book as being empty
        nasks_raw=jnp.array([job.init_orderside(100) for i in range(nparams.num_envs)]) 
        nbids_raw=jnp.array([job.init_orderside(100) for i in range(nparams.num_envs)])
        # nasks_raw=jnp.array([job.init_orderside(self.nOrdersPerSide) for i in range(nparams.num_envs)])  
        # nbids_raw=jnp.array([job.init_orderside(self.nOrdersPerSide) for i in range(nparams.num_envs)])  
        ntrades_init=jnp.array([(jnp.ones((100,6))*-1).astype(jnp.int32) for i in range(nparams.num_envs)])
        # ntrades_init=jnp.array([(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32) for i in range(nparams.num_envs)])  
        #Process the initial messages through the orderbook
        
        nordersides=jnp.array([job.scan_through_entire_array(ninit_orders[i],(nasks_raw[i],nbids_raw[i],ntrades_init[i])) for i in range(nparams.num_envs)])
        # nordersides = jnp.vstack(nordersides) # TODO, wrong here, as ordersides has three elems

        # Mid Price after init added to env nstate as the initial price --> Do not at to self as this applies to all environments.
        bestAskBid_pairs = jnp.array([job.get_best_bid_and_ask(ordersides[0],ordersides[1]) for ordersides in nordersides])
        Ms = jnp.array([(nbest_bid + nbest_ask)//2//100*100 for nbest_bid, nbest_ask in bestAskBid_pairs])
        # Ms = jnp.array([(nbest_bid + nbest_ask)//2//self.tick_size*self.tick_size for nbest_bid, nbest_ask in bestBidAsk_pairs])

        #Craft the first nstate
        # states = [NEnvState(*nordersides[i],jnp.resize(bestBidAsk_pairs[i][0],(self.stepLines,)),jnp.resize(bestBidAsk_pairs[i][0],(self.stepLines,)),time,time,0,nidx_data_window[i],0,Ms[i],self.task_size,0) for i in range(nparams.num_envs)]
        # jax.debug.print('nstate: {}',nstate)
        # jax.debug.breakpoint()
        
        ones = jnp.ones((nparams.num_envs,),dtype=jnp.int32)
        nstate = NEnvState(*map(jnp.stack,[nordersides[:,i] for i in range(nordersides.shape[0])]),*(jnp.repeat(arr,100,axis=0) for arr in jnp.vsplit(bestAskBid_pairs.T, 2)),ntime,ntime,0*ones,nidx_data_window,0*ones,Ms,200*ones,0*ones)
        # nstate = NEnvState(*map(jnp.stack,[nordersides[:,i] for i in range(nordersides.shape[0])]),*(jnp.repeat(arr,100,axis=0) for arr in jnp.vsplit(bestAskBid_pairs.T, 2)),ntime,ntime,0*ones,nidx_data_window,0*ones,Ms,self.task_size*ones,0*ones)
        

        
        # jnp.vstack(states[0])
        return self.get_obs(nstate,nparams),nstate


    def get_obs(self,  nstate: NEnvState, nparams:NEnvParams) -> chex.Array:
        
        """Return observation from raw  nstate trafo."""
        # ========= self.get_obs( nstate,nparams) =============
        # -----------------------2--------------------------
        nmid_prices = (nstate.nbest_asks +  nstate.nbest_bids)//2//100*100
        # nmid_prices = (nstate.nbest_asks +  nstate.nbest_bids)//2//self.tick_size*self.tick_size 
        nsecond_passives =  nstate.nbest_asks+100*20 if 'sell'=='sell' else  nstate.nbest_bids-100*20
        # nsecond_passives =  nstate.nbest_asks+self.tick_size*self.n_ticks_in_book if self.task=='sell' else  nstate.nbest_bids-self.tick_size*self.n_ticks_in_book
        # -----------------------3--------------------------
        ntimeOfDay =  nstate.ntime.T
        ndeltaT =  ntimeOfDay -  nstate.ninit_time.T
        # -----------------------4--------------------------
        
        ninitPrice =  nstate.ninit_price.reshape(1,-1)
        npriceDrift = nmid_prices[-1,:].reshape(1,-1) -  ninitPrice
        # -----------------------5--------------------------
        nspreads =  nstate.nbest_asks -  nstate.nbest_bids
        # -----------------------6--------------------------
        ntaskSize =  nstate.ntask_to_execute.reshape(1,-1)
        nexecuted_quant= nstate.nquant_executed.reshape(1,-1)
        # -----------------------7--------------------------
        getQuants = lambda raw_quants: jnp.where(raw_quants>0, raw_quants, 0)
        naskQuants,nbidQuants = jax.lax.map(getQuants,nstate.nask_raw_orders[:,:,1]), jax.lax.map(getQuants,nstate.nbid_raw_orders[:,:,1])
        nOFI_series = naskQuants - nbidQuants
        nlastShallowImbalance, nlastDeepImbalance = jnp.array([nOFI_series[:,0]]), jnp.array([nOFI_series.sum(axis=1)])
        # ========= self.get_obs( nstate,nparams) =============
        # --------------------------------------------------
        # the below is the original obs
        # obs = jnp.concatenate(( nstate.nbest_asks, nstate.nbest_bids,mid_prices,second_passives,spreads,timeOfDay,deltaT,jnp.array([initPrice]),jnp.array([priceDrift]),jnp.array([taskSize]),jnp.array([executed_quant]),deepImbalance))
        obs = jnp.concatenate(( nstate.nbest_asks, nstate.nbest_bids,nmid_prices,nsecond_passives,nspreads,ntimeOfDay,ndeltaT,ninitPrice,npriceDrift,ntaskSize,nexecuted_quant,nlastShallowImbalance, nlastDeepImbalance))
        # the above is the new obs with lastShallowImbalance, lastDeepImbalance
        # --------------------------------------------------
        # jax.debug.breakpoint()
        return obs



    def is_terminal(self, nstate: NEnvState, nparams: NEnvParams) -> bool:
        """Check whether nstate is terminal."""
        return ((nstate.time-nstate.init_time)[0]>nparams.episode_time) | (nstate.task_to_execute-nstate.quant_executed<0)
    
    
    def get_reward(self, nstate: NEnvState, nparams: NEnvParams) -> float:
        # ========== get_executed_piars for rewards ==========
        # TODO  no valid trades(all -1) case (might) hasn't be handled.
        #jax.debug.breakpoint()
        executed = jnp.where((nstate.trades[:, 0] > 0)[:, jnp.newaxis], nstate.trades, 0)
        
        vwap = (executed[:,0] * executed[:,1]).sum()/ executed[:1].sum() 
        mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        advantage = (agentTrades[:,0] * agentTrades[:,1]).sum() - vwap * agentTrades[:,1].sum()
        Lambda = 0.5 # FIXME shoud be moved to NEnvState or NEnvParams
        drift = agentTrades[:,1].sum() * (vwap - nstate.init_price)
        rewardValue = advantage + Lambda * drift
        reward = jnp.sign(agentTrades[0,0]) * rewardValue # if no value agentTrades then the reward is set to be zero
        # ========== get_executed_piars for rewards ==========
        reward=jnp.nan_to_num(reward)
        return reward




    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions



    
    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, nparams: NEnvParams):
        """Observation space of the environment."""
        # space = spaces.Box(-10000,99999999,(608,),dtype=jnp.int32) 
        space = spaces.Box(-10000,99999999,(510,nparams.num_envs),dtype=jnp.int32)
        return space

    #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    def state_space(self, nparams: NEnvParams) -> spaces.Dict:
        """nstate space of the environment."""
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(nparams.max_steps_in_episode),
            }
        )

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
        ATFolder = '/homes/80/kang/AlphaTrade'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,nstate=env.reset(key_reset,env_params)
    print("nstate after reset: \n",nstate)
    print("Time for reset: \n",time.time()-start)
    print(env_params.message_data.shape, env_params.book_data.shape)

    for i in range(1,100):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        test_action=env.action_space().sample(key_policy)
        # ---------- acion from trained network ----------
        ac_in = (obs[np.newaxis, :], obs[np.newaxis, :])
        ## import ** network
        from gymnax_exchange.jaxrl.ppoRnnExecCont import ActorCriticRNN
        ppo_config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 4,
            "NUM_STEPS": 2,
            "TOTAL_TIMESTEPS": 5e5,
            "UPDATE_EPOCHS": 4,
            "NUM_MINIBATCHES": 4,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ENV_NAME": "alphatradeExec-v0",
            "ANNEAL_LR": True,
            "DEBUG": True,
            "NORMALIZE_ENV": False,
            "ATFOLDER": ATFolder,
            "TASKSIDE":'buy'
        }
        # runner_state = np.load("runner_state.npy") # FIXME/TODO save the runner_state after training
        # network = ActorCriticRNN(env.action_space(env_params).shape[0], config=ppo_config)
        # hstate, pi, value = network.apply(runner_state.train_state.nparams, hstate, ac_in)
        # action = pi.sample(seed=rng) # 4*1, should be (4*4: 4actions * 4envs)
        # ==================== ACTION ====================
        
        
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,nstate,reward,done,info=env.step(key_step, nstate,test_action, env_params)
        print(f"nstate after {i} step: \n",nstate,done,file=open('output.txt','a'))
        print(f"Time for {i} step: \n",time.time()-start)

    # ####### Testing the vmap abilities ########
    
    enable_vmap=True
    if enable_vmap:
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, nstate = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, nstate, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)