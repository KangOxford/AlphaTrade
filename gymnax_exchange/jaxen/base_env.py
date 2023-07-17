# from jax import config
# config.update("jax_enable_x64",True)

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
    max_steps_in_episode: int

@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    # max_steps_in_episode: int = 100
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    




class BaseLOBEnv(environment.Environment):
    def __init__(self,alphatradePath):
        super().__init__()
        self.sliceTimeWindow = 1800 # counted by seconds, 1800s=0.5h
        self.stepLines = 100
        # self.messagePath = alphatradePath+"/data_small/Flow_10/"
        # self.orderbookPath = alphatradePath+"/data_small/Book_10/"
        self.messagePath = alphatradePath+"/data/Flow_10/"
        self.orderbookPath = alphatradePath+"/data/Book_10/"
        self.start_time = 34200  # 09:30
        self.end_time = 57600  # 16:00
        self.nOrdersPerSide=100
        self.nTradesLogged=100
        self.book_depth=10
        self.n_actions=3
        self.customIDCounter=0
        self.trader_unique_id=-9000+1
        self.tick_size=100



        # Load the image MNIST data at environment init
        def load_LOBSTER(sliceTimeWindow, stepLines, messagePath, orderbookPath, start_time, end_time):
            def preProcessingData_csv2pkl():
                return 0
            def load_files():
                from os import listdir; from os.path import isfile, join; import pandas as pd
                readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
                messageFiles, orderbookFiles = readFromPath(messagePath), readFromPath(orderbookPath)
                dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
                messageCSVs = [pd.read_csv(messagePath + file, usecols=range(6), dtype=dtype, header=None) for file in messageFiles if file[-3:] == "csv"]
                orderbookCSVs = [pd.read_csv(orderbookPath + file, header=None) for file in orderbookFiles if file[-3:] == "csv"]
                return messageCSVs, orderbookCSVs
            messages, orderbooks = load_files()

            def preProcessingMassegeOB(message, orderbook):
                def splitTimeStamp(m):
                    m[6] = m[0].apply(lambda x: int(x))
                    m[7] = ((m[0] - m[6]) * int(1e9)).astype(int)
                    m.columns = ['time','type','order_id','qty','price','direction','time_s','time_ns']
                    return m
                message = splitTimeStamp(message)
                def filterValid(message):
                    message = message[message.type.isin([1,2,3,4])]
                    valid_index = message.index.to_numpy()
                    message.reset_index(inplace=True,drop=True)
                    return message, valid_index
                message, valid_index = filterValid(message)
                def tuneDirection(message):
                    message.loc[message['type'] == 4, 'direction'] *= -1
                    return message
                message = tuneDirection(message)
                def addTraderId(message):
                    import warnings
                    from pandas.errors import SettingWithCopyWarning
                    warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
                    message['trader_id'] = message['order_id']
                    return message

                message = addTraderId(message)
                orderbook.iloc[valid_index,:].reset_index(inplace=True, drop=True)
                return message,orderbook
            pairs = [preProcessingMassegeOB(message, orderbook) for message,orderbook in zip(messages,orderbooks)]
            messages, orderbooks = zip(*pairs)

            def index_of_sliceWithoutOverlap(start_time, end_time, interval):
                indices = list(range(start_time, end_time, interval))
                return indices
            indices = index_of_sliceWithoutOverlap(start_time, end_time, sliceTimeWindow)
            def sliceWithoutOverlap(message, orderbook):
                # print("start")
                def splitMessage(message, orderbook):
                    import numpy as np
                    sliced_parts = []
                    init_OBs = []
                    for i in range(len(indices) - 1):
                        start_index = indices[i]
                        end_index = indices[i + 1]
                        index_s, index_e = message[(message['time'] >= start_index) & (message['time'] < end_index)].index[[0, -1]].tolist()
                        index_e = (index_e // stepLines + 10) * stepLines + index_s % stepLines
                        assert (index_e - index_s) % stepLines == 0, 'wrong code 31'
                        sliced_part = message.loc[np.arange(index_s, index_e)]
                        sliced_parts.append(sliced_part)
                        init_OBs.append(orderbook.iloc[index_s,:])

                    # Last sliced part from last index to end_time
                    start_index = indices[i]
                    end_index = indices[i + 1]
                    index_s, index_e = message[(message['time'] >= start_index) & (message['time'] < end_index)].index[[0, -1]].tolist()
                    index_s = (index_s // stepLines - 10) * stepLines + index_e % stepLines
                    assert (index_e - index_s) % stepLines == 0, 'wrong code 32'
                    last_sliced_part = message.loc[np.arange(index_s, index_e)]
                    sliced_parts.append(last_sliced_part)
                    init_OBs.append(orderbook.iloc[index_s, :])
                    for part in sliced_parts:
                        # print("start")
                        assert part.time_s.iloc[-1] - part.time_s.iloc[0] >= sliceTimeWindow, f'wrong code 33, {part.time_s.iloc[-1] - part.time_s.iloc[0]}, {sliceTimeWindow}'
                        assert part.shape[0] % stepLines == 0, 'wrong code 34'
                    return sliced_parts, init_OBs
                sliced_parts, init_OBs = splitMessage(message, orderbook)
                def sliced2cude(sliced):
                    columns = ['type','direction','qty','price','trader_id','order_id','time_s','time_ns']
                    cube = sliced[columns].to_numpy()
                    cube = cube.reshape((-1, stepLines, 8))
                    return cube
                # def initialOrderbook():
                slicedCubes = [sliced2cude(sliced) for sliced in sliced_parts]
                # Cube: dynamic_horizon * stepLines * 8
                slicedCubes_withOB = zip(slicedCubes, init_OBs)
                return slicedCubes_withOB
            slicedCubes_withOB_list = [sliceWithoutOverlap(message, orderbook) for message,orderbook in zip(messages,orderbooks)]
            # i = 6 ; message,orderbook = messages[i],orderbooks[i]
            # slicedCubes_list(nested list), outer_layer : day, inter_later : time of the day


            def nestlist2flattenlist(nested_list):
                import itertools
                flattened_list = list(itertools.chain.from_iterable(nested_list))
                return flattened_list
            Cubes_withOB = nestlist2flattenlist(slicedCubes_withOB_list)


            max_steps_in_episode_arr = jnp.array([m.shape[0] for m,o in Cubes_withOB],jnp.int32)


            def Cubes_withOB_padding(Cubes_withOB):
                max_m = max(m.shape[0] for m, o in Cubes_withOB)
                new_Cubes_withOB = []
                for cube, OB in Cubes_withOB:
                    def padding(cube, target_shape):
                        pad_width = np.zeros((100, 8))
                        # Calculate the amount of padding required
                        padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
                        padded_cube = np.pad(cube, padding, mode='constant', constant_values=0)
                        return padded_cube
                    cube = padding(cube, max_m)
                    new_Cubes_withOB.append((cube, OB))
                return new_Cubes_withOB
            Cubes_withOB = Cubes_withOB_padding(Cubes_withOB)
            return Cubes_withOB, max_steps_in_episode_arr

        Cubes_withOB, max_steps_in_episode_arr = load_LOBSTER(self.sliceTimeWindow,self.stepLines,self.messagePath,self.orderbookPath,self.start_time,self.end_time)
        self.max_steps_in_episode_arr = max_steps_in_episode_arr
        # # ------------------------------- TESTING ------------------------------
        # alphatradePath = '/homes/80/kang/AlphaTrade'
        # messagePath = alphatradePath+"/data_small/Flow_10/"
        # orderbookPath = alphatradePath+"/data_small/Book_10/"
        # sliceTimeWindow, stepLines, messagePath, orderbookPath, start_time, end_time=1800,100,messagePath,orderbookPath,34200,57600
        # Cubes_withOB, max_steps_in_episode_arr = load_LOBSTER(1800,100,messagePath,orderbookPath,34200,57600)
        # msgs=[jnp.array(cube) for cube, book in Cubes_withOB]
        # bks=[jnp.array(book) for cube, book in Cubes_withOB]
        # message_data, book_data = msgs[0],bks[0]
        # nOrdersPerSide, nTradesLogged, tick_size,stepLines,task_size, n_ticks_in_book= 100, 100, 100,100, 20,200
        # # ------------------------------- TESTING ------------------------------
        # print(len(msgs))
        # for message_data in msgs:
        #     print(message_data.shape)
            
            
            
        #List of message cubes 
        msgs=[jnp.array(cube) for cube, book in Cubes_withOB]
        bks=[jnp.array(book) for cube, book in Cubes_withOB]

        self.messages=jnp.array(msgs)   #4D Array: (n_windows x n_steps (max) x n_messages x n_features)
        self.books=jnp.array(bks)       #2D Array: (n_windows x [4*n_depth])

        self.n_windows=len(self.books)
        # jax.debug.breakpoint()
        
        # ==================================================================
        # ================= CAUTION NOT BELONG TO BASE ENV =================
        # ================= EPECIALLY SUPPORT FOR EXEC ENV =================
        print("START:  pre-reset in the initialization")


        nOrdersPerSide, nTradesLogged, tick_size,stepLines,task_size,n_ticks_in_book = self.nOrdersPerSide, self.nTradesLogged, self.tick_size,self.stepLines,200, 20
        
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
            asks_raw=job.init_orderside(nOrdersPerSide)
            bids_raw=job.init_orderside(nOrdersPerSide)
            trades_init=(jnp.ones((nTradesLogged,6))*-1).astype(jnp.int32)
            #Process the initial messages through the orderbook
            ordersides=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw,trades_init))

            # Mid Price after init added to env state as the initial price --> Do not at to self as this applies to all environments.
            best_ask, best_bid = job.get_best_bid_and_ask_inclQuants(ordersides[0],ordersides[1])
            M = (best_bid[0] + best_ask[0])//2//tick_size*tick_size 
            
            state = (ordersides[0],ordersides[1],ordersides[2],jnp.resize(best_ask,(stepLines,2)),jnp.resize(best_bid,(stepLines,2)),\
                time,time,0,-1,M,task_size,0,0,0,0,max_steps_in_episode)
            return state
        
        def get_obs(state):
            """Return observation from raw state trafo."""
            # ========= self.get_obs(state,params) =============
            # -----------------------1--------------------------
            best_asks=state[3][:,0]
            best_bids =state[4][:,0]
            mid_prices=(best_asks+best_bids)//2//tick_size*tick_size 
            second_passives_sell_task = best_asks+tick_size*n_ticks_in_book 
            second_passives_buy_task =  best_bids-tick_size*n_ticks_in_book
            spreads = best_asks - best_bids

            # -----------------------2--------------------------
            timeOfDay = state[6] #state.time
            deltaT = timeOfDay -  state[5] # state.init_time
            # -----------------------3--------------------------
            initPrice = state[10] # state.init_price
            priceDrift = mid_prices[-1] - initPrice
            # -----------------------4--------------------------
            # -----------------------5--------------------------
            taskSize = state[11] # state.task_to_execute
            executed_quant=state[12] # state.quant_executed
            # -----------------------7--------------------------
            bestAsksQtys = state[3][:,1]
            bestBidsQtys = state[4][:,1]
            shallowImbalance = bestAsksQtys - bestBidsQtys # ShallowImbalance
            # -----------------------8--------------------------
            step_counter=0;max_steps_in_episode=state[-1]
            # ========= self.get_obs(state,params) =============
            obs_sell = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives_sell_task,spreads,timeOfDay,deltaT,jnp.array([initPrice]),jnp.array([priceDrift]),jnp.array([taskSize]),jnp.array([executed_quant]),shallowImbalance,jnp.array([step_counter]),jnp.array([max_steps_in_episode])))
            obs_buy  = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives_buy_task, spreads,timeOfDay,deltaT,jnp.array([initPrice]),jnp.array([priceDrift]),jnp.array([taskSize]),jnp.array([executed_quant]),shallowImbalance,jnp.array([step_counter]),jnp.array([max_steps_in_episode])))
            # jax.debug.breakpoint()
            return obs_sell, obs_buy
        
        def get_state_obs(message_data, book_data,max_steps_in_episode):
            state = get_state(message_data, book_data,max_steps_in_episode)
            obs_sell, obs_buy = get_obs(state)
            return state, obs_sell, obs_buy

        state_obs = [get_state_obs(Cubes_withOB[i][0], Cubes_withOB[i][1], max_steps_in_episode_arr[i]) for i in range(len(max_steps_in_episode_arr))]
        
        def state2stateArray(state):
            state_5 = jnp.hstack((state[5],state[6],state[9],state[15]))
            padded_state = jnp.pad(state_5, (0, 100 - state_5.shape[0]), constant_values=-1)[:,jnp.newaxis]
            stateArray = jnp.hstack((state[0],state[1],state[2],state[3],state[4],padded_state))
            # jax.debug.breakpoint()
            return stateArray
        
        self.stateArray_list = jnp.array([state2stateArray(state) for state, obs_sell, obs_buy in state_obs])
        self.obs_sell_list = jnp.array([jnp.array(obs_sell) for state, obs_sell, obs_buy in state_obs])
        self.obs_buy_list =  jnp.array([jnp.array(obs_buy) for state, obs_sell, obs_buy in state_obs])
        print("FINISH: pre-reset in the initialization")
        # ================= CAUTION NOT BELONG TO BASE ENV =================
        # ================= EPECIALLY SUPPORT FOR EXEC ENV =================
        # ==================================================================
        #TODO Most of the state space should be exactly the same for the base and exec env, 
        # can we think about keeping the base part seperate from the exec part? 


        print(f"Num of data_window: {self.n_windows}")



    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(self.messages,self.books)
        # return EnvParams(self.messages,self.books,self.state_list,self.obs_sell_list,self.obs_buy_list)
    # @property
    # def default_params(self) -> EnvParams:
    #     # Default environment parameters
    #     # return EnvParams(self.messages,self.books)
    #     return EnvParams(self.messages,self.books,self.state_list,self.obs_sell_list,self.obs_buy_list)


    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #jax.debug.print("Data Messages to process \n: {}",data_messages)

        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides=((action["sides"]+1)/2).astype(jnp.int32)      #from action space
        prices=action["prices"]     #from action space
        quants=action["quantities"] #from action space
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)

        #Add to the top of the data messages 
        total_messages=jnp.concatenate([action_msgs,data_messages],axis=0)
        #jax.debug.print("Step messages to process are: \n {}", total_messages)

        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]

        #Process messages of step (action+data) through the orderbook
        ordersides=job.scan_through_entire_array(total_messages,(state.ask_raw_orders,state.bid_raw_orders,state.trades))

        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter)
        state = EnvState(ordersides[0],ordersides[1],ordersides[2],state.init_time,time,state.customIDcounter+self.n_actions,\
            state.window_index,state.step_counter+1,state.max_steps_in_episode)
        done = self.is_terminal(state,params)
        reward=0
        #jax.debug.print("Final state after step: \n {}", state)
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
        state = EnvState(*ordersides,time,time,0,idx_data_window,0,self.max_steps_in_episode_arr[idx_data_window])

        return self.get_obs(state,params),state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return (state.time-state.init_time)[0]>params.episode_time

    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        return job.get_L2_state(self.book_depth,state.ask_raw_orders,state.bid_raw_orders)

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeBase-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Dict(
            {
                "sides":spaces.Box(0,2,(self.n_actions,),dtype=jnp.int32),
                "quantities":spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32),
                "prices":spaces.Box(0,99999999,(self.n_actions,),dtype=jnp.int32)
            }
        )

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
                "bids": spaces.Box(-1,999999999,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,999999999,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,999999999,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
