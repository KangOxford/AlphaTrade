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


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    #Note: book depth and nOrdersperside must be given at init as they define shapes of jax arrays,
    #       only the self args are static, the param args are not static and so must be traceable.
    #book_depth: int = 10
    #nOrdersPerSide: int = 100
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    max_steps_in_episode: int = 100
    messages_per_step: int=1
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    




class BaseLOBEnv(environment.Environment):
    def __init__(self,alphatradePath):
        super().__init__()

       
       
       
        self.sliceTimeWindow = 1800 # counted by seconds, 1800s=0.5h
        self.stepLines = 100
        self.messagePath = alphatradePath+"/data/Flow_10/"
        self.orderbookPath = alphatradePath+"/data/Book_10/"
        self.start_time = 34200  # 09:30
        self.end_time = 57600  # 16:00


        self.nOrdersPerSide=100
        self.nTradesLogged=1000
        self.book_depth=10
        self.n_actions=3
        self.customIDCounter=0
        self.trader_unique_id=-9000+1



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
                    import numpy as np
                    message['direction'] = np.where(message['type'] == 4, message['direction'] * -1,
                                                    message['direction'])
                    return message
                message = tuneDirection(message)
                def addTraderId(message):
                    message['trader_id'] = message['order_id']
                    return message

                message = addTraderId(message)
                orderbook.iloc[valid_index,:].reset_index(inplace=True, drop=True)
                return message,orderbook
            pairs = [preProcessingMassegeOB(message, orderbook) for message,orderbook in zip(messages,orderbooks)]
            messages, orderbooks = zip(*pairs)

            # def slice_initialOB(orderbook):
            #     orderbook[indices,:]
            #
            # orderbook = orderbooks[0]


            def index_of_sliceWithoutOverlap(start_time, end_time, interval):
                indices = list(range(start_time, end_time, interval))
                return indices
            indices = index_of_sliceWithoutOverlap(start_time, end_time, sliceTimeWindow)
            def sliceWithoutOverlap(message, orderbook):
                def splitMessage(message, orderbook):
                    import numpy as np
                    sliced_parts = []
                    init_OBs = []
                    for i in range(len(indices) - 1):
                        start_index = indices[i]
                        end_index = indices[i + 1]
                        index_s, index_e = message[(message['time'] >= start_index) & (message['time'] < end_index)].index[[0, -1]].tolist()
                        index_e = (index_e // stepLines + 3) * stepLines + index_s % stepLines
                        assert (index_e - index_s) % stepLines == 0, 'wrong code 31'
                        sliced_part = message.loc[np.arange(index_s, index_e)]
                        sliced_parts.append(sliced_part)
                        init_OBs.append(orderbook.iloc[index_s,:])

                    # Last sliced part from last index to end_time
                    start_index = indices[i]
                    end_index = indices[i + 1]
                    index_s, index_e = message[(message['time'] >= start_index) & (message['time'] < end_index)].index[[0, -1]].tolist()
                    index_s = (index_s // stepLines - 3) * stepLines + index_e % stepLines
                    assert (index_e - index_s) % stepLines == 0, 'wrong code 32'
                    last_sliced_part = message.loc[np.arange(index_s, index_e)]
                    sliced_parts.append(last_sliced_part)
                    init_OBs.append(orderbook.iloc[index_s, :])
                    for part in sliced_parts:
                        assert part.time_s.iloc[-1] - part.time_s.iloc[0] >= sliceTimeWindow, 'wrong code 33'
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
            # slicedCubes_list(nested list), outer_layer : day, inter_later : time of the day


            def nestlist2flattenlist(nested_list):
                import itertools
                flattened_list = list(itertools.chain.from_iterable(nested_list))
                return flattened_list
            Cubes_withOB = nestlist2flattenlist(slicedCubes_withOB_list)
            def Cubes_withOB_padding(Cubes_withOB):
                def quantile(Cubes_withOB):
                    length = [len(cube) for cube, ob in Cubes_withOB]
                    quantile_95 = int(np.quantile(length, 0.9428))
                    return quantile_95

                quantile_95 = quantile(Cubes_withOB)
                new_Cubes_withOB = []
                for cube, OB in Cubes_withOB:
                    if cube.shape[0] <= quantile_95:
                        def padding(cube, target_shape):
                            pad_width = np.zeros((100, 8))
                            # Calculate the amount of padding required
                            padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
                            padded_cube = np.pad(cube, padding, mode='constant', constant_values=0)
                            return padded_cube

                        cube = padding(cube, quantile_95)
                        new_Cubes_withOB.append((cube, OB))
                return new_Cubes_withOB
            Cubes_withOB = Cubes_withOB_padding(Cubes_withOB)
            return Cubes_withOB
        Cubes_withOB = load_LOBSTER(self.sliceTimeWindow,self.stepLines,self.messagePath,self.orderbookPath,self.start_time,self.end_time)
        
        #List of message cubes 
        msgs=[jnp.array(cube) for cube, book in Cubes_withOB]
        bks=[jnp.array(book) for cube, book in Cubes_withOB]

        self.messages=jnp.array(msgs)   #4D Array: (n_windows x n_steps (max) x n_messages x n_features)
        self.books=jnp.array(bks)       #2D Array: (n_windows x [4*n_depth])

        self.n_windows=len(self.books)



        """
        self.num_data = int(fraction * len(labels))
        self.image_shape = images.shape[1:]
        self.images = jnp.array(images[: self.num_data])
        self.labels = jnp.array(labels[: self.num_data])
        """

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
        sides=action["sides"]       #from action space
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
        state = EnvState(ordersides[0],ordersides[1],ordersides[2],state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,state.step_counter+1)
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
        trades_init=(jnp.ones((self.nTradesLogged,5))*-1).astype(jnp.int32)
        #Process the initial messages through the orderbook
        ordersides=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw,trades_init))

        #Craft the first state
        state = EnvState(ordersides[0],ordersides[1],ordersides[2],time,time,0,idx_data_window,0)

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
                "trades": spaces.Box(-1,999999999,shape=(5,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
