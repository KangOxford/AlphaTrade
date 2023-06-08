from ast import Dict
from contextlib import nullcontext
from email import message
import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job


@struct.dataclass
class EnvState:
    bid_raw_orders: chex.Array
    ask_raw_orders: chex.Array
    trades: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int


@struct.dataclass
class EnvParams:
    #Note: book depth and nOrdersperside must be given at init as they define shapes of jax arrays,
    #       only the self args are static, the param args are not static and so must be traceable.
    #book_depth: int = 10
    #nOrdersPerSide: int = 100
    episode_time: int =  60*10 #60seconds times 10 minutes = 600seconds
    max_steps_in_episode: int = 100
    messages_per_step: int=1
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 10000000]) #10ms time delay. 




class BaseLOBEnv(environment.Environment):
    def __init__(self):
        super().__init__()
        # Load the image MNIST data at environment init
        #TODO:Define Load function based on Kangs work (though it seems that takes quite a while)
        """(book_data, message_data), _ = load_LOBSTER()"""

        def load_LOBSTER():
            def config():
                sliceTimeWindow = 1800 # counted by seconds, 1800s=0.5h
                stepLines = 100
                messagePath = "/Users/kang/Data/Whole_Flow/"
                orderbookPath = "/Users/kang/Data/Whole_Book/"
                start_time = 34200  # 09:30
                end_time = 57600  # 16:00
                return sliceTimeWindow, stepLines, messagePath, orderbookPath, start_time, end_time
            sliceTimeWindow, stepLines, messagePath, orderbookPath, start_time, end_time = config()
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
            def preProcessingMassege(message):
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
                return message
            messages = [preProcessingMassege(message) for message in messages]
            def index_of_sliceWithoutOverlap(start_time, end_time, interval):
                indices = list(range(start_time, end_time, interval))
                return indices

            indices = index_of_sliceWithoutOverlap(start_time, end_time, sliceTimeWindow)
            def sliceWithoutOverlap(message):
                def splitMessage(message):
                    sliced_parts = []
                    for i in range(len(indices) - 1):
                        start_index = indices[i]
                        end_index = indices[i + 1]
                        sliced_part = message.loc[(message['time'] >= start_index) & (message['time'] < end_index)]
                        num_rows = len(sliced_part)
                        num_rows -= num_rows % stepLines
                        sliced_part = sliced_part[:num_rows]
                        sliced_parts.append(sliced_part)

                    # Last sliced part from last index to end_time
                    last_sliced_part = message.loc[message['time'] >= indices[-1]]
                    num_rows = len(sliced_part)
                    num_rows -= num_rows % stepLines
                    last_sliced_part = last_sliced_part[:num_rows]
                    sliced_parts.append(last_sliced_part)
                    return sliced_parts
                sliced_parts = splitMessage(message)
                def sliced2cude(sliced):
                    columns = ['type','direction','qty','price','trader_id','order_id','time_s','time_ns']
                    cube = sliced[columns].to_numpy()
                    return cube
                slicedCubes = [sliced2cude(sliced) for sliced in sliced_parts]
                # slicedCube: dynamic_horizon * stepLines * 8
                return slicedCubes
            slicedCubes_list = [sliceWithoutOverlap(message) for message in messages]
            # slicedCubes_list(nested list), outer_layer : day, inter_later : time of the day
            return slicedCubes_list
        load_LOBSTER()

        #numpy load with the memmap
        book_data=0
        message_data=0
        self.book_data=book_data
        self.message_data=message_data
        #TODO:Any cleanup required...
        self.n_windows=100
        self.nOrdersPerSide=100
        self.nTradesLogged=5
        self.book_depth=10
        self.n_actions=3
        self.customIDCounter=0
        self.trader_unique_id=job.INITID+1
        """
        self.num_data = int(fraction * len(labels))
        self.image_shape = images.shape[1:]
        self.images = jnp.array(images[: self.num_data])
        self.labels = jnp.array(labels[: self.num_data])
        """

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()


    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        data_messages=job.get_data_messages()
        types=jnp.ones((self.n_actions,),jnp.int32)
        sides=action["sides"]
        prices=action["prices"]
        quants=action["quantities"]
        
        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions)
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2))
        
        action_msgs=jnp.stack([types,sides,prices,quants,trader_ids,order_ids],axis=1)
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)
        total_messages=jnp.concatenate([action_msgs,data_messages],axis=0)
        #jax.debug.print("Step messages to process are: \n {}", total_messages)
        time=total_messages[-1:][0][-2:]
        ordersides,trades=job.scan_through_entire_array(total_messages,(state.ask_raw_orders,state.bid_raw_orders))
        state = EnvState(ordersides[0],ordersides[1],trades[0],state.init_time,time,state.customIDcounter+self.n_actions)
        #reward = lax.select(correct, 1.0, -1.0)
        # Check game condition & no. steps for termination condition
        """done = self.is_terminal(state, params)"""
        """info = {"discount": self.discount(state, params)}"""
        return self.get_obs(state,params),state,0,self.is_terminal(state,params),{"discount":0}
        """lax.stop_gradient(observation),
        lax.stop_gradient(state),
        reward,
        done,
        info,"""
        

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=())
        #TODO:create a function that selects the correct ob from the hist data and turns it into a set of init messages
        #These messages need to be ready to process by the scan function, so 6x(Ndepth*2) array and the times must be chosen correctly.
        init_orders=job.get_initial_orders(self.book_data,idx_data_window)
        time=init_orders[-1:][0][-2:]
        #jax.debug.print("Reset messages to process are: \n {}", init_orders)
        asks_raw=job.init_orderside(self.nOrdersPerSide)
        bids_raw=job.init_orderside(self.nOrdersPerSide)
        ordersides,trades=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw))
        state = EnvState(ordersides[0],ordersides[1],trades[0],time,time,0)
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
                "quantities":spaces.Box(0,10000,(self.n_actions,),dtype=jnp.int32),
                "prices":spaces.Box(0,job.MAXPRICE,(self.n_actions,),dtype=jnp.int32)
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
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(5,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
