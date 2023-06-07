from ast import Dict
from contextlib import nullcontext
from email import message
import jax
import jax.numpy as jnp
import pandas as pd
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
    time: int #TODO: decide whether or not this is actually required, probs more of an obs.


@struct.dataclass
class EnvParams:
    #Note: book depth and nOrdersperside must be given at init as they define shapes of jax arrays,
    #       only the self args are static, the param args are not static and so must be traceable.
    #book_depth: int = 10
    #nOrdersPerSide: int = 100
    max_steps_in_episode: int = 100
    messages_per_step: int=1
    time_per_step: int= 0 ##Going forward, assume that 0 implies not to use time step?




class BaseLOBEnv(environment.Environment):
    def __init__(self):
        super().__init__()
        # Load the image MNIST data at environment init
        #TODO:Define Load function based on Kangs work (though it seems that takes quite a while)
        """(book_data, message_data), _ = load_LOBSTER()"""
        def load_LOBSTER():
            def csv2pickle():
                # would be removed in the official release
                messagePath = "/Users/kang/Data/Whole_Flow/"
                orderbookPath = "/Users/kang/Data/Whole_Book/"
                messagePath1 = "/Users/kang/Data/Message_Pickles/"
                orderbookPath1 = "/Users/kang/Data/OrderBook_Pickles/"
                def make_dir(out_path):
                    try: from os import listdir; listdir(out_path)
                    except: import os;os.mkdir(out_path)
                make_dir(messagePath1);make_dir(orderbookPath1)
                from os import listdir; from os.path import isfile, join
                readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
                messageFiles, orderbookFiles = readFromPath(messagePath), readFromPath(orderbookPath)
                messageCsvs = [pd.read_csv(messagePath + file) for file in messageFiles]
                orderbookCsvs = [pd.read_csv(orderbookPath + file) for file in orderbookFiles]
                [file.to_pickle()]

            def load_files():
                messagePath = "/Users/kang/Data/Message_Pickles/"
                orderbookPath = "/Users/kang/Data/OrderBook_Pickles/"
                from os import listdir; from os.path import isfile, join
                readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
                messageFiles, orderbookFiles = readFromPath(messagePath), readFromPath(orderbookPath)
                messagePkls = [pd.read_pickle(messagePath + file) for file in messageFiles]
                orderbookPkls = [pd.read_pickle(orderbookPath + file) for file in orderbookFiles]
                return messagePkls, orderbookPkls
            messageFiles, orderbookFiles = load_files()
            return
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


    #TODO: complete (1st thing to do)
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        

        
        
        
        
        correct = action == state.correct_label
        reward = lax.select(correct, 1.0, -1.0)
        
        
        
        observation = jnp.zeros(shape=self.image_shape, dtype=jnp.float32)
        
        
        state = EnvState(
            state.correct_label,
            (state.regret + params.optimal_return - reward),
            state.time + 1,
        )
        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=())
        #TODO:create a function that selects the correct ob from the hist data and turns it into a set of init messages
        init_orders=job.get_initial_orders(self.book_data,idx_data_window)
        asks_raw=job.init_orderside(self.nOrdersPerSide)
        bids_raw=job.init_orderside(self.nOrdersPerSide)
        ordersides,trades=job.scan_through_entire_array(init_orders,(asks_raw,bids_raw))
        state = EnvState(ordersides[0], ordersides[1],trades,0)
        return self.get_obs(state,params), state

    #TODO: define terminal based on number of steps taken - steps into state? 
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Every step transition is terminal! No long term credit assignment!

        return True

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

    #TODO: define obs space (4xnDepth) array of quants&prices
    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, shape=self.image_shape)

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
