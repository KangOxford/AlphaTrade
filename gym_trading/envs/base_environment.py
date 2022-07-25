# =============================================================================
import random
import numpy as np
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod
# ----------------------------------------------------------------------------
from gym import Env
from gym import spaces
# from gym.spaces.multi_discrete import MultiDiscrete
# ----------------------------------------------------------------------------
from gym_trading.utils import get_price_list, get_adjusted_obs, get_max_quantity
from gym_trading.envs.match_engine import Core
# =============================================================================

class BaseEnv(Env, ABC):
    num_steps = 1024 # size of a flow
    high = 1024

    def __init__(self, Flow) -> None:
        super().__init__()
        self.Flow = Flow
        self.core = None
        self.price_list = None
        self.max_quantity  = None
        self.action_space = spaces.Box(0, BaseEnv.high,shape =(1,),dtype = np.float32)
        # self.observation_space = spaces.MultiDiscrete([1024 for _ in range(50)], dtype=np.int64)
        self.observation_space = None
        self.reset()
    
    def setp(self, action: float = 0):
        # return observation, reward, done, info
        pass
    def reset(self):
        '''return the observation of the initial condition'''
        index_random = random.randint(0, self.Flow.shape[0]-BaseEnv.num_steps-1)
        flow = self.Flow.iloc[index_random:index_random+BaseEnv.num_steps,:]
        self.core = Core(flow)
        self.price_list = get_price_list(flow)
        self.max_quantity = get_max_quantity(flow)
        self.observation_space = spaces.MultiDiscrete(
            [self.max_quantity for _ in range(len(self.price_list))])
        stream =  flow.iloc[0,:]
        init_obs = np.array(get_adjusted_obs(stream, self.price_list)).astype(np.int64)
        return init_obs
    def _get_obs(self):
        pass