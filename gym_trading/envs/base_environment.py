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
from gym_trading.utils import * 
from gym_trading.envs.match_engine import Core
# =============================================================================

class BaseEnv(Env, ABC):
    num_steps = 1024 # size of a flow
    high = 1024
    max_quantity = 6000
    max_price = 31620700
    min_price = 31120200

    def __init__(self, Flow) -> None:
        super().__init__()
        self.Flow = Flow
        self.core = None
        self.price_list = None
        self.action_space = spaces.Box(0, BaseEnv.high,shape =(1,),dtype = np.float32)
        self.observation_space = spaces.Dict({
            'price':spaces.MultiDiscrete([BaseEnv.max_price for _ in range(10)], dtype=np.int64),
            'quantity':spaces.MultiDiscrete([BaseEnv.max_quantity for _ in range(10)], dtype=np.int64)
            })
        self.done = False
        self.running_reward = 0.0
        self.init_reward = 0.0
        
    def setp(self, action: float = 0):
        # return observation, reward, done, info
        observation = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return  observation, reward, done, info
    def reset(self):
        '''return the observation of the initial condition'''
        index_random = random.randint(0, self.Flow.shape[0]-BaseEnv.num_steps-1)
        flow = self.Flow.iloc[index_random:index_random+BaseEnv.num_steps,:]
        self.core = Core(flow)
        self._set_init_reward()
        stream =  flow.iloc[0,:]
        init_obs = np.array(get_quantity_from_stream(stream)).astype(np.int64)
        self.running_reward += self.calculate_reward(action)
        self.reset_states()
        return init_obs
    def _get_obs(self):
        pass
    def _get_reward(self):
        if not self.done:
            return 0.0
        else:
            return self.running_reward
    def _get_done(self):
        return self.done
    def _get_info(self):
        return self.info
    def reset_states(self):
        self.running_reward = 0.0
        self.done = False
    def calculate_reward(self):
        return 1.0 # TODO
    def _set_init_reward(self):
        pass
    
if __name__=="__main__":
    from gym_trading.data.data_pipeline import ExternalData
    Flow = ExternalData.get_sample_order_book_data()
    env = BaseEnv(Flow)