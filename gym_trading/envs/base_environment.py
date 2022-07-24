# =============================================================================
import random
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod
# ----------------------------------------------------------------------------
from gym import Env
from gym import spaces
# ----------------------------------------------------------------------------
from gym_trading.envs.match_engine import Core
# =============================================================================

class BaseEnv(Env, ABC):
    num_steps = 1024 # size of a flow
    high = 1024

    def __init__(self, Flow) -> None:
        super().__init__()
        self.Flow = Flow
        self.core = None
        self.action_space = spaces.Box(0, BaseEnv.high, dtype = np.float32)
        # self.observation_space = spaces.Box
    def setp(self, action: float = 0):
        # return observation, reward, done, info
        pass
    def reset(self):
        '''return the observation of the initial condition'''
        index_random = random.randint(0, self.Flow.shape[1]-BaseEnv.num_steps-1)
        flow = self.Flow.iloc[index_random:index_random+BaseEnv.num_steps,:]
        self.core = Core(flow)
        return 0
    def _get_obs(self):
        pass