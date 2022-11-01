import numpy as np
from gym import Env
from gym import spaces
from gym_exchange.trading_environment import Config
from gym_exchange.trading_environment.utils.metric import VwapEstimator
from gym_exchange.trading_environment.env_interface import SpaceParams
from gym_exchange.trading_environment.env_interface import EnvInterface
from gym_exchange.trading_environment.env_interface import State, Observation, Action

# from gym_exchange.trading_environment.action import SimpleAction
# from gym_exchange.trading_environment.action import BaseAction

class BaseSpaceParams(SpaceParams):
    class Observation:
        price_delta_size = 7
        side_size = 2
        quantity_size = 2*(Config.num2liquidate//Config.max_horizon +1) + 1

@EnvInterface.register
class BaseEnv():
    # ========================== 01 ==========================
    def __init__(self):
        super(BaseEnv, self).__init__()
        self.observation_space = EnvInterface.state_space
        self.vwap_estimator = VwapEstimator() # Used for info
    
    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        return self.obs_from_state(self.cur_state)
    # ------------------------- 02.01 ------------------------
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""
        
    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        return self.observation, self.reward, self.done, self.info
    # --------------------- 03.01 ---------------------
    @property
    def observation(self):
        pass
    # ···················· 03.01.01 ···················· 
    def obs_from_state(self, state: State) -> Observation:
        """Sample observation for given state."""
        return state
    # --------------------- 03.02 ---------------------
    @property
    def reward(self):
        difference = self.vwap_estimator.difference
        return reward
    # --------------------- 03.03  ---------------------
    @property
    def done(self):
        pass
    # --------------------- 03.04 ---------------------  
    @property
    def info(self):
        pass