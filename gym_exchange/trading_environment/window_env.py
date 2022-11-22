import numpy as np
import pandas as pd
from gym import spaces
from gym_exchange import Config
from gym_exchange.trading_environment.env_interface import SpaceParams
from gym_exchange.trading_environment.skip_env import SkipEnv

from typing import TypeVar
Action = TypeVar("Action")
State = TypeVar("State")
Observation = TypeVar("Observation")

class WindowParams(SpaceParams):
    class Observation:
        low  = SpaceParams.State.low,
        high = SpaceParams.State.high
        shape= SpaceParams.State.shape

Config.window_size

class WindowEnv(SkipEnv):
    # ========================== 01 ==========================
    def __init__(self):
        super(WindowEnv, self).__init__()
        self.observation_space=spaces.Box(
            low   = WindowParams.Observation.low,
            high  = WindowParams.Observation.high,
            shape = WindowParams.Observation.shape,
            dtype = np.int32,
        )
        
    
    # ========================== 03 ==========================
    def step(self, action):
        state, reward, done, info = super().step(action)
        observation = self.observation(action)
        return observation, reward, done, info
        
    
    # # ----------------- 03.01 ----------------- 
    # def observation(self, action): 
    #     # ···················· 03.01.01 ···················· 
    #     state = self.state(action)
    #     return self.obs_from_state(state)
    
    def obs_from_state(self, state: State) -> Observation:
        """Sample observation for given state."""
        def concat_states(state, state_memos):
            concated_states = pd.concatenate([state, state_memos]) 
            # TODO not sure for this step, havent been tested
            return concated_states
        # ···················· 03.01.02 ···················· 
        state_memos= self.state_memos[-Config.window_size-1:]
        # ···················· 03.01.03 ···················· 
        concated_states = concat_states(state, state_memos)
        return concated_states

