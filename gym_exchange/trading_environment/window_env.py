import numpy as np
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
    def __init__(self):
        super(WindowEnv, self).__init__()
        self.observation_space=spaces.Box(
            low   = WindowParams.Observation.low,
            high  = WindowParams.Observation.high,
            shape = WindowParams.Observation.shape,
            dtype = np.int32,
        )
        
    def observation(self, action): 
        state = self.state(action)
        return self.obs_from_state(state)
    # ···················· 03.01.01 ···················· 
    def obs_from_state(self, state: State) -> Observation:
        """Sample observation for given state."""
        return state
           
    
    def initial_state(self) -> np.ndarray:
        n = self._size
        corners = np.array([[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1]])
        return corners[self.rand_state.randint(4)]
    
    def obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """Returns (x, y) concatenated with Gaussian noise."""
        noise_vector = self.rand_state.randn(self._noise_length)
        return np.concatenate([state, noise_vector]).astype(np.float32)
    

