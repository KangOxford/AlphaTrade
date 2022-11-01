import numpy as np
from gym import spaces
from gym_exchange.trading_environment import Config
from gym_exchange.trading_environment.utils.metric import VwapEstimator
from gym_exchange.trading_environment.env_interface import SpaceParams
from gym_exchange.trading_environment.base_environment import BaseEnv

class TradeSpaceParams(SpaceParams):
    class Observation:
        low=np.concatenate(([0, 0], np.full(self._noise_length, -np.inf))),
        high=np.concatenate(
            ([size - 1, size - 1], np.full(self._noise_length, np.inf)),
        )


class TradeEnv(BaseEnv):
    def __init__(self):
        super(TradeEnv, self).__init__()
        self.observation_space=spaces.Box(
            low=TradeSpaceParams.Observation.low,
            high=TradeSpaceParams.Observation.high,
            dtype=np.int32,
        )
           
    
    def initial_state(self) -> np.ndarray:
        n = self._size
        corners = np.array([[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1]])
        return corners[self.rand_state.randint(4)]
    
    def obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """Returns (x, y) concatenated with Gaussian noise."""
        noise_vector = self.rand_state.randn(self._noise_length)
        return np.concatenate([state, noise_vector]).astype(np.float32)
    

