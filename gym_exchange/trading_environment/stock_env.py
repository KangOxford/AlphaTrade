import numpy as np
from gym import Env
from gym import spaces
# from gym_exchange.trading_environment.metric import Vwap
# from gym_exchange.trading_environment.action import SimpleAction
# from gym_exchange.trading_environment.action import BaseAction
from gym_exchange.trading_environment.env_interface import EnvInterface

@EnvInterface.register
class BaseEnv():
    def __init__(self):
        super(BaseEnv, self).__init__()
            state_space=spaces.MultiDiscrete([size, size]),
            action_space=spaces.Discrete(5),
            observation_space=spaces.Box(
                low=np.concatenate(([0, 0], np.full(self._noise_length, -np.inf))),
                high=np.concatenate(
                    ([size - 1, size - 1], np.full(self._noise_length, np.inf)),
                ),
                dtype=np.float32,
            )
    
    def initial_state(self) -> np.ndarray:
        n = self._size
        corners = np.array([[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1]])
        return corners[self.rand_state.randint(4)]
    
    def obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """Returns (x, y) concatenated with Gaussian noise."""
        noise_vector = self.rand_state.randn(self._noise_length)
        return np.concatenate([state, noise_vector]).astype(np.float32)