from abc import ABC
# from abc import abstractmethod
from gym import spaces
import numpy as np

# =============================================================================
from gym import Env
from match_engine import MatchEngine
# =============================================================================

class BaseEnvironment(Env, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.min_action = 0.0
        self.max_action = 100.0
        self.min_position = 
        high = np.array(
            [100],
            dtype = np.float32
        )
        self.action_space = spaces.Box(0, high, dtype = np.float32)
        self.observation_space = spaces.Box
    def setp(self, action: float = 0):
        # return super().step()
        pass
    def reset(self):
        '''return the observation of the initial condition'''
        return super().reset()
    def _get_obs(self):
        pass
    # def seed(self):
    #     pass
    # def close(self) -> None:
    #     return super().close()
    # def render(self):
    #     pass