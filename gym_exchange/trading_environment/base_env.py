import numpy as np
from gym import Env
from gym import spaces
from gym_exchange.trading_environment.metric import Vwap
from gym_exchange.trading_environment.action import SimpleAction
from gym_exchange.trading_environment.action import BaseAction
from gym_exchange.trading_environment.base_env import BaseEnv 

@EnvInterface.register
class BaseEnv():
    def __init__(self):
        super(BaseEnv, self).__init__()
        observation_space=spaces.Box(
            low=np.concatenate(([0, 0], np.full(self._noise_length, -np.inf))),
            high=np.concatenate(
                ([size - 1, size - 1], np.full(self._noise_length, np.inf)),
            ),
            dtype=np.float32,
        ),