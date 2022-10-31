import numpy as np
from gym import Env
from gym import spaces
from gym_exchange.trading_environment.metric import Vwap
from gym_exchange.trading_environment.action import SimpleAction
from gym_exchange.trading_environment.action import BaseAction
from gym_exchange.trading_environment.base_env import BaseEnv 

@EnvInterface.register
class StockEnv():
    def __init__(self):
        super(StockEnv, self).__init__()
        pass