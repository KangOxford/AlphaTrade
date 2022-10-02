import numpy as np
from gym_trading import utils
from gym_trading.envs.broker import Flag

class Twap():
    def __init__(self, string, ):
            self.string = string
    def predict(self, obs):
        return action, states

if __name__== "__main__":
    venv = utils.get_venv(string = "OptimalLiquidation-v1", num_env = 1)
    model = Twap("common", venv, verbose=1)
    obs = venv.reset()
    action, _states = model.predict(obs)