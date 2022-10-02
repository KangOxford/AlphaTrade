import numpy as np
from gym_trading import utils
from gym_trading.envs.broker import Flag
from gym_trading.data.data_pipeline import Debug
Debug.if_return_single_flie = True # if True then return Flow_list # Flase if you want to debug
Debug.if_return_part_data = False # default for debugging
Debug.if_whole_data = False # load the whole dataset

class Twap():
    def __init__(self, string, venv, verbose):
            self.string = string
            self.venv = venv
            self.verbose = verbose
            self.num_env = venv.num_envs
    def predict(self, obs):
        num_left, step_left = utils.get_numleft_and_stepleft(obs) # TODO need to be implemented according to venv
        action = Flag.num2liquidate // Flag.max_episode_steps + 35 # doesn't work
        action = Flag.num2liquidate // Flag.max_episode_steps + 40 # work
        # if step_left // 3 == step_left / 3:
        #     action += 1 #tbd only for (2000, 600)
        states = None # TODO need to be implemented
        action = [action] * self.num_env 
        return action, states

if __name__== "__main__":
    venv = utils.get_venv(string = "OptimalLiquidation-v1", num_env = 1)
    model = Twap("common", venv, verbose=1)
    obs = venv.reset()
    step = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = venv.step(action)
        step += 1
        print("step, {0}; action {1}".format(step, action))
        try:print("num_left, ", info[0]['num_left'])
        except: pass
        if dones: break