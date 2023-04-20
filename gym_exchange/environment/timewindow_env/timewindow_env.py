from gym_exchange import Config
from gym_exchange.environment.base_env.assets.action import Action

from gym_exchange.environment.base_env.interface_env import State  # types
# from gym_exchange.environment.env_interface import State, Observation # types
from gym_exchange.exchange.timewindow_exchange import TimewindowExchange
from gym_exchange.environment.basic_env.basic_env import BasicEnv
from gym_exchange.environment.base_env.base_env import BaseEnv
from gym_exchange.environment.base_env.base_env import BaseEnv

# *************************** 2 *************************** #
class TimewindowEnv(locals()[Config.train_env]):
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        self.exchange = TimewindowExchange()

    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        state, reward, done, info = super().step(action)
        return state, reward, done, info
    # --------------------- 03.01 ---------------------
    def state(self, action: Action) -> State:
        state = super().state(action)
        return state



if __name__ == "__main__":
    env = TimewindowEnv()
    env.reset();
    print("=" * 20 + " ENV RESTED " + "=" * 20)
    for i in range(int(1e6)):
        print("-" * 20 + f'=> {i} <=' + '-' * 20)  # $
        action = Action(direction='bid', quantity_delta=0, price_delta=1)  # $
        encoded_action = action.encoded
        state, reward, done, info = env.step(encoded_action)
        print(f"info: {info}")  # $
        env.render()
        if done:
            env.reset()
            break  # $

