import sys; sys.path.append('/Users/kang/AlphaTrade/')


from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange
from gym_exchange.environment.basic_env.assets.renders.plot_render import plot_render
from gym_exchange.environment.basic_env.assets.measure import OrderbookDistance
from gym_exchange.environment.basic_env.assets.info import InfoGenerator

from gym_exchange.environment.base_env.base_env import BaseEnv
from gym_exchange import Config


# *************************** 2 *************************** #
class BasicEnv(BaseEnv):
    # ========================== 01 ==========================
    def __init__(self):
        if 'exchange' not in dir(self):
            self.exchange = Exchange()
        super().__init__()
        self.observation_space = self.state_space

    # ========================== 02 ==========================
    # ========================= RESET ========================
    # ------------------------- 02.01 ------------------------
    def init_components(self):
        super().init_components()
        self.info_generator = InfoGenerator()
        self.orderbook_distance = OrderbookDistance()

    # ========================== 03 ==========================
    # --------------------- 03.04 ---------------------
    @property
    def info(self):
        super_info = super().info
        returned_info = self.info_generator.step(self)
        return returned_info



    # ========================== 04 ==========================
    def render(self, mode = 'human'):
        '''for render method'''
        super().render()
        if self.done:
            plot_render(self)
        pass


if __name__ == "__main__":
    import numpy as np
    arr = np.array([
        [1,Config.quantity_size_one_side,0],
    ])
    # arr = np.array([
    #     [0,1,0],
    #     [0,1,0]
    # ])
    arr = np.repeat(arr, 3000, axis=0)
    env = BasicEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    sum_reward = 0
    # state, reward, done, info = env.step([0,1,0])# for testing
    # state, reward, done, info = env.step([1,1,0])# for testing
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        print(f"reward: {reward}") #$
        print(f"info: {info}") #$
        sum_reward += reward
        # env.render()
        if done:
            env.reset()
            break #$
    print(sum_reward)
