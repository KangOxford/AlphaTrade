import sys; sys.path.append('/Users/kang/AlphaTrade/')


from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange
from gym_exchange.trading_environment.basic_env.assets.renders.plot_render import plot_render
from gym_exchange.trading_environment.basic_env.assets.measure import OrderbookDistance
from gym_exchange.trading_environment.basic_env.assets.info import InfoGenerator

from gym_exchange.trading_environment.base_env.base_env import BaseEnv


# *************************** 2 *************************** #
class BasicEnv(BaseEnv):
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        self.observation_space = self.state_space
        self.exchange = Exchange()

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


# if __name__ == "__main__":
    '''
    # --------------------- 05.01 --------------------- 
    # from stable_baselines3.common.env_checker import check_env
    # env = BaseEnv()
    # check_env(env)
    # print("="*20+" ENV CHECKED "+"="*20)
    # --------------------- 05.02 --------------------- 
    env = BaseEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    # import time;time.sleep(5)
    for i in range(int(1e6)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        # action = Action(direction = 'bid', quantity_delta = 5, price_delta = 1) #$ 03 tested
        # action = Action(direction = 'bid', quantity_delta = 0, price_delta = 1) #$ # 04 testesd
        action = Action(direction = 'bid', quantity_delta = 0, price_delta = 0) #$ 01 tested
        # action = Action(direction = 'ask', quantity_delta = 0, price_delta = 0) #$ 02 tested
        # action = Action(direction = 'ask', quantity_delta = 0, price_delta = 1) #$ 05 tested
        # action = Action(direction = 'ask', quantity_delta = 0, price_delta = 1) #$ 06 tested
        # action = Action(direction = 'bid', quantity_delta = 1, price_delta = 1) #$ 07 tested
        # print(f">>> delta_action: {action}") #$
        # breakpoint() #$
        encoded_action = action.encoded
        state, reward, done, info = env.step(encoded_action)
        # print(f"state: {state}") #$
        # print(f"reward: {reward}") #$
        # print(f"done: {done}") #$
        print(f"info: {info}") #$
        env.render()
        if done:
            env.reset()
            break #$
    '''
    '''
    import numpy as np
    arr = np.loadtxt("/Users/kang/AlphaTrade/gym_exchange/outputs/actions", dtype=np.int64)
    env = BaseEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        # print(f"info: {info}") #$
        print(f"reward: {reward}") #$
        env.render()
        if done:
            env.reset()
            break #$
    '''
    '''
    import numpy as np
    arr = np.loadtxt("/Users/kang/AlphaTrade/gym_exchange/outputs/actions", dtype=np.int64)
    arr = np.repeat(arr, 2, axis=0)
    env = BaseEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        print(f"reward: {reward}") #$
        print(f"info: {info}") #$
        env.render()
        if done:
            env.reset()
            break #$
    '''
if __name__ == "__main__":
    import numpy as np
    arr = np.array([
        [1,1,0],
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
