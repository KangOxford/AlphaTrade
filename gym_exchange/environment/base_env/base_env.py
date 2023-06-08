import sys; sys.path.append('/Users/kang/AlphaTrade/')
import numpy as np
import gym
from gym import spaces
from gym_exchange import Config
from gym_exchange import SpaceParams
from gym_exchange.data_orderbook_adapter.utils import brief_order_book
from gym_exchange.environment.base_env.assets.reward import RewardGenerator
from gym_exchange.environment.base_env.assets.action import OrderFlowGenerator
from gym_exchange.environment.base_env.assets.task import NumLeftProcessor
from gym_exchange.environment.base_env.assets.task import NumHoldProcessor

from gym_exchange.environment.base_env.assets.vwap_info import VwapEstimator
from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange
from gym_exchange.environment.base_env.utils import broadcast_lists


# *************************** 2 *************************** #
class BaseEnv(gym.Env):
    # ========================== 01 ==========================
    def __init__(self):
        if 'exchange' not in dir(self):
            self.exchange = Exchange()
        super().__init__()
        self.action_space, self.state_space = self.space_definition()
        self.observation_space = self.state_space
    def space_definition(self):
        action_space = spaces.Box(
              low   = SpaceParams.Action.low,
              high  = SpaceParams.Action.high,
              shape = SpaceParams.Action.shape,
              dtype = np.int32,
        )
        state_space = spaces.Box(
              low   = SpaceParams.State.low,
              high  = SpaceParams.State.high,
              shape = SpaceParams.State.shape,
              dtype = np.int64,
        )
        return action_space, state_space

    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.exchange.reset()
        self.init_components()
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, \
            f"unexpected state {self.cur_state}, \nwith shape {self.cur_state.shape}, \nshould be in {self.state_space}"
        state = self.cur_state
        assert self.cur_step == 0
        print("env reset") #$
        return state
    # ------------------------- 02.01 ------------------------
    def init_components(self):
        self.cur_step = 0
        self.task_info = np.array(
            [0, Config.max_horizon, Config.num2liquidate, Config.num2liquidate]).reshape((4, 1))
        self.vwap_estimator = VwapEstimator()
        self.reward_generator = RewardGenerator(p_0 = self.exchange.mid_prices[0]) # Used for Reward
        self.order_flow_generator = OrderFlowGenerator() # Used for Order
        self.num_left_processor = NumLeftProcessor()
        self.num_hold_processor = NumHoldProcessor()
    def initial_state(self):
        """Samples from the initial state distribution."""
        # ···················· 02.01.01 ····················
        order_book = self.exchange.order_book
        asks, bids = brief_order_book(order_book, 'ask'), brief_order_book(order_book, 'bid')
        asks, bids = np.array(asks), np.array(bids)

        # ···················· 02.01.02 ····················
        price_indexes, quantity_indexes = [2*i for i in range(Config.price_level)], [2*i +1 for i in range(Config.price_level)]
        asks = np.concatenate([asks[price_indexes],asks[quantity_indexes]]).reshape(-1,Config.price_level)
        bids = np.concatenate([bids[price_indexes],bids[quantity_indexes]]).reshape(-1,Config.price_level)
        state = np.concatenate([asks, bids]) # fixed sequence: first ask, then bid
        state = state.astype(np.int64)
        assert state.shape == (4, Config.price_level)
        return state

    # ========================== 03 ==========================
    def step(self, delta_action):
        '''input : action
           return: observation, reward, done, info'''
        print(f"delta_action {delta_action}") #$
        state, reward, done, info = self.state(delta_action), self.reward, self.done, self.info
        return state, reward, done, info

    # --------------------- 03.01 ---------------------

    def state(self, action):
        if self.cur_step == 87:
            print()
        kind = 'limit_order'
        if self.cur_step == Config.max_horizon-1:
            print(f"self.cur_step,{self.cur_step} == Config.max_horizon-1,{Config.max_horizon-1}")#$
            num_left = self.num_left_processor.num_left
            action = [action[0],num_left,1] # aggressive order
            kind = 'market_order'
        # ···················· 03.01.01 ····················
        # generate_wrapped_order_flow {
        best_ask_bid_dict = {'ask':self.exchange.order_book.get_best_ask(), 'bid':self.exchange.order_book.get_best_bid()}
        # order_flows = self.order_flow_generator.step(action, best_ask_bid_dict) # redisual policy inside # price is wrapped into action here # price list is used for PriceDelta, only one side is needed
        order_flows = self.order_flow_generator.step(action, best_ask_bid_dict, self.num_hold_processor.num_hold, kind = kind) # redisual policy inside # price is wrapped into action here # price list is used for PriceDelta, only one side is needed
        # generate_wrapped_order_flow }
        # ···················· 03.01.02 ····················
        self.exchange.step(order_flows)
        # ···················· 03.01.03 ····················
        state = broadcast_lists(*tuple(map(lambda side: brief_order_book(self.exchange.order_book, side),('ask','bid'))))
        price, quantity = state[:,::2], state[:,1::2]
        state = np.concatenate([price,quantity],axis = 1)
        state = state.reshape(4, Config.price_level).astype(np.int64)
        # ···················· 03.01.04 ····················
        # current_step, max_horizon, num_left, num2sell
        # broadcast from (4, 10) to (4, 11)
        self.task_info = np.array([self.cur_step, Config.max_horizon, self.num_left_processor.num_left, Config.num2liquidate]).reshape((4, 1))
        # tobe_appended = np.array([self.cur_step, Config.max_horizon, self.num_left_processor.num_left, Config.num2liquidate]).reshape((4, 1))
        # state = np.hstack((tobe_appended, state))
        # ···················· 03.01.04 ····················
        # self.accumulator {
        self.num_left_processor.step(self)
        self.num_hold_processor.step(self)
        self.cur_step += 1
        # self.accumulator }
        return state
    # --------------------- 03.02 ---------------------
    @property
    def reward(self):
        if self.cur_step == Config.max_horizon:
            print()#$
        self.reward_generator.update(self.exchange.executed_pairs_recoder.market_agent_executed_pairs_in_last_step, self.exchange.mid_prices[-1])
        reward = self.reward_generator.step()
        # if self.done:
        #     penalty = Config.cost_parameter * (self.exchange.mid_prices[-1] / Config.lobster_scaling * self.num_left_processor.num_left) ** 2
        #     reward -= penalty
        # reward /= 837732.857874494 #$ for scaling
        # reward /= 1070108.357874494  #$ for scaling
        return reward
    # --------------------- 03.03  ---------------------
    @property
    def done(self):
        if self.num_left_processor.num_left == 0 or self.cur_step == Config.max_horizon : return True
        else: return False

    # --------------------- 03.04 ---------------------
    @property
    def info(self):
        return {}

    # ========================== 04 ==========================
    def render(self, mode = 'human'):
        '''for render method'''
        pass

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
    env = BaseEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    sum_reward = 0
    # state, reward, done, info = env.step([1,3,0])# for testing
    for i in range(len(arr)):
        # print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        # print(f"reward: {reward}") #$
        # print(f"info: {info}") #$
        sum_reward += reward
        env.render()
        if done:
            env.reset()
            break #$
    print(sum_reward)
