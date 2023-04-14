import numpy as np
import sys; sys.path.append('/Users/kang/AlphaTrade/')
from gym_exchange import Config

from gym_exchange.data_orderbook_adapter.utils import brief_order_book

from gym_exchange.trading_environment.base_env.assets.reward import RewardGenerator
from gym_exchange.trading_environment.base_env.assets.action import Action
from gym_exchange.trading_environment.base_env.assets.orderflow import OrderFlowGenerator
from gym_exchange.trading_environment.base_env.assets.task import NumLeftProcessor

from gym_exchange.trading_environment.base_env.metrics.vwap import VwapEstimator

from gym_exchange.trading_environment.base_env.interface_env import InterfaceEnv
from gym_exchange.trading_environment.base_env.interface_env import State # types
# from gym_exchange.trading_environment.env_interface import State, Observation # types
from gym_exchange.exchange.basic_exc.autocancel_exchange import Exchange
from gym_exchange.trading_environment.base_env.utils import broadcast_lists
from gym_exchange.trading_environment.base_env.assets.renders.plot_render import plot_render
from gym_exchange.trading_environment.base_env.assets.info import InfoGenerator
from gym_exchange.trading_environment.base_env.metrics.measure import OrderbookDistance


# *************************** 2 *************************** #
class BaseEnv(InterfaceEnv):
    # ========================== 01 ==========================
    def __init__(self):
        super().__init__()
        self.observation_space = self.state_space
        self.exchange = Exchange()

    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.exchange.reset()
        self.init_components()
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        # observation = self.obs_from_state(self.cur_state)
        state = self.cur_state
        assert self.cur_step == 0
        return state
    # ------------------------- 02.01 ------------------------
    def init_components(self):
        self.vwap_estimator = VwapEstimator()
        self.reward_generator = RewardGenerator(p_0 = self.exchange.mid_prices[0]) # Used for Reward
        self.order_flow_generator = OrderFlowGenerator() # Used for Order
        self.num_left_processor = NumLeftProcessor()
        self.info_generator = InfoGenerator()
        self.orderbook_distance = OrderbookDistance()
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""
        # ···················· 02.01.01 ····················
        self.cur_step = 0
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
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        print(action)  #$
        # ···················· 03.00.03 ····················
        decoded_action = Action.decode(action)  # machine code => [side, quantity_delta, price_delta]
        state, reward, done, info = self.state(decoded_action), self.reward, self.done, self.info
        return state, reward, done, info

    # --------------------- 03.01 ---------------------

    def state(self, action: Action) -> State:
        # if self.cur_step == 3718:
        #     print()#$
        print(self.cur_step) #$
        # ···················· 03.01.01 ····················
        # generate_wrapped_order_flow {
        best_ask_bid_dict = {'ask':self.exchange.order_book.get_best_ask(), 'bid':self.exchange.order_book.get_best_bid()}
        order_flows = self.order_flow_generator.step(action, best_ask_bid_dict) # redisual policy inside # price is wrapped into action here # price list is used for PriceDelta, only one side is needed
        order_flow  = order_flows[0]  # order_flows consists of order_flow, auto_cancel
        wrapped_order_flow = self.exchange.time_wrapper(order_flow)
        # generate_wrapped_order_flow }
        self.wrapped_order_flow = wrapped_order_flow
        if self.cur_step == 156:
            print()#$
        print("wrapped_order_flow:",wrapped_order_flow) #$
        print(self.exchange.order_book) #$$
        self.exchange.step(wrapped_order_flow)
        print(self.exchange.order_book) #$$
        # ···················· 03.01.02.01 ····················
        auto_cancel = order_flows[1]  # order_flows consists of order_flow, auto_cancel
        print(auto_cancel) #$
        self.exchange.auto_cancels.add(auto_cancel) # The step of auto cancel would be in the exchange(update_task_list)
        # ···················· 03.01.02.02 ····················
        # auto_cancel2process_list = self.exchange.auto_cancels.step()
        # for auto_cancel2process in auto_cancel2process_list:
        #     self.exchange.step(auto_cancel2process)
        # ···················· 03.01.03 ····················
        state = broadcast_lists(*tuple(map(lambda side: brief_order_book(self.exchange.order_book, side),('ask','bid'))))
        # state = np.array([brief_order_book(self.exchange.order_book, side) for side in ['ask', 'bid']])
        price, quantity = state[:,::2], state[:,1::2]
        state = np.concatenate([price,quantity],axis = 1)
        state = state.reshape(4, Config.price_level).astype(np.int64)
        # ···················· 03.01.04 ····················
        # self.accumulator {
        self.num_left_processor.step(self)
        self.cur_step += 1
        # self.accumulator }
        return state
    # --------------------- 03.02 ---------------------
    @property
    def reward(self):
        self.reward_generator.update(self.exchange.executed_pairs_recoder.market_agent_executed_pairs_in_last_step, self.exchange.mid_prices[-1])
        reward = self.reward_generator.step()
        return reward
    # --------------------- 03.03  ---------------------
    @property
    def done(self):
        # if self.num_left_processor.num_left <= 0 or self.cur_step >= Config.max_horizon : return True
        if self.num_left_processor.num_left <= 0 or self.cur_step >= Config.max_horizon :
            return True #$
        else: return False

    # --------------------- 03.04 ---------------------
    @property
    def info(self):
        returned_info = self.info_generator.step(self)
        return returned_info



    # ========================== 04 ==========================
    def render(self, mode = 'human'):
        '''for render method'''
        # base_env_render(self)
        if self.done:
            plot_render(self)
        pass


if __name__ == "__main__":
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
    import numpy as np
    arr = np.array([
        [1,1,0],
        [1,1,0]
    ])
    # arr = np.array([
    #     [0,1,0],
    #     [0,1,0]
    # ])
    arr = np.repeat(arr, 2000, axis=0)
    env = BaseEnv()
    env.reset();print("="*20+" ENV RESTED "+"="*20)
    sum_reward = 0
    state, reward, done, info = env.step([1,3,0])# for testing
    for i in range(len(arr)):
        print("-"*20 + f'=> {i} <=' +'-'*20) #$
        encoded_action = arr[i]
        # if i == 320:
        #     breakpoint()
        state, reward, done, info = env.step(encoded_action)
        print(f"reward: {reward}") #$
        print(f"info: {info}") #$
        sum_reward += reward
        env.render()
        if done:
            env.reset()
            break #$
    print(sum_reward)
