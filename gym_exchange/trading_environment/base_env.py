import numpy as np

from gym_exchange.data_orderbook_adapter.utils import brief_order_book

from gym_exchange.exchange.auto_cancel_exchange import Exchange

from gym_exchange import Config

from gym_exchange.trading_environment.assets.reward import RewardGenerator
from gym_exchange.trading_environment.assets.action import Action
from gym_exchange.trading_environment.assets.action_wrapper import  OrderFlowGenerator
from gym_exchange.trading_environment.assets.task import NumLeftProcessor

from gym_exchange.trading_environment.metrics.vwap import VwapEstimator

# from gym_exchange.trading_environment.utils.action_wrapper import action_wrapper
from gym_exchange.trading_environment.env_interface import SpaceParams, EnvInterface
from gym_exchange.trading_environment.env_interface import State, Observation # types

class BaseSpaceParams(SpaceParams):
    class Observation:
        price_delta_size = 7
        side_size = 2
        quantity_size = 2*(Config.num2liquidate//Config.max_horizon +1) + 1

class BaseEnv(EnvInterface):
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
        observation = self.obs_from_state(self.cur_state)
        return observation
    # ------------------------- 02.01 ------------------------
    def init_components(self):
        self.vwap_estimator = VwapEstimator()
        self.reward_generator = RewardGenerator(p_0 = self.exchange.mid_prices[0]) # Used for Reward
        # self.state_generator = StateGenerator() # Used for State
        self.order_flow_generator = OrderFlowGenerator() # Used for Order
        self.num_left_processor = NumLeftProcessor()
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

        # ···················· 03.00.03 ···················· 
        observation, reward, done, info = self.observation(action), self.reward, self.done, self.info
        self.accumulator()
        return observation, reward, done, info
    def accumulator(self):
        self.num_left_processor.step(self)
        self.cur_step += 1
    # --------------------- 03.01 ---------------------
    def observation(self, action): 
        state = self.state(action)
        return self.obs_from_state(state)
    # ···················· 03.01.01 ···················· 
    def obs_from_state(self, state: State) -> Observation:
        """Sample observation for given state."""
        return state
    
    def state(self, action: Action) -> State:
        # ···················· 03.00.01 ····················    
        price_list = np.array(brief_order_book(self.exchange.order_book, 'bid' if action[0] == 1 else 'ask'))[::2] # slice all odd numbers    
        order_flows = self.order_flow_generator.step(action, price_list)# price list is used for PriceDelta, only one side is needed
        order_flow  = order_flows[0] # order_flows consists of order_flow, auto_cancel
        wrapped_order_flow = self.exchange.time_wrapper(order_flow)
        self.exchange.step(wrapped_order_flow)
        # ···················· 03.00.02 ···················· 
        auto_cancel = order_flows[1] # order_flows consists of order_flow, auto_cancel
        self.exchange.auto_cancels.add(auto_cancel) 
        # ···················· 03.00.03 ····················
        print(f"self.exchange.index: {self.exchange.index}")#$
        state = np.array([brief_order_book(self.exchange.order_book, side) for side in ['ask', 'bid']])
        price, quantity = state[:,::2], state[:,1::2]
        state = np.concatenate([price,quantity],axis = 1)
        state = state.reshape(4, Config.price_level).astype(np.int64)
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
        if self.num_left_processor.num_left <= 0 or self.cur_step >= Config.max_horizon : return True
        else : return False
    # --------------------- 03.04 ---------------------  
    @property
    def info(self):
        self.vwap_estimator.update(self.exchange.executed_pairs_recoder, self.done)
        step_vwap_info_dict, epoch_vwap_info_dict = self.vwap_estimator.step()
        print(f"self.done:{self.done}")#$
        if epoch_vwap_info_dict is None:
            return {}
        else:
            return {**epoch_vwap_info_dict}
        
        # if epoch_vwap_info_dict is None:
        #     return {**step_vwap_info_dict}
        # else:
        #     return {**step_vwap_info_dict, **epoch_vwap_info_dict}
        '''in an liquidation task the market_vwap ought to be
        higher, as they are not eagle to takt the liquidity, 
        and can be executed at higher price.'''
    # ========================== 04 ==========================
    def render(self, mode = 'human'):
        '''for render method'''
        
if __name__ == "__main__":
    # --------------------- 05.01 --------------------- 
    from stable_baselines3.common.env_checker import check_env
    env = BaseEnv()
    check_env(env)
    # --------------------- 05.02 --------------------- 
    env = BaseEnv()
    env.reset()
    for i in range(int(1e6)):
        action = Action(side = 'bid', quantity = 1, price_delta = 1)
        observation, reward, done, info = env.step(action.to_array)
        env.render()
        if done:
            env.reset()