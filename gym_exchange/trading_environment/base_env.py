import numpy as np
from gym import Env
from gym import spaces
from gym_exchange.data_orderbook_adapter.utils import brief_order_book
from gym_exchange.exchange.exchange import Exchange
from gym_exchange.trading_environment import Config
from gym_exchange.trading_environment.reward import RewardGenerator
from gym_exchange.trading_environment.action import Action
from gym_exchange.trading_environment.action import BaseAction
from gym_exchange.trading_environment.utils.metric import VwapEstimator
from gym_exchange.trading_environment.utils.action_wrapper import action_wrapper
from gym_exchange.trading_environment.env_interface import SpaceParams
from gym_exchange.trading_environment.env_interface import EnvInterface
from gym_exchange.trading_environment.env_interface import State, Observation, Action

# from gym_exchange.trading_environment.action import SimpleAction
# from gym_exchange.trading_environment.action import BaseAction

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
        self.vwap_estimator = VwapEstimator() # Used for info
        # self.reward_generator = RewardGenerator(self) # Used for Reward
        self.reward_generator = RewardGenerator() # Used for Reward
        # self.state_generator = StateGenerator() # Used for State
    
    # ========================== 02 ==========================
    def reset(self):
        """Reset episode and return initial observation."""
        self.exchange.reset()
        self.cur_state = self.initial_state()
        assert self.cur_state in self.state_space, f"unexpected state {self.cur_state}"
        observation = self.obs_from_state(self.cur_state)
        return observation
    # ------------------------- 02.01 ------------------------
    def initial_state(self) -> State:
        """Samples from the initial state distribution."""
        # ···················· 02.01.01 ···················· 
        self.cur_step = 0
        self.num_left = Config.num2liquidate
        order_book = self.exchange.order_book 
        asks, bids = brief_order_book(order_book, 'ask'), brief_order_book(order_book, 'bid')
        asks, bids = np.array(asks), np.array(bids)
        
        # ···················· 02.01.02 ···················· 
        price_indexes, quantity_indexes = [2*i for i in range(Config.price_level)], [2*i +1 for i in range(Config.price_level)]
        asks = np.concatenate([asks[price_indexes],asks[quantity_indexes]]).reshape(-1,Config.price_level)
        bids = np.concatenate([bids[price_indexes],bids[quantity_indexes]]).reshape(-1,Config.price_level)
        state = np.concatenate([asks, bids])
        state = state.astype(np.int64)
        assert state.shape == (4, Config.price_level)
        return state
        
    # ========================== 03 ==========================
    def step(self, action):
        '''input : action
           return: observation, reward, done, info'''
        # self.prev_state = self.cur_state
        self.exchange.step(action.to_exchange)
        observation, reward, done, info = self.observation, self.reward, self.done, self.info
        self.accumulator()
        return observation, reward, done, info
    def accumulator(self):
        self.num_left -= self.cur_action
        self.cur_step += 1
    # --------------------- 03.01 ---------------------
    @property
    def observation(self):
        # self.prev_state = self.cur_state
        
        return self.obs_from_state(self.cur_state)
    # ···················· 03.01.01 ···················· 
    def obs_from_state(self, state: State) -> Observation:
        """Sample observation for given state."""
        return state
    def state(self, action: Action) -> State:
        action = action_wrapper(action)
    # --------------------- 03.02 ---------------------
    @property
    def reward(self):
        reward = self.reward_generator.step()
        return reward
    # --------------------- 03.03  ---------------------
    @property
    def done(self):
        if self.num_left <= 0 or self.cur_step >= Config.max_horizon : return True
        else : return False
    # --------------------- 03.04 ---------------------  
    @property
    def info(self):
        MarketVwap = self.vwap_estimator.market_vwap
        AgentVwap = self.vwap_estimator.agent_vwap
        return {
            "MarketVwap"  :MarketVwap,
            "AgentVwap"   :AgentVwap,
            "VwapSlippage":MarketVwap - AgentVwap
        }
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
        action = Action(side = 'ask', quantity = 1, price_delta = 1)
        observation, reward, done, info = env.step(action.to_array)
        env.render()
        if done:
            env.reset()