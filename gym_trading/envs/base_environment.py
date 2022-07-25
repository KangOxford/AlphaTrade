# =============================================================================
import random
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod
# ----------------------------------------------------------------------------
from gym import Env
from gym import spaces
# ----------------------------------------------------------------------------
from gym_trading.utils import * 
from gym_trading.envs.match_engine import Core
from gym_trading.envs.match_engine import Broker, Utils
# =============================================================================



class BaseEnv(Env, ABC):
    num_steps = 1024 # size of a flow
    high = 1024
    max_quantity = 6000
    max_price = 31620700
    min_price = 31120200
    num2liuquidate = 300
    cost_parameter = 0.01

    def __init__(self, Flow) -> None:
        super().__init__()
        self.Flow = Flow
        self.core = None
        self.price_list = None
        self.action_space = spaces.Box(0, BaseEnv.high,shape =(1,),dtype = np.int16)
        self.observation_space = spaces.Dict({
            'price':spaces.Box(BaseEnv.min_price,BaseEnv.max_price,shape=(10,),dtype=np.int16),
            'quantity':spaces.Box(0,BaseEnv.max_quantity,shape=(10,), dtype=np.int64)
            })
        # %%
        price = spaces.Box(0,10,shape=(10,),dtype=np.int16)
        sample = price.sample()
        # print(sample)
        arr = np.array([0 for _ in range(10)]).astype(np.int16)
        arr in price
        # %%
        self.num_left = BaseEnv.num2liuquidate
        self.done = False
        self.running_reward = 0
        self.init_reward = 0
        self.info = {}
    
    def setp(self, action: float = 0):
        # return observation, reward, done, info
        observation = self._get_obs(acion)
        reward = self._get_reward(acion)
        done = self._get_done(acion)
        info = self._get_info(acion)
        return  observation, reward, done, info
        
    def _get_obs(self):
        pass
    def _get_done(self,acion):
        return self.done
    def _get_inventory_cost(self):
        inventory = 
        return BaseEnv.cost_parameter * inventory * inventory
    def _get_reward(self,acion):
        if not self.done:
            return 0
        else:
            return self.running_reward - self._get_inventory_cost()
    def _get_info(self,acion):
        return self.info      
    
    def reset(self):
        '''return the observation of the initial condition'''
        self.reset_states()
        index_random = random.randint(0, self.Flow.shape[0]-BaseEnv.num_steps-1)
        flow = self.Flow.iloc[index_random:index_random+BaseEnv.num_steps,:]
        self.core = Core(flow)
        stream =  flow.iloc[0,:]
        self._set_init_reward(stream)
        init_price = np.array(get_price_from_stream(stream)).astype(np.int64)
        init_quant = np.array(get_quantity_from_stream(stream)).astype(np.int64)
        self.running_reward += self.calculate_reward()
        init_obs= {
            'price' : init_price,
            'quantity' : init_quant
            }
        return init_obs
    def reset_states(self):
        self.running_reward = 0
        self.done = False
        self.num_left = BaseEnv.num2liuquidate



    def calculate_reward(self):
        return 1 # TODO
    def _set_init_reward(self, stream):
        num = BaseEnv.num2liuquidate
        obs = Utils.from_series2pair(stream)
        level = Broker._level_market_order_liquidating(num, obs)
        if level == 0:
            self.init_reward = 0 
        elif level == -999:
            reward,consumed = 0,0
            for i in range(len(obs)):
                reward += obs[i][0] * obs[i][1]
                consumed += obs[i][1]
            inventory_cost=(num-consumed)*(num-consumed)*BaseEnv.cost_parameter
            self.init_reward = reward - inventory_cost
        else:
            reward,consumed = 0,0
            for i in range(level-1):
                reward += obs[i][0] * obs[i][1]
                consumed += obs[i][1]
            reward += obs[level-1][0] * (num - consumed)
            self.init_reward = reward
    
if __name__=="__main__":
    from gym_trading.data.data_pipeline import ExternalData
    Flow = ExternalData.get_sample_order_book_data()
    env = BaseEnv(Flow)
    obs = env.reset()