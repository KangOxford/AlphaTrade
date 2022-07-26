# =============================================================================
import random
import numpy as np
import pandas as pd
from abc import ABC
from abc import abstractmethod
# -----------------------------------------------------------------------------
from gym import Env
from gym import spaces
# -----------------------------------------------------------------------------
from gym_trading.utils import * 
from gym_trading.envs.match_engine import Core
from gym_trading.envs.match_engine import Broker, Utils
# =============================================================================



class BaseEnv(Env, ABC):
    num_steps = 1024 # size of a flow
    max_action = 30
    max_quantity = 6000
    max_price = 31620700
    min_price = 31120200
    num2liuquidate = 300
    cost_parameter = 0.01
    
# ============================  INIT  =========================================
    def __init__(self, Flow) -> None:
        super().__init__()
        self.Flow = Flow
        self.core = None
        self.price_list = None
        self.action_space = spaces.Box(0, BaseEnv.max_action,shape =(1,),dtype = np.int16)
        self.observation_space = spaces.Dict({
            'price':spaces.Box(low=BaseEnv.min_price,high=BaseEnv.max_price,shape=(10,),dtype=np.int32),
            'quantity':spaces.Box(low=0,high=BaseEnv.max_quantity,shape=(10,), dtype=np.int32)
            })
        # ---------------------
        self.num_left = None
        self.done = False
        self.num_step = False
        self.running_reward = 0
        self.init_reward = 0
        self.info = {}
        self.previous_obs = None
    
# ============================  STEP  =========================================
    def step(self, action):
        ''' return observation, reward, done, info '''
        observation = self._get_obs(action)
        num_executed = self.core.get_executed_quantity() 
        self.num_left -= num_executed 
        self.running_reward += self._get_each_running_reward() 
        # ---------------------
        reward = self._get_reward(action)
        done = self._get_done(action)
        info = self._get_info(action)
        return  observation, reward, done, info
    # ------  1/4.OBS  ------
    def _get_obs(self, num):
        obs_ = self.core.step(num)[0] # dtype:series
        obs = from_series2obs(obs_)
        self.previous_obs = obs 
        return obs
    # ------ 2/4.DONE ------
    def _get_done(self,acion):
        '''get & set done'''
        if self.num_left <= 0 or self.num_step >= BaseEnv.num_steps:
            self.done = True
        return self.done
    # ------ 3/4.REWARD  ------
    def _get_inventory_cost(self):
        inventory = self.num_left
        return BaseEnv.cost_parameter * inventory * inventory
    def _get_reward(self,acion):
        if not self.done:
            return 0
        else:
            return self.running_reward - self._get_inventory_cost()
    def _get_each_running_reward(self):
        pairs = self.core.get_executed_pairs() # TODO
        lst_pairs = np.array(from_pairs2lst_pairs(pairs))
        return -1 * sum(lst_pairs[0]*lst_pairs[1]) 
    # ------ 4/4.INFO ------
    def _get_info(self,acion):
        return self.info   
# =============================================================================
 
# =============================  RESET  =======================================   
    def reset(self):
        '''return the observation of the initial condition'''
        self.reset_states()
        index_random = random.randint(0, self.Flow.shape[0]-BaseEnv.num_steps-1)
        flow = self.Flow.iloc[index_random:index_random+BaseEnv.num_steps,:]
        self.core = Core(flow)
        stream =  flow.iloc[0,:]
        self._set_init_reward(stream)
        init_obs = self._get_init_obs(stream)
        self.previous_obs = init_obs
        return init_obs
    def reset_states(self):
        self.running_reward = 0
        self.done = False
        self.num_left = BaseEnv.num2liuquidate
        self.num_step = 0
    def _get_init_obs(self, stream):
        init_price = np.array(get_price_from_stream(stream)).astype(np.int32)
        init_quant = np.array(get_quantity_from_stream(stream)).astype(np.int32)
        init_obs= {
            'price' : init_price,
            'quantity' : init_quant
            }
        return init_obs        
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
# =============================================================================



    
if __name__=="__main__":
    from gym_trading.data.data_pipeline import ExternalData
    Flow = ExternalData.get_sample_order_book_data()
    env = BaseEnv(Flow)
    obs = env.reset()
    action = 3
    observation, reward, done, info = env.step(3)
    print(0)