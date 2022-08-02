# =============================================================================
import time
import random
import warnings
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
warnings.filterwarnings("ignore")
# =============================================================================



class BaseEnv(Env, ABC):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    
    max_action = 30
    max_quantity = 6000
    max_price = 31620700
    min_price = 31120200
    num2liquidate = 300
    cost_parameter = int(1e8)
    
# ============================  INIT  =========================================
    def __init__(self, Flow) -> None:
        super().__init__()
        self._max_episode_steps = 10240 # to test in 10 min
        # self._max_episode_steps = 1024 # size of a flow
        self.Flow = Flow
        self.core = None
        self.price_list = None
        # self.action_space = spaces.MultiDiscrete(BaseEnv.max_action)
        self.action_space = spaces.Box(0, BaseEnv.max_action,shape =(1,),dtype = np.int32)

        self.observation_space = spaces.Dict({
            'price':spaces.Box(low=BaseEnv.min_price,high=BaseEnv.max_price,shape=(10,),dtype=np.int32),
            'quantity':spaces.Box(low=0,high=BaseEnv.max_quantity,shape=(10,), dtype=np.int32)
            })
        # ---------------------
        self.num_left = None
        self.done = False
        self.current_step = 0
        self.running_reward = 0
        self.memory = None # revenue
        self.final_reward = 0
        self.init_reward = 0
        self.info = {}
        self.previous_obs = None
        self.reset_obs = None
    
# ============================  STEP  =========================================
    def step(self, action):
        ''' return observation, reward, done, info ''' 
        # if type(action) == np.ndarray:
        #     action = action.astype(np.int32)[0] # e.g. (3,) then we take the first element
        action = np.squeeze(action).astype(np.int32)
        # TO check, perhpas here remians problem
        observation = self._get_obs(action)
        num_executed = self.core.executed_quantity
        self.num_left -= num_executed 
        # assert num_executed == 3
        print('±'*20+str(self.core.executed_pairs))
        if self.core.executed_pairs == []:
            flow = self.core.flow ##
            print("!"*20)
        if self.core.executed_pairs == [-999]:
            print('@'*20) ## TODO TO Debug
        step_reward =  self._get_each_running_reward() 
        self.running_reward += step_reward
        self.memory.append(step_reward)
        self.current_step += 1 # Take care of the location
        # ---------------------
        done = self._get_set_done(action)
        reward = self._get_reward(action)
        info = self._get_info(action)
        # print("Step : ".format(self.current_step)) ### need to be here
        return  observation, reward, done, info
    # ------  1/4.OBS  ------
    def _get_obs(self, num):
        obs_ = self.core.step(num)[0] # dtype:series
        obs = from_series2obs(obs_)
        self.previous_obs = obs 
        return obs
    # ------ 2/4.DONE ------
    def _get_set_done(self,acion):
        '''get & set done'''
        # print('num_left : ', self.num_left)
        if self.num_left <= 0 or self.current_step >= self._max_episode_steps:
            self.done = True
        return self.done
    # ------ 3/4.REWARD  ------
    def _get_inventory_cost(self):
        inventory = self.num_left
        return BaseEnv.cost_parameter * inventory * inventory
    def _get_reward(self,acion):
        if not self.done:
            return 0
        elif self.done:
            self.final_reward = self.running_reward - self._get_inventory_cost()
            return self.final_reward
    def _get_each_running_reward(self):
        pairs = self.core.executed_pairs # TODO
        lst_pairs = np.array(from_pairs2lst_pairs(pairs))
        # lst_pairs = np.squeeze(lst_pairs).astype(np.int32)
        return -1 * sum(lst_pairs[0]*lst_pairs[1]) 
    # ------ 4/4.INFO ------
    def _get_info(self,acion):
        return self.info   
# =============================================================================
 
# =============================  RESET  =======================================   
    def reset(self):
        '''return the observation of the initial condition'''
        self.reset_states()
        index_random = random.randint(0, self.Flow.shape[0]-self._max_episode_steps-1)
        flow = self.Flow.iloc[index_random:index_random+self._max_episode_steps,:]
        flow = flow.reset_index().drop("index",axis=1)
        self.core = Core(flow)

        self._set_init_reward(flow.iloc[0,:])
        self.core = Core(flow)
        # self.core.reset() ## TODO refactor, reset the self.core
        # (1) check the self.core is updated or not
        # (2) if self.core is updated by _set_init_reward, then avoid to instance core twice
        # (3) try to implement the self.core.reset to avoid reading flow again
        
        init_obs = self._get_init_obs(flow.iloc[0,:])
        self.previous_obs = init_obs
        return init_obs
    def reset_states(self):
        # BaseEnv.max_price = max(self.core.flow.iloc[:,0])
        # BaseEnv.min_price = min(self.core.flow.iloc[:,18])
        self.memory = []
        self.running_reward = 0
        self.final_reward = 0
        self.done = False
        self.num_left = BaseEnv.num2liquidate
        self.current_step = 0
    def _get_init_obs(self, stream):
        init_price = np.array(get_price_from_stream(stream)).astype(np.int32)
        init_quant = np.array(get_quantity_from_stream(stream)).astype(np.int32)
        init_obs= {
            'price' : init_price,
            'quantity' : init_quant
            }
        return init_obs        
    def _set_init_reward(self, stream):
        num = BaseEnv.num2liquidate
        obs = Utils.from_series2pair(stream)
        level, executed_num = Broker._level_market_order_liquidating(num, obs)
        # TODO to use the num_executed
        if level == 0:
            self.init_reward = 0 
        elif level == -999:
            num_left = num-executed_num
            index = 1 # TODO not sure to check it
            while True:
                diff_obs = self.core.diff(index-1)
                index += 1
                # to use Broker here
                result = Broker.pairs_market_order_liquidating(num_left, diff_obs)
                try : Quantity = [(lambda x: -1*x[1])(x) for x in result]
                except:
                    print(" ")
                # remain problem if x<0, meaning withdraw order
                Price = [(lambda x: x[0])(x) for x in result]
                Reward = [(lambda x,y:x*y)(x,y) for x,y in zip(Price,Quantity)]
                reward = sum(Reward)
                executed_num = sum(Quantity)
                num_left -= executed_num
                if num_left <=0: 
                    break
        else:
            reward,consumed = 0,0
            for i in range(level-1):
                reward += obs[i][0] * obs[i][1]
                consumed += obs[i][1]
            reward += obs[level-1][0] * (num - consumed)
            self.init_reward = reward
        assert self.init_reward >= 0

# =============================================================================
    
    def render(self, mode = 'human'):
        print('-'*30)
        print(f'Step: {self.current_step}')
        print(f'StepAvarage: {self.memory[-1]/(3)/BaseEnv.max_price}')
        print(f'TotalAvarage: {sum(self.memory)/(BaseEnv.num2liquidate - self.num_left)/BaseEnv.max_price}')
        if self.done:
            print("============================")
            print(">>> FINAL REMAINING : "+str(format(self.num_left, ',d')))
            print(">>> Running REWARD : "+str(format(self.running_reward, ',d')))
            print(">>> FINAL REWARD : "+str(format(self.final_reward,',d')))
            print(">>> INIT  REWARD : "+str(format(self.init_reward,',d')))
            print(">>> Upper REWARD : "+str(format(BaseEnv.max_price * BaseEnv.num2liquidate,',d')))
            print(">>> Lower REWARD : "+str(format(BaseEnv.min_price * BaseEnv.num2liquidate,',d')))
            print(">>> Base Point (Upper): "+str(format(10000 *(BaseEnv.max_price / BaseEnv.min_price -1)))) #(o/oo)
            print(">>> Base Point (Init): "+str(format(10000 *(self.init_reward/BaseEnv.num2liquidate/ BaseEnv.min_price -1)))) #(o/oo)
            print(">>> Base Point (RL): "+str(format(10000 *(self.running_reward/BaseEnv.num2liquidate/ BaseEnv.min_price -1)))) #(o/oo)
            time.sleep(4)


    
if __name__=="__main__":
    from gym_trading.data.data_pipeline import ExternalData
    Flow = ExternalData.get_sample_order_book_data()
    env = BaseEnv(Flow)
    obs = env.reset()
    action = 3
    for i in range(int(1e6)):
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
    print("End of main()")