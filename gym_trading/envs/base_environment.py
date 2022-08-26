# =============================================================================
import time
import random
import warnings
import numpy as np
# import cudf 
import pandas as pd
# -----------------------------------------------------------------------------
from gym import Env
from gym import spaces
# -----------------------------------------------------------------------------
from gym_trading.utils import * 
from gym_trading.envs.match_engine import Core
from gym_trading.envs.match_engine import Broker, Utils
from gym_trading.utils import exit_after
warnings.filterwarnings("ignore")
# =============================================================================



class BaseEnv(Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    
    # max_action = 70
    # max_action = 300
    max_action = 1000
    max_quantity = 6000
    max_price = 31620700
    min_price = 31120200
    scaling = 30000000
    num2liquidate = 30000
    # cost_parameter = int(1e9)
    # cost_parameter = 0 # for debugging
    cost_parameter = 5e-6 # from paper.p29 : https://epubs.siam.org/doi/epdf/10.1137/20M1382386
    
# ============================  INIT  =========================================
    def __init__(self, Flow) -> None:
        super().__init__()
        # self._max_episode_steps = 10240 # to test in 10 min, long horizon # size of a flow
        self._max_episode_steps = 102400 # to test in 1 min, short horizon
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
        self.current_step = None
        self.memory_revenue = 0
        self.memory_revenues = None # revenue
        self.memory_obs = None # observation 
        self.memory_executed_pairs = None
        self.memory_executed = None
        self.memory_numleft = None
        self.final_reward = 0
        self.init_reward = 0
        self.init_reward_bp = 0
        self.info = {}
        self.previous_obs = None
        self.reset_obs = None
    
# ============================  STEP  =========================================
    def core_step(self, action):
        ''' return observation, reward, done, info ''' 
        # if type(action) == np.ndarray:
        #     action = action.astype(np.int32)[0] # e.g. (3,) then we take the first element
        action = np.squeeze(action).astype(np.int32)
        # TO check, perhpas here remians problem
        action = min(action, self.num_left)
        observation = self._get_obs(action)
        num_executed = self.core.executed_quantity  
        return observation, num_executed
    
    def step(self, action):
        observation, num_executed =  self.core_step(action)
        
        
        self.num_left -= num_executed 
        assert self.num_left >=0, "num_left cannot be negative"
        
        
        step_revenue =  self._get_each_running_revenue() 
        self.memory_revenue += step_revenue
        self.memory_revenues.append(step_revenue)
        self.memory_obs.append(observation)
        self.memory_executed_pairs.append(self.core.executed_pairs)
        self.memory_executed.append(num_executed)
        self.memory_numleft.append(self.num_left)
        self.current_step += 1 # Take care of the location
        # ---------------------
        done = self._get_set_done()
        reward = self._get_reward()
        info = self._get_info()
        
        return  observation, reward, done, info
    # ------  1/4.OBS  ------
    def _get_obs(self, num):
        obs_ = self.core.step(num)[0] # dtype:series
        obs = from_series2obs(obs_)
        self.previous_obs = obs 
        return obs
    # ------ 2/4.DONE ------
    def _get_set_done(self):
        '''get & set done'''
        if self.num_left <= 0 or self.current_step >= self._max_episode_steps:
            self.done = True
        return self.done
    # ------ 3/4.REWARD  ------
    
    def _get_inventory_cost(self):
        inventory = self.num_left
        return BaseEnv.cost_parameter * inventory * inventory

    def _get_reward(self):
        if not self.done: return 0 # set to be real
        # if not self.done: return -0.1 # set to encourage explore but try to find the price peak
        # if not self.done: return -0.5 # set to encourage explore, accelarating turning into selling with no stock left
        elif self.done:
            # print("=============== Finished ===============") ##
            RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff = self.calculate_info()
            assert BasePointRL <= BasePointBound
            self.final_reward = BasePointDiff
            # print(f"BasePointDiff: {BasePointDiff}, num_left : {self.num_left}")
            return self.final_reward
        
    
    def _get_each_running_revenue(self):
        pairs = self.core.executed_pairs # TODO
        lst_pairs = np.array(from_pairs2lst_pairs(pairs))
        # lst_pairs = np.squeeze(lst_pairs).astype(np.int32)
        result = sum(lst_pairs[0]* -1 *lst_pairs[1]) 
        assert result >= 0
        return result
    # ------ 4/4.INFO ------
    def calculate_info(self):
        RLbp = 10000 *(self.memory_revenue/BaseEnv.num2liquidate/ BaseEnv.min_price -1)
        Boundbp = 10000 *(BaseEnv.max_price / BaseEnv.min_price -1)
        BasePointBound = 10000 *(BaseEnv.max_price / BaseEnv.min_price -1)
        BasePointInit = 10000 *(self.init_reward/BaseEnv.num2liquidate/ BaseEnv.min_price -1)
        BasePointRL = 10000 *(self.memory_revenue/BaseEnv.num2liquidate/ BaseEnv.min_price -1)
        BasePointDiff = BasePointRL - BasePointInit        
        return RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff
    def _get_info(self):
        if self.done:
            RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff = self.calculate_info()
            self.info = {"Diff" : BasePointDiff,
                         "Step" : self.current_step,
                         "Left" : self.num_left
                         }
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
        
        self.core.reset()
        self._set_init_reward() 
        self.core.reset() 
        
        init_obs = self._get_init_obs(flow.iloc[0,:])
        self.previous_obs = init_obs
        self.memory_obs.append(init_obs) # index 0th observation
        # print(">>>"*10+"env.reset done")
        return init_obs
    def reset_states(self):
        # BaseEnv.max_price = max(self.core.flow.iloc[:,0])
        # BaseEnv.min_price = min(self.core.flow.iloc[:,18])
        self.current_step = 0 
        # print(">> reset current step") ## to be deleted 
        self.memory_revenues = []
        self.memory_obs = []
        self.memory_executed = []
        self.memory_executed_pairs = []
        self.memory_numleft = []
        self.memory_revenue = 0
        self.init_reward = 0 
        self.init_reward_bp = 0
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
    
    @exit_after
    def liquidate_init_position(self):
        num2liquidate = BaseEnv.num2liquidate
        max_action = BaseEnv.max_action
        while num2liquidate > 0:
            observation, num_executed =  self.core_step(min(max_action,num2liquidate)) # observation(only for debug), num_executed for return
            num2liquidate -= num_executed
            
            executed_pairs = self.core.executed_pairs
            Quantity = [(lambda x: x[1])(x) for x in executed_pairs]# if x<0, meaning withdraw order
            assert -1 * sum(Quantity) == num_executed # the exected_pairs in the core is for each step
            Price = [(lambda x: x[0])(x) for x in executed_pairs]
            Reward = [(lambda x,y:x*y)(x,y) for x,y in zip(Price,Quantity)]   
            reward = -1 * sum(Reward)    
            self.init_reward += reward
            if self.core.done:# still left stocks after selling all in the core.flow
                inventory = num2liquidate
                self.init_reward -= BaseEnv.cost_parameter * inventory * inventory
                break
    def _set_init_reward(self):
        self.liquidate_init_position()
        self.init_reward_bp = int(self.init_reward/BaseEnv.num2liquidate)


# =============================================================================
    
    def render(self, mode = 'human'):
        if self.done:
            RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff = self.calculate_info()
            try: assert self.num_left >= 0, "Error for the negetive left quantity"
            except: 
                raise Exception("Error for the negetive left quantity")
            print("="*30)
            print(">>> FINAL REMAINING(RL) : "+str(format(self.num_left, ',d')))
            print(">>> INIT  REWARD : "+str(format(self.init_reward,',f')))
            print(">>> Upper REWARD : "+str(format(BaseEnv.max_price * BaseEnv.num2liquidate,',d')))
            print(">>> Lower REWARD : "+str(format(BaseEnv.min_price * BaseEnv.num2liquidate,',d')))
            print(">>> Base Point (Bound): "+str(format(BasePointBound))) #(o/oo)
            print(">>> Base Point (Init): "+str(format(BasePointInit))) #(o/oo)
            print(">>> Base Point (RL): "+str(format(BasePointRL))) #(o/oo)
            print(">>> Base Point (Diff): "+str(format(BasePointDiff))) #(o/oo)
            try: assert RLbp <= Boundbp, "Error for the RL Base Point"
            except:
                raise Exception("Error for the RL Base Point")
            try: assert  self.init_reward >= -1 * BaseEnv.cost_parameter * BaseEnv.num2liquidate * BaseEnv.num2liquidate
            except:
                raise Exception("Error for the Init Lower Bound")            



    
if __name__=="__main__":
    from gym_trading.data.data_pipeline import ExternalData
    Flow = ExternalData.get_sample_order_book_data()
    env = BaseEnv(Flow)
    obs = env.reset()
    action = 1
    diff_list = []
    step_list = []
    left_list = []
    for i in range(int(1e6)):
        if i//2 == i/2: observation, reward, done, info = env.step(action)
        # if i//3 == i/3: observation, reward, done, info = env.step(action)
        else: observation, reward, done, info = env.step(0)
        env.render()
        if done:
            diff_list.append(info['Diff'])
            step_list.append(info['Step'])
            left_list.append(info['Left'])
            print(">"*20+" timestep: "+str(i))
            env.reset()
    print(f"End of main(), Diff is {np.mean(diff_list)}, Step is {np.mean(step_list)}, Left is {np.mean(left_list)}")
