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
from gym_trading.tests import *
from gym_trading.debug import Debugger
from gym_trading.envs.match_engine import Core
from gym_trading.envs.broker import Flag, Broker
warnings.filterwarnings("ignore")
# =============================================================================



class BaseEnv(Env):
    """A stock trading environment based on OpenAI gym"""
    metadata = {'render.modes': ['human']}
    
# ============================  INIT  =========================================
    def __init__(self, Flow) -> None:
        super().__init__()
        self._max_episode_steps = Flag.max_episode_steps 
        self.type_of_Flow_is_list = True if type(Flow) == list else False
        if not self.type_of_Flow_is_list:
            self.Flow = Flow.to_numpy() # changing Flow from DataFrame into numpy.array
            # self.Flow = Flow
        if self.type_of_Flow_is_list:
            self.Flow_list = [flow.to_numpy() for flow in Flow] # changing Flow from DataFrame into numpy.array
            # self.Flow_list = Flow
        # above four lines is used for choosing the Flow or Flow_list
        self.core = None
        self.price_list = None
        self.action_space = spaces.Box(0, Flag.max_action,shape =(1,),dtype = np.int32)
        self.observation_space = \
        spaces.Box(
            low = np.array([Flag.min_price] * 10 +\
                           [Flag.num2liquidate]*1 +\
                           [Flag.max_episode_steps]*1 +\
                           [Flag.min_quantity]*10 + \
                           [Flag.min_num_left]*1 +\
                           [Flag.min_step_left]*1 
                           ).reshape((Flag.state_dim_1,Flag.state_dim_2)),
            high = np.array(
                           [Flag.max_price] * 10 +\
                           [Flag.num2liquidate]*1 +\
                           [Flag.max_episode_steps]*1 +\
                           [Flag.max_quantity]*10 + \
                           [Flag.max_num_left]*1 +\
                           [Flag.max_step_left]*1 
                           ).reshape((Flag.state_dim_1,Flag.state_dim_2)),
            shape = (Flag.state_dim_1,Flag.state_dim_2),
            dtype = np.int32,
        )
        # ---------------------
        self.num_left = None
        self.done = False
        self.current_step = None
        self.memory_revenue = 0
        self.memory_revenues = None # revenue
        self.memory_reward = None # total accumulative reward
        self.memory_obs = None # observation 
        self.memory_executed_pairs = None
        self.memory_executed = None
        self.memory_numleft = None
        self.final_reward = 0
        self.init_reward = 0
        self.init_reward_bp = 0
        self.info = {}
        self.reset_obs = None
        self.observation = None
        # ---------------------
        self.num_reset_called = 0
        self.action = None
        self.penalty_delta = None
        # self.info = None
        self.info = {}
    
# ============================  STEP  =========================================
    def core_step(self, action):
        ''' return observation, reward, done, info ''' 
        # if type(action) == np.ndarray:
        #     action = action.astype(np.int32)[0] # e.g. (3,) then we take the first element
        action = np.squeeze(action).astype(np.int32)
        # TO check, perhpas here remians problem
        action = min(action, self.num_left)
        observation, _, _, _ = self.core.step(action) # dtype:np.array (2,10); self.core.step(action) with shape of 4, observation, reward, done, info
        # observation, _, _, action_ceilling = self.core.step(action) # dtype:np.array (2,10); self.core.step(action) with shape of 4, observation, reward, done, info
        # self.info.update(action_ceilling) # is the ceiliing for this step or next step, it should be for next step
        num_executed = self.core.executed_quantity  
        return observation, num_executed
    
    def step(self, action):
        self.action = action # added for the use of _train_running_penalty
        # print("============================  STEP {} (BaseEnv)  ================================".format(self.current_step)) # tbd
        observation, num_executed =  self.core_step(action)
        self.observation = observation
        self.memory_executed_pairs.append(self.core.executed_pairs)
        self.memory_executed.append(num_executed)
        
        
        self.num_left -= num_executed 
        assert self.num_left >=0, "num_left cannot be negative"
        
        
        step_revenue =  self._get_each_running_revenue() 
        self.memory_revenue += step_revenue # todo to delete
        self.memory_revenues.append(step_revenue)
        self.memory_obs.append(observation)
        self.memory_numleft.append(self.num_left)
        self.current_step += 1 # Take care of the location
        # ---------------------
        done = self._get_set_done()
        reward = self._get_reward()
        self.memory_reward += reward 
        # if self.done:
        #     breakpoint() #tbd
        info = self._get_info()
        # ---------------------
        # self.set_default_observation() # set default observation 
        
        
        observation = self.extend_obs(observation)
        #  -------- check observation  --------
        if observation.shape != (Flag.state_dim_1,Flag.state_dim_2):
            observation = observation.reshape((Flag.state_dim_1,Flag.state_dim_2))
            # if observation.shape != (2,10):
            #     breakpoint()
        # -------------------------------------
        # breakpoint()
        return  observation, float(reward), done, info
        # return of the  STEP
    # ------  1/4.OBS  ------
    def _get_obs(self, num):
        def from_series_to_numpy(obs_):
            price = obs_[[2*i for i in range(len(obs_)//2)]]
            quantity = obs_[[2*i+1 for i in range(len(obs_)//2)]]
            obs = np.array([price,quantity])
            return obs
        obs_ = self.core.step(num)[0] # dtype:series
        return obs
    
    # def set_default_observation(self):
    #     if observation.shape != (10,2):
    #         return 0
    
    # ------ 2/4.DONE ------
    def _get_set_done(self):
        '''get & set done'''
        if self.num_left <= 0 or self.current_step >= self._max_episode_steps:
            self.done = True
            # print(">>>" * 10+ " DONE " + "<<<"*10) # tbd
        return self.done
    # ------ 3/4.REWARD  ------
    
    def _get_inventory_cost(self): 
        # tuning_parameter = Flag.num2liquidate * Flag.min_price / Flag.lobster_scaling / 500
        # tuning_parameter = Flag.num2liquidate * Flag.min_price / Flag.lobster_scaling / 50 # 16
        # tuning_parameter = Flag.num2liquidate * Flag.min_price / Flag.lobster_scaling # 17
        tuning_parameter = Flag.num2liquidate * Flag.min_price / Flag.lobster_scaling * 1000 # 24
        if self.num_reset_called <= Flag.pretrain_steps :
            # cost_curve = max(Flag.cost_parameter * tuning_parameter / np.log(self.num_reset_called + 2), Flag.cost_parameter)
            cost_curve = Flag.cost_parameter * tuning_parameter 
        else:
            cost_curve = Flag.cost_parameter 
        # breakpoint() #tbd
        cost = cost_curve * self.num_left 
        self.info['cost_curve'] = cost_curve
        self.info['cost'] = cost
        return cost # self.num_left is inventory
    
    def _low_dimension_penalty(self):
        num = sum(self.observation[1,:] == 0) # The number of price-quantity pairs in observation with a quantity of 0
        return Flag.low_dimension_penalty_parameter * num * num

    def _get_each_running_revenue(self):
        pairs = self.core.executed_pairs.copy() # TODO
        if pairs.size == 0 : result = 0
        else : result = -1 * (pairs[0,:] * pairs[1,:]).sum()
        assert result >= 0
        scaled_result = result / Flag.lobster_scaling # add this line the make it as the real price
        advatage_result = scaled_result - self.init_reward_bp * self.memory_executed[-1] # add this line to get the difference between the baseline, 
        # self.memory_executed[-1] is last one executed quantity
        return advatage_result

    def _get_reward(self):
        if not self.done: 
            return self.memory_revenues[-1] - self._low_dimension_penalty() 
        elif self.done:
            self.final_reward = self.memory_revenues[-1] - self._get_inventory_cost() - self._low_dimension_penalty()
            return self.final_reward
    
    # ------ 4/4.INFO ------
    def _get_info(self):
        if self.done:
            self.info['final_num_left'] =  self.num_left
            self.info['penalty_delta'] = self.penalty_delta
        self.info['num_executed'] = self.memory_executed[-1]
        self.info['num_left'] = self.num_left
        self.info['num_reset_called'] =  self.num_reset_called
        return self.info 
# =============================================================================
 
# =============================  RESET  =======================================   
    def reset(self):
        self.num_reset_called += 1 # how many times of the reset() has been called
        '''return the observation of the initial condition'''
        # print(">>> reset") ## to be deleted
        self.reset_states()
        if not self.type_of_Flow_is_list : # single file
            index_random = random.randint(0, self.Flow.shape[0]-self._max_episode_steps-1)
            # index_random = random.randint(0, self.Flow.shape[0]-self._max_episode_steps-1)
            if Debugger.switch_on :
                if Debugger.BaseEnv.start_from_index_zero:
                    index_random = 0 # tbd
            print(">>> Index_random is ",index_random) #tbd
            flow = self.Flow[index_random:index_random + (self._max_episode_steps+1) * Flag.skip ,:]
        else:
            index_random_for_list = random.randint(0, len(self.Flow_list) - 1)
            Flow = self.Flow_list[index_random_for_list]
            index_random = random.randint(0, Flow.shape[0]-self._max_episode_steps-1)
            flow = Flow[index_random:index_random+(self._max_episode_steps+1) * Flag.skip,:]
        # print("(base_environment) the length of flow is ",len(flow)) # tbd
        self.core = Core(flow)
        
        self.core.reset()
        self._set_init_reward() 
        self.core.reset() 
        
        init_obs = self._get_init_obs(flow[0,:])
        extended_obs = self.extend_obs(init_obs)
        
        self.memory_obs.append(init_obs) # index 0th observation
        # return init_obs
        return extended_obs
    
    def reset_states(self):
        self.memory_revenues = []
        self.memory_obs = []
        self.memory_executed = []
        self.memory_executed_pairs = []
        self.memory_numleft = []
        self.memory_revenue = 0
        self.memory_reward = 0
        self.init_reward = 0 
        self.init_reward_bp = 0
        self.final_reward = 0
        self.done = False
        self.num_left = Flag.num2liquidate
        self.current_step = 0
        self.observation = np.array([])
        self.action = None
        self.penalty_delta = 0
    def _get_init_obs(self, stream):
        init_obs = np.array([[stream[2*i] for i in range(len(stream)//2)], [stream[2*i+1] for i in range(len(stream)//2)]])
        return init_obs   
    
    def _set_init_reward(self):
        self.init_reward = 0
        self.init_reward_bp = self.init_reward/Flag.num2liquidate
            
            
    def render(self, mode = 'human'):
# ==========================#  tbd ============================================
        print()
        print("--"*10 + str(self.current_step) + "--"*10)
        print("self.num_left: {}, self.action {}, self.num_executed {}".format(self.num_left,self.action, self.memory_executed[-1]))
        print("self.observation: {}".format(self.observation))
        if self.done: print(">>>"*10 + " Base_Environment Render " + "<<<"*10)

# =============================================================================
        pass # tbd
        
    def extend_obs(self, init_obs):
        num_left = self.num_left
        step_left = Flag.max_episode_steps - self.current_step
        to_be_extend_obs = np.array([ Flag.num2liquidate, Flag.max_episode_steps, num_left, step_left ]).reshape((2,2))
        extended_obs = np.concatenate((init_obs, to_be_extend_obs), axis = 1)
        return extended_obs

    
if __name__=="__main__":
    # random_strategy(BaseEnv)
    zero_strategy(BaseEnv)
