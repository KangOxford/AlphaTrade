# %%
import copy
import numpy as np
# import cudf
import pandas as pd

from gym_trading.utils import Utils
from gym_trading.envs.broker import Flag, Broker
from gym_trading.data.data_pipeline import ExternalData
'''One Match Engine is corresponds to one specific Limit Order Book DataSet'''
# %%
class Core():
    init_index = 0
    def __init__(self, flow):
        # these wont be changed during step
        self._max_episode_steps = Flag.max_episode_steps 
        self.flow = flow
        self._flow = -self.flow.diff()
        # these will be changed during step
        self.index = None
        self.state = None
        self.action = None
        self.reward = None
        self.executed_pairs = None
        self.executed_quantity = None
        self.done = None

    def update(self, obs, diff_obs):
        '''update at time index based on the observation of index-1'''
        obs.extend(diff_obs)
        if len(obs) == 0 :
            return []
        elif -999 in obs: 
            return [] # TODO to implement it in right way
        else:
            return Utils.remove_replicate(sorted(obs))
    def step(self, action):
        self.action = action
        self.index += 1
        state = Utils.from_series2pair(self.state)

        assert type(action) == np.ndarray or int
        remove_zero_quantity = lambda x:[item for index, item in enumerate(x) if item[1]!=0]
        state = remove_zero_quantity(state)
        
        # action = min(action,self.num_left) ##
        
        new_obs, executed_quantity = Broker.pairs_market_order_liquidating(action, state)
        self.executed_quantity = executed_quantity
        # get_new_obs
        

        for item in new_obs:
            assert item[1] <= 0 
        self.executed_pairs = new_obs ## TODO ERROR
        ''' e.g. [[31161600, -3], [31160000, -4], [31152200, -13]] 
        all the second element should be negative, as it is the excuted and should be
        removed from the limit order book
        '''
        
        if sum([-1*item[-1] for item in new_obs]) != executed_quantity:
            '''the left is the sum of the quantity from new_obs'''
            num, obs = executed_quantity, [[item[0],-1*item[1]] for item in new_obs]
            result, executed_num = Broker.pairs_market_order_liquidating(num, obs)
            assert executed_num == executed_quantity
            self.executed_pairs = result
        # get the executed_pairs
        
        diff_obs = self.diff(self.index-1)
        to_be_updated = self.update(diff_obs, new_obs)
        updated_state = self.update(state, to_be_updated)
        if type(updated_state) == list:
            updated_state = self.check_positive(updated_state)
            updated_state = Utils.from_pair2series(updated_state)
            
        self.state = updated_state
        reward = self.reward
        self.done = self.check_done()
        return self.state, reward, self.done, {}
    
    def check_done(self):
        if self.index < self._max_episode_steps: return False
        elif self.index == self._max_episode_steps: return True
    
    def reset(self):
        self.index = Core.init_index
        self.state = self.flow.iloc[Core.init_index,:] #initial_state
        self.action = None
        self.executed_pairs = None
        self.executed_quantity
        self.done = False
        return self.state
    
    
    def get_ceilling(self):
        next_stage = self.state
        next_stage_lst = list(next_stage)
        result = 0.0
        for i in range(len(next_stage_lst)):
            if i % 2 == 1:
                result += next_stage_lst[i]
        return result
    
    def diff(self, index):
        Index = index + 1 ## !TODO not sure
        col_num = self._flow.shape[1] 
        diff_list = [] 
        for i in range(col_num):
            if i%2 == 0:
                if Index >= self._max_episode_steps: ##
                    # print(Index) ## !TODO not sure
                    break 
                if self._flow.iat[Index,i] !=0 or self._flow.iat[Index,i+1] !=0:
                    diff_list.append([self.flow.iat[Index,i],
                                      self.flow.iat[Index,i+1]])
                    diff_list.append([self.flow.iat[Index-1,i],
                                      -self.flow.iat[Index-1,i+1]])
        if len(diff_list) == 0:
            return []
        else:
            return Utils.remove_replicate(sorted(diff_list))  
    def check_positive(self, updated_state):
        for item in updated_state:
            if item[1]<0:
                item[1]=0
        return updated_state
        
# %%
if __name__ == "__main__":
    Flow = ExternalData.get_sample_order_book_data()
    flow = Flow.iloc[3:1000,:].reset_index().drop("index",axis=1)

    core = Core(flow)
    obs0 = core.reset()
    
    # ==================================================
    obs1 = core.step(min(20,core.get_ceilling()))[0]
    obs2 = core.step(min(20,core.get_ceilling()))[0]
    obs3 = core.step(min(20,core.get_ceilling()))[0]
    obs4 = core.step(min(20,core.get_ceilling()))[0]
    obs5 = core.step(min(20,core.get_ceilling()))[0]
    obs26 = core.step(300)[0]
    obs27 = core.step(30)[0]
    obs28 = core.step(30)[0]
    obs29 = core.step(30)[0]
    obs30 = core.step(30)[0]
    # ==================================================
    

    