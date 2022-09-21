# %%
import copy
import numpy as np
# import cudf
import pandas as pd

from gym_trading import utils
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
        # self._flow = -self.flow.diff()
        self._flow  = -np.diff(self.flow, axis = 0) # self._flow[index,:] comes from diff(flow[index+1,:], flow[index,:])

        # these will be changed during step
        self.index = None
        self.state = None
        self.action = None
        self.reward = None
        self.executed_pairs = None
        self.executed_quantity = None
        self.executed_sum = None
        self.done = None

    def update(self, obs, diff_obs):
        '''update at time index based on the observation of index-1'''
        if  diff_obs.shape != (0,) and obs.shape != (0,): 
            obs = np.hstack((obs.astype(np.int64),diff_obs.astype(np.int64))) 
            result = utils.remove_replicate(obs)
            '''e.g.
            obs = array([[1239,    1]])
            diff_obs = array([[19723,     2]])
            np.vstack((obs,diff_obs)) = 
            array([[ 1239,     1],
                   [19723,     2]])
            '''
        elif diff_obs.size == 0 and obs.size == 0: 
            result =  []
        elif -999 in obs: 
            raise NotImplementedError # return [] # TODO to implement it in right way
        else:
            if obs.shape != (0,): result = utils.remove_replicate(obs)
            if diff_obs.shape != (0,): result = utils.remove_replicate(diff_obs)
        return np.array(result)
            
    def step(self, action):
        # print("=" * 10 + " New Epoch " + "=" * 10) #tbd
        # print("(match_engine) current index is ",self.index) ## tbd
        self.action = action
        
        # self_state = self.state # tbd
        assert type(action) == np.ndarray or int
        state = utils.remove_zero_quantity(self.state)
        '''e.g.
        array([[31161600, 31160000, 31152200, 31151000, 31150100, 31150000,
                31140000, 31130000, 30000000, 30000000],
               [       3,        4,       16,        2,        2,      506,
                       4,        2,       0,       0]])
        this step is to remove the last two useless pairs(price, quantity)
        '''
        
        # action = min(action,self.num_left) ##
        
        new_obs, executed_quantity = Broker.pairs_market_order_liquidating(action, state)
        # new_obs, executed_quantity = Broker.pairs_market_order_liquidating(action, self_state) # tbd
        # print('-'*20+"\n"+ "(match_engine) self.action") #tbd
        # print(self.action) #tbd
        
        # print('-'*20+"\n"+ "(match_engine) new_obs:") #tbd
        # print(new_obs) #tbd
        self.executed_quantity = executed_quantity
        # get_new_obs which are new orders, not the new state
        

        if new_obs.shape != (0,):
            assert (new_obs[1,:] <= 0).all() 
            # for i in range(new_obs.shape[1]):
            #     assert new_obs[1] <= 0 
            self.executed_pairs = new_obs ## TODO ERROR
            ''' e.g. [[31161600, -3], [31160000, -4], [31152200, -13]] 
            all the second element should be negative, as it is the excuted and should be
            removed from the limit order book
            '''
            
            if -new_obs[1,:].sum() != executed_quantity:
                '''the left is the sum of the quantity from new_obs'''
                num, obs = executed_quantity, [[item[0],-1*item[1]] for item in new_obs]
                result, executed_num = Broker.pairs_market_order_liquidating(num, obs)
                assert executed_num == executed_quantity
                self.executed_pairs = result
            # get the executed_pairs
        else:
            self.executed_pairs = np.array([ ])
        
        
        # self_index = self.index # tbd
        diff_obs = self.get_difference(Flag.skip) # get incomming orders from data 2/3
        to_be_updated = self.update(diff_obs, new_obs) # new_obs is generated by action 2/3
        updated_state = self.update(state, to_be_updated) # updated state, state combined with incomming orders 3/3
        
        # print('-'*20+"\n"+ "(match_engine) state") #tbd
        # print(state) #tbd
        # print('-'*20+"\n"+ "(match_engine) diff_obs") #tbd
        # print(diff_obs) #tbd
        # try:print("Incomming orders quantity: ",sum(diff_obs[1,:])); #tbd
        # except: pass #tbd
        # print('-'*20+"\n"+ "(match_engine) to_be_updated")#tbd
        # print(to_be_updated)#tbd
        
        updated_state = utils.check_positive_and_remove_zero(updated_state) 
        updated_state = utils.keep_dimension(updated_state,Flag.price_level)
        
        # print('-'*20+"\n"+ "(match_engine) updated_state")#tbd
        # print(updated_state)#tbd
        
        # utils.check_get_difference_and_update_0(Flag.skip, self.action, self.flow[self.index + 1,:], updated_state) 
        # utils.check_get_difference_and_update_1(Flag.skip, self.action, self.index, updated_state) 

        self.state = updated_state
        reward = self.reward # todo always return None, not implemented
        self.done = self.check_done()
        self.executed_sum += self.executed_quantity # add this line to check if all liquidated
        # ---------------------
        info = self.get_info() # update info for return of the step
        # ---------------------
        self.index += Flag.skip # default = 1, change the index before return
        return self.state, reward, self.done, info
    
    def get_info(self):
        {"action_ceilling", self.action_ceilling()}
    
    def get_difference(self, skip = 1):
        Index = copy.deepcopy(self.index)
        diff_list = []
        for i in range((self._flow.shape[1])//2): # in range of col_num
            for Index in range(self.index, self.index + skip):
                if Index >= self._max_episode_steps: break # TODO should implement in right way
                try: (self._flow[Index, 2*i] != 0 or self._flow[Index, 2*i+1] !=0)
                except:
                    breakpoint()
                if self._flow[Index, 2*i] != 0 or self._flow[Index, 2*i+1] !=0:
                    diff_list.extend([
                        [self.flow[Index+1,2*i], self.flow[Index+1,2*i+1]], 
                        [self.flow[Index,2*i], -self.flow[Index,2*i+1]]
                        ])
        if len(diff_list) == 0: result =  []
        else: 
            result = utils.list_to_gym_state(diff_list)
            result = utils.arg_sort(result) 
            result = utils.remove_replicate(result) # the diff_list should be sorted here # todo write a check func
        return np.array(result)
        '''e.g.
        If Index == 3 in the single_file_debug mode, the the incomming difference order 
        from data should be [[31120200, -35], [31155000, 28]], which means withdrawing 
        35 of 31120200, and incomming 28 of 31155000.
        '''
    
    def check_done(self):
        if self.index < self._max_episode_steps: return False
        elif self.index >= self._max_episode_steps: return True
    
    def reset(self):
        self.index = Core.init_index # default = 0
        self.state = self.flow[Core.init_index,:] #initial_state
        # self.state = self.flow.iloc[Core.init_index,:] #initial_state
        self.action = None
        self.executed_pairs = None
        self.executed_quantity = None
        self.done = False
        self.executed_sum = 0
        # --------------
        self.state = utils.change_to_gym_state(self.state)
        return self.state
    
    def action_ceilling(self):
        # the sum of all the quantity at one state,
        # the action should be chosen within [0, ceilling], if randomly chosen.
        return sum(self.state[1,:])

    
    def check_positive(self, updated_state):
        for item in updated_state:
            if item[1]<0:
                item[1]=0
        return updated_state
        
# %%
if __name__ == "__main__":
    Flow = ExternalData.get_sample_order_book_data()
    flow = Flow.iloc[0:Flag.max_episode_steps,:].reset_index().drop("index",axis=1)
    flow = flow.to_numpy()
    core = Core(flow)
    
    obs0 = core.reset()
    for i in range(10):
        obs = core.step(1)
        
    obs0 = core.reset()
    for i in range(10):
        obs = core.step(0)
        
    obs0 = core.reset()
    

    