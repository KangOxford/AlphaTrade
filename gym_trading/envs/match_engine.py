# %%
import copy
import numpy as np
import pandas as pd

from gym_trading.data.data_pipeline import ExternalData
'''One Match Engine is corresponds to one specific Limit Order Book DataSet'''
# %%
class Utils():
    def from_series2pair(stream):
        num = stream.shape[0]
        previous_list = list(stream)
        previous_flow = []
        for i in range(num):
            if i%2==0:
                previous_flow.append(
                    [previous_list[i],previous_list[i+1]]
                    ) 
        return previous_flow

    def from_pair2series(stream):
        def namelist():
            name_lst = []
            for i in range(len(stream)):
                name_lst.append("bid"+str(i+1))
                name_lst.append("bid"+str(i+1)+"_quantity")
            return name_lst
        name_lst = namelist()
        stream = sorted(stream,reverse=True)
        result = []
        for item in stream:
            result.append(item[0])
            result.append(item[1])
        return pd.Series(data=result, index = name_lst)
        # TODO deal with the empty data, which is object not float

    def remove_replicate(diff_list):
        # remove_replicate
        # diff_list = sorted(diff_list) ## !TODO not sure
        diff_list_keys = []
        for item in diff_list:
            diff_list_keys.append(item[0])
        set_diff_list_keys = sorted(set(diff_list_keys))
        
        index_list = []
        for item in set_diff_list_keys:
            index_list.append(diff_list_keys.index(item))
        index_list = sorted(index_list)
        
        present_flow = []
        for i in range(len(index_list)-1):
            index = index_list[i]
            if diff_list[index][0] == diff_list[index+1][0] :
                present_flow.append([
                    diff_list[index][0], 
                    diff_list[index][1]+diff_list[index+1][1]
                    ]) 
            elif diff_list[index][0] != diff_list[index+1][0] :
                present_flow.append([
                    diff_list[index][0],
                    diff_list[index][1]
                    ])
        if index_list[-1] == len(diff_list)-1:
            present_flow.append([
                diff_list[-1][0],
                diff_list[-1][1]
                ])
        elif index_list[-1] != len(diff_list)-1:
            present_flow.append([
                diff_list[-2][0], 
                diff_list[-2][1]+diff_list[-1][1]
                ])     
        result = present_flow.copy()
        for j in range(len(present_flow)):
            if present_flow[j][1] == 0:
                result.remove(present_flow[j])
        return result
    



# %%
class Broker():
    @classmethod
    def _level_market_order_liquidating(cls, num, obs):
        # num,obs = num_left, diff_obs ##
        num = copy.deepcopy(num)
        '''observation is one row of the flow, observed at specific time t'''   
        i = 0
        result = 0
        Num = copy.deepcopy(num)
        while num>0:
            if i>=10: 
                result = -999
                break
            try :
                num = max(num-obs[i][1], 0)
            except:
                break
            i+=1
            result = i
        executed_num = Num - num # TODO use the executed_num
        # result return the price level, 
        # begin with level 1, then end with level 10
        return result, executed_num
    
    @classmethod
    def pairs_market_order_liquidating(cls, num, obs):
        # num, obs = action, state ##
        num = copy.deepcopy(num)
        # level, executed_num = Broker._level_market_order_liquidating(num, obs)##
        level, executed_num = cls._level_market_order_liquidating(num, obs)
        # TODO need the num <=609 the sum of prices at all leveles
        sum_quantity = 0
        quantity_list = []
        for i in range(len(obs)):
            sum_quantity+=obs[i][1]
            quantity_list.append(sum_quantity)
        
        result = []
        if level>1:
            for i in range(level-1):
                result.append(
                    [obs[i][0],-obs[i][1]])
            result.append(
                [obs[level-1][0],-num+quantity_list[level-2]])
        if level == 1:
            # result.append([obs[0][0],-num]) # to check it should be wrong
            result.append([obs[0][0],-executed_num])
            '''-executed_num, to be negative means the quantity is removed from the lob
            '''
        if level == 0:
            pass
        if level == -999:
            result.append(-999)
        return result, executed_num



# %%
class Core():
    init_index = 0
    def __init__(self, flow):
        self.flow = flow
        self._flow = -self.flow.diff()
        self.index = Core.init_index
        self.state = self.initial_state()
        self.action = None
        self.reward = None
        self.executed_pairs = None
        self.executed_quantity = None
    def initial_state(self):
        return self.flow.iloc[Core.init_index,:]
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
        
        new_obs, executed_quantity = Broker.pairs_market_order_liquidating(action, state)
        self.executed_quantity =executed_quantity
        # get_new_obs
        
        if executed_quantity != self.action and executed_quantity!=0:
            assert len(new_obs) == 1
        else:
            for item in new_obs:
                if not item[1] <= 0 :
                    print()
                assert item[1] <= 0 
            self.executed_pairs = new_obs ## TODO ERROR
            ''' e.g. [[31161600, -3], [31160000, -4], [31152200, -13]] 
            all the second element should be negative, as it is the excuted and should be
            removed from the limit order book
            '''
        # TODO get the executed_pairs
        
        diff_obs = self.diff(self.index-1)
        to_be_updated = self.update(diff_obs, new_obs)
        updated_state = self.update(state, to_be_updated)
        if type(updated_state) == list:
            updated_state = self.check_positive(updated_state)
            updated_state = Utils.from_pair2series(updated_state)
            
        self.state = updated_state
        reward = self.reward
        return self.state, reward, False, {}
    
    
    def reset(self):
        self.index = Core.init_index
        self.state = self.initial_state()
        self.action = None
        self.executed_pairs = None
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
                if Index >= 1024: ##
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
    ##
    obs6 = core.step(min(20,core.get_ceilling()))[0]
    obs7 = core.step(min(20,core.get_ceilling()))[0]
    obs8 = core.step(min(20,core.get_ceilling()))[0]
    obs9 = core.step(min(20,core.get_ceilling()))[0]
    obs10= core.step(min(20,core.get_ceilling()))[0]
    # ==================================================
    
    
    # ==================================================
    obs11 = core.step(min(20,core.get_ceilling()))[0]
    obs12 = core.step(min(20,core.get_ceilling()))[0]
    obs13 = core.step(min(20,core.get_ceilling()))[0]
    obs14 = core.step(min(20,core.get_ceilling()))[0]
    obs15 = core.step(min(20,core.get_ceilling()))[0]
    # # ##
    obs16 = core.step(min(20,core.get_ceilling()))[0]
    obs17 = core.step(min(20,core.get_ceilling()))[0]   
    obs18 = core.step(min(20,core.get_ceilling()))[0]
    obs19 = core.step(min(20,core.get_ceilling()))[0]
    obs20 = core.step(min(20,core.get_ceilling()))[0]
    # ==================================================

