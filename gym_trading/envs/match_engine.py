# %%
import pandas as pd
from copy import copy
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

    def remove_replicate(diff_list):
        # remove_replicate
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
        '''observation is one row of the flow, observed at specific time t'''   
        i = 0
        result = 0
        while num>0:
            if i>=10: 
                result = -999
                break
            num -= obs[i][1]
            i+=1
            result = i
        return result
    
    @classmethod
    def pairs_market_order_liquidating(cls, num, obs):
        level = cls._level_market_order_liquidating(num, obs)
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
            result.append([obs[0][0],-num])
        if level == 0:
            pass
        if level == -999:
            result.append(-999)
        return result

# %% RUN for one time
class DataPipeline():
    def __init__(self, data):
        self.count = 0
        self.namelist = self._namelist()
        self.data = data

    
    def __call__(self):
        return self.data
    def get(self, index):
        return self.data.iloc[index,:]
    def step(self):
        result = self.data.iloc[self.count,:]
        self.count += 1
        return result
    def reset(self):
        """ !TODO reset the class """
        self.count = 0
        return self.data.iloc[0,:]
    

class ExternalData():
    @classmethod
    def get_sample_order_book_data(cls):
        import pandas as pd
        def namelist():
            name_lst = []
            for i in range(40//4):
                name_lst.append("ask"+str(i+1))
                name_lst.append("ask"+str(i+1)+"_quantity")
                name_lst.append("bid"+str(i+1))
                name_lst.append("bid"+str(i+1)+"_quantity")
            return name_lst
# =============================================================================
#         url = "https://drive.google.com/file/d/1UawhjR-9bEYYns7PyoZNym_awcyVko0i/view?usp=sharing"
#         path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
# =============================================================================
        path = "/Users/kang/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv"
        
        df = pd.read_csv(path,names = namelist())
        column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
        Flow = df.iloc[:,column_numbers]
        return Flow
    
    
Flow = ExternalData.get_sample_order_book_data()
flow = Flow.iloc[3:100,:].reset_index().drop("index",axis=1)
# datapipeline = DataPipeline(ExternalData.get_sample_order_book_data())
# data = datapipeline.reset()
# data = datapipeline.step()
# %%
class Core():
    init_index = 0
    def __init__(self, flow):
        self.flow = flow
        self.index = Core.init_index
        self.state = self.initial_state()
    def initial_state(self):
        return self.flow.iloc[Core.init_index,:]
    def get_new_obs(self, num, obs):
        return Broker.pairs_market_order_liquidating(num, obs)
    def update(self, obs, diff_obs):
        '''update at time index based on the observation of index-1'''
        obs.extend(diff_obs)
        new = sorted(obs)
        result = Utils.remove_replicate(new)
        return result
    def get_reward(self):
        return 0
    def step(self, action):
        self.index += 1
        state = Utils.from_series2pair(self.state)
        new_obs = self.get_new_obs(action, state)
        diff_obs = self.diff(self.index-1)
        to_be_updated = self.update(diff_obs, new_obs)
        updated_state = self.update(state, to_be_updated)
        if type(updated_state) == list:
            updated_state = Utils.from_pair2series(updated_state)
        self.state = updated_state
        reward = self.get_reward()
        return self.state, reward, False, {}
    def reset(self):
        self.index = 1
        self.state = self.initial_state()
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
        b = -self.flow.diff()
        col_num = b.shape[1] 
        diff_list = [] 
        for i in range(col_num):
            if i%2 == 0:
                if b.iat[index,i] !=0 or b.iat[index,i+1] !=0:
                    diff_list.append([self.flow.iat[index,i],
                                      self.flow.iat[index,i+1]])
                    diff_list.append([self.flow.iat[index-1,i],
                                      -self.flow.iat[index-1,i+1]])
        
        if len(diff_list) == 0:
            return []
        else:
            return Utils.remove_replicate(diff_list)        
# %%

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
##
obs16 = core.step(min(20,core.get_ceilling()))[0]
obs17 = core.step(min(20,core.get_ceilling()))[0]
obs18 = core.step(min(20,core.get_ceilling()))[0]
obs19 = core.step(min(20,core.get_ceilling()))[0]
obs20 = core.step(min(20,core.get_ceilling()))[0]
# ==================================================

# obs3 = core.step(20)[0]
# obs4 = core.step(20)[0]
# obs5 = core.step(20)[0]
# obs6 = core.step(20)[0]
# obs7 = core.step(20)[0]
# obs8 = core.step(20)[0]
# obs9 = core.step(20)[0]
# obs10 = core.step(20)[0]
# obs11 = core.step(20)[0]
# obs12 = core.step(20)[0]
# obs13 = core.step(20)[0]
# obs14 = core.step(20)[0]
# obs15 = core.step(20)[0]
# obs16 = core.step(20)[0]
# obs17 = core.step(20)[0]
# obs18 = core.step(20)[0]
# obs19 = core.step(20)[0]
# obs20 = core.step(20)[0]

# obs1 = core.step(20)[0]
# obs2 = core.step(20)[0]
# obs3 = core.step(20)[0]
# obs4 = core.step(20)[0]
# obs5 = core.step(20)[0]
# obs6 = core.step(20)[0]
# obs7 = core.step(20)[0]
# obs8 = core.step(20)[0]
# obs9 = core.step(20)[0]
# obs10 = core.step(20)[0]


# obs1 = core.step(20)[0]
# obs2 = core.step(20)[0]
# obs3 = core.step(20)[0]
# obs4 = core.step(20)[0]
# obs5 = core.step(20)[0]
# obs6 = core.step(20)[0]
# obs7 = core.step(20)[0]
# obs8 = core.step(20)[0]
# obs9 = core.step(20)[0]
# obs10 = core.step(20)[0]

# obs1 = core.step(20)[0]
# obs2 = core.step(20)[0]
# obs3 = core.step(20)[0]


# core.step(20)
# for i in range(2):
#     print(i)
#     core.step(20)
    
# to_be_updated = core.get_diff_obs(index, obs, new_obs)
# new_updated = core.update(obs, to_be_updated) 
# it should be the updated Flow at the position of index






# %%
class MatchEngine():
    '''One Match Engine is corresponds to one specific Limit Order Book DataSet'''
    def __init__(self):
        self.state = None
    def step(self, action, observation):
        pass
        
# %%
# if __name__ == "__main__":