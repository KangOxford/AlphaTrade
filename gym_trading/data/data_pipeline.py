# -*- coding: utf-8 -*-
import os.path
import pandas as pd
class Debug():
    if_return_single_flie = True
    if_whole_data = False

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
        def namelist():
            name_lst = []
            for i in range(40//4):
                name_lst.append("ask"+str(i+1))
                name_lst.append("ask"+str(i+1)+"_quantity")
                name_lst.append("bid"+str(i+1))
                name_lst.append("bid"+str(i+1)+"_quantity")
            return name_lst
        if not Debug.if_return_single_flie:
            Flow_list = []
            mypath = "/Users/kang/Data/Learning/Training/" if not Debug.if_whole_data else "/Users/kang/Data/Learning_full/Training/"
            from os import listdir
            from os.path import isfile, join
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            ### rm /Users/kang/Data/Learning/Training/.DS_Store
            ### rm /Users/kang/Data/Learning/Testing/.DS_Store
            for path in onlyfiles:
                # if (mypath + path)[-9:]  == ".DS_Store":  import os; command = "rm " + (mypath + path); os.system(command)
                df = pd.read_csv(mypath + path,names = namelist())
                column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
                Flow = df.iloc[:,column_numbers]
                Flow_list.append(Flow)
            return Flow_list
        if Debug.if_return_single_flie: 
            path = "/Users/kang/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv"
            if not os.path.exists(path):
                url = "https://drive.google.com/file/d/1UawhjR-9bEYYns7PyoZNym_awcyVko0i/view?usp=sharing"
                path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]   
                
            df = pd.read_csv(path,names = namelist())
            column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
            Flow = df.iloc[:,column_numbers]
            return Flow
     
if __name__ == "__main__":
    Debug.if_return_single_flie = False
    Flow = ExternalData.get_sample_order_book_data()
    flow = Flow.iloc[3:1027,:].reset_index().drop("index",axis=1)
    # datapipeline = DataPipeline(ExternalData.get_sample_order_book_data())
    # data = datapipeline.reset()
    # data = datapipeline.step()
    
    # def get_price_list(flow):
    #     price_list = []
    #     column_index = [i*2 for i in range(0,10)]
    #     for i in range(flow.shape[0]):
    #         price_list.extend(flow.iloc[i,column_index].to_list())
    #     price_set = set(price_list)
    #     price_list = sorted(list(price_set), reverse = True)
    #     return price_list
    # price_list = get_price_list(flow)
    
    # def get_max_quantity(flow):
    #     price_list = []
    #     column_index = [i*2 + 1 for i in range(0,flow.shape[1]//2)]
    #     for i in range(flow.shape[0]):
    #         price_list.extend(flow.iloc[i,column_index].to_list())
    #     price_set = max(price_list)
    #     return price_set
    # max_quantity = get_max_quantity(Flow)
    
    
# =============================================================================
#     def get_min_num2liuquidate(flow):
#         datalist = []
#         for index in range(flow.shape[0]):
#             data =  sum(flow.iloc[index,[2*i+1 for i in range(10)]].to_list())
#             datalist.append(data)
#         result = min(datalist)
#         return result
#     result = get_min_num2liuquidate(Flow)
# =============================================================================
    
    
    # %%
    # num = 24
    # from gym_trading.envs.match_engine import Broker, Utils
    # obs = Utils.from_series2pair(stream)
    # level = Broker._level_market_order_liquidating(num, obs)
    # reward = 0
    # consumed = 0
    # for i in range(level-1):
    #     reward += obs[i][0] * obs[i][1]
    #     consumed += obs[i][1]
    # reward += obs[level-1][0] * (num - consumed)

        
    # index_list = [2*i+1 for i in range(level-1)]


    # %%    
    # def get_max_price(flow):
    #     price_list = []
    #     column_index = [i*2  for i in range(0,flow.shape[1]//2)]
    #     for i in range(flow.shape[0]):
    #         price_list.extend(flow.iloc[i,column_index].to_list())
    #     price_set = max(price_list)
    #     return price_set
    # max_price = get_max_price(Flow)
    
    # def get_min_price(flow):
    #     price_list = []
    #     column_index = [i*2  for i in range(0,flow.shape[1]//2)]
    #     for i in range(flow.shape[0]):
    #         price_list.extend(flow.iloc[i,column_index].to_list())
    #     price_set = min(price_list)
    #     return price_set
    # min_price = get_min_price(Flow)
    
    # max_price = 0 
    # for i in range(len(Flow_list)):
    #     max_price = max(max_price, get_max_price(Flow_list[i]))

    # min_price = 0 
    # for i in range(len(Flow_list)):
    #     min_price = max(min_price, get_min_price(Flow_list[i]))

                
    
    
    
