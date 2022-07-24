# -*- coding: utf-8 -*-
import os.path
import pandas as pd


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
        path = "/Users/kang/AMZN_2021-04-01_34200000_57600000_orderbook_10.csv"
        if not os.path.exists(path):
            url = "https://drive.google.com/file/d/1UawhjR-9bEYYns7PyoZNym_awcyVko0i/view?usp=sharing"
            path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]        
        df = pd.read_csv(path,names = namelist())
        column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
        Flow = df.iloc[:,column_numbers]
        return Flow
     
if __name__ == "__main__":
    Flow = ExternalData.get_sample_order_book_data()
    flow = Flow.iloc[3:1000,:].reset_index().drop("index",axis=1)
    # datapipeline = DataPipeline(ExternalData.get_sample_order_book_data())
    # data = datapipeline.reset()
    # data = datapipeline.step()