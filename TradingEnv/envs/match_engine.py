# %%
import pandas as pd
from copy import copy
# import numpy as np
# Dir = "/workspaces/Dissertation/TradingEnv/envs/"
Dir = "/Users/kang/GitHub/Dissertation/TradingEnv/envs/"
filename = "AMZN_2021-04-01_34200000_57600000_orderbook_10.csv"
name_lst = []
for i in range(40//4):
    name_lst.append("ask"+str(i+1))
    name_lst.append("ask"+str(i+1)+"_quantity")
    name_lst.append("bid"+str(i+1))
    name_lst.append("bid"+str(i+1)+"_quantity")
    
Flow = pd.read_csv(Dir + filename, names= name_lst)
column_numbers=[i for i in range(40) if i%4==2 or i%4==3]
flow = Flow.iloc[:,column_numbers]
# %%
sample_flow = flow.iloc[0:200,:]
a=sample_flow.diff()
b=-sample_flow.diff()
# %%
def remove_replicate(diff_list):
    present_flow = []
    i = 0
    while True:
        if i==len(diff_list)-1:
            present_flow.append(
            [diff_list[i][0],diff_list[i][1]])
            break
        elif diff_list[i][0] == diff_list[i+1][0]:
            present_flow.append(
                [diff_list[i][0], diff_list[i][1]+diff_list[i+1][1]]
                )    
            i+=2
            if i > len(diff_list): break
        else: 
            present_flow.append(
            [diff_list[i][0],diff_list[i][1]])
            i+=1
            if i > len(diff_list): break
    result = present_flow.copy()
    for j in range(len(present_flow)):
        if present_flow[j][1] == 0:
            result.remove(present_flow[j])
    return result
# new_diff_list = remove_replicate(diff_list)  
# %%
def diff(index,flow):
    b = -flow.diff()
    col_num = b.shape[1] 
    # for item in b.iloc[index,:]:
    diff_list = [] 
    for i in range(col_num):
        if i%2 == 0:
            if b.iat[index,i] !=0 and b.iat[index,i+1] !=0:
                diff_list.append([sample_flow.iat[index,i],sample_flow.iat[index,i+1]])
                diff_list.append([sample_flow.iat[index-1,i],-sample_flow.iat[index-1,i+1]])
    return remove_replicate(diff_list)
diff_list = diff(4,sample_flow)
  
# %%
def update(index,flow):
    '''update at time index based on the observation of index-1'''
    previous_list = list(flow.iloc[index-1,:])
    previous_flow = []
    for i in range(flow.shape[1]):
        if i%2==0:
            previous_flow.append(
                [previous_list[i],previous_list[i+1]]
                ) 
    # temp = diff(index,sample_flow)
    Temp = diff(index,flow)
    previous_flow.extend(Temp)
    new = sorted(previous_flow)
    result = remove_replicate(new)
    return result
updated = update(4, flow)
# %%
# if __name__ == "__main__":