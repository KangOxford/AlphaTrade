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

class utils():
    def from_series2pair(index,flow):
        previous_list = list(flow.iloc[index,:])
        previous_flow = []
        for i in range(flow.shape[1]):
            if i%2==0:
                previous_flow.append(
                    [previous_list[i],previous_list[i+1]]
                    ) 
        return previous_flow
    def from_pair2series(name_lst, flow):
        flow = sorted(flow,reverse=True)
        new_name = []
        for i in range(len(name_lst)):
            if i%4==2 or i%4==3:
                new_name.append(name_lst[i])
        result = []
        for item in flow:
            result.append(item[0])
            result.append(item[1])
        return pd.Series(data=result, index = new_name)
    

# %%
# class Diff():
def get_diff(index, flow):
    b = -flow.diff()
    col_num = b.shape[1] 
    # for item in b.iloc[index,:]:
    diff_list = [] 
    for i in range(col_num):
        if i%2 == 0:
            # if b.iat[index,i] !=0 and b.iat[index,i+1] !=0:
            #     diff_list.append([sample_flow.iat[index,i],sample_flow.iat[index,i+1]])
            #     diff_list.append([sample_flow.iat[index-1,i],-sample_flow.iat[index-1,i+1]])
            # elif b.iat[index,i] !=0 and b.iat[index,i+1] ==0:
            if b.iat[index,i] !=0 or b.iat[index,i+1] !=0:
                diff_list.append([sample_flow.iat[index,i],sample_flow.iat[index,i+1]])
                diff_list.append([sample_flow.iat[index-1,i],-sample_flow.iat[index-1,i+1]])
    return diff_list
diff_list = get_diff(4, flow)
# %%
def remove_replicate(diff_list):
    present_flow = []
    i = 0
    while True:
        print(">>> diff_list length is {}, list is {} ##### present_flow,length is {}, list is {}".format(len(diff_list), diff_list, len(present_flow), present_flow))
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
    print(">>>while loop in remove_replicate is alright")
    result = present_flow.copy()
    for j in range(len(present_flow)):
        if present_flow[j][1] == 0:
            result.remove(present_flow[j])
    print(">>>for loop in remove_replicate is alright")
    return result
new_diff_list = remove_replicate(diff_list)  
# %%
def diff(index,flow):
    return remove_replicate(get_diff(index,flow))
New_diff_list = diff(4,sample_flow)

# %%
def update(index,flow, diff_obs):
    '''update at time index based on the observation of index-1'''
    previous_flow = utils.from_series2pair(index-1,flow)
    previous_flow.extend(diff_obs)
    new = sorted(previous_flow)
    result = remove_replicate(new)
    return result
index = 4
diff_obs = diff(index, flow)
updated = update(index, flow, diff_obs)
represented4 = utils.from_pair2series(name_lst, updated)
sample_flow.iloc[index,:] == represented4
# %%
# class broker
def level_market_order_liquidating(num, obs):
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
# num, obs = 20, utils.from_series2pair(3, flow)
# level_market_order_liquidating(num, obs)
#%%
def pairs_market_order_liquidating(num, obs):
    level = level_market_order_liquidating(num, obs)
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
index = 4
num, obs = 20, utils.from_series2pair(index-1, flow)
new_obs = pairs_market_order_liquidating(num, obs)
# %%
# def broker_with_diff():
diff_obs = diff(index, flow)
diff_obs.extend(new_obs)
to_be_updated = remove_replicate(diff_obs)
# %%
new_updated = update(index, flow, to_be_updated)


    
# %%
# if __name__ == "__main__":