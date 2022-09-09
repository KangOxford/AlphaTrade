# %%
import copy
import numpy as np
# import cudf
import pandas as pd

class Flag():
    lobster_scaling = 10000 # Dollar price times 10000 (i.e., A stock price of $91.14 is given by 911400)
    max_episode_steps= 2048 # max_episode_steps = 10240 # to test in 10 min, long horizon # size of a flow
    max_action = 300
    max_quantity = 300 # TODO is it the same function with max_action?
    # max_quantity = 6000 # TODO is it the same function with max_action?
    # max_price = 31620700 # single file
    # max_price = 34595400 # whole data
    max_price = 35000000 # upper bound
    # min_price = 31120200 # single file
    # min_price = 30000000 # whole data
    min_price = 30000000 # lower bound
    min_quantity = 0
    scaling = 30000000
    num2liquidate = 1000
    cost_parameter = 5e-6 # from paper.p29 : https://epubs.siam.org/doi/epdf/10.1137/20M1382386
    skip = 1 # default = 1 from step No.n to step No.n+1
    # skip = 20 # for 1 second on average
    
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
        assert executed_num>=0
        return result, executed_num
    
    @classmethod
    def pairs_market_order_liquidating(cls, num, obs):
        # num, obs = action, state ##
        num = copy.deepcopy(num)
        # level, executed_num = Broker._level_market_order_liquidating(num_left, diff_obs)##
        level, executed_num = cls._level_market_order_liquidating(num, obs)
        # TODO need the num <=609 the sum of prices at all leveles
        
        result = []
        if level>1:
            for i in range(level-1):
                result.append([obs[i][0],-obs[i][1]])
            num_left = num + sum([item[1] for item in result])
            result.append([obs[level-1][0], -1 * (min(num_left, obs[level-1][1]))]) # apppend the last item 
            for item in result: assert item[1]<=0 ##
        if level == 1:
            # result.append([obs[0][0],-num]) # to check it should be wrong
            result.append([obs[0][0],-executed_num])
            '''-executed_num, to be negative means the quantity is removed from the lob
            '''
        if level == 0:
            result = []
        if level == -999:
            minus_list = [[item[0],-1*item[1]] for item in obs]
            result.extend(minus_list)
        assert executed_num>=0
        assert sum([-1*item[1] for item in result]) == executed_num# the result should corresponds to the real executed quantity
        return result, executed_num

if __name__ == "__main__":
    pass