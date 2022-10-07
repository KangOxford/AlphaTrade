# %%
import copy
import numpy as np
# import cudf
from tabulate import tabulate
import pandas as pd

class Flag():
    lobster_scaling = 10000 # Dollar price times 10000 (i.e., A stock price of $91.14 is given by 911400)
    max_episode_steps= 12000 # 10 mins
    # max_episode_steps= 1200 # 1 min
    # max_episode_steps= 600 # 1/2 min
    
    num2liquidate = 2000 # 10 min
    # num2liquidate = 200 # 1 min
    # num2liquidate = 100 # 1/2 min
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
    low_dimension_penalty_parameter = 1 # todo not sure
    cost_parameter = 5e-6 # from paper.p29 : https://epubs.siam.org/doi/epdf/10.1137/20M1382386
    skip = 1 # 50 miliseconds
    # skip = 2 # default = 1 from step No.n to step No.n+1
    # skip = 20 # 1 second 
    # skip = 200 # 10 seconds
    # skip = 1200 # 1 minute
    price_level = 10
    test_seed = 2022
    pretrain_steps = int(1e3)
    runing_penalty_parameter = 100
    time_window_size = 1
    min_num_left = 0
    max_num_left = num2liquidate
    min_step_left= 0
    max_step_left = max_episode_steps
    state_dim_1 = 2
    state_dim_2 = 12 # used to be 10
    state_dim_3 = time_window_size
    
    @classmethod
    def log(cls, log_string = None):
        dct = dict(cls.__dict__)
        dct = [[k,v] for k,v in dct.items() if type(v) == float or type(v) == int]
        table = tabulate(dct, headers = ('Parameters','Value'), tablefmt='psql', numalign="right")
        print(table)
        if log_string is not None:
            with open(log_string+".txt", 'w') as f:
                f.write(table)
                
        
            

    
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
                num = max(num-obs[1][i], 0)
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
        num = copy.deepcopy(num)
        level, executed_num = cls._level_market_order_liquidating(num, obs)
        # TODO need the num <=609 the sum of prices at all levels
        
        result = []
        if level>1:
            for i in range(level-1):
                result.append([obs[0][i],-obs[1][i]])
            num_left = num + sum([item[1] for item in result])
            result.append([obs[0][level-1], -1 * (min(num_left, obs[1][level-1]))]) # apppend the last item 
            for item in result: assert item[1]<=0 ##
        elif level == 1:
            # result.append([obs[0][0],-num]) # to check it should be wrong
            result.append([obs[0][0],-executed_num])
            '''-executed_num, to be negative means the quantity is removed from the lob
            '''
        elif level == 0:
            result = []
        elif level == -999:
            obs[1,:] *= -1
            result = obs.copy()
        else: raise NotImplementedError
        assert executed_num>=0
        # -------------------------
        if type(result) == list:
            assert  sum([-1*item[1] for item in result]) == executed_num# the result should corresponds to the real executed quantity
            result = np.array(result).T # keep the shape first line: price and second line: quantity
        elif type(result) == np.ndarray:
            assert  -sum(result[1,:]) == executed_num# the result should corresponds to the real executed quantity
        return result, executed_num

if __name__ == "__main__":
    log_string = "/Users/kang/GitHub/NeuralLOB/venv_rnn-v5/Sep_26/rnn_ppo_gym_trading-Mon-Sep-26-19-58-55-2022"
    Flag.log(log_string)
