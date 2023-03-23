# ========================== 01 ==========================
import abc
import numpy as np
import pandas as pd

from gym_exchange.trading_environment.basic_env.utils import vwap_price

class Vwap(abc.ABC):
    def __init__(self):
        pass
        
    def get_market_vwap(self):
        if self.market_pairs is None:
            market_vwap = None
        else:
            market_vwap = vwap_price(self.market_pairs)
        self.market_vwap = market_vwap
        return self.market_vwap
    
    def get_agent_vwap(self):
        if self.agent_pairs is None:
            agent_vwap = None
        else:
            agent_vwap = vwap_price(self.agent_pairs)
        self.agent_vwap = agent_vwap
        return agent_vwap
    
    def get_vwap_slippage(self):
        if self.agent_vwap is not None and self.market_vwap is not None:
            vwap_slippage = self.market_vwap - self.agent_vwap
        else:
            vwap_slippage = None
        self.vwap_slippage = vwap_slippage
        return vwap_slippage
    
    @abc.abstractmethod
    def update(self,executed_pairs):
        '''update the market pairs and agent pairs'''
    
    @property
    @abc.abstractmethod
    def info_dict(self):
        '''return the info dict'''
        
    # @abc.abstractmethod    
    # def step(self, executed_pairs):
    #     '''step'''
    
# class StepVwap(Vwap):
#     def __init__(self):
#         super().__init__()
    
#     def update(self, executed_pairs):
#         if len(executed_pairs.market_pairs) == 0 or len(executed_pairs.agent_pairs) == 0 :
#             if len(executed_pairs.market_pairs) == 0: self.market_vwap = 0
#             if len(executed_pairs.agent_pairs) == 0: self.agent_vwap = 0
#             print()# % check self.market_vwap 
#         else:
#             self.market_pairs = executed_pairs.market_pairs[-1]
#             self.agent_pairs  = executed_pairs.agent_pairs[-1]
            
#     @Vwap.market_vwap.setter
#     def market_vwap(self, value):
#         self._market_vwap = value
        
#     @Vwap.agent_vwap.setter
#     def agent_vwap(self, value):
#         self._agent_vwap = value
    
#     @property
#     def info_dict(self):
#         return {
#             "StepVwap/MarketVwap"  :self.market_vwap,
#             "StepVwap/AgentVwap"   :self.agent_vwap,
#             "StepVwap/VwapSlippage":self.vwap_slippage
#         }
        
    
# ========================== 02 ==========================

class EpochVwap(Vwap):
    def __init__(self):
        super().__init__()
        
    def update(self,executed_pairs):
        self.market_pairs = executed_pairs['market_pairs']
        self.agent_pairs  = executed_pairs['agent_pairs']   
        

    @property
    def info_dict(self):
        result = {
            "EpochVwap/MarketVwap"  :self.get_market_vwap(),
            "EpochVwap/AgentVwap"   :self.get_agent_vwap(),
            "EpochVwap/VwapSlippage":self.get_vwap_slippage()
        }
        return result
    
            
    
# ========================== 03 ==========================
# class StepVwap_MA():
#     '''Moving average of step vwap'''
#     def __init__(self,):
#         pass

# class VwapCurve():
#     ''' curve format
#         index | 0 | 1 | 2 | 3 | 4 | 5 |
#         pirce |0.1|0.2|0.3|0.4|0.5|0.6|'''
#     def __init__(self,):
#         self.index = 0
        
#     def to_array(self):
#         result = 0 #TODO: implement
#         return np.array(result)
    
# ========================== 04 ==========================
class VwapEstimator():
    def __init__(self):
        # self.step_vwap = StepVwap() # Used for info
        self.epoch_vwap= EpochVwap()# Used for info
    def executed_pairs_adapter(self, executed_pairs):
        market_pairs_dict = executed_pairs.market_pairs
        agent_pairs_dict = executed_pairs.agent_pairs
        concat_pairs = lambda pairs_dict: np.concatenate(list(pairs_dict.values()),axis = 1)
        return_dict = {
            "market_pairs": concat_pairs(market_pairs_dict),
            "agent_pairs" : concat_pairs(agent_pairs_dict) if len(agent_pairs_dict) !=0 else None
            }        
        return return_dict 
    def update(self, executed_pairs, done):
        self.done = done
        if done:
            executed_pairs = self.executed_pairs_adapter(executed_pairs)
            self.epoch_vwap.update(executed_pairs)
    def step(self):
        if not self.done:
            return None, None
            # return self.step_vwap.info_dict, None
        else:
            return None, self.epoch_vwap.info_dict
            # return self.step_vwap.info_dict, self.epoch_vwap.info_dict
        
        
if __name__ == "__main__":
    # vwap_price testing
    pairs = np.array([[1,2],[1,23],[1,3],[1.1,21],[0.9,3]]).T

    # ======================================= plot ============================
    import pandas as pd
    import inspect


    # ----------------------- func ---------------------
    def get_var_name(var, caller_locals=None):
        if caller_locals is None:
            caller_locals = inspect.currentframe().f_back.f_locals

        for name, value in caller_locals.items():
            if value is var:
                return name

    # ----------------------- date ---------------------
    # recorder = env.exchange.executed_pairs_recoder
    recorder = self.exchange.executed_pairs_recoder
    agent_pairs = recorder.agent_pairs
    market_pairs = recorder.market_pairs
    from gym_exchange.trading_environment.basic_env.utils import vwap_price
    agent_step_vwap = {k:vwap_price(v) for k,v in agent_pairs.items()}
    market_step_vwap = {k:vwap_price(v) for k,v in market_pairs.items()}
    mid_prices = {k:v for k,v in enumerate(self.exchange.mid_prices)}
    # ----------------------- fig ---------------------
    import matplotlib.pyplot as plt
    # plt.rcParams["figure.figsize"] = (80, 40)
    plt.rcParams["figure.figsize"] = (40, 20)
    def curve_interpolation(market_step_vwap):
        name = get_var_name(market_step_vwap, inspect.currentframe().f_back.f_locals)
        # print(name)
        # print('type: ',type(name))
        from scipy.interpolate import interp1d
        x = np.array(list(market_step_vwap.keys()))
        y = np.array(list(market_step_vwap.values()))
        f = interp1d(x, y, kind='cubic')
        x_new = np.linspace(x.min(), x.max(), num=10000000)
        y_new = f(x_new)
        plt.plot(x_new, y_new, label= name + '_interpolated')
    ask_market_step_vwap = {}
    bid_market_step_vwap = {}
    for key,value in market_step_vwap.items():
        if  value > mid_prices.get(key):
            ask_market_step_vwap[key] = value
        elif value < mid_prices.get(key):
            bid_market_step_vwap[key] = value
        else:
            raise NotImplementedError
    curve_interpolation(ask_market_step_vwap)
    curve_interpolation(bid_market_step_vwap)
    # plt.scatter(market_step_vwap.keys(),market_step_vwap.values(),label='Market')
    # plt.plot(market_step_vwap.keys(),market_step_vwap.values(),label='Market')
    # plt.plot(agent_step_vwap.keys(),agent_step_vwap.values(),label='Agent')
    plt.plot(pd.Series(mid_prices))
    plt.plot(pd.Series(ask_market_step_vwap), label= "ask_market_step_vwap" + '_interpolated')
    plt.plot(pd.Series(bid_market_step_vwap), label= "bid_market_step_vwap" + '_interpolated')
    plt.scatter(agent_step_vwap.keys(),agent_step_vwap.values(),label='Agent', color='blue')
    plt.legend()
    # plt.title("Action(direction = 'ask', quantity_delta = 0, price_delta = 1)")
    # plt.title("Action(direction = 'ask', quantity_delta = 0, price_delta = -1)")
    # plt.title("Action(direction = 'ask', quantity_delta = 0, price_delta = 0)")
    plt.title("Action(direction = 'bid', quantity_delta = 0, price_delta = 0)")
    # plt.title("Action(direction = 'bid', quantity_delta = 5, price_delta = -1)")
    # plt.savefig("")
    plt.show()

    
    
    
    
    
    
    
    
    
    
