# ========================== 01 ==========================
import abc
from gym_exchange.trading_environment.utils import vwap_price
class Vwap(abc.ABC):
    def __init__(self):
        self.market_vwap = None
        self.agent_vwap = None
        self.vwap_slippage = None
        
    def get_market_vwap(self):
        self._market_vwap = vwap_price(self.market_pairs)
        return self._market_vwap
    
    def get_agent_vwap(self):
        self._agent_vwap = vwap_price(self.agent_pairs)
        return self._agent_vwap
    
    def get_vwap_slippage(self):
        self._vwap_slippage = self.market_vwap - self.agent_vwap
        return self._vwap_slippage
    
    
    @abc.abstractmethod
    def update(self,executed_pairs):
        '''update the market pairs and agent pairs'''
    
    @property
    @abc.abstractmethod
    def info_dict(self):
        '''return the info dict'''
        
    def step(self, executed_pairs):
        self.update(executed_pairs)
        return self.info_dict    
    
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
        self.market_pairs = np.concatenate(executed_pairs.market_pairs) # TODO: test
        self.agent_pairs  = np.concatenate(executed_pairs.agent_pairs)        

    @property
    def info_dict(self):
        return {
            "EpochVwap/MarketVwap"  :self.market_vwap,
            "EpochVwap/AgentVwap"   :self.agent_vwap,
            "EpochVwap/VwapSlippage":self.vwap_slippage
        }
            
    
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
        result = 0
        print()#$
        raise NotImplementedError
        return result 
    def step(self, executed_pairs, done):
        # self.step_vwap.step(executed_pairs)
        

        if done: 
            executed_pairs = self.executed_pairs_adapter(executed_pairs)
            self.epoch_vwap.step(executed_pairs)
        
if __name__ == "__main__":
    # vwap_price testing
    import numpy as np
    pairs = np.array([[1,2],[1,23],[1,3],[1.1,21],[0.9,3]]).T
    
    
    
    
    
    
    
    
    
    
    
