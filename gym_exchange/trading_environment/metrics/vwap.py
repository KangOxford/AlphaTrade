# ========================== 01 ==========================
import abc
class Vwap(abc.ABC):
    def __init__(self):
        pass
    
    def vwap_price(self, pairs):
        ''' pairs format
        price:    array([[ 1. ,  1. ,  1. ,  1.1,  0.9],
        quantity:        [ 2. , 23. ,  3. , 21. ,  3. ]])
        '''
        vwap_price = (pairs[0]*pairs[1]).sum()/pairs[1].sum()
        return vwap_price
    
    @property
    def market_vwap(self):
        return self.vwap_price(self.market_pairs)
    
    @property
    def agent_vwap(self):
        return self.vwap_price(self.agent_pairs)
    
    @property
    def vwap_slippage(self):
        return self.market_vwap - self.agent_vwap
    
    @abc.abstractmethod
    def update(self,executed_pairs):
        '''update the market pairs and agent pairs'''
    
    @abc.abstractmethod
    @property
    def info_dict(self):
        '''return the info dict'''
        
    def step(self, executed_pairs):
        self.update(executed_pairs)
        return self.info_dict    
    
class StepVwap(Vwap):
    def __init__(self):
        super().__init__()
    
    def update(self, executed_pairs):
        self.market_pairs = executed_pairs.market_pairs[-1]
        self.agent_pairs  = executed_pairs.agent_pairs[-1]
    
    @property
    def info_dict(self):
        return {
            "StepVwap/MarketVwap"  :self.market_vwap,
            "StepVwap/AgentVwap"   :self.agent_vwap,
            "StepVwap/VwapSlippage":self.vwap_slippage
        }
        
    
# ========================== 02 ==========================
class EpochVwap(Vwap):
    def __init__(self):
        super().__init__()
        
    @property
    def info_dict(self):
        return {
            "EpochVwap/MarketVwap"  :self.market_vwap,
            "EpochVwap/AgentVwap"   :self.agent_vwap,
            "EpochVwap/VwapSlippage":self.vwap_slippage
        }
            
    def update(self,executed_pairs):
        self.market_pairs = np.concatenate(executed_pairs.market_pairs) # TODO: test
        self.agent_pairs  = np.concatenate(executed_pairs.agent_pairs)        
    
# ========================== 03 ==========================
class StepVwap_MA():
    '''Moving average of step vwap'''
    def __init__(self,):
        pass

class VwapCurve():
    ''' curve format
        index | 0 | 1 | 2 | 3 | 4 | 5 |
        pirce |0.1|0.2|0.3|0.4|0.5|0.6|'''
    def __init__(self,):
        self.index = 0
        
    def to_array(self):
        result = 0 #TODO: implement
        return np.array(result)
    
# ========================== 04 ==========================
class VwapEstimator():
    def __init__(self):
        self.step_vwap = StepVwap() # Used for info
        self.epoch_vwap= EpochVwap()# Used for info
    def step(self, executed_pairs, done):
        self.step_vwap.step(executed_pairs)
        if done: self.epoch_vwap.step(executed_pairs)
        
if __name__ == "__main__":
    # vwap_price testing
    import numpy as np
    pairs = np.array([[1,2],[1,23],[1,3],[1.1,21],[0.9,3]]).T
    
    
    
    
    
    
    
    
    
    
    
