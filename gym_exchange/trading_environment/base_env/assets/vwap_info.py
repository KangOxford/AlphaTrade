# ========================== 01 ==========================
import abc
import numpy as np
import pandas as pd

from gym_exchange.trading_environment.base_env.utils import vwap_price

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
            vwap_slippage = None # TODO whether it would be called
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

    
# ========================== 02 ==========================

class StepVwap(Vwap):
    def __init__(self):
        super().__init__()

    def update(self, executed_pairs):
        if executed_pairs['market_pairs'] is None:
            self.market_vwap = 0
        if executed_pairs['agent_pairs'] is None:
            self.agent_vwap = 0
        if executed_pairs['agent_pairs'] is not None and executed_pairs['market_pairs'] is not None:
            self.market_pairs = executed_pairs['market_pairs']
            self.agent_pairs  = executed_pairs['agent_pairs'] # TODO not sure whether need [-1](original)
            # assert self.market_pairs.shape == self.market_pairs.shape and self.market_pairs.shape == (2,1) # TODO not sure whether need [-1](original)
            self.market_vwap = self.get_market_vwap()
            self.agent_vwap = self.get_agent_vwap()
        self.vwap_slippage = self.get_vwap_slippage()

    @property
    def info_dict(self):
        return {
            "StepVwap/MarketVwap"  :self.market_vwap,
            "StepVwap/AgentVwap"   :self.agent_vwap,
            "StepVwap/VwapSlippage":self.vwap_slippage
        }

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
class VwapEstimator():
    def __init__(self):
        self.step_vwap = StepVwap() # Used for info
        self.epoch_vwap= EpochVwap()# Used for info
    def executed_pairs_adapter(self, executed_pairs):
        market_pairs_dict = executed_pairs.market_pairs
        agent_pairs_dict = executed_pairs.agent_pairs
        concat_pairs = lambda pairs_dict: np.concatenate(list(pairs_dict.values()),axis = 1)
        return_dict = {
            "market_pairs": concat_pairs(market_pairs_dict) if len(agent_pairs_dict) !=0 else None,
            "agent_pairs" : concat_pairs(agent_pairs_dict) if len(agent_pairs_dict) !=0 else None
            }        
        return return_dict 
    def update(self, executed_pairs, done):
        self.done = done
        self.step_vwap.update(executed_pairs.market_agent_executed_pairs_in_last_step)
        if done:
            self.epoch_vwap.update(executed_pairs = self.executed_pairs_adapter(executed_pairs))
    def step(self):
        if not self.done:
            # return None, None
            return self.step_vwap.info_dict, None
        else:
            # return None, self.epoch_vwap.info_dict
            return self.step_vwap.info_dict, self.epoch_vwap.info_dict
        
        
if __name__ == "__main__":
    # vwap_price testing
    pairs = np.array([[1,2],[1,23],[1,3],[1.1,21],[0.9,3]]).T


    
    
    
    
    
    
    
    
    
    
