#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 00:00:09 2022

@author: kang
"""
from gym_trading.utils import * 
from gym_trading.tests import *
from gym_trading.envs.broker import Flag
from gym_trading.envs.optimal_liquidation_v1 import OptimalLiquidation

class OptimalLiquidation_Render(OptimalLiquidation):
    # ===============================  Init  =======================================
    def __init__(self, Flow) -> None:
        super().__init__(Flow)
    # =============================== Render =======================================
    def render_v1(self):
        if self.done:
            RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff = self.calculate_info()
            try: assert self.num_left >= 0, "Error for the negetive left quantity"
            except: 
                raise Exception("Error for the negetive left quantity")
            print("="*30)
            print(">>> FINAL REMAINING(RL) : "+str(format(self.num_left, ',d')))
            print(">>> INIT  REWARD : "+str(format(self.init_reward,',d')))
            print(">>> Upper REWARD : "+str(format(Flag.max_price * Flag.num2liquidate,',d')))
            print(">>> Lower REWARD : "+str(format(Flag.min_price * Flag.num2liquidate,',d')))
            print(">>> Base Point (Bound): "+str(format(BasePointBound))) #(o/oo)
            print(">>> Base Point (Init): "+str(format(BasePointInit))) #(o/oo)
            print(">>> Base Point (RL): "+str(format(BasePointRL))) #(o/oo)
            print(">>> Base Point (Diff): "+str(format(BasePointDiff))) #(o/oo)
            print(">>> Value (Init): "+str(format(self.init_reward,',f'))) #(o/oo)
            print(">>> Value (RL)  : "+str(format(self.memory_revenue,',f'))) #(o/oo)
            print(">>> Value (Performance): "+str(format( (self.memory_revenue/self.init_reward - 1)*100,',f'))) #(o/oo)
            print(">>> Number (Diff): "+str(format( (self.memory_revenue-self.init_reward)/Flag.min_price,',f'))) #(o/oo)

            try: assert RLbp <= Boundbp, "Error for the RL Base Point"
            except:
                raise Exception("Error for the RL Base Point")
            try: assert  self.init_reward >= -1 * Flag.cost_parameter * Flag.num2liquidate * Flag.num2liquidate
            except:
                raise Exception("Error for the Init Lower Bound") 
    def render_v2(self):
        if self.done:
            RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff = self.calculate_info()
            print("="*15 + " BEGIN " + "="*15)
            print(">>> FINAL REMAINING(RL) : "+str(format(self.num_left, ',d')))
            print(">>> Epoch   Length  : "+str(format(self.current_step, ',d')))
            print(">>> Horizon Length  : "+str(format(Flag.max_episode_steps , ',d')))
            print("-"*30)
            print(">>> INIT  REWARD : "+str(format(self.init_reward,',.2f')))
            print(">>> Upper REWARD : "+str(format(Flag.max_price * Flag.num2liquidate/Flag.lobster_scaling,',.2f')))
            print(">>> Lower REWARD : "+str(format(Flag.min_price * Flag.num2liquidate/Flag.lobster_scaling,',.2f')))
            print("-"*30)
            print(">>> TOTAL REWARD : "+str(format(self.memory_reward  + self.init_reward,',.2f'))) # (no inventory) considered ## todo some error
            pairs = self.memory_executed_pairs
            try:    print(f">>> Advantage    : $ {self.memory_reward}, for selling {Flag.num2liquidate} shares of stocks at price {int(get_avarage_price(pairs)/Flag.lobster_scaling)}")
            except: pass
            print("="*15 + "  END  " + "="*15)
            print()
    def render(self, mode = "human"):
        self.render_v2()
        
    # ===============================  Info  =======================================
    def calculate_info(self):
        RLbp = 10000 *(self.memory_revenue/Flag.num2liquidate/ Flag.min_price -1)
        Boundbp = 10000 *(Flag.max_price / Flag.min_price -1)
        BasePointBound = 10000 *(Flag.max_price / Flag.min_price -1)
        BasePointInit = 10000 *(self.init_reward/Flag.num2liquidate/ Flag.min_price -1)
        BasePointRL = 10000 *(self.memory_revenue/Flag.num2liquidate/ Flag.min_price -1)
        BasePointDiff = BasePointRL - BasePointInit        
        return RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff
    def _get_info_v1(self):
        if self.done:
            RLbp, Boundbp, BasePointBound, BasePointInit, BasePointRL, BasePointDiff = self.calculate_info()
            self.info = {"Diff" : BasePointDiff,
                         "Step" : self.current_step,
                         "Left" : self.num_left,
                         "Performance" : (self.memory_revenue/self.init_reward -1 ) * 100 # Performanceormance o/o
                         }
        return self.info 
    def _get_info_v2(self):
        if self.done:
            self.info = {
                         "Step" : self.current_step,
                         "Left" : self.num_left,
                         "Advantage" : self.memory_reward
                         }
        return self.info 
    def _get_info(self):
        return self._get_info_v2()   

    
if __name__=="__main__":
    random_strategy(OptimalLiquidation_Render)
