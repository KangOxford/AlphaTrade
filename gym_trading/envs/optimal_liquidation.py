#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 00:00:09 2022

@author: kang
"""
from gym_trading.utils import * 
from gym_trading.tests import *
from gym_trading.envs.broker import Flag
from gym_trading.envs.base_environment import BaseEnv



class OptimalLiquidation_v1(BaseEnv):
    # ===============================  Init  =======================================
    def __init__(self, Flow) -> None:
        super().__init__(Flow)
    # =============================== Reward =======================================
    def _train_running_penalty(self):
        if self.num_reset_called <= Flag.pretrain_steps:
            def get_twap_num_left(x):
                return Flag.num2liquidate - Flag.num2liquidate/Flag.max_episode_steps * x
            # penalty_delta = max(self.num_left-get_twap_num_left(self.current_step), 0)
            twap_delta = Flag.num2liquidate//Flag.max_episode_steps+1
            # penalty_delta = min(2*twap_delta, max( twap_delta - self.action ,0))
            if self.num_reset_called <= 300:
                penalty_delta = max(twap_delta - self.action , -2*twap_delta)
            else: 
                penalty_delta = max(twap_delta - self.action , 0)
            if type(Flag.runing_penalty_parameter * penalty_delta) == int: result = (Flag.runing_penalty_parameter * penalty_delta)
            else : result = (Flag.runing_penalty_parameter * penalty_delta)[0] 
            # breakpoint()
            self.penalty_delta += result
            return result
        else:
            return 0
    def _get_reward(self):
        if not self.done: 
            return self.memory_revenues[-1] - self._low_dimension_penalty() - self._train_running_penalty()
        elif self.done:
            self.final_reward = self.memory_revenues[-1] - self._get_inventory_cost() - self._low_dimension_penalty()
            return self.final_reward
        
class OptimalLiquidation_v2(BaseEnv):
    # ===============================  Init  =======================================
    def __init__(self, Flow) -> None:
        super().__init__(Flow)
    # =============================== Reward =======================================
    def _get_reward(self):
        if not self.done: 
            return self.memory_revenues[-1] - self._low_dimension_penalty() 
        elif self.done:
            self.final_reward = self.memory_revenues[-1] - self._get_inventory_cost() - self._low_dimension_penalty()
            return self.final_reward
        
class OptimalLiquidation_v3(BaseEnv):  
    # ===============================  Init  =======================================
    def __init__(self, Flow) -> None:
        super().__init__(Flow)  
    # =============================== Init Reward =======================================
    def liquidate_base_func(self, action):
        '''
        Function:
        -------
        Set properties
        
        Modified:
        -------
        self.init_reward
        '''
        num2liquidate = Flag.num2liquidate
        max_action = Flag.max_action
        # count = 0 #tbd
        while num2liquidate > 0:
            observation, num_executed =  self.core_step(min(action, num2liquidate)) # observation(only for debug), num_executed for return
            num2liquidate -= num_executed
            
            executed_pairs = self.core.executed_pairs
            if executed_pairs.size != 0:
                Quantity = executed_pairs[1,:]# if x<TWAPw0, meaning withdraw order
                assert -sum(Quantity) == num_executed # the exected_pairs in the core is for each step
                reward = -1 * sum(executed_pairs[0,:] * executed_pairs[1,:]) 
            else:
                reward = 0
            # get reward
            
            self.init_reward += reward
            # self_init_reward = self.init_reward #tbd
            # self_core_done = self.core.done #tbd
            if self.core.done:# still left stocks after selling all in the core.flow
                inventory = num2liquidate
                self.init_reward -= Flag.cost_parameter * inventory * inventory
                break
            # count += 1 #tbd
            # if count >= 1000: #tbd
            #     breakpoint() #tbd
            #     print()
        self.init_reward /= Flag.lobster_scaling # add this line to convert it to the dollar measure # self.core.executed_sum
     
    @exit_after    
    def liquidate_init_position(self):
        self.liquidate_base_func(Flag.max_action)
    
    @exit_after
    def liquidate_twap(self):
        avarage_action = Flag.num2liquidate//Flag.max_episode_steps + 1
        self.liquidate_base_func(avarage_action)
        
    def liquidate_vanilla(self):
        self.init_reward = 0
        
    def liquidate_zero(self):
        self.liquidate_base_func(0)

    def _set_init_reward(self):
        # self.liquidate_init_position() # policy #1 : choose to sell all at init time
        # self.liquidate_twap() # policy #2 : choose to sell averagely across time
        self.liquidate_vanilla() # policy #3 : choose to sell nothing should be default setting
        # self.liquidate_zero() # policy #4 : choose to debug
        
        self.init_reward_bp = self.init_reward/Flag.num2liquidate
            
if __name__=="__main__":
    random_strategy(OptimalLiquidation)
