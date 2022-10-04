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

class OptimalLiquidation(BaseEnv):
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
            
if __name__=="__main__":
    random_strategy(OptimalLiquidation)
