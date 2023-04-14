# -*- coding: utf-8 -*-
import numpy as np
from gym_exchange import Config

class NumLeftProcessor():
    def __init__(self):
        self.num_left = Config.num2liquidate
        self.agent_executed_pairs_in_last_step = None # used for the first step as there is not forward step
        self.num_executed_in_last_step = 0 # used for the first step as there is not forward step
    def step(self,Self):
        # self.num_left -= self.cur_action
        agent_executed_pairs_in_last_step = Self.exchange.executed_pairs_recoder.market_agent_executed_pairs_in_last_step['agent_pairs']
        # print(f"agent_executed_pairs_in_last_step:{agent_executed_pairs_in_last_step}")#$
        if agent_executed_pairs_in_last_step is None:
            # print("*** no agent_executed_pairs in the last step")
            pass
        else:
            self.num_left -= agent_executed_pairs_in_last_step[1,:].sum()
            # if agent_executed_pairs_in_last_step[1,:].__len__() != 1:
            #     breakpoint()
        try:
            assert self.num_left >= 0
            self.num_executed_in_last_step = 0 if agent_executed_pairs_in_last_step is None else agent_executed_pairs_in_last_step[1, :].sum()
        except:
            self.num_executed_in_last_step = agent_executed_pairs_in_last_step[1, :].sum() + self.num_left # at this point negative

# class NumHoldProcessor():
#     def __init__(self):
#         self.num_hold = Config.num2liquidate
#     def step(self,Self):
#         pass
