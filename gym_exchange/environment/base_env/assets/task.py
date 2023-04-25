# -*- coding: utf-8 -*-
import numpy as np
from gym_exchange import Config
def get_slice(aps, real_executed):
    total_quantity = np.sum(aps[1])

    if real_executed > total_quantity:
        print("Not enough quantity available.")
    else:
        quantity_needed = real_executed
        slice_start = 0
        slice_end = 0
        for i in range(aps.shape[1]):
            if quantity_needed > aps[1,i]:
                quantity_needed -= aps[1,i]
                slice_end += 1
            else:
                slice_end += 1
                break
        aps_slice = aps[:,slice_start:slice_end]
        aps_slice[1,-1] = quantity_needed
        print(aps_slice)
        return aps_slice

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
            last_executed = agent_executed_pairs_in_last_step[1,:].sum()
            real_executed = min(self.num_left, last_executed)
            self.num_left = max(0, self.num_left - last_executed)
            if self.num_left == 0:
                aps = Self.exchange.executed_pairs_recoder.market_agent_executed_pairs_in_last_step['agent_pairs']
                aps_slice = get_slice(aps, real_executed)
                Self.exchange.executed_pairs_recoder.market_agent_executed_pairs_in_last_step['agent_pairs'] = aps_slice
                print() #$
            # self.num_left -= last_executed
            # if self.num_left < 0:
            #     real_executed = self.num_left + last_executed
            #     print() #$
            # if agent_executed_pairs_in_last_step[1,:].__len__() != 1:
            #     breakpoint()
        self.num_executed_in_last_step = 0 if agent_executed_pairs_in_last_step is None else real_executed
        assert self.num_left >= 0

# class NumHoldProcessor():
#     def __init__(self):
#         self.num_hold = Config.num2liquidate
#     def step(self,Self):
#         pass
