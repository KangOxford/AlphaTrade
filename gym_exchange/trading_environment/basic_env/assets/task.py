# -*- coding: utf-8 -*-
from gym_exchange import Config

class NumLeftProcessor():
    def __init__(self):
        self.num_left = Config.num2liquidate
    def step(self,Self):
        # self.num_left -= self.cur_action
        agent_executed_pairs = Self.exchange.executed_pairs_recoder.agent_pairs
        if len(agent_executed_pairs) == 0:
            print("*** no agent_executed_pairs in the last step")
        else:
            self.num_left -= 1
        pass
        print()
