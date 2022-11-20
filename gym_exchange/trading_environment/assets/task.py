# -*- coding: utf-8 -*-
from gym_exchange import Config

class NumLeftProcessor():
    def __init__(self):
        self.num_left = Config.num2liquidate
    def step(self,Self):
        # self.num_left -= self.cur_action
        self.num_left -= 1
        pass
        print()