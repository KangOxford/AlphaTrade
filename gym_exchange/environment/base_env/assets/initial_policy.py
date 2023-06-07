import numpy as np
import pandas as pd

from gym_exchange import Config
import random;random.seed(Config.seed)


class Twap():
    def __init__(self):
        self.initialize() # return self.num_list
        self.step_index = 0
    def initialize(self):
        # # baseline {
        # new_arr = np.full(Config.max_horizon, 1)
        # new_arr[0:490:7] -= 1
        # new_arr
        # # baseline }
        # new_arr = np.full(Config.max_horizon, 0) #$ masked for testing

        # # baseline {
        # # quantity, price
        # new_arr = np.full((Config.max_horizon,2), 1)
        # new_arr[0:490:7,0] -= 1
        # new_arr[::2,1] -= 1
        # # baseline }
        def baseline():
            # baseline {
            # quantity, price
            quantity = round(Config.num2liquidate//Config.max_horizon * 5.75)
            new_arr = np.full((Config.max_horizon,2), (quantity,1)) # quantity, price
            new_arr[::2, 1] -= 1 # price adjustment, passive and aggressive orders one by one
            # baseline }
            return new_arr

        def vwap():
            # vwap {
            # quantity, price
            qty = pd.read_csv("~/vwap_qty.csv").iloc[:,-1][:Config.max_horizon]
            qty = qty/qty.sum()
            factor = 1
            # factor = 100
            # quantity = (round(qty * Config.num2liquidate * 1.70)).astype(np.int64) # aggressive
            quantity = (round(qty * Config.num2liquidate * factor)).astype(np.int64) # passive
            quantity.iloc[-1] = quantity.iloc[-1] + (Config.num2liquidate - quantity.sum())
            assert all(quantity >= 0)
            assert quantity.sum() == Config.num2liquidate
            aggressive = np.full(Config.max_horizon,1) # quantity, price, 1 means aggressive
            passive = np.full(Config.max_horizon,0) # quantity, price, 0 means passive
            # new_arr = np.vstack([quantity,aggressive]).T
            new_arr = np.vstack([quantity,passive]).T
            '''eg.
           [3, 1],
           [3, 1],
           [3, 1],
           [3, 1],
           [3, 1]])
            '''
            # vwap }
            return new_arr
        # new_arr = baseline()
        new_arr = vwap()
        self.num_list = new_arr
    @property
    def done(self):
        if self.step_index < Config.max_horizon: return False
        else: return True
        
    def step(self) -> int:
        """Return the simple actions: numbers to sell/buy"""
        self.action = self.num_list[self.step_index]
        self.step_index +=1
        return self.action, self.done
    
class ResidualPolicy_Factory():
    '''factory methods/ or interface methods
    TWAP is one implemention of ResidualPolicy'''
    @staticmethod
    def produce(name):
        if name == "Twap":
            return Twap()
        else: raise NotImplementedError
    
if __name__ == '__main__':
    action_list = []
    twap = Twap()
    done = False
    while True:
        action, done = twap.step()
        action_list.append(action)
        if done: break
    print(sum(action_list) == Config.num2liquidate)


