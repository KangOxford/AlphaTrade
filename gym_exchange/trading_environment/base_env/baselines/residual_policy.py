from fractions import Fraction
import numpy as np
from gym_exchange import Config
import random;random.seed(Config.seed)


class Twap():
    def __init__(self):
        self.initialize() # return self.num_list
        self.step_index = 0
    def initialize(self):
        '''

        step_integer = Config.num2liquidate // (Config.max_horizon//3)
        arr = np.full(Config.max_horizon//3, step_integer)
        selected_indices = random.sample(range(len(arr)), Config.num2liquidate - Config.max_horizon//3 * step_integer)
        arr[selected_indices] += 1
        assert arr.sum() == Config.num2liquidate
        new_arr = np.pad(arr, (0, Config.max_horizon - len(arr)), mode='constant')
        '''
        # new_arr = np.full(Config.max_horizon, 0) #$ masked for testing
        '''
        assert len(new_arr) == Config.max_horizon
        # for i in range(len(new_arr)):
        #     print(new_arr[i])#$
        assert new_arr.sum() == Config.num2liquidate
        '''
        '''
        # baseline {
        new_arr = np.full(Config.max_horizon, 9)
        new_arr[1::2] += 3
        new_arr[2::3] += 1
        # baseline }
        '''
        # new_arr = np.full(Config.max_horizon, 5)
        # new_arr[1::2] += 1
        # new_arr[2::3] += 1 # for testing train0.5
        # baseline {
        new_arr = np.full(Config.max_horizon, 10)
        new_arr[1::2] += 1
        # new_arr[2::3] += 1
        # new_arr[3::4] += 1
        new_arr[3::6] += 1
        new_arr[3::41] += 1
        # new_arr[1000:1100] += 1
        # baseline }
        self.num_list = new_arr
    # def initialize(self):
    #     step_integer = Config.num2liquidate // Config.max_horizon
    #     arr = np.full(Config.max_horizon, step_integer)
    #     selected_indices = random.sample(range(len(arr)), Config.num2liquidate - Config.max_horizon * step_integer)
    #     arr[selected_indices] += 1
    #     assert arr.sum() == Config.num2liquidate
    #     # arr = np.full(Config.max_horizon, 0) #$ masked for testing
    #     self.num_list = arr

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


