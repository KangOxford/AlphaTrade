from fractions import Fraction
from gym_exchange.trading_environment import Config


class Twap():
    def __init__(self):
        self.step_integer, self.numerator, self.denominator = self.initialize()
        self.numerator_list = [i for i in range(self.numerator)]
        self.step_index = 0
    def initialize(self):
        step_integer = Config.num2liquidate//Config.max_horizon
        step_fraction= Fraction(Config.num2liquidate, Config.max_horizon)
        numerator = (step_fraction - step_integer).numerator
        denominator = (step_fraction - step_integer).denominator
        return step_integer, numerator, denominator
    
    @property
    def supplement(self):
        if self.step_index % self.denominator in self.numerator_list: return 1
        else: return 0
        
    @property
    def done(self):
        if self.step_index < Config.max_horizon: return False
        else: return True
        
    def step(self) -> int:
        """Return the simple actions: numbers to sell/buy"""
        action = self.step_integer + self.supplement
        self.step_index +=1
        return action, self.done
    
    
    
if __name__ == '__main__':
    action_list = []
    twap = Twap()
    done = False
    while True:
        action, done = twap.step()
        action_list.append(action)
        if done: break
    print(sum(action_list) == Config.num2liquidate)