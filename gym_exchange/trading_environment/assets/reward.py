    
class RewardFunctional():
    '''functional'''
    def __init__(self, p_0, p_market, lambda_):
        self.p_0 = p_0
        self.p_market = p_market
        self.lambda_ = lambda_
        
    def p(self, i):
        pass
    
    def advantage(self):
        sum_ = 0
        for i in range(num_own_trades):
            sum_ += q(i) * (p(i) - self.p_market)
        return sum_
    
    def drift(self):
        sum_ = 0
        for i in range(num_own_trades):
            sum_ += q(i) * (self.p_market - self.p_0)
    
    def __call__(self):
        reward = advantage + self.lambda_ * drift
        return reward

class RewardGenerator():
    def __init__(self, lambda_ = 0.5):
        self.lambda_ = lambda_
        
    def update(self, excuted):
        self.excuted = excuted
        signals = {
            "p_0" : p_0,
            "p_market" : p_market,
            "lambda_" : lambda_,
        }
        self.reward_functional = RewardFunctional(**signals)
        
    def step(self):
        reward = self.reward_functional()
        return reward
        
if __name__ == "__main__":
    pass