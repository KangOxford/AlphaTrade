from gym_exchange.trading_environment.utils import vwap_price

    
class RewardFunctional():
    '''functional'''
    def __init__(self, p_0, p_market, lambda_, agent_pairs):
        self.p_0 = p_0
        self.p_market = p_market
        self.lambda_ = lambda_
        self.agent_pairs = agent_pairs
        self.num_own_trades = agent_pairs.shape[1] # TODO : check
        
    def p(self, i):
        return self.agent_pairs[0][i] # TODO : check
    def q(self, i):
        return self.agent_pairs[1][i] # TODO : check
    
    @property
    def advantage(self):
        sum_ = 0
        for i in range(self.num_own_trades):
            sum_ += self.q(i) * (self.p(i) - self.p_market)
        return sum_
    
    @property
    def drift(self):
        sum_ = 0
        for i in range(self.num_own_trades):
            sum_ += self.q(i) * (self.p_market - self.p_0)
    
    def __call__(self):
        reward = self.advantage + self.lambda_ * self.drift
        return reward


class RewardGenerator():
    def __init__(self, p_0, lambda_ = 0.5):
        self.p_0 = p_0
        self.lambda_ = lambda_
        
    def update(self, executed_pairs, mid_price):
        def get_p_market(executed_pairs, mid_price):
            if len(executed_pairs.market_pairs) == 0:
                p_market = 0 # TODO 
            else:
                p_market = vwap_price(executed_pairs.market_pairs)
            return p_market
        p_market = get_p_market(executed_pairs, mid_price)
        signals = {
            "p_0" : self.p_0,
            "p_market" : p_market,
            "lambda_" : self.lambda_,
            "agent_pairs":executed_pairs.agent_pairs
        }
        self.reward_functional = RewardFunctional(**signals)
        
    def step(self):
        reward = self.reward_functional()
        return reward
        
if __name__ == "__main__":
    pass