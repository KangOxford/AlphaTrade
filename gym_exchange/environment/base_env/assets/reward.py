from gym_exchange.environment.base_env.utils import vwap_price
from gym_exchange import Config
    
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
        return sum_
    
    def __call__(self):
        reward = self.advantage + self.lambda_ * self.drift
        # reward = (self.agent_pairs[0,:] * self.agent_pairs[1,:]).sum()/Config.sum_reward
        return reward


class RewardGenerator():
    def __init__(self, p_0, lambda_ = 0.5):
        self.p_0 = p_0
        self.lambda_ = lambda_
        
    def update(self, executed_pairs_bigram, mid_price):
        if executed_pairs_bigram['market_pairs'] is not None and executed_pairs_bigram['agent_pairs'] is not None:
            p_market = vwap_price(executed_pairs_bigram['market_pairs'])
            signals = {
                "p_0": self.p_0,
                "p_market": p_market,
                "lambda_": self.lambda_,
                "agent_pairs": executed_pairs_bigram['agent_pairs']
            }
            self.reward_functional = RewardFunctional(**signals)
        else:
            self.reward_functional = -1
        '''
        if executed_pairs_bigram['agent_pairs'] is not None:
            signals = {
                "p_0": 0,
                "p_market": 0,
                "lambda_": 0,
                "agent_pairs": executed_pairs_bigram['agent_pairs']
            }
            self.reward_functional = RewardFunctional(**signals)
        else:
            self.reward_functional = -1
        '''


    def step(self):
        if self.reward_functional == -1 : reward = 0 # TODO:check if the executed_pairs is all the pairs recorded or only from one step
        else:   reward = self.reward_functional()
        return  reward
        
if __name__ == "__main__":
    pass
