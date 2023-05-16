from gym_exchange.environment.base_env.utils import vwap_price
from gym_exchange import Config
import numpy as np

class RewardFunctional():
    '''functional'''
    def __init__(self, p_0, p_market, lambda_, agent_pairs, type):
        self.p_0 = p_0
        self.p_market = p_market
        self.lambda_ = lambda_
        self.agent_pairs = agent_pairs
        self.num_own_trades = agent_pairs.shape[1] # TODO : check
        self.type = type
        
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
            # sum_ += self.q(i) * (self.p_market - self.p_0) # Peer
            sum_ += self.q(i) * (self.p(i) - self.p_0) # Kang
        return sum_

    @property
    def peer_reward(self):
        # self.lambda_ = 0.0 # for testing
        if self.type == "agent_market":
            self.lambda_ = 0.01 # for less trend rewarding
            reward = self.advantage + self.lambda_ * self.drift
            # reward = (self.agent_pairs[0,:] * self.agent_pairs[1,:]).sum()/Config.sum_reward
        # elif self.type is "agent_market"
        else:
            reward = 0
        return reward


    def __call__(self):
        revenue = (self.agent_pairs[0, :] * self.agent_pairs[1, :]).sum()
        regularity = self.peer_reward
        # normed(revenue, regularity) {
        # revenue = (revenue - 12825558.401639344)/18586214.00096323
        # regularity = (regularity-2192.844995644455)/5409.271445768599
        # ---------------------------------
        # mean_reward= 0; std_reward= 1.0
        # # mean_reward=318112370.6779661; std_reward=484791277.90215284
        # revenue = (revenue - mean_reward) / std_reward
        # # revenue = (revenue - 13428500.214592274) / 15486530.002679234
        # ---------------------------------
        sum_reward = 939273430800.0
        # sum_reward = 187686298700.0
        revenue = revenue / sum_reward
        # print(f"{revenue}, {regularity},` {Config.mu_regularity * regularity}")
        # normed(revenue, regularity) }
        reward = revenue + Config.mu_regularity * regularity
        return reward



class RewardGenerator():
    def __init__(self, p_0, lambda_ = 0.5):
        self.p_0 = p_0
        self.lambda_ = lambda_
        
    def update(self, executed_pairs_bigram, mid_price):
        if executed_pairs_bigram['market_pairs'] is not None and executed_pairs_bigram['agent_pairs'] is not None:
            p_market = vwap_price(executed_pairs_bigram['market_pairs'])
            type = "agent_market"
        elif executed_pairs_bigram['market_pairs'] is None and executed_pairs_bigram['agent_pairs'] is not None:
            p_market = mid_price ## TODO remain discussion
            type = "agent_only"
        elif executed_pairs_bigram['market_pairs'] is not None and executed_pairs_bigram['agent_pairs'] is None:
            p_market = mid_price ## TODO remain discussion
            type = "market_only"
        elif executed_pairs_bigram['market_pairs'] is None and executed_pairs_bigram['agent_pairs'] is None:
            p_market = mid_price ## TODO remain discussion
            type = "no_trading"
        else:
            raise NotImplementedError
        zeros = np.zeros((2,2))
        agent_pairs = executed_pairs_bigram.get('agent_pairs')
        signals = {
            "p_0": self.p_0,
            "p_market": p_market,
            "lambda_": self.lambda_,
            # "agent_pairs": executed_pairs_bigram['agent_pairs'],
            # "agent_pairs": executed_pairs_bigram.get('agent_pairs',zeros),
            "agent_pairs": zeros if agent_pairs is None else agent_pairs,
            "type": type
        }
        self.reward_functional = RewardFunctional(**signals)


    def step(self):
        # if self.reward_functional == -1 : reward = 0 # TODO:check if the executed_pairs is all the pairs recorded or only from one step
        reward = self.reward_functional()
        return  reward
        
if __name__ == "__main__":
    pass
