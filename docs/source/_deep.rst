.. _deep:

Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy
====================

1. Authors: 
--------------------

Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid

2. Affiliation: 
--------------------

Dept. of Statistics, Columbia University

3. Keywords: 
--------------------

Deep reinforcement learning, Markov Decision Process, automated stock trading, ensemble strategy, actor-critic framework

4. Urls: 
--------------------

https://ieeexplore.ieee.org/abstract/document/8324327 or Github: None 

5. Summary:
--------------------

(1): This article focuses on the challenge of designing a profitable stock trading strategy in a complex and dynamic market, and proposes an ensemble strategy based on deep reinforcement learning.

 

(2): Past methods include a traditional approach that computes expected stock return and covariance matrix, and a Markov Decision Process-based approach using dynamic programming, but they have limitations. The approach proposed in this paper is well-motivated as it uses deep reinforcement learning with an ensemble of three actor-critic based algorithms, inheriting the best features of each to adapt to various market situations.

 

(3): The research methodology employs an ensemble of three deep reinforcement learning algorithms and finds the optimal trading strategy for stocks. The three algorithms used are Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG). The ensemble strategy integrates the best features of the three algorithms and is evaluated on 30 Dow Jones stocks that have adequate liquidity.

  

(4): The performance achieved by the proposed deep ensemble strategy is shown to outperform three individual algorithms and two baselines in terms of the risk-adjusted return measured by the Sharpe ratio. Therefore, the performance supports their goal of designing a profitable stock trading strategy using deep reinforcement learning.

6. Conclusion:
--------------------

(1): This piece of work proposes an ensemble strategy based on deep reinforcement learning for automated stock trading, which achieves superior performance than traditional methods and shows the potential of using actor-critic based algorithms in learning stock trading strategies.

(2): In terms of innovation point, this article innovatively proposes an ensemble strategy that combines the strengths of three actor-critic based algorithms in automated stock trading. As for performance, the proposed deep ensemble strategy outperforms three individual algorithms and two baselines in terms of risk-adjusted return measured by the Sharpe ratio. However, the workload of this article might be high as the ensemble of three deep reinforcement learning algorithms requires a significant computational resource and time.

