.. _deeprl:

Deep Reinforcement Learning in Agent Based Financial Market Simulation
====================

1. Authors: 
--------------------

Ikuya Morikawa, Daisuke Deguchi, Hideyuki Takahashi, Takanori Hayashi, Hiroyuki Matsui, Masahiro Kotosaka, Kenta Inoue and Atsushi Kawamoto

2. Affiliation: 
--------------------

Ikuya Morikawa: R&D Group, AI and FinTech Department, Nomura Research Institute, Ltd., Tokyo, Japan

3. Keywords: 
--------------------

deep reinforcement learning; financial market simulation; agent based simulation

4. URLs: 
--------------------

Paper: https://www.mdpi.com/1911-8074/13/4/71/htm; Github: None

5. Summary:
--------------------

(1): The article aims to propose a framework for training deep reinforcement learning models in agent-based artificial price-order-book simulations that yield non-trivial policies under diverse conditions with market impact.

 

(2): The past methods suffered from unknowable state space, limited observations, and the inability to model the impact of investment actions on the market, which can often be prohibitive when trying to find investment strategies using deep reinforcement learning. The approach proposed in this article is well motivated as it overcomes these limitations by augmenting real market data with agent-based artificial market simulation.

  

(3): The proposed methodology involves designing a reward function that maximizes capital accumulation without excessive risk, and training deep reinforcement learning models in an agent-based artificial market simulation using this function. The simulation is designed to reproduce realistic market features, create unobserved market states, and model the impact of investment actions on the market itself.

  

(4): The methods achieve robust investment strategies with an attractive risk-return profile, which are consistent with investment strategies used in real markets. The proposed framework can optimize strategies that adapt to realistic market features and enable creation of various deep reinforcement learning agents that perform well in the real world. The performance supports their goals of using deep reinforcement learning to optimize investment strategies in financial markets.

6. Conclusion:
--------------------

(1): The significance of this piece of work is to propose a framework using deep reinforcement learning for optimizing investment strategies in financial markets. The proposed methodology overcomes the limitations of previous methods and achieves robust investment strategies with an attractive risk-return profile.

(2): In terms of innovation point, the article presents a novel approach by augmenting real market data with agent-based artificial market simulation for training deep reinforcement learning models. The proposed methodology enables creation of various deep reinforcement learning agents that perform well in the real world. In terms of performance, the methods achieve robust investment strategies that are consistent with investment strategies used in real markets. However, the article lacks detailed analysis and discussion on the proposed reward function and the impact of investment actions on the market. In terms of workload, the article provides sufficient information on the methodology and results, but lacks analysis on the computational resources required for training the deep reinforcement learning models.

