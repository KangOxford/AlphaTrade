.. _peer:

Asynchronous Deep Double Duelling Q-Learning for Trading-Signal Execution in Limit Order Book Markets, February 2023
====================

1. Authors: 
--------------------

Peer Nagy, Jan-Peter Calliess, Stefan Zohren

2. Affiliation: 
--------------------

Oxford-Man Institute of Quantitative Finance, University of Oxford

3. Keywords: 
--------------------

Limit Order Books, Quantitative Finance, Reinforcement Learning, LOBSTER

4. Url: 
--------------------

arXiv:2301.08688v1  [q-fin.TR]  20 Jan 2023, Github: None

5. Summary: 
--------------------

(1): The research background of this article is to effectively translate high-frequency trading signals into a trading strategy that can be executed in the limit order book markets.

(2): Past methods have relied on predicting prices over short time periods, but this doesn't always lead to trading profits due to transaction costs, implementation details, and time delays. The proposed approach is motivated by utilizing deep reinforcement learning to maximize trading returns in a realistic trading environment for NASDAQ equities.

(3): The research methodology proposed in this paper is to use Deep Duelling Double Q-learning with the APEX architecture to train a trading agent that observes the current limit order book state, its recent history, and a short-term directional forecast. The agent's performance is optimized through using synthetic alpha signals obtained by perturbing forward-looking returns with varying levels of noise.

(4): The methods proposed in this paper achieve an effective trading strategy for inventory management and order placing that outperforms a heuristic benchmark trading strategy having access to the same signal. The performance of RL for adaptive trading is investigated independently from a concrete forecasting algorithm.

6. Conclusion:
--------------------

(1): The significance of this piece of work is to address the challenges of effectively translating high-frequency trading signals into trading strategies for execution in limit order book markets. The proposed approach utilizes deep reinforcement learning to optimize trading returns in a realistic trading environment for NASDAQ equities.

(2): Innovation point: The proposed methodology utilizes Deep Duelling Double Q-learning with the APEX architecture to train a trading agent that observes the current limit order book state, its recent history, and a short-term directional forecast. The use of synthetic alpha signals obtained by perturbing forward-looking returns with varying levels of noise is an innovative idea.

(3): Performance: The methods proposed in this paper achieve an effective trading strategy for inventory management and order placing that outperforms a heuristic benchmark trading strategy having access to the same signal.

(4): Workload: The paper requires a technical understanding of reinforcement learning, and the experiments and analyses require significant computational resources, which could be a weakness for those without access to such resources.

