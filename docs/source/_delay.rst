.. _delay:

Optimal execution with stochastic delay
====================

1. Authors: 
--------------------

Álvaro Cartea, Leandro Sánchez-Betancourt

2. Affiliation: 
--------------------

Álvaro Cartea is affiliated with the Mathematical Institute, University of Oxford and the Oxford-Man Institute of Quantitative Finance, Oxford, UK.

3. Keywords: 
--------------------

Algorithmic trading, high-frequency trading, stochastic delay, latency

4. Url: 
--------------------

https://doi.org/10.1007/s00780-022-00491-w

5. Summary: 
--------------------

(1): The paper studies how traders use marketable limit orders (MLOs) to liquidate a position over a trading window when there is latency in the marketplace.

(2): The past methods have mainly focused on executing trades at the best available price, without taking into account the impact of latency on execution. The approach in this paper is motivated by the fact that delay times in trading can be quite stochastic, and thus need to be modeled as such to optimize trading decisions.

(3): The paper proposes a model based on impulse control with stochastic latency, where the trader controls the times and price limits of the MLOs sent to the exchange. The study analyzes the behavior of impatient and patient liquidity takers, and shows how patient traders use their speed to complete the execution program with as many speculative MLOs as possible.

(4): The proposed random-latency-optimal strategy outperforms the benchmarks for patient traders by an amount greater than the transaction costs paid by liquidity takers in foreign exchange markets. The superiority of the strategy is due to both the speculative MLOs that are filled and the price protection of the MLOs. Around news announcements, the value of the outperformance is between two and ten times the value of the transaction costs.


6. Conclusion:

--------------------

(1): This paper proposes a novel optimal execution strategy for traders in the presence of stochastic latency. The proposed model of impulse control with stochastic latency takes into account both the times and price limits of marketable limit orders (MLOs) to optimize trading decisions. The study highlights the significance of considering the impact of latency on execution and shows that the proposed strategy outperforms benchmarks in foreign exchange markets.

(2): Innovation point: The paper introduces a new approach to optimal trading execution that considers the impact of latency on marketable limit orders. The model of impulse control with stochastic latency is an innovative way to address this issue.


(3): Performance: The random-latency-optimal strategy proposed in the paper outperforms benchmarks in foreign exchange markets, indicating that it is a viable and effective trading strategy.


(4): Workload: The paper provides a thorough analysis and mathematical proofs of the proposed model and strategy, which may require a significant workload for readers to understand, especially those without a strong background in mathematical finance.

