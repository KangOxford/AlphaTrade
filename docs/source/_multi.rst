.. _multi:

Multi-Agent Reinforcement Learning in a Realistic Limit Order Book Market Simulation
====================


1. Authors:
--------------------

Michaël Karpe, Jin Fang, Zhongyao Ma, Chen Wang


2. Affiliation:
--------------------

University of California, Berkeley


3. Keywords:
--------------------

high-frequency trading, limit order book, market simulation, multi-agent reinforcement learning, optimal execution


4. Urls:
--------------------

https://doi.org/10.1145/3383455.3422570, Github: None

                 

5. Summary:
--------------------

(1): The article discusses the challenges posed by the complex and unknown market dynamics in developing and validating optimal execution strategies in high-level trading strategies involving large volumes of orders in financial investments.

(2): Existing simulation methods are based on sound assumptions about the statistical properties of the market environment and the impact of transactions on the prices of financial instruments, but they generally show lower profitability when implemented in real markets. Therefore, the paper proposes a model-free approach by training Reinforcement Learning (RL) agents in a realistic market simulation environment with multiple agents.

(3): The paper configures a multi-agent historical order book simulation environment for execution tasks built on an Agent-Based Interactive Discrete Event Simulation (ABIDES). It formulates the problem of optimal execution in an RL setting where an intelligent agent can make order execution and placement decisions based on market microstructure trading signals in HFT. It develops and trains an RL execution agent using the Double Deep Q-Learning (DDQL) algorithm in the ABIDES environment.

(4): The simulation with the RL agent is evaluated by comparing it with a market replay simulation using real market Limit Order Book (LOB) data. The results show that the RL agent converges towards a Time-Weighted Average Price (TWAP) strategy, and it outperforms other execution strategies and the classical benchmarks by reducing transaction costs and slippage. The performance achieved supports their goals of developing an optimal execution strategy in HFT.

6. Conclusion:
--------------------

(1): This research proposes a multi-agent reinforcement learning approach to optimize execution tasks in high-frequency trading, which can improve transaction cost and slippage and enhance the profitability of financial investments. This study is significant for developing and validating optimal execution strategies in real financial markets, as simulation-based methods show lower profitability when implemented in actual markets.

(2): Innovation point: The paper adopts a model-free approach to learn optimal execution strategies by training reinforcement learning agents in a realistic market simulation environment with multiple agents, which improves the realism and effectiveness of execution tasks in high-frequency trading. 

(3): Performance: The results show that the proposed method outperforms other execution strategies and classical benchmarks by reducing transaction costs and slippage, which demonstrates the effectiveness and suitability of the multi-agent reinforcement learning approach.

(4): Workload: The paper configures a multi-agent historical order book simulation environment for execution tasks built on an Agent-Based Interactive Discrete Event Simulation, which requires a certain amount of workload to set up and train the reinforcement learning agent. However, the workload is reasonable and manageable considering the improvement in execution performance brought by the proposed approach.

