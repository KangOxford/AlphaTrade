.. _simple:

A simple learning agent interacting with an agent-based market model, Septermber 2022
====================

1. Authors: 
--------------------

Matthew Dicks, Tim Gebbie

2. Affiliation: 
--------------------

Department of Statistical Sciences, University of Cape Town, Rondebosch 7701, South Africa

3. Keywords: 
--------------------

strategic order-splitting, reinforcement learning, market simulation, matching engine, agent-based model

4. Urls: 
--------------------

https://arxiv.org/abs/2208.10434

5. Summary: 
--------------------

(1): The paper's research background is on reinforcement learning in trading agents that interact with agent-based market models. 

(2): Previous research in agent-based modeling faced problems due to having to make multiple assumptions in data-sets, leading to multiple researchers relying on machine learning to address the problem. However, the paper highlights that the existing literature on optimizing trading algorithms with reinforcement learning has predominantly focused on the application of a single optimal execution agent that trades with a pre-existing order book, whereas, in reality, trading is still done in a centralized matching engine. To address this problem, the paper presents a simple learning agent that interacts with an agent-based market model asynchronously through a matching engine in event time. 

(3): The paper proposes a calibration approach to explore the impact of the optimal execution agent's learning dynamics on the agent-based model and market. The agent-based model and the optimal execution agent are calibrated and explored at different levels of initial order-sizes and state spaces. Convergence, volume trajectory, and action trace plots are used to visualize the learning dynamics. 

(4): The paper evaluates the methodology proposed by exploring changes in the empirical stylized facts and price impact curves. They find that the moments of the model are robust to the impact of the learning agents except for the Hurst exponent, which was lowered by the introduction of strategic order-splitting. The paper shows that the introduction of the learning agent preserves the shape of the price impact curves but can reduce the trade-sign autocorrelations when their trading volumes increase. Overall, the performance supports their goals of demonstrating the effectiveness of a simple learning agent interacting with an agent-based market model.

6. Methods: 
--------------------

(1): The paper uses reinforcement learning to address the problem of optimizing trading algorithms in agent-based market models. Unlike previous research that focused on the application of a single optimal execution agent, the paper proposes a simple learning agent that interacts with an agent-based market model asynchronously through a matching engine in event time.

(2): The paper proposes a calibration approach to explore the impact of the optimal execution agent's learning dynamics on the agent-based model and market. The agent-based model and the optimal execution agent are calibrated and explored at different levels of initial order-sizes and state spaces. Convergence, volume trajectory, and action trace plots are used to visualize the learning dynamics.

(3): The paper evaluates the methodology proposed by exploring changes in the empirical stylized facts and price impact curves. The method-of-moments with simulated minimum distance (MM-SMD) is used for calibration, and the Nelder-Mead algorithm with threshold accepting is used to optimize the objective function. The calibrated parameters and associated confidence intervals are presented, and the moments used to characterize the price paths are listed. The simulated and empirical moments are compared to assess the goodness of the agent-based market model.

7. Conclusion:
--------------------

(1): This article is significant in proposing a simple learning agent that interacts with an agent-based market model asynchronously through a matching engine in event time. It addresses the problem of optimizing trading algorithms in agent-based market models and explores the impact of the optimal execution agent's learning dynamics on the agent-based model and market. 

(2): Innovation point: The article proposes a unique approach of combining a realistic but minimally intelligent agent-based model with a learning agent with bounded rationality in event time which required moving the market clearing mechanism into one that is event-driven. (3): Performance: The methodology proposed is evaluated through changes in the empirical stylized facts and price impact curves, and the results show that the moments of the model are robust except for the Hurst exponent, which was lowered by the introduction of strategic order-splitting. (4): Workload: The paper uses reinforcement learning and calibration approach to explore the impact of the optimal execution agent's learning dynamics on the agent-based model and market which can be time-consuming.

