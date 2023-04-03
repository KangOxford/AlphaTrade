.. _towards_fully:

Towards a fully RL-based Market Simulator
=============

1. Authors:
--------------

Leo Ardon, Nelson Vadori, Thomas Spooner, Mengda Xu, Jared Vann, and Sumitra Ganesh

2. Affiliation:
--------------

J.P. Morgan AI Research

3. Keywords:
--------------

multi-agent, reinforcement learning, market making

4. Urls:
--------------

https://doi.org/10.1145/3490354.3494372

5. Summary:
--------------

(1): The research focuses on understanding the dynamics of the financial market, which is challenging due to the large number of actors and behaviors involved. The goal is to build a market simulator that can replicate complex market conditions and study the dynamics of the financial market under various scenarios.

(2): Past methods include statistical approaches that consider different actors in isolation and make assumptions about the rest of the market, as well as modeling the market as a multi-agent system (MAS) where the participants are represented as independent entities able to interact between each other. However, hand-coded policies driven by business experience and common sense are typically used but are either not complex enough to truly characterize the agent’s behavior or are too difficult to calibrate with the data available. The approach of using reinforcement learning (RL)-based multi-agent systems as learning agents provides a less opinionated setting to study the dynamics of the market without having to make hard assumptions on the policy to apply.

(3): The researchers build upon the previous work with their use of RL to model the Liquidity Provider (or market maker) and introduce a new financial framework composed of RL-based agents able to learn how to react to a change in the market. The proposed methodology involves a parametrized reward formulation and the use of Deep RL for each group of agents, with a shared policy able to generalize and interpolate over a wide range of behaviors.

(4): The methods in this paper are evaluated on the market making problem, where the Liquidity Providers and Liquidity Takers learn simultaneously to satisfy their objective. The results show that the proposed approach is able to achieve better performance than the baseline methods, supporting the goal of building a fully RL-based market simulator that can replicate complex market conditions. The performance of the methods is evaluated based on the trade-off between quantity and PnL.

6. Conclusion:
--------------

(1): The significance of this work lies in the development of a fully RL-based market simulator that can replicate complex market conditions and study the dynamics of the financial market under various scenarios. This work provides a promising direction for future research in the field of financial market modeling and analysis.

(2): Innovation point: The use of reinforcement learning based multi-agent systems to model the financial market is a promising innovation. The proposed methodology involves a parameterized reward formulation and the use of Deep RL for each group of agents, with a shared policy able to generalize and interpolate over a wide range of behaviors.

(3): Performance: The results show that the proposed approach is able to achieve better performance than the baseline methods, supporting the goal of building a fully RL-based market simulator that can replicate complex market conditions. The performance of the methods is evaluated based on the trade-off between quantity and PnL.

(4): Workload: The workload involved in developing and testing the proposed methodology is significant, as it requires the use of Deep RL algorithms and extensive data input. Further work is needed to explore ways to reduce the workload while maintaining the accuracy and effectiveness of the approach.




