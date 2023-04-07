.. _many:

Many learning agents interacting with an agent-based market model
====================

1. Authors: 
--------------------

Matthew Dicks, Andrew Paskaramoothy, Tim Gebbie

2. Affiliation: 
--------------------

Department of Statistical Sciences, University of Cape Town, Rondebosch 7701, South Africa

3. Keywords: 
--------------------

agent-based model, reinforcement learning, multi-agent, market simulation, order splitting, stylised facts, order flow, price impact, volatility clustering, market micro-structure, optimal execution

4. Urls: 
--------------------

arXiv:2303.07393v2 [q-fin.TR] 25 Mar 2023

5. Summary:
--------------------

(1): The article is dedicated to the study of the dynamics and interaction of multiple reinforcement learning optimal execution trading agents interacting with a reactive Agent-Based Model (ABM) of a financial market in event time.

(2): Past methods mostly investigated subsets of various low-frequency micro-structural stylized facts arising from the interactions of heterogeneous agents, with chartists and fundamentalists most often utilized within a minority game setting. However, stylized facts like long-memory of order flows and absolute returns, the power-law of price impact, etc. require the investigation of the strategic behavior of different classes of heterogeneous agents, operating at different time scales and under asymmetric information. ABMs investigating market micro-structure required the inclusion of optimal execution agents to conform more with empirical data.

(3): The research methodology proposed in this paper is the use of multiple competing learning agents that impact a minimally intelligent market simulation as functions of the number of agents, the size of agents’ initial orders, and the state spaces used for learning. Further, the use of phase space plots enables the examination of the dynamics of the ABM when various specifications of learning agents are included.

(4): The task of the methods in this paper is to demonstrate how multiple competing learning agents impact a minimally intelligent market simulation, and examine whether the inclusion of optimal execution agents that can learn results in dynamics that have the same complexity as empirical data. The performance achieved by the methods in this paper indicates that the inclusion of optimal execution agents changes the stylized facts produced by ABM to conform more with empirical data and are a necessary inclusion for ABMs investigating market micro-structure. However, including execution agents to chartist-fundamentalist-noise ABMs is insufficient to recover the complexity observed in empirical data.

6. Conclusion:
--------------------

(1): The significance of this piece of work lies in the investigation of the strategic behavior of different classes of heterogeneous agents interacting with a reactive Agent-Based Model (ABM) of a financial market in event time. It also demonstrates the importance of including optimal execution agents to conform with empirical data and produce stylized facts associated with order flow and the cost of trading.

(2): Innovation point: By introducing multiple competing learning agents, this article innovatively explores the impact of optimal execution agents on market simulation.

(3): Performance: The performance achieved by the methods in this paper indicates that the inclusion of optimal execution agents changes the stylized facts produced by ABM to conform more with empirical data, but including execution agents to chartist-fundamentalist-noise ABMs is insufficient to recover the complexity observed in empirical data.

(4): Workload: The methods proposed in this article require the use of phase space plots to examine the dynamics of the ABM and the investigation of the strategic behavior of different classes of heterogeneous agents, which may increase the workload of implementation and analysis.

