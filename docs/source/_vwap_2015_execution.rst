.. _vwap_2015_execution:

Volume Weighted Average Price Optimal Execution, 2015 
====================

                 
1. Authors:
--------------------

Claudio Busetti, Andrea Consiglio, Roberto Reno

                 

2. Affiliation:
--------------------

None



3. Keywords:
--------------------

Optimal Execution, Volume Weighted Average Price, Mean-Variance, Slippage, NYSE



4. Urls:
--------------------


Paper, Github: None



5. Summary:
--------------------


(1): The paper studies the problem of optimal execution of a trading order under the Volume Weighted Average Price (VWAP) benchmark, from the point of view of a risk-averse broker.



(2): The paper points out that most related works in the literature eschew the issue of imperfect knowledge of the total market volume. They instead incorporate it in their model, and validate their method by extensive simulation of order execution on real NYSE market data. The paper is well motivated and the approach is novel.



(3): The paper devises multiple ways to solve the problem, in particular by studying how to incorporate the information coming from the market during the schedule. The methodology proposed in this paper involves the minimization of mean-variance of the slippage, with quadratic transaction costs.



(4): The method proposed in this paper, using a simple model for market volumes, reduces by 10% the VWAP deviation RMSE of the standard "static" solution (and can simultaneously reduce transaction costs), achieving their performance goals.

6. Conclusion:
--------------------

(1): This article proposes a novel method for optimal execution of a trading order under the Volume Weighted Average Price (VWAP) benchmark from the perspective of a risk-averse broker. The authors incorporate the issue of imperfect knowledge of the total market volume into their model and validate their method through extensive simulations on real NYSE market data. 

(2): The innovation point of this work lies in incorporating the information coming from the market during the schedule and minimizing the mean-variance of the slippage with quadratic transaction costs. The method proposed in this paper achieves a 10% reduction in the VWAP deviation RMSE of the standard "static" solution while simultaneously reducing transaction costs. 

(3): In terms of performance, the methodology proposed in this paper outperforms the existing methods with a higher degree of accuracy in achieving the expected performance goals. The workload required to implement this method is not explicitly stated in the article. However, it is reasonable to assume that it may require a significant amount of computational power and resources to execute effectively.

