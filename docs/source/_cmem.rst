.. _cmem:

Intra-daily Volume Modeling and Prediction for Algorithmic Trading
====================



1. Authors: 
--------------------

Brownlees, C.T., Cipollini, F., & Gallo, G.M.

2. Affiliation: 
--------------------

Department of Finance, Stern School of Business, NYU (for the first author)

3. Keywords: 
--------------------

Algorithmic trading, Volume prediction, Intra-daily periodicity, Component Multiplicative Error Model, Generalized Method of Moments, VWAP

4. Urls: 
--------------------

Paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1759636, Github:None

5. Summary:
--------------------

(1): The paper's research background is the increasing prevalence of algorithmic trading, where automated trading strategies require intra-daily volume predictions to minimize transaction costs by optimally placing trades.

(2): Past methods have used Rolling Means or Complex Autoregressive Models to model intra-daily volume, which fail to accurately capture intra-daily periodicity and volume asymmetry. This leads to imprecise volume predictions for the VWAP tracking trading exercise. The dynamic model proposed in this paper, the Component Multiplicative Error Model, is well-motivated to model intra-daily periodicity and volume asymmetry, and is simultaneously estimated using Generalized Method of Moments.

(3): The research methodology proposed is a dynamic model with different components which captures the behavior of traded volumes viewed from daily and intra-daily time perspectives. The Component Multiplicative Error Model is estimated using Generalized Method of Moments, and is applied to three major Exchange Traded Funds (ETFs) to show that both static and dynamic VWAP replication strategies generally outperform the na¨ıve method of rolling means for intra-daily volumes.

(4): The methods in this paper achieve significantly more precise predictions for intra-daily volume and deliver superior performance in the VWAP tracking trading exercise, outperforming common volume forecasting methods. The results support the paper's goal of proposing a dynamic model for intra-daily volume forecasting in algorithmic trading.

6. Conclusion:
--------------------

(1): The significance of this piece of work is to propose a dynamic model for intra-daily volume forecasting in algorithmic trading, which achieves more precise predictions for intra-daily volume and delivers superior performance in the VWAP tracking trading exercise.

(2): Innovation point: The paper proposes a dynamic model, the Component Multiplicative Error Model, to accurately capture intra-daily periodicity and volume asymmetry in intra-daily volume forecasting for algorithmic trading. 


(3): Performance: The proposed model outperforms common volume forecasting methods and achieves significantly more precise predictions for intra-daily volume, delivering superior performance in the VWAP tracking trading exercise. 

(4): Workload: The research methodology proposed in this paper is complex and requires expertise in finance and econometric modeling. However, the paper provides clear explanations of the methodology and implementation steps.

