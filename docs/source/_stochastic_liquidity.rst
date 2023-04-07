.. _stochastic_liquidity:

Stochastic Liquidity as a Proxy for Nonlinear Price Impact, December 2022
====================

 

1. Authors: 
--------------------

Rama Cont, Andreea Minca, Yubo Tao

 

2. Affiliation: 
--------------------

Rama Cont University of Oxford

 

3. Keywords: 
--------------------

market liquidity, price impact, trading algorithms, stochastic liquidity, nonlinear propagator model

 

4. URLs: 
--------------------

https://arxiv.org/abs/2106.15708, Github: None

 

5. Summary:
--------------------

(1): The article focuses on measuring price impact empirically and incorporating it into trading algorithms. Nonlinear price impact models have been found to fit empirical data better, but they are intractable from an analytical point of view, leading to an emphasis on linear models. A gap between empirical results and linear models challenged the authors to find a bridge between the two.

 

(2): Past methods have relied on linear models, which can be analyzed with powerful mathematical tools, but do not fit empirical data as closely as nonlinear models. Nonlinear models are empirically superior but lack analytical tractability. The authors propose a novel approach to relate nonlinear impact to stochastic liquidity, offering a way to incorporate nonlinear price impact into quantitative models without direct access to trading data. The approach is well-motivated, as it ties empirical and analytical models while allowing for more accurate pricing.

 

(3): The authors build a bridge between empirical and analytical models by deriving the continuous-time limit of a discrete nonlinear propagator model, resulting in an Ornstein-Uhlenbeck process driven by order flow. The continuous-time limit is a linear model, and the authors introduce stochastic liquidity to capture the nonlinear effects of market impact on trading costs. For a given level of trading costs, the authors show how to derive the corresponding level of stochastic liquidity.

 

(4): The authors apply their approach to a case study and compare it to alternative models. Their stochastic model improves upon a standard constant parameter model and outperforms alternative dynamic models. The performance supports their goals of incorporating nonlinear price impact into quantitative models and improving the accuracy of pricing.

6. Conclusion:
--------------------

(1): This piece of work proposes a novel approach to relate nonlinear price impact to stochastic liquidity, offering a way to incorporate nonlinear price impact into quantitative models without direct access to trading data. The approach is well-motivated and improves the accuracy of pricing. 

(2): Innovation point: The authors build a bridge between empirical and analytical models by deriving the continuous-time limit of a discrete nonlinear propagator model, resulting in an Ornstein-Uhlenbeck process driven by order flow. This approach ties empirical and analytical models together while allowing for more accurate pricing. 

(3): Performance: The authors apply their approach to a case study and compare it to alternative models. Their stochastic model improves upon a standard constant parameter model and outperforms alternative dynamic models, supporting the goal of improving the accuracy of pricing. 

(4): Workload: The article is well-written and well-organized, and the authors provide detailed proofs in the appendix. The workload required to understand the article is moderate, as it requires a basic understanding of mathematical concepts and quantitative finance.

