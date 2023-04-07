.. _model_based_env:

Model-based gym environments for limit order book trading, Octorber 2022
====================

1. Authors: 
--------------------

Joseph Jerome, Leandro Sánchez-Betancourt, Rahul Savani, and Martin Herdegen.

2. Affiliation: 
--------------------

Department of Computer Science, University of Liverpool (Joseph Jerome); Department of Mathematics, King’s College London (Leandro Sánchez-Betancourt); Department of Computer Science, University of Liverpool (Rahul Savani); and Department of Statistics, University of Warwick (Martin Herdegen).

3. Keywords: 
--------------------

Limit order book, market-making, optimal execution, liquidity provision, inventory risk, and reinforcement learning.

5. Url(s): https://arxiv.org/abs/2209.07823 (paper), https://github.com/jsalon-maths/mbt-gym (Github).

5. Summary: 
--------------------

(1): This article discusses the use of reinforcement learning for solving model-based limit order book (LOB) trading problems in mathematical finance.

(2): In the past, the Hamilton-Jacobi-Bellman equation using Euler schemes has been employed for solving LOB trading problems. However, the method is tailored to specific stochastic processes of a model and suffers from the curse of dimensionality, which limits the types of models that can be considered. The article proposes using model-free RL to address these issues and provides a benchmark module that implements a range of model-based LOB trading problems. The approach is motivated by the potential of RL for solving richer and more realistic models, and its ability to solve problems where other methods fail.

(3): The proposed methodology involves the use of mbt_gym, a Python module that provides a suite of gym environments for training RL agents to solve model-based trading problems. The module is set up in an extensible way to allow different aspects of different models to be combined. Highly efficient implementations of vectorized environments are supported to enable faster training of RL agents. The module is provided as an open source repository on GitHub to serve as a reference for RL research in model-based algorithmic trading.

(4): The methods in this paper are shown to solve standard and non-standard problems from the literature. The highly efficient implementation allows the approach to achieve close-to-optimal solutions to the benchmark problems. The performance achieved supports the goal of using RL as a complementary solution method to PDE approaches and enabling the solution of richer and more realistic models.

6. Conclusion:
--------------------

(1): This piece of work is significant because it proposes a novel approach for solving limit order book trading problems using reinforcement learning. The proposed approach can potentially solve richer and more realistic models that traditional methods cannot.

 

(2): Innovation point: The article proposes using model-free RL to solve limit order book trading problems, which is a novel approach. The mbt_gym module also provides a benchmark suite of model-based LOB trading problems that can be used for RL research in algorithmic trading. 

(3): Performance: The methods proposed in the article are shown to solve standard and non-standard problems from the literature, achieving near-optimal performance. The highly efficient implementation of vectorized environments allows for faster training of RL agents.

(4): Workload: The mbt_gym module is set up in an extensible way to allow different aspects of different models to be combined. However, further contributions from the wider community are welcome to enhance the module's functionalities.

