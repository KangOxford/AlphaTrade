.. _double_q:

Double Deep Q-Learning for Optimal Execution, August 2022
====================

1. Authors: 
--------------------

Brian Ning, Franco Ho Ting Lin & Sebastian Jaimungal

2. Affiliation: 
--------------------

None

3. Keywords: 
--------------------

algorithmic trading, reinforcement learning, optimal execution, DDQN

4. Urls: 
--------------------

https://www.tandfonline.com/doi/abs/10.1080/1350486X.2022.2077783

5. Summary: 
--------------------

(1): This article focuses on the problem of optimal trade execution faced by traders in algorithmic trading. 

(2): Existing research on optimal execution uses model assumptions and stochastic control methods. However, this article proposes a model-free approach using Double Deep Q-Learning, which estimates the optimal actions of a trader. This approach is motivated by the need for a simpler and more flexible approach while still achieving high performance. 

(3): The proposed methodology uses a fully connected Neural Network trained using Experience Replay and Double DQN with input features given by the current state of the limit order book, other trading signals, and available execution actions. The output is the Q-value function estimating the future rewards under an arbitrary action. 

(4): The proposed approach is applied to nine different stocks and shows improved performance compared to the standard benchmark approach in terms of mean and median out-performance, probability of out-performance, and gain-loss ratios. The performance supports the goal of achieving optimal trade execution in algorithmic trading.

6. Methods: 
--------------------

(1): The article proposes a model-free approach for the problem of optimal trade execution using Double Deep Q-Learning. 

(2): A fully connected Neural Network is trained using Experience Replay and Double DQN, with input features including the current state of the limit order book, other trading signals, and available execution actions. The output is the Q-value function estimating the future rewards under an arbitrary action.

(3): The proposed approach is applied to nine different stocks, and the performance is evaluated using metrics such as mean and median out-performance, probability of out-performance, and gain-loss ratios, compared to a standard benchmark approach.

(4): The methodology also includes hyperparameters such as the size of replay memory, rate of decay of  exploration factor, rate of updating target and main network, and quadratic penalty coefficient, which are carefully selected based on the specific problem.

(5): The state space is defined by a set of features observed from the market, including time, inventory, price, and quadratic variation. These features are transformed into the range [-1,1] to increase stability of the algorithm.

(6): Pre-training is conducted on a set of boundary action cases using randomly selected price intervals from the full data set, to increase network stability during training.

(7): Experiments are conducted on all trading days from 2-January-2017 to 30-March-2018, for each of the nine different stocks, and evaluated using Profit and Loss with Transaction Cost, compared to a Time-Weighted Average Price strategy.

7. Conclusion: 
--------------------

(1): The significance of this piece of work is proposing a novel model-free approach using Double Deep Q-Learning for the problem of optimal trade execution in algorithmic trading. The proposed methodology shows improvements in performance compared to existing methods, and supports the goal of achieving optimal trade execution.

(2): Innovation point: The article proposes a model-free approach using Double Deep Q-Learning, which is a novel method for the problem of optimal trade execution.

(3): Performance: The proposed approach shows improved performance compared to a standard benchmark method in terms of mean and median out-performance, probability of out-performance, and gain-loss ratios.

(4): Workload: The proposed methodology includes carefully selected hyperparameters and requires the training of a fully connected Neural Network, which may increase computational workload compared to simpler methods.

