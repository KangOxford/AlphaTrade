.. _towards:

Towards Realistic Market Simulations: a Generative Adversarial Networks Approach
====================

1. Authors: 
--------------------

Andrea Coletta, Matteo Prata, Michele Conti, Emanuele Mercanti, Novella Bartolini, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch.

2. Affiliation: 
--------------------

Sapienza University of Rome

3. Keywords: 
--------------------

Reinforcement learning, financial markets, generative adversarial networks, simulations.

4. Urls: 
--------------------

Paper: https://arxiv.org/abs/2110.13287; Github: None

5. Summary: 
--------------------

1. The article deals with the problem of creating realistic simulations of financial markets, which is important for evaluating trading strategies before approaching the actual market. 

2. The existing approach of backtesting does not consider the market response to experimental agents' actions, while traditional simulation methods use pools of hand-crafted experimental agents to mimic market complexity. However, these methods often result in poorly realistic simulations. The authors propose a synthetic market generator based on a Conditional Generative Adversarial Network (CGAN) trained on real aggregate-level historical data to address this problem. 

3. The authors train a CGAN to generate limit orders conditioned by the observed market situation, allowing a pre-trained "world" agent to produce orders to be processed by a simulation environment. They integrate their synthetic market generator into an open source simulator of financial markets called ABIDES. 

4. The authors show through extensive simulations that their proposal outperforms previous work in terms of stylized facts reflecting market responsiveness and realism. The performance achieved by the methods in the paper supports the authors' goals of creating more realistic market simulations.

6. Methods: 
--------------------

1. The authors propose a synthetic market generator based on a Conditional Generative Adversarial Network (CGAN), which is trained on real aggregate-level historical data to create realistic simulations of financial markets. 

2. The CGAN is trained to generate limit orders conditioned by the observed market situation, allowing a pre-trained "world" agent to produce orders to be processed by a simulation environment. The synthetic market generator is integrated into an open source simulator of financial markets called ABIDES. 

3. The authors conduct extensive simulations and evaluate the proposed method based on stylized facts reflecting market responsiveness and realism. The performance achieved by the proposed method outperforms previous work, demonstrating the effectiveness of the synthetic market generator in creating more realistic market simulations.

7. Conclusion: 
--------------------

1. The significance of this piece of work lies in the creation of a synthetic market generator based on a Conditional Generative Adversarial Network (CGAN) trained on real aggregate-level historical data, which provides more realistic market simulations for testing trading strategies.

2. Innovation point: The authors propose a new method to create more realistic simulations of financial markets by training a CGAN to generate limit orders conditioned by the observed market situation. This method improves upon traditional simulation methods and outperforms previous work in terms of market responsiveness and realism. Performance: The extensive simulations conducted by the authors demonstrate the effectiveness of their proposed method in creating more realistic market simulations. Workload: While the proposed method may require more computational resources than traditional simulation methods, it provides more accurate results and can save time and costs compared to testing in the real market. Overall, the strengths of this article lie in its innovative approach and improved performance, outweighed by the potential increased workload.

