.. _generating:

Generating Realistic Stock Market Order Streams
====================

1. Authors: 
--------------------

Junyi Li, Xintong Wang, Yaoyang Lin, Arunesh Sinha, Michael P. Wellman

2. Affiliation: 
--------------------

University of Pittsburgh, University of Michigan, Harvard University, Singapore Management University, University of Michigan

3. Keywords: 
--------------------

Reinforcement Learning, Generative Adversarial Networks, Finance, Stock Market, Order Streams

4. Urls: 
--------------------

Paper: https://www.aaai.org/ojs/index.php/AAAI/article/view/6934/6786, Github: None

5. Summary:
--------------------

(1): The research background is the need to develop a model capable of producing high-fidelity and high-realism stock market data to support a range of market design and analysis problems.

 

(2): Prior work on market modeling has been limited by low realism and inadequate simulation capabilities. The proposed approach uses generative adversarial networks (GANs) and a conditional Wasserstein GAN to capture time-dependence of order streams, with a generator design that includes components to approximate the market’s auction mechanism and order-book constructions to improve the generation task. These innovations address shortcomings of prior approaches and are well motivated.

 

(3): The research methodology proposed is an approach to generate realistic and high-fidelity stock market data based on a GAN architecture. The model employs a conditional Wasserstein GAN to capture history dependence of orders, and includes specifically crafted components such as a separate neural network to approximate double auction mechanism and order-book information to improve conditioning history of the network. The generator's learned distribution is mathematically characterized, and statistics are proposed to measure the quality of generated orders.

 

(4): The proposed approach achieves close-to-real-data performance compared to other generative models when tested with synthetic and actual market data. The generated data is evaluated using statistics of distribution of price and quantity of orders, inter-arrival times of orders, and best bid and best ask evolution over time. The research methodology supports the goals of generating high-fidelity and high-realism stock market data to support market design and analysis problems.

6. Conclusion:
--------------------

(1): This work proposes a novel approach using generative adversarial networks (GANs) and a conditional Wasserstein GAN to generate high-fidelity and high-realism stock market data to support market design and analysis problems. The approach achieves close-to-real-data performance compared to other generative models, providing a means for conducting research on sensitive stock market data without access to the real data.

(2): Innovation point: The proposed approach uses GANs and a conditional Wasserstein GAN to capture time-dependence of order streams, with a generator design that includes components to approximate the market’s auction mechanism and order-book constructions to improve the generation task. This addresses shortcomings of prior approaches and is well motivated. 

(3): Performance: The approach achieves close-to-real-data performance compared to other generative models when tested with synthetic and actual market data. The generated data is evaluated using statistics of distribution of price and quantity of orders, inter-arrival times of orders, and best bid and best ask evolution over time.

(4): Workload: The workload required to implement the proposed approach is not explicitly discussed. However, the authors acknowledge that future work involves testing the effectiveness of the approach on more stocks.

