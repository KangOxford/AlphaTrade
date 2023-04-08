.. _pg_stockgan:

Policy Gradient Stock GAN for Realistic Discrete Order Data Generation in Financial Markets, April 2022
====================

1. Authors: 
--------------------

Masanori HIRANO, Hiroki SAKAJI, Kiyoshi IZUMI

2. Affiliation: 
--------------------

School of Engineering, The University of Tokyo (Masanori HIRANO, Hiroki SAKAJI, Kiyoshi IZUMI)

3. Keywords: 
--------------------

Generative adversarial networks (GAN), Financial markets, Policy gradient, Order generation

4. Urls: 
--------------------

https://arxiv.org/abs/2204.13338v1 

5. Summary:
--------------------

(1): The article aims to generate realistic orders for financial markets. Due to the insufficiency of data, GANs are employed to augment past data.

(2): Previous works for stock markets generated fake orders in continuous spaces, which are not acceptable in the order systems in reality. The article aims to address this issue by placing generated fake orders into discrete spaces using policy gradient, which, due to GAN architectures' learning limitations, has not been employed in the past.

(3): The proposed model employs policy gradient for the learning algorithm to generate realistic orders in financial markets. The entropy of the generated policy can be used to check GAN learning status.

(4): The proposed model outperforms previous models in generated order distribution. The performance of the proposed model supports their goals of generating realistic orders in financial markets.

6. Methods: 
--------------------

(1): The proposed model employs a GAN framework to generate realistic orders for financial markets by learning from past data. The generator module takes in the noise input and generates fake order samples, while the discriminator tries to distinguish real and fake orders. 

(2): To address the issue of generating orders in discrete spaces, the policy gradient method is employed to assign probabilities to each possible order at each time step, and the orders are sampled from the probabilities. The model is optimized using a combination of adversarial objective and policy gradient loss. 

(3): In order to check the learning status of the GAN architecture, the entropy of the generated policy is used as a metric. Higher entropy indicates more diverse order generation, which resembles the real-world order distribution. 

(4): The proposed model is evaluated on a real-world dataset, and the performance is compared with previous models based on various metrics, including order distribution, simulated P/L, and trade frequency. Results show that the proposed model outperforms previous models in all metrics, demonstrating its effectiveness for generating realistic orders in financial markets.

7. Conclusion:
--------------------

(1): This piece of work proposes a Policy Gradient Stock GAN for Realistic Discrete Order Data Generation in Financial Markets, which aims to generate realistic orders for financial markets and address the issue of generating orders in discrete spaces using policy gradient. 

(2): Innovation Point: the article employs policy gradient for the learning algorithm to generate realistic orders in financial markets, which has not been explored in the past due to GAN architectures' learning limitations. (3): Performance: the proposed model outperforms previous models in generated order distribution, simulated P/L, and trade frequency, demonstrating its effectiveness for generating realistic orders in financial markets. (4): Workload: the study uses a real-world dataset for evaluation and proposes a combination of adversarial objective and policy gradient loss for optimization, but the workloads are not explicitly mentioned in the article.

