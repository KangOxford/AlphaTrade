# Dissertation

**`Reinforcement learning in optimal execution`**

Supervised by Prof. Ben Hambly and Prof. Jakob Foerster.

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/10jy4Z4Vb_D0miQY21NnhhNnac0cTEwCI/view?usp=sharing) -->

* Basic Components
  * [`Dissertation`](https://www.overleaf.com/read/mswtggqkvywb) Overleaf
  * [`Documents`](https://drive.google.com/drive/folders/1Ta5N33J8PjD9tZH2OyeXcnMtxd1vS-mT?usp=sharing) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Google drive folder, with all the shard files.
  * [`Experiments`](https://colab.research.google.com/drive/1QZjz5Q6rIEoTdrgC0N56EiTSorW-Xaqq?usp=sharing) &nbsp; To run it on the `Colab-Vscode`

 <br> 

<H1> Part II, RL Strategies</H1>

## `Week.07 Jun.06~Jun.12`
**RL Strategies and Dissertation Draft**.
</br>`2022.Jun.07, 02:30PM~03:30PM`, with `Chriss` and `Timon` at `Informaion Engineering`.
</br>`2022.Jun.08`, with `Prof. Hambly` and `Huining Yang` at `Mathematical Institute`.


## `Week.06 May.30~Jun.05`

**RL Strategies**.
</br>`2022.May.27, 04:00PM~04:45PM`, with `Prof. Hambly`, `Dr. Christian`, and `Prof. Foerster` at `Mathematical Institute`.
</br>`2022.May.28, 02:00PM~03:00PM`, with `Chriss` and `Timon` at `Informaion Engineering`.
* Policy gradient methods, NPG, TRPO and PPO
  * Fisher matrix can define a norm as it is a positive definite matrix. Under this norm, the steepest descent direction is NPG and NPG is invariant to the coordinate system we have chosen. Another advantage of NPG is insensitive to the parametrized family of policy we have chosen and thus the training is more stable. <!--   * With the Fisher matrix in the NPG, which is a kind of norm as it is a positive definite matrix, we can get a `Homeomorphism mapping`. It is insensitive to the coordinate system we have chosen. -->
* Model free RL for the experiment.


**Meeting Record** 
</br>`Wednesday Meeting` at the `S0.29`, with `Prof. Hambly`, `Dr. Christian`, and `Prof. Foerster` at `Mathematical Institute`. 
</br> <ins>Still confused about the underlined parts</ins>

**In Summary** 
1. We have what exactly was the environment that only uses historical data without modelling any response from other agents to our actions for now.
2. And then we are going to give the agent the ability to eat into the order book, to excute the trades.
3. But we are still going to use the fact that there are changes to the order book based on the incoming orders.
4. So if I eat into the order book, and somebody else happens to like, throw more orders, I can keep excuting. <ins>So the arrival of new orders into the order book is going to be assumed to be unchanged.</ins>
5. GPU simulation already built: turn order flow into order book.
6. <ins>Then I can add to the order flow as the agent</ins> The agent can add to the order flow, and then we should be good because just pretend that the historical flow is the agent. Then gradually, we could replace more agents with <ins>it</ins> to have the response. <ins>And that gives us stays keeps us close to the data.</ins>

**In Details**
1. It isn't an interesting way to generate the environment, basically, it's basically the environment actually consists of real data.
2. Some heuristic for how the real data changes based on the interactions of the agent. So it's basically saying, and we don't need to make it because that was a linearity assumption. 
3. We don't really have to go and generate all the data. We just want to generate a difference between what happened in it.
4. However, if we interact in the market, there's going to be a tiny $\delta$, we are just going to try and like have a model for the delta between the data based upon the agents actions, as a result, and you've done this in your in your paper, right, because you've said there is a linear there is a delta is delta that is linear. 
5. We only have to have a model for the perturbation of the data. And obviously, that means that the overall <ins>approximation error, because I have to first order in correct</ins>. And if I'm like one of many people playing in the market, hopefully the market isn't gonna look completely different based on my actions.
6. We then have to be sure that we have enough real data so that we can <ins>produce enough samples and the agent doesn't just memorise these trajectories.</ins>
7. If you take out a big chunk, then people are likely to throw things in because they think they're gonna get executed.
8. <ins>There's no approximations at all.</ins>
9. For an agent: observation $\tau$ is the 10 minites order book history. And also dollar: how much you want to sell.
10. The reward is how much money we get for this.
11. The action space is in the next 10 minutes, at any time point we can take an action. From execution, we have determined a period to excute.
12. At the end of 10 minutes, everything else will get sold. We actually have to clean the position.
13. Challenge: Do we have enough data?
14. And then later on, maybe once you have this agent, you can throw a bunch of these into the environment that want to buy and sell different amounts of stuff. And that generates response.
15. Then you have a model for response among these RL agents but always <ins>grounded around the historical data<ins>. <ins>And the response emerges quite naturally because there's other agents</ins>.


 <br> 
 
# Part I, Market Simulation

## `Week.05 May.23~May.29`

* **Market Simulation Implemention Based on GAN**.
* Difference between $P_{data}$ and $P_{G}$ ($Generated$) in GAN
  * $f$-GAN, with Fenchel Conjugate, choose different $f-divergence$
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/37290277/172031861-cd6b8278-f801-412c-a163-b0e7c8d526c1.png">
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/37290277/172031871-abafdad8-e4d2-4a37-bafc-8853427e1ed2.png">
  * Wasserstein distance
* Techniques in GAN training
  * Ensemble, to figure out the problem of low diversity, `Standard ensemble`, `Self-ensemble` and `Cascade of GAN`
  * Typical issues to be aware of while designing the GAN architecture: `Model Collapse`, `Missing Mode`, `Memory GAN`
  * Evalutation: `likelihood`(kernal density estimation), evaluated by other networks, eg. Inception score(IS)，Fréchet Inception Distance(FID). 
  * Different GAN architechtures for distinct tasks with specific purpose
    * `Pitch GAN` and `Stack GAN` for large single_data_size generation
    * `InfoGAN`, `VAE-GAN`, `BiGAN` for Feature extraction
    * Conditional GAN (on LOB). Generated data based on the historical data.
    * Domain-adversarial training for traing and testing data are in different doimains
    * Feature disentangle, UNIT(based on Coupled GAN) and Cycle-GAN(same thing: dual/disco GAN) for style transfer
* Are GANs creared equal? (GANs have not so big difference. GAN outperforms VAE to a great extent.)
  </br> <img height="135" alt="image" src="https://user-images.githubusercontent.com/37290277/172035714-8f4a36e5-59c4-4046-85bd-347c5f3f2cf7.png"> <img height="135" alt="image" src="https://user-images.githubusercontent.com/37290277/172035780-7d7b9afe-4584-468f-88cc-e290497d2871.png">
  </br> <img width="800" alt="image" src="https://user-images.githubusercontent.com/37290277/172035651-45d83015-249c-4952-97b9-c408a8f5e11b.png"> 
* `2022.Jun.05, 06:00PM~07:30PM`, with `Chao Zhang` from `Department of Statistics`.
  * `LOB model, with the focus of forecasting` (Time Series and DL)
  * `Time GAN`, GAN applied in time series genetation
  * `Quant GAN`. Time Series Data Synthesis
* `Implemention` of the common GANs `from scratch`
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [SimpleGAN](https://github.com/KangOxford/Dissertation/tree/main/GANs/SimpleGAN)
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [WGAN](https://github.com/KangOxford/Dissertation/tree/main/GANs/WGAN)
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> WGAN-GP
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> DCGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> EBGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> InfoGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> Conditional GAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> Context Conditional GAN
* `Implemention` of Stock GAN for Realistic Discrete Order Data Generation in Financial Markets `from scratch`
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [PGSGAN](https://github.com/KangOxford/Dissertation/blob/main/GANs/PGSGAN/PGSGAN.ipynb) 
<!-- &nbsp; -->
<!--   * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> CycleGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> StyleGAN -->
<!-- * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> BEGAN -->

## `Week.04 May.16~May.22`

* **Market Simulation Implemention and Benchmark Seeking**.
  </br>`2022.May.18, 04:00PM~04:45PM`, with `Prof. Hambly`, and `Huining Yang` at `Mathematical Institute`.
* Testing of performance of the rama lob model.
  * Meeting with `Zhangcheng.sun` at `Qube`
* Data synthesis via GAN.
* Optimal Liquidation
  * Use the Huining's LQR model as a benchmark.
    * It can work with unknown parameters and outperfoms the AC model.
  * `Dynamic` and `Static` LOB
    * the difference is that the Dynamic one can have the price impact
  * Imitation learning via AIRL and GAIL
    * `AIRL` [Learning Robust Rewards with Adverserial Inverse Reinforcement Learning](https://drive.google.com/file/d/195Km4VxW9AVKWitmCcSlbuXurbsIyeL1/view?usp=sharing)
    * [Adversarial recovery of agent rewards from latent spaces of the limit order book](https://drive.google.com/file/d/11oXnRRrwfighxG5ifVE6ENPJB43_TYmK/view?usp=sharing)
      * Inverse RL, reward recovery may conduct to different sets of rewards.
      * We should instead, recover the policy from the expert trajectories.
* [A stochastic model for order book dynamics](https://drive.google.com/file/d/125ZhxZKB-42gNbCmUfjtWgKntsAUUwin/view?usp=sharing) `Implemented`

## `Week.03 May.09~May.15`

* **Market Simulation Implemention and Benchmark Seeking**.
  </br>`2022.May.09, 11:00AM~12:15Noon`, with `Prof. Hambly`, `Prof. Foerster`, `Huining Yang`, and `Dr. Christian` at `Mathematical Institute`.
  </br>`May.17 04:00P.M, Next Tuesday` with `Prof. Hambly`, and `Huining Yang`
  </br>`May.23 11:00A.M.` with  `Prof. Hambly`, `Prof. Foerster`, `Huining Yang`, and `Dr. Christian`.
  * `Setting_1` Queue-reactive model, with quantity solved by `Monte Carlo` simulation to the stochastic process, or `High(2N) dimensional PDEs`, solved by `Fourie Neural Operater`(2020). Implemention refer to this [repository](https://github.com/KangOxford/Fourier-Transformer).
  * `#TODO` This week
    * Implement PPO agents in the setting of this paper: [`Setting_2` Get Real: Realism Metrics for Robust Limit Order Book Market Simulations](https://drive.google.com/file/d/1QpmPRC4Wm32QfS8uvjhY66T0AQTSu6LM/view?usp=sharing) ,with no need to turn to the GPU version unless coming across the situation we really need the speed.
    * Find the Benchmark for the `Setting_2`.
  * `Setting_3` RL agents with preset reward functions(parameters calibrated from historical data), rather than fixed action policies.
  * `Setting_4` Generative Models
    * [Generating Realistic Stock Market Order Streams](https://drive.google.com/file/d/1zX1tQfpPaMCSeK7KWcx8x2UhSRsvbQYh/view?usp=sharing)
    * [Deep Learning of the Order Flow for Modelling Price Formation](https://drive.google.com/file/d/1ohi-dCJf7uiGsYSh8x6Jkd_eYr0sFr2D/view?usp=sharing)
  * Data requirements
    * `Pending` [Lobster](https://lobsterdata.com/) data from Prof. Hambly
    * `Solved ` Amazon LOB [data](https://drive.google.com/file/d/13aazGzyp6MqhwZDs--TRR2WQu9uo5TI7/view?usp=sharing) from Huining Yang
* `New Criterion Found` : **KS-distance or Wasserstein Distance** in [Generating Realistic Stock Market Order Streams](https://drive.google.com/file/d/1UYb4mNsqTcfmy25oxcqRqy8qoAXbSuOr/view?usp=sharing)
  * [Shared folders from Cornell University](https://drive.google.com/drive/folders/1lxM0_JTE4FrqS-2AJgyqRVxfIyVAqQmm?usp=sharing)

## `Week.02 May.02~May.08`

* [Week_2 Slides in Latex](https://www.overleaf.com/9548188445gqpcppdvxbrf), **Market Simulation Literature Review**. Here are the [papers](https://drive.google.com/drive/folders/15qHlvRmFMd_oaMlqXxRFkLD64gnkAPcy?usp=sharing) for this review. It has been [`revised`](https://www.overleaf.com/9548188445gqpcppdvxbrf) with meeting records.
  </br>`2022.May.02, 11:30AM~12:00Noon`, with `Prof. Foerster`, `Dr. Christian`, and `Timon`, at `Information Engineering Department`.
  </br>`2022.May.04, 12:00AM~12:45Noon`, with `Prof. Hambly`, and `Huining Yang`, at `Mathematical Institute`.
  </br> ~~`2022.May.04, 04:30PM~05:00PM`, with `Timon`, and `Chris`, at `Mathematical Institute`~~ `Rearranged`. [Slides](https://www.overleaf.com/7776181439bdmxqpfsrsgp) for `Introduction of the math part`.
  * Math Intro to the Optimail Liquidation
    * [Algorithmic and High-Frequency Trading](https://drive.google.com/file/d/1RROKPIsICQnoc6jPudeSk4wTkgsh777h/view?usp=sharing)
    * [Market Microstructure and Algorithmic Trading](https://drive.google.com/file/d/1wBDATDeOrFL10NuHgy7gdA3AYIinx_y7/view?usp=sharing)
  * `#TODO` this week
    * get familiar with the [GPU version of LOB](https://github.com/KangOxford/Dissertation/blob/main/market%20sim.ipynb) this week.
    * read the following paper:
      * [Simulating and analyzing order book data: The queue-reactive model](https://drive.google.com/file/d/1rJd7TxzcZSoQipzNI9asJ_DwCUhqEVB4/view?usp=sharing)
      * [Deep Reinforcement Learning in Agent Based Financial Market Simulation](https://drive.google.com/file/d/1k8URGCP06wvm2J5Se7m11iq9KAkkqv0B/view?usp=sharing)
      * [Generative models of limit order books](https://drive.google.com/file/d/1_rxMUxZsnmNJG4ytfgnenvw0HTTy-P9D/view?usp=sharing)
      * **[`Important` Get Real: Realism Metrics for Robust Limit Order Book Market Simulations](https://drive.google.com/file/d/1QpmPRC4Wm32QfS8uvjhY66T0AQTSu6LM/view?usp=sharing)**
  * [![Week 2 Slides](https://github.com/KangOxford/Dissertation/blob/main/static/Snipaste_2022-05-02_05-45-57.png?raw=true)](https://drive.google.com/file/d/1TqkdstZFrnRDiwFLoJaF078_EWiFs2pn/view?usp=sharing)

## `Week.01 Apr.25~May.01`

* [Week_1 Slides in Latex](https://www.overleaf.com/8586558697psrwmhswmvyc), **Introduction**, `revised` with meeting records.
  </br>`2022.April.25, 11:00AM~12:00Noon`, with `Prof. Foerster` and `Prof. Hambly`, at `Information Engineering Department`.
  * [![Week 1 Slides](https://github.com/KangOxford/Dissertation/blob/main/static/Snipaste_2022-05-01_17-09-15.png?raw=true)](https://drive.google.com/file/d/1gqLcS46IOkqJgZbIBukd3N5wH4IrHiGd/view?usp=sharing)
