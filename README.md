# Dissertation

**`Reinforcement learning in optimal execution`**

Supervised by Prof. Ben Hambly and Prof. Jakob Foerster.

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/10jy4Z4Vb_D0miQY21NnhhNnac0cTEwCI/view?usp=sharing) -->

* Basic Components
  * [`Dissertation`](https://www.overleaf.com/read/mswtggqkvywb) Overleaf
  * [`Documents`](https://drive.google.com/drive/folders/1Ta5N33J8PjD9tZH2OyeXcnMtxd1vS-mT?usp=sharing) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Google drive folder, with all the shard files. &nbsp; 
  * [`Data`](https://drive.google.com/drive/folders/1Hj6sB3eKQi_SpJWncFln8OSVjhlrHGnq?usp=sharing) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [The Discriptions](https://github.com/KangOxford/Dissertation/tree/main/Data)
  * [`Experiments`](https://colab.research.google.com/drive/1QZjz5Q6rIEoTdrgC0N56EiTSorW-Xaqq?usp=sharing) &nbsp; To run it on the `Colab-Vscode`
* Timeline 
  * Jun.01~Jun.31
    * Jun.01-Jun.15 Experiments Implemention.
      * Experiment.01 
        * PartA Turn the Order Flow in to Order Book, with the cancellation type taken into consideration.
        * PartB [Implemented](https://github.com/KangOxford/Dissertation/blob/main/Model_Free_LQR-type1.ipynb) the Model-free LQR form Huining Yang's [paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=DknyNcIAAAAJ&citation_for_view=DknyNcIAAAAJ:u5HHmVD_uO8C). 
      * Experiment.02 Test different RL strategies to do the optimal liquidation.
      * Experiment.03 Compare RL strategies and LQR strategies.
    * Jun.16-Jun.30 Periodical eassy for the works have been done before. 
* Four different settings
  * `Setting_1` Queue-reactive model, with quantity solved by `Monte Carlo` simulation to the stochastic process.
  * `Setting_2` `Implemented` <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [Simulated markets based on PPO agents interaction](https://github.com/KangOxford/Dissertation/blob/main/market%20sim.ipynb). [Get Real: Realism Metrics for Robust Limit Order Book Market Simulations](https://drive.google.com/file/d/1QpmPRC4Wm32QfS8uvjhY66T0AQTSu6LM/view?usp=sharing) 
  * `Setting_3` RL agents with preset reward functions(parameters calibrated from historical data), rather than fixed action policies.
  * `Setting_4` Generative Models. [Generating Realistic Stock Market Order Streams](https://drive.google.com/file/d/1zX1tQfpPaMCSeK7KWcx8x2UhSRsvbQYh/view?usp=sharing)
   
 <br> 

<H1> Part II, RL Strategies</H1>

## `Week.07 Jun.06~Jun.12`
**RL Strategies and Dissertation Draft**.
</br>`2022.Jun.07, 02:30PM~03:30PM`, with `Chriss` and `Timon` at `Informaion Engineering`.
</br>`2022.Jun.08`, with `Prof. Hambly` and `Huining Yang` at `Mathematical Institute`.

**State Representation**<br>
<img height="300" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset3.png">
* In the figure, the observation is the limit order book at one time point. We can add time horizion into it by combining more LOB from different time points 

**Data Augmentation** and **Pretrain by Imitation Learnning**<br>
<img height="300" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset4.v2.png"> &nbsp; &nbsp; &nbsp; &nbsp; <img height="300" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset5.v2.png">
* In the situation imitation learning, the `Benchmark_models` is then converted to the `Baseline_models`.
* There remains a condition, that is the rewards/policy of the `expert model` and `rl model` must keep the same. If this condition is satisfied then the succeeding rl model tend to have a better performence than the baseline model.
* For pretraining, we tend to apply the offline algorithms to train, rather than apply the rl method, which needs to interact with the environment, such as `GAIL` and `AIRL`.
*  `In Implemention` <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> Two kinds of Behaviour Cloning for pretrain with imitation learning: [MSE based](https://github.com/KangOxford/Dissertation/tree/main/ImitationLearning), [MLE based](https://github.com/KangOxford/Dissertation/tree/main/ImitationLearning).

[**Web App by Flask**](https://github.com/KangOxford/Dissertation/tree/main/WebApp) &nbsp; Make it `OpenAI Gym` like style. <br>
<img height="400" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset8.v5.png"> <img height="400" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset9.png">


## `Week.06 May.30~Jun.05`

**RL Strategies**.
</br>`2022.May.27, 04:00PM~04:45PM`, with `Prof. Hambly`, `Dr. Christian`, and `Prof. Foerster` at `Mathematical Institute`.
</br>`2022.May.28, 02:00PM~03:00PM`, with `Chriss` and `Timon` at `Informaion Engineering`.
* Policy gradient methods, NPG, TRPO and PPO
  * Fisher matrix can define a norm as it is a positive definite matrix. Under this norm, the steepest descent direction is NPG and NPG is invariant to the coordinate system we have chosen. Another advantage of NPG is insensitive to the parametrized family of policy we have chosen and thus the training is more stable. <!--   * With the Fisher matrix in the NPG, which is a kind of norm as it is a positive definite matrix, we can get a `Homeomorphism mapping`. It is insensitive to the coordinate system we have chosen. -->
* Model free RL for the experiment.


[**Meeting Record**](https://github.com/KangOxford/Dissertation/tree/main/MeetingRecords/Week6)
</br>`Wednesday Meeting` at the `S0.29`, with `Prof. Hambly`, `Dr. Christian`, and `Prof. Foerster` at `Mathematical Institute`. 

**Structure of `Experiment01.v2`**<br>
<img width="500" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset2.png">
<img width="700" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset6.v2.png">

**Structure of `Experiment01.v1`**<br>
<img height="300" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset7.v3.png">
<img width="450" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset1.png">
<!-- <img height="200" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Asset7.png"> -->
 
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
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [DCGAN](https://github.com/KangOxford/Dissertation/tree/main/GANs/DCGAN)
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [WGAN](https://github.com/KangOxford/Dissertation/tree/main/GANs/WGAN)
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [WGAN-GP](https://github.com/KangOxford/Dissertation/tree/main/GANs/WGAN-GP)
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> ACGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> EBGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> InfoGAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> Conditional GAN
  * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> Contextural GAN
* Generative model for the `setting3`
  * StockGAN 
  * PGSGAN: Policy Gradient Stock GAN for Realistic Discrete Order Data Generation in Financial Markets 
    * `Implemention` of Stock GAN(PG-StockGAN) for Realistic Discrete Order Data Generation in Financial Markets `from scratch`
      * <img width="12" alt="image" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"> [PGSGAN](https://github.com/KangOxford/Dissertation/blob/main/GANs/PGSGAN/PGSGAN.ipynb)
      * The paper for the code: [Policy Gradient Stock GAN for Realistic Discrete Order Data Generation in Financial Markets](https://drive.google.com/file/d/1JSwaeL8hP9UVeiGfgI63F3oROKTS4aAm/view?usp=sharing)

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
  * [Shared folders from Cornell University](https://drive.google.com/drive/folders/1lxM0_JTE4FrqS-2AJgyqRVxfIyVAqQmm?usp=sharing)(Prof. Andreea C. Minca)

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
  * [<img width="600" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Snipaste_2022-05-02_05-45-57.png?raw=true">](https://drive.google.com/file/d/1TqkdstZFrnRDiwFLoJaF078_EWiFs2pn/view?usp=sharing)

## `Week.01 Apr.25~May.01`

* [Week_1 Slides in Latex](https://www.overleaf.com/8586558697psrwmhswmvyc), **Introduction**, `revised` with meeting records.
  </br>`2022.April.25, 11:00AM~12:00Noon`, with `Prof. Foerster` and `Prof. Hambly`, at `Information Engineering Department`.
  * [<img width="600" alt="image" src="https://github.com/KangOxford/Dissertation/blob/main/static/Snipaste_2022-05-01_17-09-15.png?raw=true">](https://drive.google.com/file/d/1gqLcS46IOkqJgZbIBukd3N5wH4IrHiGd/view?usp=sharing)
