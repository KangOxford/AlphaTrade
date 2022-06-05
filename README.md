# Dissertation

Reinforcement learning in optimal execution

Supervised by Prof. Ben Hambly and Prof. Jakob Foerster.

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/10jy4Z4Vb_D0miQY21NnhhNnac0cTEwCI/view?usp=sharing) -->

* Basic Components
  * Here is the link to dissertation [overleaf](https://www.overleaf.com/8586558697psrwmhswmvyc)
  * Here is the link to the [google drive folder](https://drive.google.com/drive/folders/1Ta5N33J8PjD9tZH2OyeXcnMtxd1vS-mT?usp=sharing), with all the shard files.
  * Here is the link to run it on the [Colab-Vscode](https://colab.research.google.com/drive/1QZjz5Q6rIEoTdrgC0N56EiTSorW-Xaqq?usp=sharing)

## `Week.06 May.30~Jun.05`
* **RL Strategies**. 

## `Week.05 May.23~May.29`
* **Market Simulation Implemention**. 
</br>`2022.May.27, 04:00PM~04:45PM`, with `Prof. Hambly`, `Dr. Christian`, and `Prof. Foerster` at `Mathematical Institute`.
</br>`2022.May.28, 02:00PM~03:00PM`, with `Chriss` and `Timon` at `Informaion Engineering`.
* Critic for $P_{data}$ and $P_{G}$
  * Fenchel Conjugate, choose different $f-divergence$
  * Wasserstein distance
* Techniques in GAN training
  * PitchGAN and StackGAN 

## `Week.04 May.16~May.22`
* **Market Simulation Implemention and Benchmark Seeking**. 
 </br>`2022.May.18, 04:00PM~04:45PM`, with `Prof. Hambly`, and `Huining Yang` at `Mathematical Institute`.
* Testing of performance of the rama lob model.
  * Meeting with `Zhangcheng.sun` at `Qube `
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
  * `Setting_1` Queue-reactive model, with quantity solved by `Monte Carlo` simulation to the stochastic process, or `High(2N) dimensional PDEs`, solved by `Fourie Neural Operater`(2020). Implemention refer to this [repository](https://github.com/KangOxford/Fourier-Transformer).
  * `#TODO` This week
    * Implement PPO agents in the setting of this paper: [`Setting_2` Get Real: Realism Metrics for Robust Limit Order Book Market Simulations](https://drive.google.com/file/d/1QpmPRC4Wm32QfS8uvjhY66T0AQTSu6LM/view?usp=sharing) ,with no need to turn to the GPU version unless coming across the situation we really need the speed. 
    * Find the Benchmark for the `Setting_2`.
  * `Setting_3` RL agents with preset reward functions(parameters calibrated from historical data), rather than fixed action policies.
  * `Setting_4` Generative Models
    * [Generating Realistic Stock Market Order Streams](https://drive.google.com/file/d/1zX1tQfpPaMCSeK7KWcx8x2UhSRsvbQYh/view?usp=sharing) 
    * [Deep Learning of the Order Flow for Modelling Price Formation](https://drive.google.com/file/d/1ohi-dCJf7uiGsYSh8x6Jkd_eYr0sFr2D/view?usp=sharing)
  * Sheduled meeting : 
    * `May.17 04:00P.M, Next Tuesday` with `Prof. Hambly`, and `Huining Yang`.
    * Discuss with  `Prof. Foerster` via email after the meeting.
    * `May.23 11:00A.M.` with  `Prof. Hambly`, `Prof. Foerster`, `Huining Yang`, and `Dr. Christian`.
  * Data requirements
    * `Pending` [Lobster](https://lobsterdata.com/) data from Prof. Hambly
    * `Solved ` Amazon LOB [data](https://drive.google.com/file/d/13aazGzyp6MqhwZDs--TRR2WQu9uo5TI7/view?usp=sharing) from Huining Yang
* `New Criterion Found` : **KS-distance or Wasserstein Distance** in [Generating Realistic Stock Market Order Streams](https://drive.google.com/file/d/1UYb4mNsqTcfmy25oxcqRqy8qoAXbSuOr/view?usp=sharing)
  * [Shared folders from Cornell University](https://drive.google.com/drive/folders/1lxM0_JTE4FrqS-2AJgyqRVxfIyVAqQmm?usp=sharing)


## `Week.02 May.02~May.08`
* [Week_2 Slides in Latex](https://www.overleaf.com/9548188445gqpcppdvxbrf), **Market Simulation Literature Review**. Here are the [papers](https://drive.google.com/drive/folders/15qHlvRmFMd_oaMlqXxRFkLD64gnkAPcy?usp=sharing) for this review. It has been [`revised`](https://www.overleaf.com/9548188445gqpcppdvxbrf) with meeting records.
  </br>`2022.May.02, 11:30AM~12:00Noon`, with `Prof. Foerster`, `Dr. Christian`, and `Timon`, at `Information Engineering Department`.
  </br>`2022.May.04, 12:00AM~12:45Noon`, with `Prof. Hambly`, and `Huining Yang`, at `Mathematical Institute`.
  </br>~~`2022.May.04, 04:30PM~05:00PM`, with `Timon`, and `Chris`, at `Mathematical Institute`~~ `Rearranged`. [Slides](https://www.overleaf.com/7776181439bdmxqpfsrsgp) for `Introduction of the math part`.
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





