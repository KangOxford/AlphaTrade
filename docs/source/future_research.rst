Future Research Topics
=====


Optimal Execution 
----------------


For example:

>>> import gym_exchange
>>> gym_exchange.make()

Optimal Scheduling with Predicted Trading Volume
---------------
* In vwap replicate strategies, split the task into smaller sizes according to the predicted trading volume.

Order-flow Generating 
---------
* Mathematical Perspective: Order flow as a general spatial point process
* Time-series Forecasting for Order Flow
* Order Flow Generating by Large Language Models

Representation Learning
----------
* Representation Learning for the States
* VAE, GAN and other encoder models.


Price Impact Research through Market Clearing
----------
* Mathematical Perspective: Market clearing as a deterministic operator acting on the distributions of buy and sell orders.
* Calculate the price impact without the assumption of impact function


Indirect Market Impact
----------------------
* Agent's Impact on Triggering the Modification of other Agents' Actions
    * Different from the price impact, which is the direct maret impact.

Agent Based Modelling/Simulation
-----------------------
* Generative adversarial network approach simulation
* Market Simulation

Unsupervised Environment Design
----------
* Adversarial Learning by the differentiable environment



Related Papers
------
* Related Sections
   * Simulated Markets
   * Learning Trading Strategies
   * Forecasting Financial Data
* ICAIF2022
   * High Related
       * :doc:`_dyn`
       * :doc:`_learn`
   * Mid Related
       * Cost-Efficient Reinforcement Learning for Optimal Trade Execution on Dynamic Market Environment
       * Market Making under Order Stacking Framework: A Deep Reinforcement Learning Approach
   * Low Related
       * Graph and tensor-train recurrent neural networks for high-dimensional models of limit order books
       * Computationally Efficient Feature Significance and Importance for Predictive Models
       * LaundroGraph: Self-Supervised Graph Representation Learning for Anti-Money Laundering
       * Deep Hedging: Continuous Reinforcement Learning for Hedging of General Portfolios across Multiple Risk Aversions
       * Efficient Calibration of Multi-Agent Simulation Models from Output Series with Bayesian Optimization
* ICAIF2021
   * High Related
      * :doc:`_towards_fully`
      * :doc:`_towards`
      * FinRL: deep reinforcement learning framework to automate trading in quantitative finance
      * Bit by bit: how to realistically simulate a crypto-exchange
      * Deep Q-learning market makers in a multi-agent simulated stock market
      * :doc:`_learning`
   * Mid Related
      * Sig-wasserstein GANs for time series generation
      * Agent-based markets: equilibrium strategies and robustness
      * Intelligent trading systems: a sentiment-aware reinforcement learning approach
      * High frequency automated market making algorithms with adverse selection risk control via reinforcement learning
   * Low Realted
      * An automated portfolio trading system with feature preprocessing and recurrent reinforcement learning
      * Monte carlo tree search for trading and hedging
      * Visual time series forecasting: an image-driven approach
      * Trading via selective classification
      * Timing is money: the impact of arrival order in beta-bernoulli prediction markets
      * An agent-based model of strategic adoption of real-time payments
      * FinRL-podracer: high performance and scalable deep reinforcement learning for quantitative finance
      * Stability effects of arbitrage in exchange traded funds: an agent-based model
* ICAIF2020
   * Get real: realism metrics for robust limit order book market simulations
   * Multi-agent reinforcement learning in a realistic limit order book market simulation
   * Deep reinforcement learning for automated stock trading: an ensemble strategy
   * A tabular sarsa-based stock market agent
   * Dynamic prediction length for time series with sequence to sequence network

Related Techniques
----------
* Long Sequence Modelling
   * :doc:`_s5`
