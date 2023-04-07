Future Research Topics
=====

**************
Related Topics
**************

* **Optimal Execution**
   * Optimal liquidation or acqurization within 15-30 minutes with agent trained on the simulated environment.
* **Optimal Scheduling with Predicted Trading Volume**
   * In :doc:`_vwap_strategies`, split the task into smaller sizes according to the :doc:`_volume`.
* **Order-flow Generating**
   * Mathematical Perspective: Order flow as a general spatial point process
   * Time-series Forecasting for Order Flow
   * Order Flow Generating by Large Language Models
* Representation Learning
   * Representation Learning for the States/Observations
   * In the Optimal Execution, e.g. the all snapshots of limit order book in the past 30 seconds.
   * VAE, GAN and other encoder models.
* Price Impact Research through Market Clearing
   * Mathematical Perspective: Market clearing as a deterministic operator acting on the distributions of buy and sell orders.
   * Calculate the price impact without the assumption of impact function
* Indirect Market Impact
   * Agent's Impact on Triggering the Modification of other Agents' Actions
   * Different from the price impact, which is the direct maret impact.
* Agent Based Modelling/Simulation
   * Generative adversarial network approach simulation
   * Market Simulation
* Recover Trader's Reward Function
   * Recover Trader's Reward Function by Inverse RL
* Unsupervised Environment Design
   * Adversarial Learning by the differentiable environment


**************
Related Papers
**************

All related ICAIF and OMI Research Newsletter papers are included.

* Related Sections
   * Simulated Markets
   * Learning Trading Strategies
   * Forecasting Financial Data

High Related ICAIF Papers

* ICAIF2022
    * :doc:`_dyn`
    * :doc:`_learn`
    * :doc:`_cost`
* ICAIF2021
   * High Related
   * :doc:`_towards_fully`
   * :doc:`_towards`
   * :doc:`_learning`
   * :doc:`_bit`
* ICAIF2020
   * :doc:`_get`
   * :doc:`_multi`
   * :doc:`_deep`
* :doc:`_mid_related_icaif`
* OMI Research Newsletter
   * Market Microstructure
      * :doc:`_many`
      * :doc:`_peer`
      * :doc:`_liquidation_signals`
      * :doc:`_stochastic_liquidity`
      * :doc:`_exogenous`
      * :doc:`_towards_multi`
      * :doc:`_model_based_env`
      * :doc:`_simple`
   * Time Series Forecasting
      * :doc:`_scale`
      * :doc:`_lob_hft`
      * :doc:`_generative_tsf`
      * :doc:`_shortterm_predictability`
* Other related papers
   * :doc:`_stock`
   * :doc:`_generating`
   * :doc:`_deeprl`



**************
Related Techniques
**************
* Transformers
   * Time Series Forecasting with Transformers
      * :doc:`_transformers_tsf`
   * Transformer in Low Signal-noise Ratio System
      * Sparse Transfomer: :doc:`_sparse_tf`
* Long Sequence Modelling
   * :doc:`_efficiently`
   * :doc:`_s5`
* Unsupervised Environment Design
   * :doc:`_ued`
* Behavior Cloning
   * :doc:`_bc`


**************
Related Issues
**************
* **Hard to generalize**. There might be several reasons jointly contribute to this situation:
   1. The **signal-to-noise ratio** of financial market data is much lower than that of other artificial intelligence fields.
   2. The financial market is not a closed system and will **evolve** on its own.
   3. The financial market is a derivative of the economy and therefore can be impacted by **external factors**.
