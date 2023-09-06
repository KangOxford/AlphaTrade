# AlphaTrade

## Research in Limit Order Book for Optimal Execution


![image](https://user-images.githubusercontent.com/37290277/233871831-cd0f3afd-62c0-4e0f-a16e-71e932784211.png)

* perception, predicting, planning&controlling

### Architecture of the package:
* `gymnax_exchange`: the `GPU` version of rl_environment
  * `jaxob`: Jax limit order book
  * `jaxen`: Jax trading_environment
  * `jaxrl`: Jax training loop

### Outlines for our work:
* Converting into Jax
* Baseline Models(TWAP, VWAP), Pre-train(Imitation Learning, GAIL)
* Representing the order book
* Generating Order Flows by Tockenizing with Large Language Models


* Anyone who might conribute codes to this repositry, please send email to me: kang.li@stats.ox.ac.uk 
