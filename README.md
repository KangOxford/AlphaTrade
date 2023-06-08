# AlphaTrade

## Research in Limit Order Book for Optimal Execution


This is a lean version of the `main` branch, primarily intended for the development of simulation environments.

Previous document refers to `main` branch



![image](https://user-images.githubusercontent.com/37290277/233871831-cd0f3afd-62c0-4e0f-a16e-71e932784211.png)



### Architecture of the package:
* `gym_exchange`: the `CPU` version of rl_environment
  * `order_book`: order_book as the container
  * `order_book_adapter`: convert data into trading_signal, then to order_flows
  * `exchange`: simulated stock exchange
  * `environment`: rl_environment
* `gymnax_exchange`: the `GPU` version of rl_environment
  * `jaxob`: Jax limit order book
  * `jaxes`: Jax exchange
  * `jaxen`: Jax rl_environment

### Outlines for our work:
* Converting into Jax
* Baseline Models(TWAP, VWAP), Pre-train(Imitation Learning, GAIL)
* Representing the order book
* Generating Order Flows by Tockenizing with Large Language Models

### Slides for `27.Feb~05.Mar` Meeting 

### Slides for `20.Feb~26.Feb` Meeting 
* [Overleaf](https://www.overleaf.com/7842834529bwxpvqnsdqsv)
* Anyone who might conribute codes to this repositry, please send email to me: kang.li@stats.ox.ac.uk 
