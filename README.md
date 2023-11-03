# JAX-LOB

## JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading

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
