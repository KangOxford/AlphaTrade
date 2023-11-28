# JAX-LOB

## JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading

### Architecture of the package:
* `gymnax_exchange`: the `GPU` version of rl_environment
  * `jaxob`: Jax limit order book
  * `jaxen`: Jax trading_environment
  * `jaxrl`: Jax training loop

## Dependencies

```
pip install jax[cuda]==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html distrax brax chex flax optax gymnax wandb  
```

## Installation

To install the latest version of Jaxlob, run:

```bash
pip install jaxlob
```

## Citation

If you find this project useful, please cite:

```
@misc{frey2023jaxlob,
      title={JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading}, 
      author={Sascha Frey and Kang Li and Peer Nagy and Silvia Sapora and Chris Lu and Stefan Zohren and Jakob Foerster and Anisoara Calinescu},
      year={2023},
      eprint={2308.13289},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR}
}
```
## Contribution

* Anyone who might contribute codes to this repository, please email me: kang@robots.ox.ac.uk 
