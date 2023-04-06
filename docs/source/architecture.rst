Architecture of the package
========



* ``gym_exchange``: the ``CPU`` version of rl_trading_environment

  * ``order_book``: order_book as the container

  * ``order_book_adapter``: convert data into trading_signal, then to order_flows

  * ``exchange``: simulated stock exchange

  * ``trading_environment``: rl_trading_environment

* ``gymnax_exchange``: the ``GPU`` version of rl_trading_environment

  * ``jaxob``: Jax limit order book

  * ``jaxes``: Jax exchange

  * ``jaxen``: Jax rl_trading_environment





