Introduction
========

:doc:`_framework` proposed a mathematical framework to define orderbook dynamics, which involves two crucial components: market clearing and order flow generating. In the following, I will elaborate on this package from these two perspectives.

The package comprises three main components.

* The first part is the Order-Book, which acts as the matching engine for market clearing.

* The second part is the exchange, which supplements the Order-Book by providing additional supporting information, such as executed pairs, and extending its functionality to handle "smart orders". For instance, it can automatically cancel orders if they are not executed within a specified time frame.

* The final part is the trading environment, which builds on the Order-Book and exchange to provide auxiliary information for agents. This includes submitting smart orders, accessing Order-Book snapshots from the last 30 seconds, calculating the volume-weighted average price (VWAP) for agents and other market participants, and providing slippage information. All three parts are based on Google Jax, a framework similar to PyTorch.

These parts are highly vectorized and differentiable. And it runs fast on both GPU and CPU.

In addition, if we trained deep learning algorithms on this package and view it as a project, it also enables Order Generating. After training a deep learning agent to perform a specific task that is comparable to human performance, we can replicate this agent, introduce random factors, and fine-tune it for different tasks. This group of agents can interact and compete with each other, having the potential to simulate the market effectively, which is why Order Generating can also be implemented in this project. By contrast, traditional methods of defining agents are overly simplistic and lack the flexibility to handle multiple scenarios and respond appropriately.
