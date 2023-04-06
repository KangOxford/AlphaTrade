Usage
=====

.. _installation:

Installation
------------

To use AlphaTrade, first install it using pip:

.. code-block:: console

   (.venv) $ pip install gym_exchange

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``gym_exchange.make()`` function:

.. autofunction:: gym_exchange.make()

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`gym_exchange.make()`
will raise an exception.

.. autoexception:: gym_exchange.make()

For example:

>>> import gym_exchange
>>> gym_exchange.make()

['shells', 'gorgonzola', 'parsley']

