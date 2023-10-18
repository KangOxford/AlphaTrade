# ===================================== #
# ******* Config your own func ******** #
# ===================================== #

from gymnax_exchange.jaxob.JaxOrderBookArrays import *
@partial(jax.jit)
def get_best_bid(asks, bids):
    L2_state = get_L2_state(1, asks, bids)
    return L2_state[0]

@partial(jax.jit)
def get_best_bid(asks, bids):
    L2_state = get_L2_state(1, asks, bids)
    return L2_state[1]
