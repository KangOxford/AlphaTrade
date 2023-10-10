from functools import partial
import importlib
from os import remove
from readline import remove_history_item
from typing import Dict, NamedTuple, Optional, Tuple
from unicodedata import bidirectional
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
job = importlib.reload(job)
import jax
from jax import numpy as jnp
from jax import lax



class LobState(NamedTuple):
    asks: jnp.ndarray
    bids: jnp.ndarray
    trades: jnp.ndarray


class OrderBook():
    def __init__(
            self: 'OrderBook',
            nOrders: int = 100,
            nTrades: int = 100
        ) -> None:
        self.nOrders = nOrders
        self.nTrades = nTrades

    @jax.jit
    def init(self: 'OrderBook') -> LobState:
        asks = (jnp.ones((self.nOrders, 6)) * -1).astype(jnp.int32)
        bids = (jnp.ones((self.nOrders, 6)) * -1).astype(jnp.int32)
        trades = (jnp.ones((self.nTrades, 6)) * -1).astype(jnp.int32)
        return LobState(asks, bids, trades)

    @jax.jit
    def reset(
            self: 'OrderBook',
            l2_book: Optional[jnp.ndarray] = None,
        ) -> LobState:
        """"""
        state = self.init()
        if l2_book is not None:
            msgs = job.init_msgs_from_l2(l2_book)
            state = self.process_orders_array(state, msgs)
        return state

    @jax.jit
    def process_order(
            self: 'OrderBook',
            state: LobState,
            quote: Dict,
            from_data: bool = False,
            verbose: bool = False
        ) -> LobState:
        '''Wrapper function for the object class that takes a Dict Object as the quote,
         ensures the order is conserved and turns the values into a jnp array which is passed to the JNP ProcessOrder function'''
        #Type, Side,quant,price
        inttype = 5
        intside = -1
        if quote['side'] == 'bid':
            intside = 1 

        if quote['type'] == 'limit':
            inttype = 1
        elif quote['type'] == 'cancel':
            inttype = 2
        elif quote['type'] == 'delete':
            inttype = 2
        elif quote['type'] == 'market':
            inttype = 1
            intside = intside * -1

        msg = jnp.array([
            inttype,
            intside,
            quote['quantity'],
            quote['price'],
            quote['trade_id'],
            quote['order_id'],
            int(quote['timestamp'].split('.')[0]),
            int(quote['timestamp'].split('.')[1])
        ], dtype=jnp.int32)

        ordersides, _ = job.cond_type_side(state, msg)
        return LobState(*ordersides)

    @jax.jit
    def process_order_array(
            self: 'OrderBook',
            state: LobState,
            quote: jax.Array,
            from_data: bool = False,
            verbose: bool = False
        ) -> LobState:
        '''Same as process_order but quote is an array.'''
        ordersides, _ = job.cond_type_side(state, quote)
        return LobState(*ordersides)

    @jax.jit
    def process_orders_array(
        self: 'OrderBook',
        state: LobState,
        msgs: jax.Array,
    ) -> LobState:
        '''Wrapper function for the object class that takes a JNP Array of messages (Shape=Nx8), and applies them, in sequence, to the orderbook'''
        return LobState(*job.scan_through_entire_array(msgs, tuple(state)))

    @jax.jit
    def process_orders_array_l2(
            self: 'OrderBook',
            state: LobState,
            msgs: jax.Array,
            n_levels: int
        ) -> Tuple[LobState, jax.Array]:
        all_asks, all_bids, trades = job.scan_through_entire_array_save_states(msgs, state, msgs.shape[0])
        state = LobState(all_asks[-1], all_bids[-1], trades)
        # calculate l2 states
        l2_states = job.vmap_get_L2_state(all_asks, all_bids, n_levels)
        return state, l2_states

    @partial(jax.jit, static_argnums=(2,))
    def get_volume_at_price(
            self: 'OrderBook',
            state: LobState,
            side: int,
            price: int
        ) -> int:
        if side == 0:
            side_array = state.bids
        elif side == 1:
            side_array = state.asks
        else:
            raise ValueError('Side must be 0 or 1')

        volume = jnp.sum(
            jnp.where(
                side_array[:,0] == price,
                side_array[:,1],
                0
            )
        )
        return volume

    @partial(jax.jit, static_argnums=(1,))
    def get_best_price(
            self: 'OrderBook',
            state: LobState,
            side: int
        ) -> int:
        # sell / asks
        if side == 0:
            return self.get_best_ask(state)
        # buy / bids
        elif side == 1:
            return self.get_best_bid(state)
        else:
            raise ValueError('Side must be 0 or 1')
    
    @jax.jit
    def get_best_bid(
            self: 'OrderBook',
            state: LobState
        ) -> int:
        return job.get_best_bid(state.bids)

    @jax.jit
    def get_best_ask(
            self: 'OrderBook',
            state: LobState
        ) -> int:
        return job.get_best_ask(state.asks)

    @jax.jit
    def get_best_bid_and_ask_inclQuants(
            self: 'OrderBook',
            state: LobState
        ) -> Tuple[jax.Array, jax.Array]:
        return job.get_best_bid_and_ask_inclQuants(state.asks, state.bids)

    @partial(jax.jit, static_argnums=(2,))
    def get_L2_state(
            self: 'OrderBook',
            state: LobState,
            n_levels: int
        ) -> jax.Array:
        return job.get_L2_state(state.asks, state.bids, n_levels)
    
    #Flatten and Unflatten functions so that methods can be appropriately jitted. 
    def _tree_flatten(self: 'OrderBook'):
        children = ()  # arrays / dynamic values
        aux_data = {'nOrders': self.nOrders, 'nTrades':self.nTrades}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(
    OrderBook,
    OrderBook._tree_flatten,
    OrderBook._tree_unflatten
)
