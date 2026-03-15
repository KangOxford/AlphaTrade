"""CUDA Matching Engine — JAX FFI Integration.

Drop-in replacement for _match_against_ask/bid_orders.
Mirrors the Triton kernel interface exactly: same signature, same vmap rules.

Usage:
    export JAXOB_USE_CUDA_MATCHING=1
    python your_script.py

Build first:
    cd gymnax_exchange/jaxob/cuda_matching && bash build.sh
"""

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_batching

# ─── Load shared library and register FFI target ────────────

_LIB_PATH = os.path.join(os.path.dirname(__file__), "libcuda_matching.so")
_CUDA_MATCHING_AVAILABLE = False

try:
    if os.path.exists(_LIB_PATH):
        _lib = ctypes.cdll.LoadLibrary(_LIB_PATH)
        jax.ffi.register_ffi_target(
            "cuda_match_orders",
            jax.ffi.pycapsule(_lib.CudaMatchOrders),
            platform="CUDA",
        )
        _CUDA_MATCHING_AVAILABLE = True
except Exception as e:
    import warnings
    warnings.warn(f"CUDA matching kernel load failed: {e}", stacklevel=2)


# ─── Core implementation ─────────────────────────────────────

def _match_against_orders_cuda_impl(
    orderside, qtm, price, trade,
    agrOID, time, time_ns, agrTID, side,
    is_bid: bool,
):
    """Call the CUDA matching kernel via JAX FFI.

    Packs 7 scalars into a buffer (same as Triton approach) since
    traced JAX values cannot be FFI attrs.
    """
    n_orders = orderside.shape[0]
    n_trades = trade.shape[0]

    # Pack scalars into buffer (matches Triton's `incoming` array)
    incoming = jnp.array(
        [qtm, price, agrOID, time, time_ns, agrTID, side],
        dtype=jnp.int32,
    )

    # ffi_call returns a callable; attrs go to the returned function, not ffi_call
    _cuda_fn = jax.ffi.ffi_call(
        "cuda_match_orders",
        (
            jax.ShapeDtypeStruct(orderside.shape, jnp.int32),
            jax.ShapeDtypeStruct(trade.shape, jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        vmap_method="sequential",
    )
    orderside_out, trade_out, qtm_arr = _cuda_fn(
        orderside,
        trade,
        incoming,
        n_orders=np.int32(n_orders),
        n_trades=np.int32(n_trades),
        is_bid=np.int32(1 if is_bid else 0),
    )

    return (orderside_out, jnp.squeeze(qtm_arr, axis=0), price, trade_out)


# ─── vmap rule (mirrors Triton exactly) ──────────────────────

def _match_against_orders_cuda_vmap_rule(
    axis_size, in_batched,
    orderside, qtm, price, trade,
    agrOID, time, time_ns, agrTID, side,
    is_bid: bool,
):
    def _broadcast_if_needed(x, batched):
        if batched:
            return x
        return jnp.broadcast_to(x, (axis_size,) + jnp.shape(x))

    orderside_b = _broadcast_if_needed(orderside, in_batched[0])
    qtm_b       = _broadcast_if_needed(qtm, in_batched[1])
    price_b     = _broadcast_if_needed(price, in_batched[2])
    trade_b     = _broadcast_if_needed(trade, in_batched[3])
    agrOID_b    = _broadcast_if_needed(agrOID, in_batched[4])
    time_b      = _broadcast_if_needed(time, in_batched[5])
    time_ns_b   = _broadcast_if_needed(time_ns, in_batched[6])
    agrTID_b    = _broadcast_if_needed(agrTID, in_batched[7])
    side_b      = _broadcast_if_needed(side, in_batched[8])

    out = jax.lax.map(
        lambda xs: _match_against_orders_cuda_impl(
            xs[0], xs[1], xs[2], xs[3],
            xs[4], xs[5], xs[6], xs[7], xs[8],
            is_bid,
        ),
        (orderside_b, qtm_b, price_b, trade_b,
         agrOID_b, time_b, time_ns_b, agrTID_b, side_b),
    )
    return out, (True, True, True, True)


# ─── Public API ───────────────────────────────────────────────

@custom_batching.custom_vmap
def match_against_ask_orders_cuda(
    orderside, qtm, price, trade,
    agrOID, time, time_ns, agrTID, side,
):
    """CUDA kernel: match incoming bid against standing asks."""
    return _match_against_orders_cuda_impl(
        orderside, qtm, price, trade,
        agrOID, time, time_ns, agrTID, side,
        True,
    )


@match_against_ask_orders_cuda.def_vmap
def _ask_vmap(axis_size, in_batched,
              orderside, qtm, price, trade,
              agrOID, time, time_ns, agrTID, side):
    return _match_against_orders_cuda_vmap_rule(
        axis_size, in_batched,
        orderside, qtm, price, trade,
        agrOID, time, time_ns, agrTID, side,
        True,
    )


@custom_batching.custom_vmap
def match_against_bid_orders_cuda(
    orderside, qtm, price, trade,
    agrOID, time, time_ns, agrTID, side,
):
    """CUDA kernel: match incoming ask against standing bids."""
    return _match_against_orders_cuda_impl(
        orderside, qtm, price, trade,
        agrOID, time, time_ns, agrTID, side,
        False,
    )


@match_against_bid_orders_cuda.def_vmap
def _bid_vmap(axis_size, in_batched,
              orderside, qtm, price, trade,
              agrOID, time, time_ns, agrTID, side):
    return _match_against_orders_cuda_vmap_rule(
        axis_size, in_batched,
        orderside, qtm, price, trade,
        agrOID, time, time_ns, agrTID, side,
        False,
    )
