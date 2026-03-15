"""CUDA Matching Engine — JAX FFI Integration.

Two levels:
  V1: Per-match kernel (replaces _match_against_ask/bid_orders)
  V2: Batched scan kernel (replaces scan_through_entire_array)

Usage:
    export JAXOB_USE_CUDA_MATCHING=1    # V1 per-match
    export JAXOB_USE_CUDA_BATCH=1       # V2 batched scan (recommended)

Build first:
    cd gymnax_exchange/jaxob/cuda_matching && bash build.sh
"""

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_batching

_DIR = os.path.dirname(__file__)

# ─── Load V1: per-match kernel ────────────────────────────────

_LIB_PATH = os.path.join(_DIR, "libcuda_matching.so")
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
    warnings.warn(f"CUDA V1 matching kernel load failed: {e}", stacklevel=2)

# ─── Load V2: batched scan kernel ─────────────────────────────

_BATCH_LIB_PATH = os.path.join(_DIR, "libcuda_batch.so")
_CUDA_BATCH_AVAILABLE = False

try:
    if os.path.exists(_BATCH_LIB_PATH):
        _batch_lib = ctypes.cdll.LoadLibrary(_BATCH_LIB_PATH)
        jax.ffi.register_ffi_target(
            "cuda_batch_process_messages",
            jax.ffi.pycapsule(_batch_lib.CudaBatchProcessMessages),
            platform="CUDA",
        )
        _CUDA_BATCH_AVAILABLE = True
except Exception as e:
    import warnings
    warnings.warn(f"CUDA V2 batch kernel load failed: {e}", stacklevel=2)


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


# ═════════════════════════════════════════════════════════════
# V2: Batched scan — replaces scan_through_entire_array
# ═════════════════════════════════════════════════════════════

def scan_through_entire_array_cuda(cfg, key, msg_array, book_state):
    """CUDA V2: Process all messages in one kernel launch.

    Drop-in replacement for scan_through_entire_array.
    Handles: limit, cancel, delete, match (type 1-4), noop (type 0).
    Supports: GENERAL_EXCHANGE mode, IOC type4, INCLUDE_INITS cancel.

    Args:
        cfg: JAXLOB_Configuration (used for init_id)
        key: PRNGKey (unused — CUDA cancel doesn't use randomness)
        msg_array: (n_msgs, 8) int32 messages
        book_state: tuple (asks, bids, trades)

    Returns:
        (asks, bids, trades) after processing all messages
    """
    asks, bids, trades = book_state
    n_orders = asks.shape[0]
    n_trades = trades.shape[0]
    n_msgs = msg_array.shape[0]

    # Single env → n_envs=1, add batch dim, call batched kernel, squeeze
    asks_b = asks[None, ...]      # (1, n_orders, 6)
    bids_b = bids[None, ...]
    trades_b = trades[None, ...]
    msgs_b = msg_array[None, ...]  # (1, n_msgs, 8)

    _fn = jax.ffi.ffi_call(
        "cuda_batch_process_messages",
        (
            jax.ShapeDtypeStruct(asks_b.shape, jnp.int32),
            jax.ShapeDtypeStruct(bids_b.shape, jnp.int32),
            jax.ShapeDtypeStruct(trades_b.shape, jnp.int32),
        ),
        vmap_method="sequential",
    )
    asks_out, bids_out, trades_out = _fn(
        asks_b, bids_b, trades_b, msgs_b,
        n_envs=np.int32(1),
        n_orders=np.int32(n_orders),
        n_trades=np.int32(n_trades),
        n_msgs=np.int32(n_msgs),
        init_id=np.int32(cfg.init_id),
    )
    return (asks_out[0], bids_out[0], trades_out[0])


def vscan_through_entire_array_cuda(cfg, keys, msg_arrays, book_states):
    """CUDA V2 batched: Process multiple envs in one kernel launch.

    Drop-in replacement for vmap(scan_through_entire_array).
    grid=(n_envs,) — each CUDA block handles one independent book.
    NO vmap needed — explicit batch dimension.

    Args:
        cfg: JAXLOB_Configuration
        keys: (n_envs,) PRNGKeys (unused)
        msg_arrays: (n_envs, n_msgs, 8) int32
        book_states: tuple of (n_envs, n_orders, 6), (n_envs, n_orders, 6), (n_envs, n_trades, 8)

    Returns:
        (asks_batch, bids_batch, trades_batch)
    """
    asks_b, bids_b, trades_b = book_states
    n_envs = asks_b.shape[0]
    n_orders = asks_b.shape[1]
    n_trades = trades_b.shape[1]
    n_msgs = msg_arrays.shape[1]

    _fn = jax.ffi.ffi_call(
        "cuda_batch_process_messages",
        (
            jax.ShapeDtypeStruct(asks_b.shape, jnp.int32),
            jax.ShapeDtypeStruct(bids_b.shape, jnp.int32),
            jax.ShapeDtypeStruct(trades_b.shape, jnp.int32),
        ),
        vmap_method="sequential",
    )
    return _fn(
        asks_b, bids_b, trades_b, msg_arrays,
        n_envs=np.int32(n_envs),
        n_orders=np.int32(n_orders),
        n_trades=np.int32(n_trades),
        n_msgs=np.int32(n_msgs),
        init_id=np.int32(cfg.init_id),
    )
