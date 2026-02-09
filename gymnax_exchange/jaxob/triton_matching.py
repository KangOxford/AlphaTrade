"""Triton matching kernels for JAX order book arrays.

This module provides drop-in replacements for:
  - _match_against_ask_orders (incoming bid matches standing asks)
  - _match_against_bid_orders (incoming ask matches standing bids)

The implementation keeps the matching loop fully on GPU.

Adapted for jax-lob-triton branch:
  - 10-parameter signature (adds agrTID, side vs original 7)
  - 8-column trade arrays (adds passTID, agrTID vs original 6)
  - Signed trade quantity: -side * matched_qty
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax_triton as jt
from jax import custom_batching
import triton
import triton.language as tl


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _match_orders_kernel(
    orderside_ptr,
    trade_ptr,
    incoming_ptr,
    orderside_out_ptr,
    trade_out_ptr,
    qtm_out_ptr,
    N_ORDERS: tl.constexpr,
    N_TRADES: tl.constexpr,
    IS_BID: tl.constexpr,
    BLOCK_ORDERS: tl.constexpr,
    BLOCK_TRADES: tl.constexpr,
):
    max_int = 2_147_483_647
    order_cols = 6
    trade_cols = 8  # Modified: 6 → 8 (added passTID, agrTID)

    # Row ids for side arrays and trade arrays.
    row_ids = tl.arange(0, BLOCK_ORDERS)
    trade_ids = tl.arange(0, BLOCK_TRADES)
    row_mask = row_ids < N_ORDERS
    trade_mask = trade_ids < N_TRADES

    # Load incoming order scalars (7 scalars: qtm, price, agrOID, time, time_ns, agrTID, side).
    qtm = tl.load(incoming_ptr + 0)
    incoming_price = tl.load(incoming_ptr + 1)
    agr_oid = tl.load(incoming_ptr + 2)
    time_s = tl.load(incoming_ptr + 3)
    time_ns = tl.load(incoming_ptr + 4)
    agr_tid = tl.load(incoming_ptr + 5)   # New: agrTID
    side = tl.load(incoming_ptr + 6)      # New: side (+1 or -1)

    # Load standing book into registers.
    base = row_ids * order_cols
    prices = tl.load(orderside_ptr + base + 0, mask=row_mask, other=-1)
    qtys = tl.load(orderside_ptr + base + 1, mask=row_mask, other=-1)
    oids = tl.load(orderside_ptr + base + 2, mask=row_mask, other=-1)
    tids = tl.load(orderside_ptr + base + 3, mask=row_mask, other=-1)
    times_s = tl.load(orderside_ptr + base + 4, mask=row_mask, other=-1)
    times_ns = tl.load(orderside_ptr + base + 5, mask=row_mask, other=-1)

    # Load trade buffer into registers (8 columns).
    tbase = trade_ids * trade_cols
    t_price = tl.load(trade_ptr + tbase + 0, mask=trade_mask, other=-1)
    t_qty = tl.load(trade_ptr + tbase + 1, mask=trade_mask, other=-1)
    t_pass_oid = tl.load(trade_ptr + tbase + 2, mask=trade_mask, other=-1)
    t_agr_oid = tl.load(trade_ptr + tbase + 3, mask=trade_mask, other=-1)
    t_time_s = tl.load(trade_ptr + tbase + 4, mask=trade_mask, other=-1)
    t_time_ns = tl.load(trade_ptr + tbase + 5, mask=trade_mask, other=-1)
    t_pass_tid = tl.load(trade_ptr + tbase + 6, mask=trade_mask, other=-1)  # New: passTID
    t_agr_tid_buf = tl.load(trade_ptr + tbase + 7, mask=trade_mask, other=-1)  # New: agrTID

    active = qtm > 0

    # Iterate with fixed bound; updates are gated by `active`.
    for _ in tl.static_range(0, BLOCK_ORDERS):
        if IS_BID:
            # Incoming bid: pick lowest ask, excluding empty rows.
            search_prices = tl.where(prices == -1, max_int, prices)
            best_price = tl.min(search_prices, axis=0)
            price_ok = (best_price <= incoming_price) & (best_price != max_int)
            is_best_price = prices == best_price
        else:
            # Incoming ask: pick highest bid.
            best_price = tl.max(prices, axis=0)
            price_ok = (best_price >= incoming_price) & (best_price != -1)
            is_best_price = prices == best_price

        cand_times = tl.where(is_best_price, times_s, max_int)
        best_time = tl.min(cand_times, axis=0)
        is_best_time = is_best_price & (times_s == best_time)

        cand_ns = tl.where(is_best_time, times_ns, max_int)
        best_ns = tl.min(cand_ns, axis=0)
        is_best = is_best_time & (times_ns == best_ns)

        best_idx = tl.min(tl.where(is_best, row_ids, BLOCK_ORDERS), axis=0)
        valid_idx = best_idx < N_ORDERS
        do_match = active & price_ok & valid_idx

        idx_mask = row_ids == best_idx

        standing_price = tl.sum(tl.where(idx_mask, prices, 0), axis=0)
        standing_qty = tl.sum(tl.where(idx_mask, qtys, 0), axis=0)
        standing_oid = tl.sum(tl.where(idx_mask, oids, 0), axis=0)
        standing_tid = tl.sum(tl.where(idx_mask, tids, 0), axis=0)  # Extract standing trader ID

        new_qty = tl.maximum(0, standing_qty - qtm)
        matched_qty = standing_qty - new_qty

        qtm = qtm - tl.where(do_match, standing_qty, 0)
        new_qty_effective = tl.where(do_match, new_qty, standing_qty)

        set_qty_mask = idx_mask & do_match
        qtys = tl.where(set_qty_mask, new_qty_effective, qtys)

        remove_row = do_match & (new_qty_effective <= 0)
        remove_mask = idx_mask & remove_row
        prices = tl.where(remove_mask, -1, prices)
        qtys = tl.where(remove_mask, -1, qtys)
        oids = tl.where(remove_mask, -1, oids)
        tids = tl.where(remove_mask, -1, tids)
        times_s = tl.where(remove_mask, -1, times_s)
        times_ns = tl.where(remove_mask, -1, times_ns)

        # Append trade row.
        empty_candidate = tl.min(
            tl.where(t_price == -1, trade_ids, BLOCK_TRADES),
            axis=0,
        )
        empty_idx = tl.where(
            empty_candidate < N_TRADES,
            empty_candidate,
            N_TRADES - 1,
        )
        write_trade_mask = (trade_ids == empty_idx) & do_match

        t_price = tl.where(write_trade_mask, standing_price, t_price)
        # Modified: quantity is now signed (-side * matched_qty)
        t_qty = tl.where(write_trade_mask, -side * matched_qty, t_qty)
        t_pass_oid = tl.where(write_trade_mask, standing_oid, t_pass_oid)
        t_agr_oid = tl.where(write_trade_mask, agr_oid, t_agr_oid)
        t_time_s = tl.where(write_trade_mask, time_s, t_time_s)
        t_time_ns = tl.where(write_trade_mask, time_ns, t_time_ns)
        # New: write passTID (standing trader ID) and agrTID (aggressor trader ID)
        t_pass_tid = tl.where(write_trade_mask, standing_tid, t_pass_tid)
        t_agr_tid_buf = tl.where(write_trade_mask, agr_tid, t_agr_tid_buf)

        active = do_match & (qtm > 0)

    # Store side output.
    tl.store(orderside_out_ptr + base + 0, prices, mask=row_mask)
    tl.store(orderside_out_ptr + base + 1, qtys, mask=row_mask)
    tl.store(orderside_out_ptr + base + 2, oids, mask=row_mask)
    tl.store(orderside_out_ptr + base + 3, tids, mask=row_mask)
    tl.store(orderside_out_ptr + base + 4, times_s, mask=row_mask)
    tl.store(orderside_out_ptr + base + 5, times_ns, mask=row_mask)

    # Store trade output (8 columns).
    tl.store(trade_out_ptr + tbase + 0, t_price, mask=trade_mask)
    tl.store(trade_out_ptr + tbase + 1, t_qty, mask=trade_mask)
    tl.store(trade_out_ptr + tbase + 2, t_pass_oid, mask=trade_mask)
    tl.store(trade_out_ptr + tbase + 3, t_agr_oid, mask=trade_mask)
    tl.store(trade_out_ptr + tbase + 4, t_time_s, mask=trade_mask)
    tl.store(trade_out_ptr + tbase + 5, t_time_ns, mask=trade_mask)
    tl.store(trade_out_ptr + tbase + 6, t_pass_tid, mask=trade_mask)  # New column
    tl.store(trade_out_ptr + tbase + 7, t_agr_tid_buf, mask=trade_mask)  # New column

    # Return remaining quantity (can be negative, matching current JAX logic).
    tl.store(qtm_out_ptr + 0, qtm)


@partial(jax.jit, static_argnums=(9,))
def _match_against_orders_triton_impl(
    orderside,
    qtm,
    price,
    trade,
    agrOID,
    time,
    time_ns,
    agrTID,      # New parameter (position 7)
    side,        # New parameter (position 8, jnp.int32 scalar)
    is_bid: bool,  # Static parameter (position 9)
):
    n_orders = orderside.shape[0]
    n_trades = trade.shape[0]
    block_orders = _next_power_of_two(n_orders)
    block_trades = _next_power_of_two(n_trades)

    # Modified: 7 scalars (added agrTID and side)
    incoming = jnp.array([qtm, price, agrOID, time, time_ns, agrTID, side], dtype=jnp.int32)
    qtm_out = jnp.zeros((1,), dtype=jnp.int32)

    orderside_out, trade_out, qtm_arr = jt.triton_call(
        orderside,
        trade,
        incoming,
        kernel=_match_orders_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(orderside.shape, jnp.int32),
            jax.ShapeDtypeStruct(trade.shape, jnp.int32),
            jax.ShapeDtypeStruct(qtm_out.shape, jnp.int32),
        ],
        grid=(1,),
        N_ORDERS=n_orders,
        N_TRADES=n_trades,
        IS_BID=is_bid,
        BLOCK_ORDERS=block_orders,
        BLOCK_TRADES=block_trades,
    )
    return (orderside_out, jnp.squeeze(qtm_arr, axis=0), price, trade_out)


def _match_against_orders_triton_vmap_rule(
    axis_size,
    in_batched,
    orderside,
    qtm,
    price,
    trade,
    agrOID,
    time,
    time_ns,
    agrTID,      # New parameter
    side,        # New parameter
    is_bid: bool,
):
    def _broadcast_if_needed(x, batched):
        if batched:
            return x
        return jnp.broadcast_to(x, (axis_size,) + jnp.shape(x))

    orderside_b = _broadcast_if_needed(orderside, in_batched[0])
    qtm_b = _broadcast_if_needed(qtm, in_batched[1])
    price_b = _broadcast_if_needed(price, in_batched[2])
    trade_b = _broadcast_if_needed(trade, in_batched[3])
    agrOID_b = _broadcast_if_needed(agrOID, in_batched[4])
    time_b = _broadcast_if_needed(time, in_batched[5])
    time_ns_b = _broadcast_if_needed(time_ns, in_batched[6])
    agrTID_b = _broadcast_if_needed(agrTID, in_batched[7])  # New
    side_b = _broadcast_if_needed(side, in_batched[8])      # New

    out = jax.lax.map(
        lambda xs: _match_against_orders_triton_impl(
            xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7], xs[8], is_bid
        ),
        (orderside_b, qtm_b, price_b, trade_b, agrOID_b, time_b, time_ns_b, agrTID_b, side_b),
    )
    out_batched = (True, True, True, True)
    return out, out_batched


@custom_batching.custom_vmap
def match_against_ask_orders_triton(orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side):
    """Triton implementation for matching incoming bid against standing asks.

    Parameters match jax-lob-triton branch convention:
      orderside: (N, 6) int32 array of standing orders
      qtm: int32 scalar, incoming quantity
      price: int32 scalar, incoming price
      trade: (M, 8) int32 array of trade buffer
      agrOID: int32 scalar, aggressor order ID
      time: int32 scalar, time in seconds
      time_ns: int32 scalar, time in nanoseconds
      agrTID: int32 scalar, aggressor trader ID
      side: int32 scalar, trade direction (+1 or -1)
    """
    # Incoming bid matches standing asks.
    return _match_against_orders_triton_impl(
        orderside,
        qtm,
        price,
        trade,
        agrOID,
        time,
        time_ns,
        agrTID,
        side,
        True,
    )


@match_against_ask_orders_triton.def_vmap
def _match_against_ask_orders_triton_vmap_rule(
    axis_size,
    in_batched,
    orderside,
    qtm,
    price,
    trade,
    agrOID,
    time,
    time_ns,
    agrTID,
    side,
):
    return _match_against_orders_triton_vmap_rule(
        axis_size,
        in_batched,
        orderside,
        qtm,
        price,
        trade,
        agrOID,
        time,
        time_ns,
        agrTID,
        side,
        True,
    )


@custom_batching.custom_vmap
def match_against_bid_orders_triton(orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side):
    """Triton implementation for matching incoming ask against standing bids.

    Parameters match jax-lob-triton branch convention:
      orderside: (N, 6) int32 array of standing orders
      qtm: int32 scalar, incoming quantity
      price: int32 scalar, incoming price
      trade: (M, 8) int32 array of trade buffer
      agrOID: int32 scalar, aggressor order ID
      time: int32 scalar, time in seconds
      time_ns: int32 scalar, time in nanoseconds
      agrTID: int32 scalar, aggressor trader ID
      side: int32 scalar, trade direction (+1 or -1)
    """
    # Incoming ask matches standing bids.
    return _match_against_orders_triton_impl(
        orderside,
        qtm,
        price,
        trade,
        agrOID,
        time,
        time_ns,
        agrTID,
        side,
        False,
    )


@match_against_bid_orders_triton.def_vmap
def _match_against_bid_orders_triton_vmap_rule(
    axis_size,
    in_batched,
    orderside,
    qtm,
    price,
    trade,
    agrOID,
    time,
    time_ns,
    agrTID,
    side,
):
    return _match_against_orders_triton_vmap_rule(
        axis_size,
        in_batched,
        orderside,
        qtm,
        price,
        trade,
        agrOID,
        time,
        time_ns,
        agrTID,
        side,
        False,
    )
