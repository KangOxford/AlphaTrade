"""
Benchmark: JAX while_loop vs Triton kernel for order matching.

Measures three levels:
  1. Isolated matching function (_match_against_ask_orders)
  2. Full scan_through_entire_array (complete message processing pipeline)
  3. vmap'd multi-environment scaling (1, 10, 100, 1000 books)

Usage:
  JAXOB_USE_TRITON_MATCHING=0 python benchmark_matching_engine.py  # JAX
  JAXOB_USE_TRITON_MATCHING=1 python benchmark_matching_engine.py  # Triton
"""

import argparse
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Must set env var BEFORE importing jaxob (it reads at import time)
_backend_label = "Triton" if os.environ.get("JAXOB_USE_TRITON_MATCHING", "0") == "1" else "JAX"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gymnax_exchange.jaxob.JaxOrderBookArrays import (
    _match_against_ask_orders,
    _match_against_bid_orders,
    scan_through_entire_array,
    cond_type_side,
    add_order,
)
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst


def make_config(n_orders=100, n_trades=100):
    return JAXLOB_Configuration(nOrders=n_orders, nTrades=n_trades)


def generate_realistic_book(key, n_orders, fill_fraction=0.6):
    """Generate a partially-filled order book with realistic price distribution."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_filled = int(n_orders * fill_fraction)
    empty = jnp.full((n_orders, 6), -1, dtype=jnp.int32)

    # Asks: prices around 10000-10500 (100 = 1 cent, so $100.00-$105.00)
    ask_prices = jax.random.randint(k1, (n_filled,), 10000, 10500)
    ask_qtys = jax.random.randint(k2, (n_filled,), 1, 100)
    ask_oids = jnp.arange(1, n_filled + 1)
    ask_tids = jax.random.randint(k3, (n_filled,), 1, 50)
    ask_times = jnp.full((n_filled,), 34200, dtype=jnp.int32)
    ask_ns = jnp.arange(n_filled, dtype=jnp.int32) * 1000

    asks = empty.at[:n_filled].set(
        jnp.stack([ask_prices, ask_qtys, ask_oids, ask_tids, ask_times, ask_ns], axis=1)
    )

    # Bids: prices around 9500-10000
    bid_prices = jax.random.randint(k1, (n_filled,), 9500, 10000)
    bid_qtys = jax.random.randint(k2, (n_filled,), 1, 100)
    bid_oids = jnp.arange(n_filled + 1, 2 * n_filled + 1)
    bid_tids = jax.random.randint(k4, (n_filled,), 1, 50)
    bid_times = jnp.full((n_filled,), 34200, dtype=jnp.int32)
    bid_ns = jnp.arange(n_filled, dtype=jnp.int32) * 1000

    bids = empty.at[:n_filled].set(
        jnp.stack([bid_prices, bid_qtys, bid_oids, bid_tids, bid_times, bid_ns], axis=1)
    )

    trades = jnp.full((100, 8), -1, dtype=jnp.int32)
    return asks, bids, trades


def generate_message_sequence(key, n_msgs, n_filled_orders):
    """Generate a realistic mix of limit orders and cancels.

    Mix: ~40% bid limit, ~40% ask limit, ~10% bid cancel, ~10% ask cancel
    Some limit orders will cross the spread and trigger matching.
    """
    keys = jax.random.split(key, 6)

    rand = jax.random.uniform(keys[0], (n_msgs,))
    types = jnp.where(rand < 0.8, 1, 2).astype(jnp.int32)
    sides = jnp.where(jax.random.uniform(keys[1], (n_msgs,)) < 0.5, 1, -1).astype(jnp.int32)

    base_prices = jnp.where(
        sides == 1,
        jax.random.randint(keys[2], (n_msgs,), 9500, 10000),
        jax.random.randint(keys[2], (n_msgs,), 10000, 10500),
    )
    cross_mask = jax.random.uniform(keys[3], (n_msgs,)) < 0.2
    crossing_prices = jnp.where(
        sides == 1,
        jax.random.randint(keys[2], (n_msgs,), 10000, 10300),
        jax.random.randint(keys[2], (n_msgs,), 9700, 10000),
    )
    prices = jnp.where(cross_mask & (types == 1), crossing_prices, base_prices)

    qtys = jax.random.randint(keys[4], (n_msgs,), 1, 50).astype(jnp.int32)
    oids = jnp.arange(10000, 10000 + n_msgs, dtype=jnp.int32)
    tids = jax.random.randint(keys[5], (n_msgs,), 100, 200).astype(jnp.int32)
    time_s = jnp.full((n_msgs,), 34300, dtype=jnp.int32)
    time_ns = jnp.arange(n_msgs, dtype=jnp.int32) * 10000

    cancel_oids = jax.random.randint(keys[5], (n_msgs,), 1, max(n_filled_orders, 2)).astype(jnp.int32)
    final_oids = jnp.where(types == 2, cancel_oids, oids)

    msgs = jnp.stack([types, sides, qtys, prices, final_oids, tids, time_s, time_ns], axis=1)
    return msgs


def bench_scan_pipeline(cfg, asks, bids, trades, msgs, n_warmup, n_trials):
    """Benchmark full scan_through_entire_array."""
    key = jax.random.PRNGKey(42)
    book_state = (asks, bids, trades)
    fn = jax.jit(partial(scan_through_entire_array, cfg))

    for _ in range(n_warmup):
        result = fn(key, msgs, book_state)
        jax.block_until_ready(result)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = fn(key, msgs, book_state)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    return np.array(times)


def bench_vmap_scaling(cfg, n_books_list, n_msgs, n_orders, n_warmup, n_trials):
    """Benchmark vmap'd scan over multiple environments."""
    key = jax.random.PRNGKey(0)
    n_filled = int(n_orders * 0.6)

    results = {}
    for n_books in n_books_list:
        print(f"  vmap n_books={n_books} ...", end=" ", flush=True)

        keys = jax.random.split(key, n_books)
        asks_batch = jnp.stack([generate_realistic_book(k, n_orders)[0] for k in keys])
        bids_batch = jnp.stack([generate_realistic_book(k, n_orders)[1] for k in keys])
        trades_batch = jnp.full((n_books, 100, 8), -1, dtype=jnp.int32)

        msg_keys = jax.random.split(jax.random.PRNGKey(1), n_books)
        msgs_batch = jnp.stack([generate_message_sequence(k, n_msgs, n_filled) for k in msg_keys])

        book_states = (asks_batch, bids_batch, trades_batch)
        prng_keys = jax.random.split(jax.random.PRNGKey(2), n_books)

        vfn = jax.jit(jax.vmap(partial(scan_through_entire_array, cfg)))

        for _ in range(n_warmup):
            result = vfn(prng_keys, msgs_batch, book_states)
            jax.block_until_ready(result)

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            result = vfn(prng_keys, msgs_batch, book_states)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - t0)

        arr = np.array(times)
        results[n_books] = arr
        print(f"median={np.median(arr)*1000:.2f}ms")

    return results


def format_stats(times_arr):
    ms = times_arr * 1000
    return f"median={np.median(ms):8.3f}ms  p5={np.percentile(ms, 5):8.3f}ms  p95={np.percentile(ms, 95):8.3f}ms  std={np.std(ms):7.3f}ms"


def main():
    parser = argparse.ArgumentParser(description="JAXOB Matching Engine Benchmark")
    parser.add_argument("--n_orders", type=int, default=100)
    parser.add_argument("--n_msgs", type=int, default=500)
    parser.add_argument("--n_books", type=str, default="1,10,100,1000",
                        help="Comma-separated list of book counts for vmap scaling")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    n_books_list = [int(x) for x in args.n_books.split(",")]

    print(f"{'='*60}")
    print(f"  JAXOB Matching Engine Benchmark")
    print(f"  Backend: {_backend_label}")
    print(f"  n_orders={args.n_orders}  n_msgs={args.n_msgs}")
    print(f"  warmup={args.warmup}  trials={args.trials}")
    print(f"  n_books={args.n_books}")
    print(f"{'='*60}")
    print()

    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX version: {jax.__version__}")
    print()

    cfg = make_config(n_orders=args.n_orders)
    key = jax.random.PRNGKey(42)
    n_filled = int(args.n_orders * 0.6)

    asks, bids, trades = generate_realistic_book(key, args.n_orders)
    msgs = generate_message_sequence(jax.random.PRNGKey(1), args.n_msgs, n_filled)

    # ─── Level 1: Isolated Matching ───────────────────────────────
    print("=" * 60)
    print("LEVEL 1: Isolated matching (_match_against_ask_orders)")
    print("  Incoming bid vs standing asks at various quantities")
    print("=" * 60)

    for qty_label, qty_val in [("small (qty=1)", 1), ("medium (qty=10)", 10), ("large (qty=500)", 500)]:
        qtm = jnp.int32(qty_val)
        price = jnp.int32(10500)
        agrOID = jnp.int32(99999)
        time_val = jnp.int32(34300)
        time_ns = jnp.int32(0)
        agrTID = jnp.int32(100)
        side_val = jnp.int32(1)

        fn = jax.jit(partial(_match_against_ask_orders, cfg))

        for _ in range(args.warmup):
            r = fn(asks, qtm, price, trades, agrOID, time_val, time_ns, agrTID, side_val)
            jax.block_until_ready(r)

        times = []
        for _ in range(args.trials):
            t0 = time.perf_counter()
            r = fn(asks, qtm, price, trades, agrOID, time_val, time_ns, agrTID, side_val)
            jax.block_until_ready(r)
            times.append(time.perf_counter() - t0)

        arr = np.array(times)
        remaining_qty = int(jax.device_get(r[1]))
        matched = qty_val - max(0, remaining_qty)
        print(f"  {qty_label:20s} (matched {matched:3d} orders): {format_stats(arr)}")

    print()

    # ─── Level 2: Full Scan Pipeline ──────────────────────────────
    print("=" * 60)
    print(f"LEVEL 2: scan_through_entire_array ({args.n_msgs} messages)")
    print("  ~80% limit orders, ~20% cancels, ~20% of limits cross spread")
    print("=" * 60)

    times = bench_scan_pipeline(cfg, asks, bids, trades, msgs, args.warmup, args.trials)
    print(f"  {format_stats(times)}")
    msgs_per_sec = args.n_msgs / np.median(times)
    print(f"  Throughput: {msgs_per_sec:,.0f} msgs/sec (single book)")
    print()

    # ─── Level 3: vmap Scaling ────────────────────────────────────
    print("=" * 60)
    print("LEVEL 3: vmap scaling (multi-environment)")
    print(f"  {args.n_msgs} msgs x N books, total wall time")
    print("=" * 60)

    vmap_results = bench_vmap_scaling(
        cfg, n_books_list, args.n_msgs, args.n_orders, args.warmup, args.trials
    )

    print()
    print("  +----------+------------+------------+------------+--------------+")
    print("  | n_books  | median(ms) |   p5(ms)   |  p95(ms)   | msgs/sec/bk  |")
    print("  +----------+------------+------------+------------+--------------+")
    for nb, arr in sorted(vmap_results.items()):
        ms = arr * 1000
        med = np.median(ms)
        p5 = np.percentile(ms, 5)
        p95 = np.percentile(ms, 95)
        total_msgs_per_sec = (args.n_msgs * nb) / np.median(arr)
        per_book = total_msgs_per_sec / nb
        print(f"  | {nb:>8d} | {med:>10.2f} | {p5:>10.2f} | {p95:>10.2f} | {per_book:>12,.0f} |")
    print("  +----------+------------+------------+------------+--------------+")

    # Scaling efficiency
    if len(vmap_results) >= 2:
        base_nb = min(vmap_results.keys())
        base_time = np.median(vmap_results[base_nb])
        print()
        print(f"  Scaling efficiency (vs n_books={base_nb}):")
        for nb, arr in sorted(vmap_results.items()):
            if nb == base_nb:
                continue
            ratio = nb / base_nb
            time_ratio = np.median(arr) / base_time
            efficiency = ratio / time_ratio * 100
            print(f"    {base_nb:>4d} -> {nb:>4d} ({ratio:.0f}x books): "
                  f"time {time_ratio:.2f}x -> efficiency {efficiency:.1f}%")

    print()
    print(f"Backend: {_backend_label}")
    print(f"JAXOB_USE_TRITON_MATCHING={os.environ.get('JAXOB_USE_TRITON_MATCHING', '0')}")
    print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
