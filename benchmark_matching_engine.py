"""
Matching Engine Microbenchmark
==============================
Isolates the JAX order book matching engine and measures:
  1. Time per message (single order processed through the book)
  2. Messages/sec throughput (single book)
  3. Messages/sec throughput (vmap'd across N parallel books)
  4. Breakdown by message type (limit, cancel, matching limit)

Usage:
  python benchmark_matching_engine.py [--n_orders 100] [--n_msgs 500] [--n_books 1,10,100,1000] [--warmup 3] [--trials 10]
"""

import os
import sys
import time
import argparse
from functools import partial

# Ensure we can import the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from jax import vmap

# Print device info early
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState

# ─────────────────────────────────────────────
# Synthetic data generators
# ─────────────────────────────────────────────

def make_empty_book(cfg: JAXLOB_Configuration):
    """Create an empty order book state."""
    asks = (jnp.ones((cfg.nOrders, 6)) * -1).astype(jnp.int32)
    bids = (jnp.ones((cfg.nOrders, 6)) * -1).astype(jnp.int32)
    trades = (jnp.ones((cfg.nTrades, 8)) * -1).astype(jnp.int32)
    return (asks, bids, trades)


def make_populated_book(cfg: JAXLOB_Configuration, n_levels=10, base_price=100000, tick=100):
    """Create a book pre-populated with n_levels on each side."""
    asks, bids, trades = make_empty_book(cfg)
    mid = base_price
    for i in range(n_levels):
        # Ask side: price increasing from mid + tick
        ask_price = mid + tick * (i + 1)
        asks = asks.at[i, :].set(jnp.array([ask_price, 100, -(i+1), -2, 34200, i*1000], dtype=jnp.int32))
        # Bid side: price decreasing from mid - tick
        bid_price = mid - tick * (i + 1)
        bids = bids.at[i, :].set(jnp.array([bid_price, 100, -(n_levels+i+1), -2, 34200, i*1000], dtype=jnp.int32))
    return (asks, bids, trades)


def make_limit_messages(n_msgs, base_price=100000, tick=100, key=None):
    """Generate synthetic limit order messages (no crossing = no matching).
    Mix of bid and ask limit orders at non-crossing prices.
    Message format: [type, side, quantity, price, orderid, traderid, time_s, time_ns]
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    sides = jax.random.choice(keys[0], jnp.array([-1, 1]), shape=(n_msgs,))
    # Asks above mid, bids below mid (no crossing)
    offsets = jax.random.randint(keys[1], (n_msgs,), 1, 20)
    prices = base_price + sides * offsets * tick  # asks go up, bids go down...
    # Actually: side=-1 is ask, side=1 is bid
    # For no crossing: asks should be ABOVE mid, bids BELOW mid
    # side=-1 (ask): price = base + offset*tick
    # side=1 (bid): price = base - offset*tick
    prices = base_price - sides * offsets * tick

    quantities = jax.random.randint(keys[2], (n_msgs,), 10, 200)
    oids = jnp.arange(1000, 1000 + n_msgs)
    tids = jnp.arange(5000, 5000 + n_msgs)
    times_s = jnp.full(n_msgs, 34200) + jnp.arange(n_msgs)
    times_ns = jax.random.randint(keys[3], (n_msgs,), 0, 999999999)

    types = jnp.ones(n_msgs, dtype=jnp.int32)  # type=1 = limit

    msgs = jnp.stack([types, sides, quantities, prices, oids, tids, times_s, times_ns], axis=-1).astype(jnp.int32)
    return msgs


def make_crossing_messages(n_msgs, base_price=100000, tick=100, key=None):
    """Generate limit orders that WILL cross the spread and trigger matching.
    Alternates: place ask at mid+tick, then aggressive bid at mid+2*tick (crosses).
    """
    if key is None:
        key = jax.random.PRNGKey(99)
    keys = jax.random.split(key, 2)

    msgs = []
    for i in range(n_msgs):
        if i % 2 == 0:
            # Place a passive ask at mid + small offset
            msg = jnp.array([1, -1, 50, base_price + tick, 2000+i, 6000+i, 34200+i, i*1000], dtype=jnp.int32)
        else:
            # Place aggressive bid that crosses (price >= best ask)
            msg = jnp.array([1, 1, 30, base_price + tick * 5, 2000+i, 6000+i, 34200+i, i*1000], dtype=jnp.int32)
        msgs.append(msg)
    return jnp.stack(msgs)


def make_cancel_messages(n_msgs, book_state, cfg):
    """Generate cancel messages for orders that exist in the book."""
    asks, bids, _ = book_state
    msgs = []
    for i in range(min(n_msgs, cfg.nOrders)):
        side_arr = asks if i % 2 == 0 else bids
        side_val = -1 if i % 2 == 0 else 1
        idx = i // 2
        if idx < cfg.nOrders and side_arr[idx, 0] != -1:
            msg = jnp.array([2, side_val, 10, side_arr[idx, 0].item(),
                           side_arr[idx, 2].item(), side_arr[idx, 3].item(),
                           34300 + i, i * 1000], dtype=jnp.int32)
            msgs.append(msg)
    if len(msgs) == 0:
        # Fallback: generate dummy cancels
        return make_limit_messages(n_msgs)
    while len(msgs) < n_msgs:
        msgs.append(msgs[len(msgs) % len(msgs[:max(1, len(msgs))])])
    return jnp.stack(msgs[:n_msgs])


def make_mixed_messages(n_msgs, base_price=100000, tick=100, key=None):
    """Realistic mix: 40% passive limits, 30% crossing limits, 30% cancels-as-limits.
    (Cancels need existing OIDs so we approximate with passive limits for simplicity)
    """
    if key is None:
        key = jax.random.PRNGKey(77)
    k1, k2, k3 = jax.random.split(key, 3)

    n_passive = n_msgs * 4 // 10
    n_crossing = n_msgs * 3 // 10
    n_cancel_like = n_msgs - n_passive - n_crossing

    passive = make_limit_messages(n_passive, base_price, tick, k1)
    crossing = make_crossing_messages(n_crossing, base_price, tick, k2)
    cancel_like = make_limit_messages(n_cancel_like, base_price, tick, k3)

    all_msgs = jnp.concatenate([passive, crossing, cancel_like], axis=0)
    # Shuffle
    perm = jax.random.permutation(k3, n_msgs)
    return all_msgs[perm]


# ─────────────────────────────────────────────
# Benchmark functions
# ─────────────────────────────────────────────

def benchmark_single_book(cfg, msgs, book_state, key, warmup=3, trials=10):
    """Benchmark scan_through_entire_array on a single book."""

    @jax.jit
    def run(book_state, key, msgs):
        return job.scan_through_entire_array(cfg, key, msgs, book_state)

    # Warmup (includes JIT compilation)
    for i in range(warmup):
        k = jax.random.PRNGKey(i)
        result = run(book_state, k, msgs)
        jax.block_until_ready(result)

    # Timed trials
    times = []
    for i in range(trials):
        k = jax.random.PRNGKey(1000 + i)
        t0 = time.perf_counter()
        result = run(book_state, k, msgs)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def benchmark_vmap_books(cfg, msgs, book_state, key, n_books, warmup=3, trials=10):
    """Benchmark vmap'd scan across n_books parallel order books."""

    # Stack n_books copies
    asks, bids, trades = book_state
    asks_batch = jnp.tile(asks[None, :, :], (n_books, 1, 1))
    bids_batch = jnp.tile(bids[None, :, :], (n_books, 1, 1))
    trades_batch = jnp.tile(trades[None, :, :], (n_books, 1, 1))
    book_batch = (asks_batch, bids_batch, trades_batch)

    # Same msgs for all books (broadcast)
    msgs_batch = jnp.tile(msgs[None, :, :], (n_books, 1, 1))
    keys_batch = jax.random.split(key, n_books)

    @jax.jit
    def run(book_batch, keys_batch, msgs_batch):
        return vmap(partial(job.scan_through_entire_array, cfg))(
            keys_batch, msgs_batch, book_batch
        )

    # Warmup
    for i in range(warmup):
        ks = jax.random.split(jax.random.PRNGKey(i), n_books)
        result = run(book_batch, ks, msgs_batch)
        jax.block_until_ready(result)

    # Timed trials
    times = []
    for i in range(trials):
        ks = jax.random.split(jax.random.PRNGKey(2000 + i), n_books)
        t0 = time.perf_counter()
        result = run(book_batch, ks, msgs_batch)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def benchmark_single_message(cfg, book_state, key, msg_type="limit", warmup=5, trials=50):
    """Benchmark processing a SINGLE message through cond_type_side."""

    base_price = 100000
    tick = 100

    if msg_type == "limit_passive":
        # Non-crossing limit
        msg = jnp.array([1, 1, 50, base_price - tick * 5, 9999, 9999, 34200, 0], dtype=jnp.int32)
    elif msg_type == "limit_crossing":
        # Crossing limit (will trigger matching)
        msg = jnp.array([1, 1, 50, base_price + tick * 5, 9999, 9999, 34200, 0], dtype=jnp.int32)
    elif msg_type == "cancel":
        msg = jnp.array([2, -1, 10, base_price + tick, -1, -2, 34200, 0], dtype=jnp.int32)
    elif msg_type == "noop":
        msg = jnp.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    else:
        raise ValueError(f"Unknown msg_type: {msg_type}")

    @jax.jit
    def run(book_state, key, msg):
        return job.cond_type_side(cfg, book_state, (key, msg))

    # Warmup
    for i in range(warmup):
        k = jax.random.PRNGKey(i)
        result = run(book_state, k, msg)
        jax.block_until_ready(result)

    # Timed trials
    times = []
    for i in range(trials):
        k = jax.random.PRNGKey(3000 + i)
        t0 = time.perf_counter()
        result = run(book_state, k, msg)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 1e-6:
        return f"{seconds*1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.1f} us"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def print_stats(label, times, n_msgs=1):
    import numpy as np
    arr = np.array(times)
    median = np.median(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    p5 = np.percentile(arr, 5)
    p95 = np.percentile(arr, 95)
    per_msg = median / n_msgs
    msgs_per_sec = n_msgs / median if median > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Messages:      {n_msgs}")
    print(f"  Trials:        {len(times)}")
    print(f"  Median total:  {fmt_time(median)}")
    print(f"  Mean total:    {fmt_time(mean)} +/- {fmt_time(std)}")
    print(f"  P5-P95:        {fmt_time(p5)} - {fmt_time(p95)}")
    print(f"  Per message:   {fmt_time(per_msg)}")
    print(f"  Throughput:    {msgs_per_sec:,.0f} msgs/sec")
    print(f"{'='*60}")

    return {
        "label": label,
        "n_msgs": n_msgs,
        "median_s": median,
        "mean_s": mean,
        "std_s": std,
        "per_msg_s": per_msg,
        "msgs_per_sec": msgs_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Matching Engine Microbenchmark")
    parser.add_argument("--n_orders", type=int, default=100, help="Book depth (nOrders per side)")
    parser.add_argument("--n_trades", type=int, default=100, help="Trade array size")
    parser.add_argument("--n_msgs", type=int, default=500, help="Messages per scan batch")
    parser.add_argument("--n_books", type=str, default="1,10,100,1000", help="Comma-sep list of vmap book counts")
    parser.add_argument("--warmup", type=int, default=3, help="JIT warmup iterations")
    parser.add_argument("--trials", type=int, default=10, help="Timed trial iterations")
    parser.add_argument("--skip_vmap", action="store_true", help="Skip vmap benchmarks")
    args = parser.parse_args()

    n_books_list = [int(x) for x in args.n_books.split(",")]

    cfg = JAXLOB_Configuration(
        nOrders=args.n_orders,
        nTrades=args.n_trades,
    )

    print(f"\n{'#'*60}")
    print(f"  MATCHING ENGINE MICROBENCHMARK")
    print(f"{'#'*60}")
    print(f"  nOrders={cfg.nOrders}, nTrades={cfg.nTrades}")
    print(f"  n_msgs={args.n_msgs}, warmup={args.warmup}, trials={args.trials}")
    print(f"  Triton matching: USE={os.environ.get('JAXOB_USE_TRITON_MATCHING','0')}")
    print(f"  vmap book counts: {n_books_list}")
    print()

    key = jax.random.PRNGKey(0)

    # Create populated book
    book_state = make_populated_book(cfg, n_levels=min(10, cfg.nOrders))

    # ── Section 1: Single message benchmarks ──
    print("\n" + "━"*60)
    print("  SECTION 1: Single Message Latency")
    print("━"*60)

    results = []
    for msg_type in ["noop", "limit_passive", "limit_crossing", "cancel"]:
        times = benchmark_single_message(cfg, book_state, key, msg_type,
                                          warmup=args.warmup, trials=max(args.trials, 30))
        r = print_stats(f"Single msg: {msg_type}", times, n_msgs=1)
        results.append(r)

    # ── Section 2: Batch scan benchmarks ──
    print("\n" + "━"*60)
    print("  SECTION 2: Batch Scan (scan_through_entire_array)")
    print("━"*60)

    for label, gen_fn in [
        ("Passive limits (no matching)", lambda: make_limit_messages(args.n_msgs, key=jax.random.PRNGKey(10))),
        ("Crossing limits (triggers matching)", lambda: make_crossing_messages(args.n_msgs, key=jax.random.PRNGKey(20))),
        ("Mixed (realistic)", lambda: make_mixed_messages(args.n_msgs, key=jax.random.PRNGKey(30))),
    ]:
        msgs = gen_fn()
        times = benchmark_single_book(cfg, msgs, book_state, key,
                                       warmup=args.warmup, trials=args.trials)
        r = print_stats(f"Batch scan: {label}", times, n_msgs=args.n_msgs)
        results.append(r)

    # ── Section 3: vmap benchmarks ──
    if not args.skip_vmap:
        print("\n" + "━"*60)
        print("  SECTION 3: VMAP Parallel Books")
        print("━"*60)

        msgs = make_mixed_messages(args.n_msgs, key=jax.random.PRNGKey(50))

        for n_books in n_books_list:
            try:
                times = benchmark_vmap_books(cfg, msgs, book_state, key, n_books,
                                              warmup=args.warmup, trials=args.trials)
                total_msgs = args.n_msgs * n_books
                r = print_stats(f"VMAP {n_books} books x {args.n_msgs} msgs", times, n_msgs=total_msgs)
                r["n_books"] = n_books
                results.append(r)
            except Exception as e:
                print(f"\n  VMAP {n_books} books: FAILED - {e}")

    # ── Summary table ──
    print("\n\n" + "="*80)
    print("  SUMMARY TABLE")
    print("="*80)
    print(f"  {'Test':<45} {'Per Msg':>12} {'Throughput':>16}")
    print(f"  {'-'*45} {'-'*12} {'-'*16}")
    for r in results:
        print(f"  {r['label']:<45} {fmt_time(r['per_msg_s']):>12} {r['msgs_per_sec']:>13,.0f} /s")
    print("="*80)

    # ── Comparison with C++ ──
    print("\n\n" + "="*80)
    print("  COMPARISON WITH C++ MATCHING ENGINES")
    print("="*80)

    # Use the mixed single-book result as reference
    mixed_result = [r for r in results if "Mixed" in r.get("label", "")]
    if mixed_result:
        jax_throughput = mixed_result[0]["msgs_per_sec"]
        cpp_throughput = 132_000_000  # PIYUSH C++20 engine (Binance replay)
        rust_throughput = 11_300_000   # matching-engine-rs

        print(f"  {'Engine':<30} {'Throughput':>16} {'vs JAX':>10}")
        print(f"  {'-'*30} {'-'*16} {'-'*10}")
        print(f"  {'JAX (this benchmark)':<30} {jax_throughput:>13,.0f} /s {'1x':>10}")
        print(f"  {'C++20 (PIYUSH, Binance)':<30} {cpp_throughput:>13,.0f} /s {cpp_throughput/max(1,jax_throughput):>9,.0f}x")
        print(f"  {'Rust (matching-engine-rs)':<30} {rust_throughput:>13,.0f} /s {rust_throughput/max(1,jax_throughput):>9,.0f}x")
    print("="*80)


if __name__ == "__main__":
    main()
