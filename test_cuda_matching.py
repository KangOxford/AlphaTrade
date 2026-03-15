"""Correctness test + benchmark: CUDA matching kernel vs JAX reference.

Run on compute node:
  srun --nodes=1 --gpus-per-node=1 --time=00:15:00 --account=brics.s5e \
       python test_cuda_matching.py
"""

import os
import sys
import time

# Force JAX as baseline (not CUDA) for reference comparisons
os.environ["JAXOB_USE_CUDA_MATCHING"] = "0"
os.environ["JAXOB_USE_TRITON_MATCHING"] = "0"

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gymnax_exchange.jaxob.JaxOrderBookArrays import (
    _match_against_ask_orders_jax,
    _match_against_bid_orders_jax,
)
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration

# Import CUDA directly (bypassing env var dispatch)
from gymnax_exchange.jaxob.cuda_matching import (
    match_against_ask_orders_cuda,
    match_against_bid_orders_cuda,
    _CUDA_MATCHING_AVAILABLE,
)

CFG = JAXLOB_Configuration(nOrders=100, nTrades=100)

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"CUDA matching available: {_CUDA_MATCHING_AVAILABLE}")
print()


def make_ask_book(n_filled=10, base_price=10000, tick=100):
    """Asks at base+tick, base+2*tick, ..."""
    asks = jnp.full((100, 6), -1, dtype=jnp.int32)
    for i in range(n_filled):
        p = base_price + tick * (i + 1)
        asks = asks.at[i].set(jnp.array(
            [p, 50, -(i+1), -2, 34200, i * 1000], dtype=jnp.int32
        ))
    return asks


def make_bid_book(n_filled=10, base_price=10000, tick=100):
    """Bids at base-tick, base-2*tick, ..."""
    bids = jnp.full((100, 6), -1, dtype=jnp.int32)
    for i in range(n_filled):
        p = base_price - tick * (i + 1)
        bids = bids.at[i].set(jnp.array(
            [p, 50, -(100+i+1), -2, 34200, i * 1000], dtype=jnp.int32
        ))
    return bids


def empty_trades():
    return jnp.full((100, 8), -1, dtype=jnp.int32)


def compare_results(label, jax_r, cuda_r):
    """Compare JAX and CUDA results, return True if matching."""
    os_match = np.array_equal(np.array(jax_r[0]), np.array(cuda_r[0]))
    qtm_match = int(jax_r[1]) == int(cuda_r[1])
    tr_match = np.array_equal(np.array(jax_r[3]), np.array(cuda_r[3]))

    ok = os_match and qtm_match and tr_match
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        if not os_match:
            print(f"         orderside mismatch")
        if not qtm_match:
            print(f"         qtm: JAX={int(jax_r[1])}, CUDA={int(cuda_r[1])}")
        if not tr_match:
            # Find first differing trade
            for t in range(100):
                if not np.array_equal(np.array(jax_r[3][t]), np.array(cuda_r[3][t])):
                    print(f"         trade[{t}]: JAX={jax_r[3][t].tolist()}, CUDA={cuda_r[3][t].tolist()}")
                    break
    return ok


# ─── Correctness Tests ───────────────────────────────────────

def run_correctness_tests():
    print("=" * 60)
    print("  CORRECTNESS TESTS: CUDA vs JAX")
    print("=" * 60)

    if not _CUDA_MATCHING_AVAILABLE:
        print("  SKIP: CUDA matching not available (libcuda_matching.so not built)")
        return False

    all_pass = True

    # Test 1: No match (bid below all asks)
    asks = make_ask_book(n_filled=5)
    trades = empty_trades()
    args = (asks, jnp.int32(50), jnp.int32(9000), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))
    jax_r = _match_against_ask_orders_jax(CFG, *args)
    cuda_r = match_against_ask_orders_cuda(*args)
    all_pass &= compare_results("No match (bid below asks)", jax_r, cuda_r)

    # Test 2: Single level match
    args = (asks, jnp.int32(30), jnp.int32(10100), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))
    jax_r = _match_against_ask_orders_jax(CFG, *args)
    cuda_r = match_against_ask_orders_cuda(*args)
    all_pass &= compare_results("Single level match (qty=30, price=10100)", jax_r, cuda_r)

    # Test 3: Multi-level sweep
    args = (asks, jnp.int32(200), jnp.int32(10500), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))
    jax_r = _match_against_ask_orders_jax(CFG, *args)
    cuda_r = match_against_ask_orders_cuda(*args)
    all_pass &= compare_results("Multi-level sweep (qty=200, 5 levels)", jax_r, cuda_r)

    # Test 4: Exact fill
    asks1 = make_ask_book(n_filled=1)
    args = (asks1, jnp.int32(50), jnp.int32(10100), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))
    jax_r = _match_against_ask_orders_jax(CFG, *args)
    cuda_r = match_against_ask_orders_cuda(*args)
    all_pass &= compare_results("Exact fill (qty=50, standing=50)", jax_r, cuda_r)

    # Test 5: Empty book
    empty_asks = jnp.full((100, 6), -1, dtype=jnp.int32)
    args = (empty_asks, jnp.int32(50), jnp.int32(10500), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))
    jax_r = _match_against_ask_orders_jax(CFG, *args)
    cuda_r = match_against_ask_orders_cuda(*args)
    all_pass &= compare_results("Empty book (no orders)", jax_r, cuda_r)

    # Test 6: Bid side matching (incoming ask vs standing bids)
    bids = make_bid_book(n_filled=3)
    args = (bids, jnp.int32(80), jnp.int32(9800), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(-1))
    jax_r = _match_against_bid_orders_jax(CFG, *args)
    cuda_r = match_against_bid_orders_cuda(*args)
    all_pass &= compare_results("Bid side: ask@9800 vs bids@9900,9800,9700", jax_r, cuda_r)

    # Test 7: Partial fill
    asks2 = make_ask_book(n_filled=2)
    args = (asks2, jnp.int32(120), jnp.int32(10200), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))
    jax_r = _match_against_ask_orders_jax(CFG, *args)
    cuda_r = match_against_ask_orders_cuda(*args)
    all_pass &= compare_results("Partial fill (qty=120, 2 levels x 50)", jax_r, cuda_r)

    # Test 8: vmap over 10 books
    print()
    print("  vmap test (10 books):")
    n_books = 10
    asks_batch = jnp.stack([make_ask_book(n_filled=5) for _ in range(n_books)])
    trades_batch = jnp.stack([empty_trades() for _ in range(n_books)])
    qtm_batch = jnp.full(n_books, 30, dtype=jnp.int32)
    price_batch = jnp.full(n_books, 10100, dtype=jnp.int32)
    oid_batch = jnp.arange(n_books, dtype=jnp.int32) + 1000
    time_batch = jnp.full(n_books, 34300, dtype=jnp.int32)
    tns_batch = jnp.zeros(n_books, dtype=jnp.int32)
    tid_batch = jnp.full(n_books, 100, dtype=jnp.int32)
    side_batch = jnp.ones(n_books, dtype=jnp.int32)

    try:
        vmap_result = jax.vmap(match_against_ask_orders_cuda)(
            asks_batch, qtm_batch, price_batch, trades_batch,
            oid_batch, time_batch, tns_batch, tid_batch, side_batch
        )
        # Check all books give same result
        consistent = all(
            np.array_equal(np.array(vmap_result[0][0]), np.array(vmap_result[0][i]))
            for i in range(1, n_books)
        )
        status = "PASS" if consistent else "FAIL"
        print(f"  [{status}] vmap 10 books: all results consistent")
        all_pass &= consistent
    except Exception as e:
        print(f"  [FAIL] vmap 10 books: {e}")
        all_pass = False

    print()
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# ─── Benchmark ────────────────────────────────────────────────

def run_benchmark():
    print()
    print("=" * 60)
    print("  BENCHMARK: CUDA vs JAX matching latency")
    print("=" * 60)

    if not _CUDA_MATCHING_AVAILABLE:
        print("  SKIP: CUDA matching not available")
        return

    asks = make_ask_book(n_filled=60)
    trades = empty_trades()
    args = (asks, jnp.int32(100), jnp.int32(10500), trades,
            jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

    n_warmup = 10
    n_trials = 50

    # Warmup both
    for _ in range(n_warmup):
        r = match_against_ask_orders_cuda(*args)
        jax.block_until_ready(r)
    for _ in range(n_warmup):
        r = _match_against_ask_orders_jax(CFG, *args)
        jax.block_until_ready(r)

    # CUDA timing
    times_cuda = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        r = match_against_ask_orders_cuda(*args)
        jax.block_until_ready(r)
        times_cuda.append(time.perf_counter() - t0)

    # JAX timing
    times_jax = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        r = _match_against_ask_orders_jax(CFG, *args)
        jax.block_until_ready(r)
        times_jax.append(time.perf_counter() - t0)

    cuda_us = np.median(times_cuda) * 1e6
    jax_us = np.median(times_jax) * 1e6
    speedup = jax_us / cuda_us if cuda_us > 0 else float('inf')

    print(f"  CUDA:    {cuda_us:8.1f} us (median, {n_trials} trials)")
    print(f"  JAX:     {jax_us:8.1f} us (median, {n_trials} trials)")
    print(f"  Speedup: {speedup:8.1f}x")
    print()

    # Also bench single-call with varying quantities
    print("  Per-quantity breakdown (asks filled=60, price=10500):")
    for qty_label, qty_val in [("qty=1", 1), ("qty=10", 10), ("qty=100", 100), ("qty=500", 500)]:
        args_q = (asks, jnp.int32(qty_val), jnp.int32(10500), trades,
                  jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        for _ in range(5):
            r = match_against_ask_orders_cuda(*args_q)
            jax.block_until_ready(r)
        times_c = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = match_against_ask_orders_cuda(*args_q)
            jax.block_until_ready(r)
            times_c.append(time.perf_counter() - t0)

        for _ in range(5):
            r = _match_against_ask_orders_jax(CFG, *args_q)
            jax.block_until_ready(r)
        times_j = []
        for _ in range(30):
            t0 = time.perf_counter()
            r = _match_against_ask_orders_jax(CFG, *args_q)
            jax.block_until_ready(r)
            times_j.append(time.perf_counter() - t0)

        c_us = np.median(times_c) * 1e6
        j_us = np.median(times_j) * 1e6
        sp = j_us / c_us if c_us > 0 else 0
        matched = qty_val - max(0, int(jax.device_get(r[1])))
        print(f"    {qty_label:8s} (matched {matched:3d}): CUDA={c_us:7.1f}us  JAX={j_us:7.1f}us  {sp:.1f}x")


if __name__ == "__main__":
    passed = run_correctness_tests()
    run_benchmark()
    sys.exit(0 if passed else 1)
