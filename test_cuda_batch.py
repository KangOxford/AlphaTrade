"""Test + Benchmark: CUDA V2 batched scan kernel.

Compares scan_through_entire_array_cuda vs JAX scan_through_entire_array.
Tests correctness for single env and vmap multi-env scaling.

Run:
  srun --nodes=1 --gpus-per-node=1 --time=00:15:00 --account=brics.s5e python test_cuda_batch.py
"""

import os
import sys
import time

os.environ["JAXOB_USE_CUDA_MATCHING"] = "0"
os.environ["JAXOB_USE_TRITON_MATCHING"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gymnax_exchange.jaxob.JaxOrderBookArrays import scan_through_entire_array
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
from gymnax_exchange.jaxob.cuda_matching import (
    scan_through_entire_array_cuda,
    vscan_through_entire_array_cuda,
    _CUDA_BATCH_AVAILABLE,
)

CFG = JAXLOB_Configuration(nOrders=100, nTrades=100)

print(f"JAX: {jax.__version__} | Devices: {jax.devices()}")
print(f"CUDA batch available: {_CUDA_BATCH_AVAILABLE}")
print()


def make_book(key, n_orders=100, n_filled=10, base_price=10000, tick=100):
    asks = jnp.full((n_orders, 6), -1, dtype=jnp.int32)
    bids = jnp.full((n_orders, 6), -1, dtype=jnp.int32)
    trades = jnp.full((100, 8), -1, dtype=jnp.int32)
    for i in range(n_filled):
        asks = asks.at[i].set(jnp.array(
            [base_price + tick*(i+1), 50, -(i+1), -2, 34200, i*1000], dtype=jnp.int32))
        bids = bids.at[i].set(jnp.array(
            [base_price - tick*(i+1), 50, -(100+i+1), -2, 34200, i*1000], dtype=jnp.int32))
    return asks, bids, trades


def make_msgs(key, n_msgs=50):
    """Realistic message mix: limits, crossing limits, cancels."""
    keys = jax.random.split(key, 5)
    rand = jax.random.uniform(keys[0], (n_msgs,))
    types = jnp.where(rand < 0.8, 1, 2).astype(jnp.int32)
    sides = jnp.where(jax.random.uniform(keys[1], (n_msgs,)) < 0.5, 1, -1).astype(jnp.int32)

    # Mix of crossing and non-crossing
    prices = jnp.where(
        sides == 1,
        jax.random.randint(keys[2], (n_msgs,), 9500, 10300),
        jax.random.randint(keys[2], (n_msgs,), 9700, 10500),
    )
    qtys = jax.random.randint(keys[3], (n_msgs,), 1, 50).astype(jnp.int32)
    oids = jnp.arange(1000, 1000 + n_msgs, dtype=jnp.int32)
    # For cancels, use OIDs that exist in the book
    cancel_oids = jax.random.randint(keys[4], (n_msgs,), 1, 10).astype(jnp.int32) * -1  # init order IDs
    final_oids = jnp.where(types == 2, cancel_oids, oids)

    tids = jax.random.randint(keys[4], (n_msgs,), 100, 200).astype(jnp.int32)
    time_s = jnp.full((n_msgs,), 34300, dtype=jnp.int32)
    time_ns = jnp.arange(n_msgs, dtype=jnp.int32) * 10000

    return jnp.stack([types, sides, qtys, prices, final_oids, tids, time_s, time_ns], axis=1)


# ─── Correctness ─────────────────────────────────────────────

def run_correctness():
    print("=" * 60)
    print("  CORRECTNESS: CUDA batch scan vs JAX scan")
    print("=" * 60)

    if not _CUDA_BATCH_AVAILABLE:
        print("  SKIP: CUDA batch kernel not built")
        return False

    all_pass = True
    key = jax.random.PRNGKey(42)

    for label, n_msgs in [("10 msgs", 10), ("50 msgs", 50), ("200 msgs", 200), ("500 msgs", 500)]:
        asks, bids, trades = make_book(key, n_filled=10)
        msgs = make_msgs(key, n_msgs)
        book_state = (asks, bids, trades)

        # JAX reference
        jax_result = scan_through_entire_array(CFG, key, msgs, book_state)

        # CUDA batch
        cuda_result = scan_through_entire_array_cuda(CFG, key, msgs, book_state)

        asks_ok = np.array_equal(np.array(jax_result[0]), np.array(cuda_result[0]))
        bids_ok = np.array_equal(np.array(jax_result[1]), np.array(cuda_result[1]))
        trades_ok = np.array_equal(np.array(jax_result[2]), np.array(cuda_result[2]))

        ok = asks_ok and bids_ok and trades_ok
        status = "PASS" if ok else "FAIL"
        detail = ""
        if not ok:
            if not asks_ok: detail += " asks-mismatch"
            if not bids_ok: detail += " bids-mismatch"
            if not trades_ok: detail += " trades-mismatch"
        print(f"  [{status}] {label}{detail}")
        all_pass &= ok

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# ─── Benchmark ────────────────────────────────────────────────

def run_benchmark():
    print()
    print("=" * 60)
    print("  BENCHMARK: CUDA batch vs JAX scan")
    print("=" * 60)

    if not _CUDA_BATCH_AVAILABLE:
        print("  SKIP: CUDA batch kernel not built")
        return

    n_orders = 100
    n_msgs = 500
    n_warmup = 5
    n_trials = 20

    key = jax.random.PRNGKey(0)
    asks, bids, trades = make_book(key, n_orders=n_orders, n_filled=60)
    msgs = make_msgs(key, n_msgs)
    book_state = (asks, bids, trades)

    # ── Single env ──
    print(f"\n  Single env ({n_msgs} msgs):")

    # JAX warmup + bench
    jax_fn = jax.jit(partial(scan_through_entire_array, CFG))
    for _ in range(n_warmup):
        r = jax_fn(key, msgs, book_state)
        jax.block_until_ready(r)
    times_jax = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        r = jax_fn(key, msgs, book_state)
        jax.block_until_ready(r)
        times_jax.append(time.perf_counter() - t0)

    # CUDA warmup + bench
    for _ in range(n_warmup):
        r = scan_through_entire_array_cuda(CFG, key, msgs, book_state)
        jax.block_until_ready(r)
    times_cuda = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        r = scan_through_entire_array_cuda(CFG, key, msgs, book_state)
        jax.block_until_ready(r)
        times_cuda.append(time.perf_counter() - t0)

    j_ms = np.median(times_jax) * 1000
    c_ms = np.median(times_cuda) * 1000
    speedup = j_ms / c_ms if c_ms > 0 else 0
    j_mps = n_msgs / np.median(times_jax)
    c_mps = n_msgs / np.median(times_cuda)

    print(f"    JAX:   {j_ms:8.2f} ms  ({j_mps:>10,.0f} msgs/sec)")
    print(f"    CUDA:  {c_ms:8.2f} ms  ({c_mps:>10,.0f} msgs/sec)")
    print(f"    Speedup: {speedup:.1f}x")

    # ── Multi-env vmap scaling ──
    print(f"\n  Multi-env scaling ({n_msgs} msgs each):")
    print(f"  {'n_envs':>8} | {'JAX (ms)':>10} | {'CUDA (ms)':>10} | {'Speedup':>8} | {'CUDA msgs/s':>14}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*14}")

    for n_envs in [1, 10, 100, 1000]:
        keys = jax.random.split(key, n_envs)
        asks_b = jnp.stack([make_book(k, n_orders=n_orders, n_filled=60)[0] for k in keys])
        bids_b = jnp.stack([make_book(k, n_orders=n_orders, n_filled=60)[1] for k in keys])
        trades_b = jnp.full((n_envs, 100, 8), -1, dtype=jnp.int32)
        msg_keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
        msgs_b = jnp.stack([make_msgs(k, n_msgs) for k in msg_keys])
        book_b = (asks_b, bids_b, trades_b)

        # JAX vmap
        jax_vfn = jax.jit(jax.vmap(partial(scan_through_entire_array, CFG)))
        try:
            for _ in range(n_warmup):
                r = jax_vfn(keys, msgs_b, book_b)
                jax.block_until_ready(r)
            tj = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                r = jax_vfn(keys, msgs_b, book_b)
                jax.block_until_ready(r)
                tj.append(time.perf_counter() - t0)
            j_ms = np.median(tj) * 1000
        except Exception as e:
            j_ms = float('nan')

        # CUDA batch (no vmap!)
        try:
            for _ in range(n_warmup):
                r = vscan_through_entire_array_cuda(CFG, keys, msgs_b, book_b)
                jax.block_until_ready(r)
            tc = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                r = vscan_through_entire_array_cuda(CFG, keys, msgs_b, book_b)
                jax.block_until_ready(r)
                tc.append(time.perf_counter() - t0)
            c_ms = np.median(tc) * 1000
        except Exception as e:
            c_ms = float('nan')
            print(f"  CUDA {n_envs} envs error: {e}")

        sp = j_ms / c_ms if c_ms > 0 else 0
        total_msgs = n_envs * n_msgs
        c_mps = total_msgs / (c_ms / 1000) if c_ms > 0 else 0
        print(f"  {n_envs:>8} | {j_ms:>10.2f} | {c_ms:>10.2f} | {sp:>7.1f}x | {c_mps:>14,.0f}")


if __name__ == "__main__":
    passed = run_correctness()
    run_benchmark()
    sys.exit(0 if passed else 1)
