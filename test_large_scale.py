"""Large-scale CUDA V2 benchmark: 1 → 8000 envs.

Matches the L1-L4 benchmark format for direct comparison.
Tests both CUDA V2 batch kernel and JAX baseline at scale.
"""

import os, sys, time
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

N_ORDERS = 100
N_MSGS = 500
N_WARMUP = 3
N_TRIALS = 10
ENV_COUNTS = [1, 10, 100, 1000, 2000, 4000, 8000]

CFG = JAXLOB_Configuration(nOrders=N_ORDERS, nTrades=100)

print(f"JAX: {jax.__version__} | Devices: {jax.devices()}")
print(f"CUDA batch: {_CUDA_BATCH_AVAILABLE}")
print(f"Config: nOrders={N_ORDERS}, nMsgs={N_MSGS}, warmup={N_WARMUP}, trials={N_TRIALS}")
print(f"Env counts: {ENV_COUNTS}")
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


def make_msgs(key, n_msgs=500):
    keys = jax.random.split(key, 5)
    rand = jax.random.uniform(keys[0], (n_msgs,))
    types = jnp.where(rand < 0.8, 1, 2).astype(jnp.int32)
    sides = jnp.where(jax.random.uniform(keys[1], (n_msgs,)) < 0.5, 1, -1).astype(jnp.int32)
    prices = jnp.where(
        sides == 1,
        jax.random.randint(keys[2], (n_msgs,), 9500, 10300),
        jax.random.randint(keys[2], (n_msgs,), 9700, 10500),
    )
    qtys = jax.random.randint(keys[3], (n_msgs,), 1, 50).astype(jnp.int32)
    oids = jnp.arange(1000, 1000 + n_msgs, dtype=jnp.int32)
    cancel_oids = jax.random.randint(keys[4], (n_msgs,), 1, 10).astype(jnp.int32) * -1
    final_oids = jnp.where(types == 2, cancel_oids, oids)
    tids = jax.random.randint(keys[4], (n_msgs,), 100, 200).astype(jnp.int32)
    time_s = jnp.full((n_msgs,), 34300, dtype=jnp.int32)
    time_ns = jnp.arange(n_msgs, dtype=jnp.int32) * 10000
    return jnp.stack([types, sides, qtys, prices, final_oids, tids, time_s, time_ns], axis=1)


def build_batch(n_envs, n_orders, n_msgs):
    """Pre-build batched data using vectorized JAX ops (no Python loops)."""
    n_filled = 60

    # Vectorized book generation: tile a single template book across n_envs
    template_asks = jnp.full((n_orders, 6), -1, dtype=jnp.int32)
    template_bids = jnp.full((n_orders, 6), -1, dtype=jnp.int32)
    idxs = jnp.arange(n_filled)
    ask_prices = 10000 + 100 * (idxs + 1)
    bid_prices = 10000 - 100 * (idxs + 1)
    qtys = jnp.full(n_filled, 50, dtype=jnp.int32)
    ask_oids = -(idxs + 1)
    bid_oids = -(100 + idxs + 1)
    tids = jnp.full(n_filled, -2, dtype=jnp.int32)
    times = jnp.full(n_filled, 34200, dtype=jnp.int32)
    nss = idxs * 1000

    ask_rows = jnp.stack([ask_prices, qtys, ask_oids, tids, times, nss], axis=1)
    bid_rows = jnp.stack([bid_prices, qtys, bid_oids, tids, times, nss], axis=1)
    template_asks = template_asks.at[:n_filled].set(ask_rows)
    template_bids = template_bids.at[:n_filled].set(bid_rows)

    # Tile across envs
    asks_b = jnp.tile(template_asks[None, :, :], (n_envs, 1, 1))
    bids_b = jnp.tile(template_bids[None, :, :], (n_envs, 1, 1))
    trades_b = jnp.full((n_envs, 100, 8), -1, dtype=jnp.int32)

    # Keys for JAX vmap benchmark
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, n_envs)

    # Vectorized message generation: vmap make_msgs
    msg_keys = jax.random.split(jax.random.PRNGKey(1), n_envs)
    msgs_b = jax.vmap(lambda k: make_msgs(k, n_msgs))(msg_keys)
    return keys, asks_b, bids_b, trades_b, msgs_b


def bench_jax_vmap(n_envs, keys, asks_b, bids_b, trades_b, msgs_b):
    """Benchmark JAX vmap baseline."""
    book_b = (asks_b, bids_b, trades_b)
    vfn = jax.jit(jax.vmap(partial(scan_through_entire_array, CFG)))

    try:
        for _ in range(N_WARMUP):
            r = vfn(keys, msgs_b, book_b)
            jax.block_until_ready(r)
        times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            r = vfn(keys, msgs_b, book_b)
            jax.block_until_ready(r)
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1000
    except Exception as e:
        print(f"    JAX {n_envs} envs ERROR: {e}")
        return float('nan')


def bench_cuda_batch(n_envs, keys, asks_b, bids_b, trades_b, msgs_b):
    """Benchmark CUDA V2 batch kernel."""
    book_b = (asks_b, bids_b, trades_b)

    try:
        for _ in range(N_WARMUP):
            r = vscan_through_entire_array_cuda(CFG, keys, msgs_b, book_b)
            jax.block_until_ready(r)
        times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            r = vscan_through_entire_array_cuda(CFG, keys, msgs_b, book_b)
            jax.block_until_ready(r)
            times.append(time.perf_counter() - t0)
        return np.median(times) * 1000
    except Exception as e:
        print(f"    CUDA {n_envs} envs ERROR: {e}")
        return float('nan')


def main():
    print("=" * 80)
    print("  LARGE-SCALE BENCHMARK: CUDA V2 vs JAX Baseline")
    print("  nOrders=100, nMsgs=500, GH200 GPU")
    print("=" * 80)
    print()

    results = []

    for n_envs in ENV_COUNTS:
        print(f"  n_envs={n_envs:>5d} ...", end=" ", flush=True)

        # Build data
        t_build = time.perf_counter()
        keys, asks_b, bids_b, trades_b, msgs_b = build_batch(n_envs, N_ORDERS, N_MSGS)
        build_ms = (time.perf_counter() - t_build) * 1000
        print(f"data built ({build_ms:.0f}ms) ...", end=" ", flush=True)

        # Memory estimate
        mem_mb = (asks_b.nbytes + bids_b.nbytes + trades_b.nbytes + msgs_b.nbytes) / 1e6
        print(f"mem={mem_mb:.0f}MB ...", end=" ", flush=True)

        # JAX benchmark
        j_ms = bench_jax_vmap(n_envs, keys, asks_b, bids_b, trades_b, msgs_b)

        # CUDA benchmark
        c_ms = bench_cuda_batch(n_envs, keys, asks_b, bids_b, trades_b, msgs_b)

        speedup = j_ms / c_ms if c_ms > 0 and not np.isnan(c_ms) else float('nan')
        total_msgs = n_envs * N_MSGS
        cuda_mps = total_msgs / (c_ms / 1000) if c_ms > 0 else 0
        jax_mps = total_msgs / (j_ms / 1000) if j_ms > 0 else 0

        results.append({
            'n_envs': n_envs,
            'jax_ms': j_ms,
            'cuda_ms': c_ms,
            'speedup': speedup,
            'cuda_mps': cuda_mps,
            'jax_mps': jax_mps,
            'mem_mb': mem_mb,
        })

        print(f"JAX={j_ms:.1f}ms  CUDA={c_ms:.1f}ms  {speedup:.1f}x")

    # Summary table
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'n_envs':>8} | {'JAX (ms)':>10} | {'CUDA (ms)':>10} | {'Speedup':>8} | {'CUDA msgs/s':>14} | {'Mem (MB)':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*14}-+-{'-'*8}")
    for r in results:
        print(f"  {r['n_envs']:>8} | {r['jax_ms']:>10.1f} | {r['cuda_ms']:>10.1f} | {r['speedup']:>7.1f}x | {r['cuda_mps']:>14,.0f} | {r['mem_mb']:>8.0f}")

    # CUDA scaling analysis
    if len(results) >= 2:
        base = results[0]
        print()
        print("  CUDA scaling (time growth vs env count growth):")
        for r in results[1:]:
            env_ratio = r['n_envs'] / base['n_envs']
            time_ratio = r['cuda_ms'] / base['cuda_ms'] if base['cuda_ms'] > 0 else float('nan')
            efficiency = env_ratio / time_ratio * 100 if time_ratio > 0 else 0
            print(f"    {base['n_envs']:>5} → {r['n_envs']:>5} ({env_ratio:>6.0f}x envs): "
                  f"time {time_ratio:>5.1f}x → parallel efficiency {efficiency:>5.1f}%")

    # Comparison with other session's baseline
    print()
    print("  Cross-session comparison (other session baseline at nOrders=100, 500 msgs):")
    print("  ┌──────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ Approach             │ 1000 env │ 4000 env │ 8000 env │")
    print("  ├──────────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Baseline (other ses) │ 2,917 ms │10,276 ms │20,038 ms │")
    print(f"  │ L1 Sorted (other)   │ 1,656 ms │ 3,766 ms │  killed  │")
    r1k = next((r for r in results if r['n_envs'] == 1000), None)
    r4k = next((r for r in results if r['n_envs'] == 4000), None)
    r8k = next((r for r in results if r['n_envs'] == 8000), None)
    c1k = f"{r1k['cuda_ms']:.0f} ms" if r1k else "—"
    c4k = f"{r4k['cuda_ms']:.0f} ms" if r4k else "—"
    c8k = f"{r8k['cuda_ms']:.0f} ms" if r8k else "—"
    print(f"  │ CUDA V2 (this ses)  │ {c1k:>8} │ {c4k:>8} │ {c8k:>8} │")
    print("  └──────────────────────┴──────────┴──────────┴──────────┘")

    print()
    print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
