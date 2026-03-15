"""Debug: find exactly where V2 CUDA batch diverges from JAX.

Strategy: process messages one at a time and compare after each.
"""
import os, sys
os.environ["JAXOB_USE_CUDA_MATCHING"] = "0"
os.environ["JAXOB_USE_TRITON_MATCHING"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gymnax_exchange.jaxob.JaxOrderBookArrays import scan_through_entire_array, cond_type_side
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
from gymnax_exchange.jaxob.cuda_matching import (
    scan_through_entire_array_cuda, _CUDA_BATCH_AVAILABLE,
)

CFG = JAXLOB_Configuration(nOrders=100, nTrades=100)

print(f"JAX: {jax.__version__} | CUDA batch: {_CUDA_BATCH_AVAILABLE}")

# Simple book setup
asks = jnp.full((100, 6), -1, dtype=jnp.int32)
bids = jnp.full((100, 6), -1, dtype=jnp.int32)
trades = jnp.full((100, 8), -1, dtype=jnp.int32)
# 5 asks at 10100-10500, 5 bids at 9900-9500
for i in range(5):
    asks = asks.at[i].set(jnp.array([10100+i*100, 50, -(i+1), -2, 34200, i*1000], dtype=jnp.int32))
    bids = bids.at[i].set(jnp.array([9900-i*100, 50, -(100+i+1), -2, 34200, i*1000], dtype=jnp.int32))

key = jax.random.PRNGKey(0)

# Test messages one at a time
test_msgs = [
    # type, side, qty, price, oid, tid, time_s, time_ns
    [1,  1,  30, 10100, 1001, 100, 34300, 1000],   # 0: bid limit @ 10100 (crosses ask@10100)
    [1, -1,  20, 9900,  1002, 101, 34300, 2000],   # 1: ask limit @ 9900 (crosses bid@9900)
    [1,  1,  10, 9000,  1003, 102, 34300, 3000],   # 2: bid limit @ 9000 (no cross, adds to book)
    [2,  1,  10, 9000,  1003, 102, 34300, 4000],   # 3: cancel bid OID 1003
    [1, -1,  15, 11000, 1004, 103, 34300, 5000],   # 4: ask limit @ 11000 (no cross, adds to book)
    [0,  0,   0,     0,    0,   0,     0,    0],   # 5: noop
]

print(f"\nProcessing {len(test_msgs)} messages one-by-one to find divergence:\n")

jax_asks, jax_bids, jax_trades = asks, bids, trades
cuda_book = (asks, bids, trades)

for i, msg_data in enumerate(test_msgs):
    msg_arr = jnp.array([msg_data], dtype=jnp.int32)  # (1, 8)

    # JAX: process 1 message
    jax_result = scan_through_entire_array(CFG, key, msg_arr, (jax_asks, jax_bids, jax_trades))
    jax_asks, jax_bids, jax_trades = jax_result

    # CUDA: process 1 message
    cuda_result = scan_through_entire_array_cuda(CFG, key, msg_arr, cuda_book)
    cuda_asks, cuda_bids, cuda_trades = cuda_result
    cuda_book = cuda_result

    # Compare
    asks_ok = np.array_equal(np.array(jax_asks), np.array(cuda_asks))
    bids_ok = np.array_equal(np.array(jax_bids), np.array(cuda_bids))
    trades_ok = np.array_equal(np.array(jax_trades), np.array(cuda_trades))

    type_names = {0: "noop", 1: "limit", 2: "cancel", 3: "delete", 4: "match"}
    side_names = {-1: "ask", 0: "none", 1: "bid"}
    t = msg_data[0]
    s = msg_data[1]
    label = f"msg[{i}]: {type_names.get(t,'?')} {side_names.get(s,'?')} qty={msg_data[2]} price={msg_data[3]}"

    if asks_ok and bids_ok and trades_ok:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}")
        if not asks_ok:
            for r in range(100):
                ja = np.array(jax_asks[r])
                ca = np.array(cuda_asks[r])
                if not np.array_equal(ja, ca):
                    print(f"         asks[{r}]: JAX={ja.tolist()} CUDA={ca.tolist()}")
        if not bids_ok:
            for r in range(100):
                jb = np.array(jax_bids[r])
                cb = np.array(cuda_bids[r])
                if not np.array_equal(jb, cb):
                    print(f"         bids[{r}]: JAX={jb.tolist()} CUDA={cb.tolist()}")
        if not trades_ok:
            for r in range(100):
                jt = np.array(jax_trades[r])
                ct = np.array(cuda_trades[r])
                if not np.array_equal(jt, ct):
                    print(f"         trades[{r}]: JAX={jt.tolist()} CUDA={ct.tolist()}")
        # Stop at first failure for clarity
        break

print("\n=== Done ===")
