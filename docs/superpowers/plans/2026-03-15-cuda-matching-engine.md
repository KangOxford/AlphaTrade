# CUDA Matching Engine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the JAX `while_loop`-based order matching kernel with a CUDA C++ kernel via JAX FFI, achieving 50-200x single-book speedup while preserving GPU-parallel multi-book scaling via vmap.

**Architecture:** Write a CUDA kernel that implements price-time priority matching using bitset-indexed flat arrays (ported from PIYUSH C++20 engine). Register it with JAX via `jax.ffi.ffi_call` + `jax.ffi.register_ffi_target`. Drop into the existing dispatch mechanism at `_match_against_ask/bid_orders` (same replacement surface as the Triton kernel). Custom vmap rules enable multi-book parallelism.

**Tech Stack:** CUDA C++17, nvcc (CUDA 12.6), JAX 0.9.0.1 FFI, Python 3.12, GH200 GPU (sm_90)

---

## Replacement Surface

```
What stays (JAX):                      What gets replaced (→ CUDA):
─────────────────                      ────────────────────────────
jax.lax.scan(cond_type_side, ...)      _match_against_ask_orders_jax() ─→ CUDA kernel
  ├─ cond_type_side()                  _match_against_bid_orders_jax() ─→ CUDA kernel
  │   ├─ jax.lax.switch(5 branches)     (including _get_top_*_order_idx,
  │   │   ├─ bid_lim()                   _check_before_matching_*,
  │   │   │   ├─ _match_against_ask ◄──   match_order, while_loop)
  │   │   │   └─ add_order()
  │   │   ├─ ask_lim()
  │   │   │   ├─ _match_against_bid ◄──
  │   │   │   └─ add_order()
  │   │   ├─ ask_cancel()              (cancel stays in JAX — uses PRNGKey)
  │   │   ├─ bid_cancel()
  │   │   └─ doNothing()
```

**Input/Output contract** (identical to Triton kernel):

```
Input:
  orderside    (nOrders, 6)  int32  — standing orders [price, qty, oid, tid, time_s, time_ns]
  qtm          scalar        int32  — incoming quantity to match
  price        scalar        int32  — incoming limit price
  trade        (nTrades, 8)  int32  — trade buffer [price, qty, pass_oid, agr_oid, time_s, time_ns, pass_tid, agr_tid]
  agrOID       scalar        int32  — aggressor order ID
  time         scalar        int32  — timestamp seconds
  time_ns      scalar        int32  — timestamp nanoseconds
  agrTID       scalar        int32  — aggressor trader ID
  side         scalar        int32  — +1 (bid) or -1 (ask)

Output:
  orderside    (nOrders, 6)  int32  — standing orders after matching
  qtm_remain   scalar        int32  — remaining unmatched quantity (can be negative)
  price        scalar        int32  — passthrough
  trade        (nTrades, 8)  int32  — trade buffer with new trades appended
```

## File Structure

```
gymnax_exchange/jaxob/
├── cuda_matching/                       # NEW — CUDA matching engine package
│   ├── __init__.py                      # Python API: register_cuda_matching()
│   ├── cuda_matching_kernel.cu          # CUDA C++ kernel source
│   ├── build.sh                         # Build script: nvcc → libcuda_matching.so
│   └── libcuda_matching.so              # Compiled shared library (gitignored)
├── JaxOrderBookArrays.py                # MODIFY — add CUDA dispatch branch (lines 59-77, 355-376)
├── triton_matching.py                   # UNCHANGED — existing Triton path
├── jorderbook.py                        # UNCHANGED
├── jaxob_config.py                      # UNCHANGED
└── jaxob_constants.py                   # UNCHANGED

tests/
└── test_cuda_matching.py                # NEW — correctness + benchmark tests
```

---

## Chunk 1: CUDA Kernel

### Task 1: CUDA kernel — core matching loop

**Files:**
- Create: `gymnax_exchange/jaxob/cuda_matching/cuda_matching_kernel.cu`

The kernel implements the exact same matching semantics as `_match_against_ask_orders_jax` / `_match_against_bid_orders_jax`, but using PIYUSH-style bitset price indexing instead of O(N) linear scans.

- [ ] **Step 1: Create the CUDA kernel file**

```cuda
// cuda_matching_kernel.cu
//
// CUDA matching kernel for JAX order book.
// Registered with JAX via XLA FFI (jax.ffi.register_ffi_target).
//
// Build: see build.sh
//
// Architecture:
//   - Single thread per order book (no intra-book parallelism needed)
//   - Bitset-indexed price levels for O(1) best-price lookup
//   - Sequential message matching within each book
//   - Multi-book parallelism via JAX vmap (each call = 1 book)

#include <cstdint>
#include <cstring>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ─── Constants ───────────────────────────────────────────────
// Price range: we support prices 0..MAX_PRICE-1
// For LOBSTER data, prices are ~$100 = 10000 (in 1-cent ticks)
// We use a 2-level bitset: word index = price / 64, bit = price % 64
constexpr int MAX_PRICE = 200000;  // covers prices up to $2000.00
constexpr int BITSET_WORDS = (MAX_PRICE + 63) / 64;
constexpr int EMPTY_SLOT = -1;

// ─── Device helpers ──────────────────────────────────────────

// Find lowest set bit in word (0-indexed), returns -1 if word is 0
__device__ __forceinline__ int find_lowest_bit(uint64_t word) {
    if (word == 0) return -1;
    return __ffsll(word) - 1;  // __ffsll returns 1-indexed
}

// Find highest set bit in word (0-indexed), returns -1 if word is 0
__device__ __forceinline__ int find_highest_bit(uint64_t word) {
    if (word == 0) return -1;
    return 63 - __clzll(word);
}

// ─── Matching kernel ─────────────────────────────────────────

// Process one matching request: incoming order at `price` with `qtm` quantity
// matches against standing orders on `orderside`.
//
// is_bid=true:  incoming is a BID, matching against ASKs (find LOWEST ask price)
// is_bid=false: incoming is an ASK, matching against BIDs (find HIGHEST bid price)
__global__ void match_orders_kernel(
    // Inputs
    const int32_t* __restrict__ orderside_in,   // (N_ORDERS, 6) flattened row-major
    const int32_t* __restrict__ trade_in,        // (N_TRADES, 8) flattened row-major
    int32_t qtm,
    int32_t price,
    int32_t agrOID,
    int32_t time_s,
    int32_t time_ns,
    int32_t agrTID,
    int32_t side,               // +1 for bid, -1 for ask
    int32_t n_orders,
    int32_t n_trades,
    int32_t is_bid,             // 1 = incoming is bid (match vs asks), 0 = incoming is ask (match vs bids)
    // Outputs
    int32_t* __restrict__ orderside_out,  // (N_ORDERS, 6)
    int32_t* __restrict__ trade_out,       // (N_TRADES, 8)
    int32_t* __restrict__ qtm_out          // (1,) remaining quantity
) {
    // Copy inputs to outputs (we modify in-place on the output)
    for (int i = 0; i < n_orders * 6; i++) {
        orderside_out[i] = orderside_in[i];
    }
    for (int i = 0; i < n_trades * 8; i++) {
        trade_out[i] = trade_in[i];
    }

    // Build bitset from standing orders
    // Using stack-allocated bitset (MAX_PRICE/64 = 3125 uint64_t = 25KB)
    // For shared memory: move to __shared__ if needed
    uint64_t bitset[BITSET_WORDS];
    memset(bitset, 0, sizeof(bitset));

    for (int i = 0; i < n_orders; i++) {
        int32_t p = orderside_out[i * 6 + 0];  // price column
        if (p != EMPTY_SLOT && p >= 0 && p < MAX_PRICE) {
            int word_idx = p / 64;
            int bit_idx = p % 64;
            bitset[word_idx] |= (1ULL << bit_idx);
        }
    }

    // Matching loop
    while (qtm > 0) {
        // Find best price on the standing side
        int best_price = -1;

        if (is_bid) {
            // Incoming bid matches against asks: find LOWEST ask price <= incoming price
            for (int w = 0; w <= price / 64 && w < BITSET_WORDS; w++) {
                uint64_t word = bitset[w];
                if (w == price / 64) {
                    // Mask out bits above our price
                    int bit = price % 64;
                    word &= ((1ULL << (bit + 1)) - 1);
                }
                if (word != 0) {
                    best_price = w * 64 + find_lowest_bit(word);
                    break;
                }
            }
        } else {
            // Incoming ask matches against bids: find HIGHEST bid price >= incoming price
            for (int w = (MAX_PRICE - 1) / 64; w >= price / 64 && w >= 0; w--) {
                uint64_t word = bitset[w];
                if (w == price / 64) {
                    // Mask out bits below our price
                    int bit = price % 64;
                    word &= ~((1ULL << bit) - 1);
                }
                if (word != 0) {
                    best_price = w * 64 + find_highest_bit(word);
                    break;
                }
            }
        }

        if (best_price == -1) break;  // No more matchable orders

        // Find the earliest order at best_price (time priority)
        int best_idx = -1;
        int32_t best_time_s = 0x7FFFFFFF;
        int32_t best_time_ns = 0x7FFFFFFF;

        for (int i = 0; i < n_orders; i++) {
            int32_t p = orderside_out[i * 6 + 0];
            int32_t q = orderside_out[i * 6 + 1];
            int32_t ts = orderside_out[i * 6 + 4];
            int32_t tns = orderside_out[i * 6 + 5];

            if (p == best_price && q > 0) {
                if (ts < best_time_s || (ts == best_time_s && tns < best_time_ns)) {
                    best_time_s = ts;
                    best_time_ns = tns;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) {
            // Price level empty (stale bitset), clear bit and retry
            int word_idx = best_price / 64;
            int bit_idx = best_price % 64;
            bitset[word_idx] &= ~(1ULL << bit_idx);
            continue;
        }

        // Execute match
        int32_t standing_qty = orderside_out[best_idx * 6 + 1];
        int32_t match_qty = (qtm < standing_qty) ? qtm : standing_qty;
        int32_t standing_oid = orderside_out[best_idx * 6 + 2];
        int32_t standing_tid = orderside_out[best_idx * 6 + 3];

        // Update standing order
        orderside_out[best_idx * 6 + 1] -= match_qty;
        if (orderside_out[best_idx * 6 + 1] <= 0) {
            // Remove order (set all columns to -1)
            for (int c = 0; c < 6; c++) {
                orderside_out[best_idx * 6 + c] = EMPTY_SLOT;
            }
            // Check if this was the last order at this price
            bool price_still_active = false;
            for (int i = 0; i < n_orders; i++) {
                if (orderside_out[i * 6 + 0] == best_price && orderside_out[i * 6 + 1] > 0) {
                    price_still_active = true;
                    break;
                }
            }
            if (!price_still_active) {
                int word_idx = best_price / 64;
                int bit_idx = best_price % 64;
                bitset[word_idx] &= ~(1ULL << bit_idx);
            }
        }

        // Record trade — find first empty trade slot
        for (int t = 0; t < n_trades; t++) {
            if (trade_out[t * 8 + 2] == EMPTY_SLOT) {  // PASS_OID column
                trade_out[t * 8 + 0] = best_price;                    // price
                trade_out[t * 8 + 1] = -side * match_qty;             // signed qty
                trade_out[t * 8 + 2] = standing_oid;                  // passive OID
                trade_out[t * 8 + 3] = agrOID;                        // aggressor OID
                trade_out[t * 8 + 4] = time_s;                        // time
                trade_out[t * 8 + 5] = time_ns;                       // time_ns
                trade_out[t * 8 + 6] = standing_tid;                  // passive TID
                trade_out[t * 8 + 7] = agrTID;                        // aggressor TID
                break;
            }
        }

        qtm -= match_qty;
    }

    qtm_out[0] = qtm;
}

// ─── XLA FFI Handler ─────────────────────────────────────────

ffi::Error MatchOrdersHandler(
    cudaStream_t stream,
    // Inputs
    ffi::Buffer<ffi::DataType::S32> orderside_in,
    ffi::Buffer<ffi::DataType::S32> trade_in,
    // Scalar attrs
    int32_t qtm,
    int32_t price,
    int32_t agrOID,
    int32_t time_s,
    int32_t time_ns,
    int32_t agrTID,
    int32_t side,
    int32_t n_orders,
    int32_t n_trades,
    int32_t is_bid,
    // Outputs
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> orderside_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> trade_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> qtm_out
) {
    match_orders_kernel<<<1, 1, 0, stream>>>(
        orderside_in.typed_data(),
        trade_in.typed_data(),
        qtm, price, agrOID, time_s, time_ns, agrTID, side,
        n_orders, n_trades, is_bid,
        orderside_out->typed_data(),
        trade_out->typed_data(),
        qtm_out->typed_data()
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// Register the handler with XLA FFI
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaMatchOrders,
    MatchOrdersHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        // Input buffers
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // orderside
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // trade
        // Scalar attributes
        .Attr<int32_t>("qtm")
        .Attr<int32_t>("price")
        .Attr<int32_t>("agrOID")
        .Attr<int32_t>("time_s")
        .Attr<int32_t>("time_ns")
        .Attr<int32_t>("agrTID")
        .Attr<int32_t>("side")
        .Attr<int32_t>("n_orders")
        .Attr<int32_t>("n_trades")
        .Attr<int32_t>("is_bid")
        // Output buffers
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // orderside_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // trade_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // qtm_out
);
```

- [ ] **Step 2: Create build script**

Create `gymnax_exchange/jaxob/cuda_matching/build.sh`:

```bash
#!/bin/bash
# Build CUDA matching kernel for JAX FFI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Get XLA FFI include path from JAX
XLA_INCLUDE=$(python3 -c "import jax; print(jax.ffi.include_dir())")

echo "XLA include: ${XLA_INCLUDE}"
echo "Building CUDA matching kernel..."

nvcc -shared -o libcuda_matching.so \
    -std=c++17 \
    -Xcompiler -fPIC \
    -I"${XLA_INCLUDE}" \
    --gpu-architecture=sm_90 \
    -O3 \
    cuda_matching_kernel.cu

echo "Built: ${SCRIPT_DIR}/libcuda_matching.so"
ls -lh libcuda_matching.so
```

- [ ] **Step 3: Build and verify compilation**

```bash
cd gymnax_exchange/jaxob/cuda_matching
chmod +x build.sh
# Must run on compute node (needs nvcc + GPU headers)
# Submit via sbatch or srun
srun --nodes=1 --gpus-per-node=1 --time=00:10:00 --account=brics.s5e bash build.sh
```

Expected: `libcuda_matching.so` created successfully.

- [ ] **Step 4: Commit**

```bash
git add gymnax_exchange/jaxob/cuda_matching/
echo "libcuda_matching.so" >> .gitignore
git add .gitignore
git commit -m "feat(jaxob): add CUDA matching kernel with bitset price indexing"
```

---

## Chunk 2: Python FFI Bindings

### Task 2: Python package for CUDA matching

**Files:**
- Create: `gymnax_exchange/jaxob/cuda_matching/__init__.py`

- [ ] **Step 1: Create the Python FFI binding module**

```python
"""
CUDA Matching Engine — JAX FFI Integration

Usage:
    from gymnax_exchange.jaxob.cuda_matching import match_against_ask_orders_cuda, match_against_bid_orders_cuda

    # Same signature as Triton/JAX matching functions:
    orderside_out, qtm_remain, price_out, trade_out = match_against_ask_orders_cuda(
        orderside, qtm, price, trade, agrOID, time_s, time_ns, agrTID, side
    )

Env var: JAXOB_USE_CUDA_MATCHING=1 to enable dispatch in JaxOrderBookArrays.py
"""

import ctypes
import os
import functools
import numpy as np

import jax
import jax.numpy as jnp
from jax import custom_batching

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
    warnings.warn(f"CUDA matching kernel failed to load: {e}", stacklevel=2)


def _match_orders_cuda_impl(orderside, qtm, price, trade,
                             agrOID, time_s, time_ns, agrTID, side,
                             is_bid):
    """Call the CUDA matching kernel via JAX FFI."""
    n_orders = orderside.shape[0]
    n_trades = trade.shape[0]

    # Define output shapes
    out_orderside = jax.ShapeDtypeStruct(orderside.shape, jnp.int32)
    out_trade = jax.ShapeDtypeStruct(trade.shape, jnp.int32)
    out_qtm = jax.ShapeDtypeStruct((1,), jnp.int32)

    orderside_out, trade_out, qtm_out = jax.ffi.ffi_call(
        "cuda_match_orders",
        (out_orderside, out_trade, out_qtm),
        orderside,
        trade,
        # Scalar attributes (must be numpy scalars, not JAX arrays)
        qtm=np.int32(qtm),
        price=np.int32(price),
        agrOID=np.int32(agrOID),
        time_s=np.int32(time_s),
        time_ns=np.int32(time_ns),
        agrTID=np.int32(agrTID),
        side=np.int32(side),
        n_orders=np.int32(n_orders),
        n_trades=np.int32(n_trades),
        is_bid=np.int32(1 if is_bid else 0),
        vmap_method="sequential",
    )

    return orderside_out, qtm_out[0], price, trade_out


# ─── Public API with custom vmap rules ────────────────────────

@custom_batching.custom_vmap
def match_against_ask_orders_cuda(orderside, qtm, price, trade,
                                   agrOID, time_s, time_ns, agrTID, side):
    """Match incoming BID against standing ASKs (CUDA kernel)."""
    return _match_orders_cuda_impl(
        orderside, qtm, price, trade,
        agrOID, time_s, time_ns, agrTID, side,
        is_bid=True,
    )


@match_against_ask_orders_cuda.def_vmap
def _ask_vmap_rule(axis_size, in_batched,
                   orderside, qtm, price, trade,
                   agrOID, time_s, time_ns, agrTID, side):
    # Sequential map over batch dimension (same as Triton approach)
    out = jax.lax.map(
        lambda xs: _match_orders_cuda_impl(
            xs[0], xs[1], xs[2], xs[3],
            xs[4], xs[5], xs[6], xs[7], xs[8],
            is_bid=True,
        ),
        (orderside, qtm, price, trade, agrOID, time_s, time_ns, agrTID, side),
    )
    return out, (True, True, True, True)


@custom_batching.custom_vmap
def match_against_bid_orders_cuda(orderside, qtm, price, trade,
                                   agrOID, time_s, time_ns, agrTID, side):
    """Match incoming ASK against standing BIDs (CUDA kernel)."""
    return _match_orders_cuda_impl(
        orderside, qtm, price, trade,
        agrOID, time_s, time_ns, agrTID, side,
        is_bid=False,
    )


@match_against_bid_orders_cuda.def_vmap
def _bid_vmap_rule(axis_size, in_batched,
                   orderside, qtm, price, trade,
                   agrOID, time_s, time_ns, agrTID, side):
    out = jax.lax.map(
        lambda xs: _match_orders_cuda_impl(
            xs[0], xs[1], xs[2], xs[3],
            xs[4], xs[5], xs[6], xs[7], xs[8],
            is_bid=False,
        ),
        (orderside, qtm, price, trade, agrOID, time_s, time_ns, agrTID, side),
    )
    return out, (True, True, True, True)
```

- [ ] **Step 2: Commit**

```bash
git add gymnax_exchange/jaxob/cuda_matching/__init__.py
git commit -m "feat(jaxob): add Python FFI bindings for CUDA matching kernel"
```

---

### Task 3: Integrate dispatch in JaxOrderBookArrays.py

**Files:**
- Modify: `gymnax_exchange/jaxob/JaxOrderBookArrays.py` (lines 57-77, 355-376)

- [ ] **Step 1: Add CUDA dispatch flag and import (lines 57-77)**

After the existing Triton import block, add:

```python
_USE_CUDA_MATCHING = os.environ.get("JAXOB_USE_CUDA_MATCHING", "0") == "1"
try:
    from gymnax_exchange.jaxob.cuda_matching import (
        match_against_ask_orders_cuda as _cuda_match_against_ask_orders,
        match_against_bid_orders_cuda as _cuda_match_against_bid_orders,
        _CUDA_MATCHING_AVAILABLE,
    )
except Exception:
    _CUDA_MATCHING_AVAILABLE = False
    if _USE_CUDA_MATCHING:
        import warnings
        warnings.warn(
            "JAXOB_USE_CUDA_MATCHING=1 but CUDA matching is not available. "
            "Falling back to JAX matching. Build libcuda_matching.so first.",
            stacklevel=2,
        )
```

- [ ] **Step 2: Add CUDA branch to dispatch functions (lines 355-376)**

Replace the two dispatch functions:

```python
def _match_against_bid_orders(cfg, orderside, qtm, price, trade,
                               agrOID, time, time_ns, agrTID, side):
    if _USE_CUDA_MATCHING and _CUDA_MATCHING_AVAILABLE:
        return _cuda_match_against_bid_orders(
            orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side)
    if _USE_TRITON_MATCHING and _TRITON_MATCHING_AVAILABLE:
        return _triton_match_against_bid_orders(
            orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side)
    return _match_against_bid_orders_jax(
        cfg, orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side)

def _match_against_ask_orders(cfg, orderside, qtm, price, trade,
                               agrOID, time, time_ns, agrTID, side):
    if _USE_CUDA_MATCHING and _CUDA_MATCHING_AVAILABLE:
        return _cuda_match_against_ask_orders(
            orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side)
    if _USE_TRITON_MATCHING and _TRITON_MATCHING_AVAILABLE:
        return _triton_match_against_ask_orders(
            orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side)
    return _match_against_ask_orders_jax(
        cfg, orderside, qtm, price, trade, agrOID, time, time_ns, agrTID, side)
```

- [ ] **Step 3: Commit**

```bash
git add gymnax_exchange/jaxob/JaxOrderBookArrays.py
git commit -m "feat(jaxob): add CUDA matching dispatch path (JAXOB_USE_CUDA_MATCHING=1)"
```

---

## Chunk 3: Testing & Benchmarking

### Task 4: Correctness tests

**Files:**
- Create: `tests/test_cuda_matching.py`

- [ ] **Step 1: Write correctness test — compare CUDA vs JAX results**

```python
"""
Test CUDA matching kernel produces identical results to JAX matching.

Run on compute node:
  srun --nodes=1 --gpus-per-node=1 --time=00:10:00 --account=brics.s5e \
       python -m pytest tests/test_cuda_matching.py -v
"""
import os
import sys
import pytest
import numpy as np

# Force both backends available
os.environ["JAXOB_USE_CUDA_MATCHING"] = "0"
os.environ["JAXOB_USE_TRITON_MATCHING"] = "0"

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gymnax_exchange.jaxob.JaxOrderBookArrays import (
    _match_against_ask_orders_jax,
    _match_against_bid_orders_jax,
)
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration

# Import CUDA functions directly
from gymnax_exchange.jaxob.cuda_matching import (
    match_against_ask_orders_cuda,
    match_against_bid_orders_cuda,
    _CUDA_MATCHING_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not _CUDA_MATCHING_AVAILABLE,
    reason="CUDA matching kernel not built"
)

CFG = JAXLOB_Configuration(nOrders=100, nTrades=100)


def make_book(key, n_orders=100, n_filled=10, base_price=10000, tick=100):
    """Create a test order book with known orders."""
    asks = jnp.full((n_orders, 6), -1, dtype=jnp.int32)
    trades = jnp.full((100, 8), -1, dtype=jnp.int32)

    for i in range(n_filled):
        p = base_price + tick * (i + 1)
        asks = asks.at[i].set(jnp.array([p, 50, -(i+1), -2, 34200, i*1000], dtype=jnp.int32))

    return asks, trades


class TestCudaMatchingCorrectness:
    """Compare CUDA kernel output to JAX reference implementation."""

    def test_no_match(self):
        """Bid price below all asks — no matching should occur."""
        asks, trades = make_book(None)
        # Bid at 9000, all asks at 10100+ → no match
        args = (asks, jnp.int32(50), jnp.int32(9000), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        jax_result = _match_against_ask_orders_jax(CFG, *args)
        cuda_result = match_against_ask_orders_cuda(*args)

        np.testing.assert_array_equal(jax_result[0], cuda_result[0])  # orderside
        assert int(jax_result[1]) == int(cuda_result[1])               # remaining qty
        np.testing.assert_array_equal(jax_result[3], cuda_result[3])  # trades

    def test_single_match(self):
        """Bid crosses exactly one ask level."""
        asks, trades = make_book(None, n_filled=5, base_price=10000, tick=100)
        # Bid at 10100, matches the lowest ask at 10100
        args = (asks, jnp.int32(30), jnp.int32(10100), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        jax_result = _match_against_ask_orders_jax(CFG, *args)
        cuda_result = match_against_ask_orders_cuda(*args)

        np.testing.assert_array_equal(jax_result[0], cuda_result[0])
        assert int(jax_result[1]) == int(cuda_result[1])
        np.testing.assert_array_equal(jax_result[3], cuda_result[3])

    def test_multi_level_match(self):
        """Bid crosses multiple ask levels."""
        asks, trades = make_book(None, n_filled=5, base_price=10000, tick=100)
        # Bid at 10500, qty=200 — should sweep through all 5 levels (50 qty each = 250 available)
        args = (asks, jnp.int32(200), jnp.int32(10500), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        jax_result = _match_against_ask_orders_jax(CFG, *args)
        cuda_result = match_against_ask_orders_cuda(*args)

        np.testing.assert_array_equal(jax_result[0], cuda_result[0])
        assert int(jax_result[1]) == int(cuda_result[1])
        np.testing.assert_array_equal(jax_result[3], cuda_result[3])

    def test_full_fill(self):
        """Qty exactly equals available — should fully match with 0 remaining."""
        asks, trades = make_book(None, n_filled=1, base_price=10000, tick=100)
        args = (asks, jnp.int32(50), jnp.int32(10100), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        jax_result = _match_against_ask_orders_jax(CFG, *args)
        cuda_result = match_against_ask_orders_cuda(*args)

        assert int(jax_result[1]) == 0
        assert int(cuda_result[1]) == 0
        np.testing.assert_array_equal(jax_result[0], cuda_result[0])

    def test_bid_side_matching(self):
        """Test _match_against_bid_orders (incoming ASK vs standing BIDs)."""
        bids = jnp.full((100, 6), -1, dtype=jnp.int32)
        trades = jnp.full((100, 8), -1, dtype=jnp.int32)
        # BIDs at 9900, 9800, 9700
        for i in range(3):
            p = 9900 - i * 100
            bids = bids.at[i].set(jnp.array([p, 50, -(i+1), -2, 34200, i*1000], dtype=jnp.int32))

        # Incoming ask at 9800, qty=80 — should match bid@9900(50) + bid@9800(30)
        args = (bids, jnp.int32(80), jnp.int32(9800), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(-1))

        jax_result = _match_against_bid_orders_jax(CFG, *args)
        cuda_result = match_against_bid_orders_cuda(*args)

        np.testing.assert_array_equal(jax_result[0], cuda_result[0])
        assert int(jax_result[1]) == int(cuda_result[1])
        np.testing.assert_array_equal(jax_result[3], cuda_result[3])

    def test_empty_book(self):
        """No standing orders — nothing to match."""
        asks = jnp.full((100, 6), -1, dtype=jnp.int32)
        trades = jnp.full((100, 8), -1, dtype=jnp.int32)
        args = (asks, jnp.int32(50), jnp.int32(10500), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        jax_result = _match_against_ask_orders_jax(CFG, *args)
        cuda_result = match_against_ask_orders_cuda(*args)

        assert int(jax_result[1]) == 50  # full qty remains
        assert int(cuda_result[1]) == 50


class TestCudaMatchingVmap:
    """Test that vmap works with CUDA matching."""

    def test_vmap_basic(self):
        """vmap over 10 independent books."""
        n_books = 10
        asks_batch = jnp.stack([make_book(None, n_filled=5)[0] for _ in range(n_books)])
        trades_batch = jnp.full((n_books, 100, 8), -1, dtype=jnp.int32)
        qtm_batch = jnp.full(n_books, 30, dtype=jnp.int32)
        price_batch = jnp.full(n_books, 10100, dtype=jnp.int32)
        oid_batch = jnp.arange(n_books, dtype=jnp.int32) + 1000
        time_batch = jnp.full(n_books, 34300, dtype=jnp.int32)
        tns_batch = jnp.zeros(n_books, dtype=jnp.int32)
        tid_batch = jnp.full(n_books, 100, dtype=jnp.int32)
        side_batch = jnp.ones(n_books, dtype=jnp.int32)

        result = jax.vmap(match_against_ask_orders_cuda)(
            asks_batch, qtm_batch, price_batch, trades_batch,
            oid_batch, time_batch, tns_batch, tid_batch, side_batch
        )

        # All books identical → all results should be identical
        for i in range(1, n_books):
            np.testing.assert_array_equal(result[0][0], result[0][i])
            assert int(result[1][0]) == int(result[1][i])
```

- [ ] **Step 2: Write benchmark comparison test**

Add to the same file:

```python
class TestCudaBenchmark:
    """Performance comparison: CUDA vs JAX matching."""

    def test_single_match_speed(self):
        """Measure single-call latency for CUDA vs JAX."""
        import time

        asks, trades = make_book(None, n_filled=60)
        args = (asks, jnp.int32(100), jnp.int32(10500), trades,
                jnp.int32(999), jnp.int32(34300), jnp.int32(0), jnp.int32(100), jnp.int32(1))

        # Warmup
        for _ in range(5):
            r = match_against_ask_orders_cuda(*args)
            jax.block_until_ready(r)
            r = _match_against_ask_orders_jax(CFG, *args)
            jax.block_until_ready(r)

        # CUDA
        times_cuda = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = match_against_ask_orders_cuda(*args)
            jax.block_until_ready(r)
            times_cuda.append(time.perf_counter() - t0)

        # JAX
        times_jax = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = _match_against_ask_orders_jax(CFG, *args)
            jax.block_until_ready(r)
            times_jax.append(time.perf_counter() - t0)

        cuda_median = np.median(times_cuda) * 1e6
        jax_median = np.median(times_jax) * 1e6
        speedup = jax_median / cuda_median

        print(f"\n{'='*50}")
        print(f"CUDA matching: {cuda_median:.1f} μs (median)")
        print(f"JAX matching:  {jax_median:.1f} μs (median)")
        print(f"Speedup:       {speedup:.1f}x")
        print(f"{'='*50}")

        # We expect at least some speedup; flag if CUDA is slower
        assert speedup > 0.5, f"CUDA is {1/speedup:.1f}x SLOWER than JAX — investigate"
```

- [ ] **Step 3: Create sbatch test script**

Create `tests/run_cuda_matching_tests.batch`:

```bash
#!/bin/bash
#SBATCH --job-name=cuda-match-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --account=brics.s5e
#SBATCH --output=logs/cuda_match_test_%j.out

export CONDA_PREFIX=/projects/s5e/quant/miniforge3
export PATH=$CONDA_PREFIX/bin:$PATH
module load cuda/12.6

cd /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/Alphatrade

# Step 1: Build the kernel
echo "=== Building CUDA kernel ==="
cd gymnax_exchange/jaxob/cuda_matching
bash build.sh
cd /lus/lfs1aip2/projects/s5e/quant/AlphaTrade/LOBS5/Alphatrade

# Step 2: Run correctness tests
echo "=== Running tests ==="
python -m pytest tests/test_cuda_matching.py -v -s

echo "=== Done: $(date) ==="
```

- [ ] **Step 4: Submit test job**

```bash
sbatch --job-name=cuda-match-test tests/run_cuda_matching_tests.batch
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_cuda_matching.py tests/run_cuda_matching_tests.batch
git commit -m "test(jaxob): add correctness and benchmark tests for CUDA matching"
```

---

### Task 5: Run full benchmark comparison

- [ ] **Step 1: Run benchmark with CUDA backend**

```bash
# In benchmark_matching_engine.batch, add Run 3:
echo "=== Run 3: CUDA matching ==="
export JAXOB_USE_CUDA_MATCHING=1
export JAXOB_USE_TRITON_MATCHING=0
python benchmark_matching_engine.py \
    --n_orders 100 \
    --n_msgs 500 \
    --n_books 1,10,100,1000 \
    --warmup 5 \
    --trials 20
```

- [ ] **Step 2: Compare results and update findings.md**

Expected results table:

```
| Backend | Single msg (crossing) | Batch scan (mixed) | VMAP 1000 books |
|---------|----------------------|--------------------|-----------------|
| JAX     | ~206 μs              | ~63 μs / 15.8K /s  | ~2.5M /s        |
| CUDA    | ~?? μs               | ~?? μs / ??K /s    | ~?? /s          |
| Target  | <10 μs               | <5 μs / >100K /s   | >10M /s         |
```

- [ ] **Step 3: Commit results**

```bash
git add findings.md progress.md
git commit -m "perf(jaxob): benchmark CUDA matching kernel results"
```

---

## Chunk 4: Optimization (if needed)

### Task 6: Optimize CUDA kernel based on profiling

Only proceed if initial benchmark shows kernel is slower than expected.

- [ ] **Step 1: Profile with nsys**

```bash
srun --nodes=1 --gpus-per-node=1 --time=00:10:00 --account=brics.s5e \
    nsys profile -o cuda_match_profile \
    python -c "
from gymnax_exchange.jaxob.cuda_matching import match_against_ask_orders_cuda
import jax.numpy as jnp
asks = jnp.full((100, 6), -1, dtype=jnp.int32)
# ... setup and call 1000 times
"
```

- [ ] **Step 2: Potential optimizations based on profile**

| Optimization | When to Apply | Expected Impact |
|-------------|---------------|-----------------|
| Move bitset to `__shared__` memory | If global memory is bottleneck | 2-5x for bitset ops |
| Use warp-level `__ballot_sync` for price scan | If linear scan dominates | O(1) best-price lookup |
| Batch multiple messages per kernel launch | If kernel launch overhead dominates | Amortize launch cost |
| Process N messages sequentially within kernel | If lax.scan overhead dominates | Eliminate N-1 kernel launches |

The most impactful optimization is likely **batching multiple messages per kernel launch** — replacing the JAX `lax.scan` over messages with a single CUDA kernel that processes all messages sequentially. This eliminates the per-message kernel launch overhead (~2-5 μs × 500 messages = 1-2.5 ms of pure overhead).

- [ ] **Step 3: Commit optimizations**

```bash
git commit -m "perf(jaxob): optimize CUDA matching kernel based on profiling"
```

---

## Chunk 5: Integration with Training Pipeline

### Task 7: End-to-end validation

- [ ] **Step 1: Run a short training job with CUDA matching**

```bash
# Add to node_wrapper.sh or batch script:
export JAXOB_USE_CUDA_MATCHING=1

# Submit a short training test (1 epoch, curtailed)
CURTAIL_EPOCHS=300 sbatch --nodes=1 --time=00:30:00 --job-name=cuda-match-train train_full_autoreg.batch
```

- [ ] **Step 2: Compare loss trajectory with JAX baseline**

The loss after 300 steps should be identical (within floating point tolerance) to a JAX-matching run with the same seed and data.

- [ ] **Step 3: Commit integration**

```bash
git commit -m "feat(jaxob): validate CUDA matching in training pipeline"
```

---

## Summary: Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Replacement surface | `_match_against_*_orders` only | Minimal change, same as Triton, avoids PRNGKey in cancel path |
| Integration API | JAX FFI (`jax.ffi.ffi_call`) | Modern API, built-in vmap support, no deprecated features |
| Kernel threading | 1 thread per book, 1 block per call | Matching is sequential within a book; parallelism comes from vmap |
| Price lookup | Bitset (`uint64_t[]` with `__ffsll`) | O(1) vs O(N) linear scan — main source of speedup |
| vmap strategy | `@custom_batching.custom_vmap` with `jax.lax.map` | Same pattern as existing Triton kernel, proven to work |
| Build system | Simple `nvcc` script | Minimal complexity, no cmake/meson needed for single file |
| Dispatch mechanism | `JAXOB_USE_CUDA_MATCHING` env var | Consistent with existing `JAXOB_USE_TRITON_MATCHING` pattern |

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| FFI scalar attrs don't work with traced JAX values | Attrs are compile-time constants; dynamic values must be buffers. May need to pack scalars into a buffer (like Triton does). |
| Kernel launch overhead dominates | Batch messages: process all N messages in one kernel call instead of N separate calls. |
| `nvcc` fails on ARM aarch64 | GH200 nodes have ARM CPUs; verify nvcc cross-compilation works. CUDA 12.6 should support this natively. |
| Bitset stack allocation too large for GPU | 25KB per book is fine for a single thread. For warp-level, use shared memory. |
| vmap sequential map is slow | Same limitation as Triton. Future: batch multiple books in one kernel launch using CUDA grid. |
