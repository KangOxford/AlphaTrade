// CUDA Matching Kernel for JAX Order Book (XLA FFI)
//
// Drop-in replacement for _match_against_ask/bid_orders.
// Keeps the entire matching loop on GPU with no D2H sync.
//
// Input buffers:
//   orderside_in  (n_orders * 6,)  int32  — standing orders, row-major
//   trade_in      (n_trades * 8,)  int32  — trade buffer, row-major
//   incoming      (7,)             int32  — [qtm, price, agrOID, time_s, time_ns, agrTID, side]
//
// Output buffers:
//   orderside_out (n_orders * 6,)  int32
//   trade_out     (n_trades * 8,)  int32
//   qtm_out       (1,)             int32  — remaining unmatched quantity
//
// Attrs (compile-time):
//   n_orders  int32
//   n_trades  int32
//   is_bid    int32   — 1 = incoming bid vs standing asks, 0 = incoming ask vs standing bids
//
// Build: nvcc -shared -o libcuda_matching.so -std=c++17 -Xcompiler -fPIC
//        -I$(python3 -c "import jax; print(jax.ffi.include_dir())")
//        --gpu-architecture=sm_90 -O3 cuda_matching_kernel.cu

#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

constexpr int ORDER_COLS = 6;
constexpr int TRADE_COLS = 8;
constexpr int32_t EMPTY_SLOT = -1;
constexpr int32_t MAX_INT = 2147483647;

// Column indices for orderside
constexpr int COL_PRICE = 0;
constexpr int COL_QTY   = 1;
constexpr int COL_OID   = 2;
constexpr int COL_TID   = 3;
constexpr int COL_TIME  = 4;
constexpr int COL_TNS   = 5;

// Column indices for trades
constexpr int T_PRICE    = 0;
constexpr int T_QTY      = 1;
constexpr int T_PASS_OID = 2;
constexpr int T_AGR_OID  = 3;
constexpr int T_TIME     = 4;
constexpr int T_TNS      = 5;
constexpr int T_PASS_TID = 6;
constexpr int T_AGR_TID  = 7;

// Incoming scalar indices
constexpr int IN_QTM     = 0;
constexpr int IN_PRICE   = 1;
constexpr int IN_AGROID  = 2;
constexpr int IN_TIME    = 3;
constexpr int IN_TNS     = 4;
constexpr int IN_AGRTID  = 5;
constexpr int IN_SIDE    = 6;


__global__ void match_orders_kernel(
    const int32_t* __restrict__ orderside_in,
    const int32_t* __restrict__ trade_in,
    const int32_t* __restrict__ incoming,
    int32_t* __restrict__ orderside_out,
    int32_t* __restrict__ trade_out,
    int32_t* __restrict__ qtm_out,
    int32_t n_orders,
    int32_t n_trades,
    int32_t is_bid
) {
    // Copy orderside and trades to output (modify in-place on output)
    int os_size = n_orders * ORDER_COLS;
    int tr_size = n_trades * TRADE_COLS;
    for (int i = threadIdx.x; i < os_size; i += blockDim.x) {
        orderside_out[i] = orderside_in[i];
    }
    for (int i = threadIdx.x; i < tr_size; i += blockDim.x) {
        trade_out[i] = trade_in[i];
    }
    __syncthreads();

    // Only thread 0 does the matching (sequential logic)
    if (threadIdx.x != 0) return;

    // Unpack incoming scalars
    int32_t qtm       = incoming[IN_QTM];
    int32_t price     = incoming[IN_PRICE];
    int32_t agr_oid   = incoming[IN_AGROID];
    int32_t time_s    = incoming[IN_TIME];
    int32_t time_ns   = incoming[IN_TNS];
    int32_t agr_tid   = incoming[IN_AGRTID];
    int32_t side      = incoming[IN_SIDE];

    // Matching loop: iterate up to n_orders times (bounded, like Triton's static_range)
    for (int iter = 0; iter < n_orders; iter++) {
        if (qtm <= 0) break;

        // Find best price order with time priority
        int best_idx = -1;
        int32_t best_price;
        int32_t best_time_s = MAX_INT;
        int32_t best_time_ns = MAX_INT;

        if (is_bid) {
            // Incoming bid: find LOWEST ask price <= incoming price
            best_price = MAX_INT;
            for (int i = 0; i < n_orders; i++) {
                int32_t p = orderside_out[i * ORDER_COLS + COL_PRICE];
                int32_t q = orderside_out[i * ORDER_COLS + COL_QTY];
                if (p == EMPTY_SLOT || q <= 0) continue;
                if (p > price) continue;  // ask price too high for our bid

                int32_t ts = orderside_out[i * ORDER_COLS + COL_TIME];
                int32_t tns = orderside_out[i * ORDER_COLS + COL_TNS];

                // Price priority first, then time priority
                if (p < best_price ||
                    (p == best_price && ts < best_time_s) ||
                    (p == best_price && ts == best_time_s && tns < best_time_ns)) {
                    best_price = p;
                    best_time_s = ts;
                    best_time_ns = tns;
                    best_idx = i;
                }
            }
        } else {
            // Incoming ask: find HIGHEST bid price >= incoming price
            best_price = -1;
            for (int i = 0; i < n_orders; i++) {
                int32_t p = orderside_out[i * ORDER_COLS + COL_PRICE];
                int32_t q = orderside_out[i * ORDER_COLS + COL_QTY];
                if (p == EMPTY_SLOT || q <= 0) continue;
                if (p < price) continue;  // bid price too low for our ask

                int32_t ts = orderside_out[i * ORDER_COLS + COL_TIME];
                int32_t tns = orderside_out[i * ORDER_COLS + COL_TNS];

                if (p > best_price ||
                    (p == best_price && ts < best_time_s) ||
                    (p == best_price && ts == best_time_s && tns < best_time_ns)) {
                    best_price = p;
                    best_time_s = ts;
                    best_time_ns = tns;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) break;  // No matchable orders

        // Execute match
        int32_t standing_qty = orderside_out[best_idx * ORDER_COLS + COL_QTY];
        int32_t standing_oid = orderside_out[best_idx * ORDER_COLS + COL_OID];
        int32_t standing_tid = orderside_out[best_idx * ORDER_COLS + COL_TID];

        int32_t matched_qty = (qtm < standing_qty) ? qtm : standing_qty;
        int32_t new_qty = standing_qty - matched_qty;

        // Update standing order
        if (new_qty <= 0) {
            // Remove entire order
            for (int c = 0; c < ORDER_COLS; c++) {
                orderside_out[best_idx * ORDER_COLS + c] = EMPTY_SLOT;
            }
        } else {
            orderside_out[best_idx * ORDER_COLS + COL_QTY] = new_qty;
        }

        // Record trade — find first empty slot
        for (int t = 0; t < n_trades; t++) {
            if (trade_out[t * TRADE_COLS + T_PASS_OID] == EMPTY_SLOT) {
                trade_out[t * TRADE_COLS + T_PRICE]    = best_price;
                trade_out[t * TRADE_COLS + T_QTY]      = -side * matched_qty;
                trade_out[t * TRADE_COLS + T_PASS_OID] = standing_oid;
                trade_out[t * TRADE_COLS + T_AGR_OID]  = agr_oid;
                trade_out[t * TRADE_COLS + T_TIME]     = time_s;
                trade_out[t * TRADE_COLS + T_TNS]      = time_ns;
                trade_out[t * TRADE_COLS + T_PASS_TID] = standing_tid;
                trade_out[t * TRADE_COLS + T_AGR_TID]  = agr_tid;
                break;
            }
        }

        qtm -= matched_qty;
    }

    qtm_out[0] = qtm;
}


// ─── XLA FFI Handler ─────────────────────────────────────────

ffi::Error MatchOrdersImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> orderside_in,
    ffi::Buffer<ffi::DataType::S32> trade_in,
    ffi::Buffer<ffi::DataType::S32> incoming,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> orderside_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> trade_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> qtm_out,
    int32_t n_orders,
    int32_t n_trades,
    int32_t is_bid
) {
    // Launch with 32 threads: thread 0 does matching, threads 0-31 help copy data
    match_orders_kernel<<<1, 32, 0, stream>>>(
        orderside_in.typed_data(),
        trade_in.typed_data(),
        incoming.typed_data(),
        orderside_out->typed_data(),
        trade_out->typed_data(),
        qtm_out->typed_data(),
        n_orders,
        n_trades,
        is_bid
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// Register via XLA FFI symbol export (loaded by ctypes)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaMatchOrders,
    MatchOrdersImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // orderside_in
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // trade_in
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // incoming (7 scalars packed)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // orderside_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // trade_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // qtm_out
        .Attr<int32_t>("n_orders")
        .Attr<int32_t>("n_trades")
        .Attr<int32_t>("is_bid")
);
