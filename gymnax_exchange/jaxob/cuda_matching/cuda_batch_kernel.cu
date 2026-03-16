// Batched CUDA Kernel: processes an ENTIRE message sequence in one launch.
//
// Replaces scan_through_entire_array (lax.scan over cond_type_side).
// One kernel launch processes n_msgs messages sequentially per env.
// For multi-env: grid=(n_envs,), each block handles one independent book.
//
// Eliminates:
//   - 500× FFI dispatch overhead (one launch instead of 500)
//   - lax.switch 5-branch overhead (real if/else in CUDA)
//   - vmap overhead (explicit batch dimension via grid)
//
// Supports: GENERAL_EXCHANGE mode, IOC type4, INCLUDE_INITS cancel mode.

#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

constexpr int ORDER_COLS = 6;
constexpr int TRADE_COLS = 8;
constexpr int MSG_COLS = 8;
constexpr int32_t EMPTY = -1;
constexpr int32_t MAXINT = 2147483647;

// Message columns: [type, side, qty, price, oid, tid, time_s, time_ns]
constexpr int M_TYPE = 0, M_SIDE = 1, M_QTY = 2, M_PRICE = 3;
constexpr int M_OID = 4, M_TID = 5, M_TIME = 6, M_TNS = 7;

// Order columns: [price, qty, oid, tid, time_s, time_ns]
constexpr int O_PRICE = 0, O_QTY = 1, O_OID = 2, O_TID = 3, O_TIME = 4, O_TNS = 5;

// Trade columns: [price, qty, pass_oid, agr_oid, time_s, time_ns, pass_tid, agr_tid]
constexpr int T_PRICE = 0, T_QTY = 1, T_POID = 2, T_AOID = 3;
constexpr int T_TIME = 4, T_TNS = 5, T_PTID = 6, T_ATID = 7;


// ─── Device helper: matching loop ────────────────────────────
// Matches incoming order against standing orders on one side.
// Returns remaining qtm (can be negative, matching JAX semantics).
__device__ int32_t match_against_orders(
    int32_t* side_arr,          // the opposing side array (modified in place)
    int32_t* trade_arr,
    int32_t qtm,
    int32_t price,
    int32_t agr_oid,
    int32_t time_s,
    int32_t time_ns,
    int32_t agr_tid,
    int32_t side,               // +1 or -1 (for trade qty sign)
    int32_t is_bid,             // 1 = incoming bid vs asks, 0 = incoming ask vs bids
    int32_t n_orders,
    int32_t n_trades
) {
    for (int iter = 0; iter < n_orders; iter++) {
        if (qtm <= 0) break;

        int best_idx = -1;
        int32_t best_price = is_bid ? MAXINT : -1;
        int32_t best_ts = MAXINT, best_tns = MAXINT;

        for (int i = 0; i < n_orders; i++) {
            int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
            int32_t q = side_arr[i * ORDER_COLS + O_QTY];
            if (p == EMPTY || q <= 0) continue;

            if (is_bid) {
                if (p > price) continue;
                int32_t ts = side_arr[i * ORDER_COLS + O_TIME];
                int32_t tns = side_arr[i * ORDER_COLS + O_TNS];
                if (p < best_price ||
                    (p == best_price && ts < best_ts) ||
                    (p == best_price && ts == best_ts && tns < best_tns)) {
                    best_price = p; best_ts = ts; best_tns = tns; best_idx = i;
                }
            } else {
                if (p < price) continue;
                int32_t ts = side_arr[i * ORDER_COLS + O_TIME];
                int32_t tns = side_arr[i * ORDER_COLS + O_TNS];
                if (p > best_price ||
                    (p == best_price && ts < best_ts) ||
                    (p == best_price && ts == best_ts && tns < best_tns)) {
                    best_price = p; best_ts = ts; best_tns = tns; best_idx = i;
                }
            }
        }

        if (best_idx == -1) break;

        int32_t sq = side_arr[best_idx * ORDER_COLS + O_QTY];
        int32_t s_oid = side_arr[best_idx * ORDER_COLS + O_OID];
        int32_t s_tid = side_arr[best_idx * ORDER_COLS + O_TID];
        int32_t diff = sq - qtm;
        int32_t new_qty = (diff > 0) ? diff : 0;
        int32_t matched = sq - new_qty;

        if (new_qty <= 0) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[best_idx * ORDER_COLS + c] = EMPTY;
        } else {
            side_arr[best_idx * ORDER_COLS + O_QTY] = new_qty;
        }

        // Record trade (find by T_PRICE == -1)
        for (int t = 0; t < n_trades; t++) {
            if (trade_arr[t * TRADE_COLS + T_PRICE] == EMPTY) {
                trade_arr[t * TRADE_COLS + T_PRICE] = best_price;
                trade_arr[t * TRADE_COLS + T_QTY]   = -side * matched;
                trade_arr[t * TRADE_COLS + T_POID]  = s_oid;
                trade_arr[t * TRADE_COLS + T_AOID]  = agr_oid;
                trade_arr[t * TRADE_COLS + T_TIME]  = time_s;
                trade_arr[t * TRADE_COLS + T_TNS]   = time_ns;
                trade_arr[t * TRADE_COLS + T_PTID]  = s_tid;
                trade_arr[t * TRADE_COLS + T_ATID]  = agr_tid;
                break;
            }
        }

        qtm -= sq;  // subtract full standing qty (can go negative)
    }
    return qtm;
}


// ─── Device helper: add order (matches JAX add_order exactly) ──
// JAX finds empty slot via jnp.where(orderside==-1) which checks
// ANY column for -1 (not just price). qty is capped to max(0, qty).
__device__ void add_order(
    int32_t* side_arr,
    int32_t price, int32_t qty, int32_t oid, int32_t tid,
    int32_t time_s, int32_t time_ns,
    int32_t n_orders
) {
    // Find first row where ANY column is -1 (matches JAX's jnp.where)
    int empty_idx = -1;
    for (int i = 0; i < n_orders && empty_idx == -1; i++) {
        for (int c = 0; c < ORDER_COLS; c++) {
            if (side_arr[i * ORDER_COLS + c] == EMPTY) {
                empty_idx = i;
                break;
            }
        }
    }
    // Fallback: overwrite last row (JAX uses fill_value=-1 → Python -1 index)
    if (empty_idx == -1) empty_idx = n_orders - 1;

    int32_t safe_qty = (qty > 0) ? qty : 0;
    side_arr[empty_idx * ORDER_COLS + O_PRICE] = price;
    side_arr[empty_idx * ORDER_COLS + O_QTY]   = safe_qty;
    side_arr[empty_idx * ORDER_COLS + O_OID]   = oid;
    side_arr[empty_idx * ORDER_COLS + O_TID]   = tid;
    side_arr[empty_idx * ORDER_COLS + O_TIME]  = time_s;
    side_arr[empty_idx * ORDER_COLS + O_TNS]   = time_ns;
}

// ─── Device helper: removeZeroNegQuant (matches JAX exactly) ──
// Clears ALL rows where qty (column 1) <= 0 to all -1.
__device__ void removeZeroNegQuant(int32_t* side_arr, int32_t n_orders) {
    for (int i = 0; i < n_orders; i++) {
        if (side_arr[i * ORDER_COLS + O_QTY] <= 0) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[i * ORDER_COLS + c] = EMPTY;
        }
    }
}

// ─── Device helper: check_book_fill (matches JAX bid_lim/ask_lim) ──
// When book is full (no empty slots), remove worst price to make room.
// For bids: worst = lowest price. For asks: worst = highest price.
// Runs BEFORE add_order.
__device__ void check_book_fill_bids(int32_t* side_arr, int32_t n_orders) {
    bool full = true;
    for (int i = 0; i < n_orders; i++) {
        if (side_arr[i * ORDER_COLS + O_PRICE] < 0) { full = false; break; }
    }
    if (!full) return;

    int32_t worst = MAXINT;
    for (int i = 0; i < n_orders; i++) {
        int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
        if (p < worst) worst = p;
    }
    for (int i = 0; i < n_orders; i++) {
        if (side_arr[i * ORDER_COLS + O_PRICE] == worst) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[i * ORDER_COLS + c] = EMPTY;
        }
    }
}

__device__ void check_book_fill_asks(int32_t* side_arr, int32_t n_orders) {
    bool full = true;
    for (int i = 0; i < n_orders; i++) {
        if (side_arr[i * ORDER_COLS + O_PRICE] < 0) { full = false; break; }
    }
    if (!full) return;

    int32_t worst = -1;
    for (int i = 0; i < n_orders; i++) {
        int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
        if (p > worst) worst = p;
    }
    for (int i = 0; i < n_orders; i++) {
        if (side_arr[i * ORDER_COLS + O_PRICE] == worst) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[i * ORDER_COLS + c] = EMPTY;
        }
    }
}


// ─── Device helper: cancel order (INCLUDE_INITS mode) ────────
__device__ void cancel_order(
    int32_t* side_arr,
    int32_t price, int32_t qty, int32_t oid,
    int32_t n_orders,
    int32_t init_id    // cfg.init_id, default -2
) {
    // Try exact OID match first
    int idx = -1;
    for (int i = 0; i < n_orders; i++) {
        if (side_arr[i * ORDER_COLS + O_OID] == oid) {
            idx = i;
            break;
        }
    }

    // Fallback: init order match (price + init_id range + qty check)
    if (idx == -1) {
        int book_depth = 10;  // default
        for (int i = 0; i < n_orders; i++) {
            int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
            int32_t o = side_arr[i * ORDER_COLS + O_OID];
            int32_t q = side_arr[i * ORDER_COLS + O_QTY];
            if (p == price && o <= init_id && o >= init_id - (book_depth * 2) && q >= qty) {
                idx = i;
                break;
            }
        }
    }

    if (idx == -1) return;  // Order not found, skip

    side_arr[idx * ORDER_COLS + O_QTY] -= qty;
    // Remove if qty <= 0
    if (side_arr[idx * ORDER_COLS + O_QTY] <= 0) {
        for (int c = 0; c < ORDER_COLS; c++)
            side_arr[idx * ORDER_COLS + c] = EMPTY;
    }
}


// ─── Main batched kernel ─────────────────────────────────────
__global__ void batch_process_messages_kernel(
    const int32_t* __restrict__ asks_in,     // (n_envs, n_orders, 6) or (n_orders, 6)
    const int32_t* __restrict__ bids_in,
    const int32_t* __restrict__ trades_in,   // (n_envs, n_trades, 8) or (n_trades, 8)
    const int32_t* __restrict__ msgs,        // (n_envs, n_msgs, 8) or (n_msgs, 8)
    int32_t* __restrict__ asks_out,
    int32_t* __restrict__ bids_out,
    int32_t* __restrict__ trades_out,
    int32_t n_orders,
    int32_t n_trades,
    int32_t n_msgs,
    int32_t init_id      // cfg.init_id for cancel matching
) {
    int env_id = blockIdx.x;

    // Per-env offsets
    int ask_offset = env_id * n_orders * ORDER_COLS;
    int bid_offset = env_id * n_orders * ORDER_COLS;
    int trade_offset = env_id * n_trades * TRADE_COLS;
    int msg_offset = env_id * n_msgs * MSG_COLS;

    int ask_size = n_orders * ORDER_COLS;
    int bid_size = n_orders * ORDER_COLS;
    int trade_size = n_trades * TRADE_COLS;

    // Parallel copy input → output
    for (int i = threadIdx.x; i < ask_size; i += blockDim.x)
        asks_out[ask_offset + i] = asks_in[ask_offset + i];
    for (int i = threadIdx.x; i < bid_size; i += blockDim.x)
        bids_out[bid_offset + i] = bids_in[bid_offset + i];
    for (int i = threadIdx.x; i < trade_size; i += blockDim.x)
        trades_out[trade_offset + i] = trades_in[trade_offset + i];
    __syncthreads();

    // Only thread 0 processes messages sequentially
    if (threadIdx.x != 0) return;

    int32_t* asks = asks_out + ask_offset;
    int32_t* bids = bids_out + bid_offset;
    int32_t* trades = trades_out + trade_offset;
    const int32_t* msg_base = msgs + msg_offset;

    for (int m = 0; m < n_msgs; m++) {
        const int32_t* msg = msg_base + m * MSG_COLS;
        int32_t type    = msg[M_TYPE];
        int32_t side    = msg[M_SIDE];
        int32_t qty     = msg[M_QTY];
        int32_t price   = msg[M_PRICE];
        int32_t oid     = msg[M_OID];
        int32_t tid     = msg[M_TID];
        int32_t time_s  = msg[M_TIME];
        int32_t time_ns = msg[M_TNS];

        // Flip side for type 4 (match/execution orders)
        int32_t eff_side = (type == 4) ? -side : side;

        if ((type == 0 && side == 0) || type == 5 || type == 6 || type == 7) {
            // Noop / hidden / auction / halt — skip
            continue;
        }

        if ((type == 1 || type == 4) && eff_side == -1) {
            // ASK LIMIT (ask_lim): match against bids, add to asks
            int32_t qtm = match_against_orders(
                bids, trades, qty, price, oid, time_s, time_ns, tid,
                eff_side, 0 /*is_bid=false: incoming ask vs bids*/,
                n_orders, n_trades
            );
            if (type != 4) {
                // Full ask_lim flow: check_book_fill → add_order → cleanup
                check_book_fill_asks(asks, n_orders);
                add_order(asks, price, qtm, oid, tid, time_s, time_ns, n_orders);
                removeZeroNegQuant(asks, n_orders);
            }
            // type 4 IOC: match only, no add (JAX restores pre-add state)
        }
        else if ((type == 1 || type == 4) && eff_side == 1) {
            // BID LIMIT (bid_lim): match against asks, add to bids
            int32_t qtm = match_against_orders(
                asks, trades, qty, price, oid, time_s, time_ns, tid,
                eff_side, 1 /*is_bid=true: incoming bid vs asks*/,
                n_orders, n_trades
            );
            if (type != 4) {
                // Full bid_lim flow: check_book_fill → add_order → cleanup
                check_book_fill_bids(bids, n_orders);
                add_order(bids, price, qtm, oid, tid, time_s, time_ns, n_orders);
                removeZeroNegQuant(bids, n_orders);
            }
        }
        else if ((type == 2 || type == 3) && side == -1) {
            // ASK CANCEL
            cancel_order(asks, price, qty, oid, n_orders, init_id);
        }
        else if ((type == 2 || type == 3) && side == 1) {
            // BID CANCEL
            cancel_order(bids, price, qty, oid, n_orders, init_id);
        }
        // else: unrecognized → skip
    }
}


// ─── XLA FFI Handler ─────────────────────────────────────────
ffi::Error BatchProcessMessagesImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> asks_in,
    ffi::Buffer<ffi::DataType::S32> bids_in,
    ffi::Buffer<ffi::DataType::S32> trades_in,
    ffi::Buffer<ffi::DataType::S32> msgs,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> asks_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> bids_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> trades_out,
    int32_t n_envs,
    int32_t n_orders,
    int32_t n_trades,
    int32_t n_msgs,
    int32_t init_id
) {
    batch_process_messages_kernel<<<n_envs, 32, 0, stream>>>(
        asks_in.typed_data(),
        bids_in.typed_data(),
        trades_in.typed_data(),
        msgs.typed_data(),
        asks_out->typed_data(),
        bids_out->typed_data(),
        trades_out->typed_data(),
        n_orders, n_trades, n_msgs, init_id
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CudaBatchProcessMessages,
    BatchProcessMessagesImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // asks_in
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // bids_in
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // trades_in
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // msgs
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // asks_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // bids_out
        .Ret<ffi::Buffer<ffi::DataType::S32>>()    // trades_out
        .Attr<int32_t>("n_envs")
        .Attr<int32_t>("n_orders")
        .Attr<int32_t>("n_trades")
        .Attr<int32_t>("n_msgs")
        .Attr<int32_t>("init_id")
);
