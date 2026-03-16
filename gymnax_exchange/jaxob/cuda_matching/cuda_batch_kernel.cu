// Batched CUDA Kernel V3: WARP-PARALLEL message processing.
//
// All 32 threads in a warp cooperate on every operation:
//   - Finding best price: parallel scan + warp shuffle reduction
//   - Finding empty slots: parallel scan + warp ballot
//   - removeZeroNegQuant: parallel cleanup
//   - Data copy: strided parallel copy
//
// grid=(n_envs,), block=(32,) — one warp per order book.
// NO idle threads. Every thread participates in every message.

#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFF;

constexpr int ORDER_COLS = 6;
constexpr int TRADE_COLS = 8;
constexpr int MSG_COLS = 8;
constexpr int32_t EMPTY = -1;
constexpr int32_t MAXINT = 2147483647;

constexpr int M_TYPE = 0, M_SIDE = 1, M_QTY = 2, M_PRICE = 3;
constexpr int M_OID = 4, M_TID = 5, M_TIME = 6, M_TNS = 7;

constexpr int O_PRICE = 0, O_QTY = 1, O_OID = 2, O_TID = 3, O_TIME = 4, O_TNS = 5;

constexpr int T_PRICE = 0, T_QTY = 1, T_POID = 2, T_AOID = 3;
constexpr int T_TIME = 4, T_TNS = 5, T_PTID = 6, T_ATID = 7;


// ─── Warp-level reduction: find best order index ─────────────
// Each thread has a candidate (local_idx, local_price, local_ts, local_tns).
// Reduce across warp to find the global best.
// For bids (is_bid=1): best = lowest price.
// For asks (is_bid=0): best = highest price.
// Tie-break by time_s, then time_ns, then index (lowest wins).
__device__ int warp_reduce_best(
    int local_idx,
    int32_t local_price,
    int32_t local_ts,
    int32_t local_tns,
    int is_bid
) {
    // Pack: each thread has (price, ts, tns, idx). We compare pairwise.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other_idx = __shfl_down_sync(FULL_MASK, local_idx, offset);
        int32_t other_price = __shfl_down_sync(FULL_MASK, local_price, offset);
        int32_t other_ts = __shfl_down_sync(FULL_MASK, local_ts, offset);
        int32_t other_tns = __shfl_down_sync(FULL_MASK, local_tns, offset);

        // Determine if other is better
        bool other_wins;
        if (local_idx == -1 && other_idx == -1) {
            other_wins = false;
        } else if (local_idx == -1) {
            other_wins = true;
        } else if (other_idx == -1) {
            other_wins = false;
        } else if (is_bid) {
            // Best bid match = lowest ask price
            other_wins = (other_price < local_price) ||
                         (other_price == local_price && other_ts < local_ts) ||
                         (other_price == local_price && other_ts == local_ts && other_tns < local_tns);
        } else {
            // Best ask match = highest bid price
            other_wins = (other_price > local_price) ||
                         (other_price == local_price && other_ts < local_ts) ||
                         (other_price == local_price && other_ts == local_ts && other_tns < local_tns);
        }

        if (other_wins) {
            local_idx = other_idx;
            local_price = other_price;
            local_ts = other_ts;
            local_tns = other_tns;
        }
    }
    // Broadcast winner from lane 0
    return __shfl_sync(FULL_MASK, local_idx, 0);
}


// ─── Warp-level: find first index where condition is true ────
// Returns the smallest index across all threads, or fallback if none.
__device__ int warp_find_first(int local_found_idx, int fallback) {
    // Each thread has its local best found index, or n_orders if not found
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(FULL_MASK, local_found_idx, offset);
        if (other < local_found_idx) local_found_idx = other;
    }
    int result = __shfl_sync(FULL_MASK, local_found_idx, 0);
    return (result >= 1000000) ? fallback : result;  // sentinel for "not found"
}


// ─── Warp-parallel matching ──────────────────────────────────
__device__ int32_t match_against_orders_warp(
    int tid,                    // threadIdx.x (0-31)
    int32_t* side_arr,
    int32_t* trade_arr,
    int32_t qtm,
    int32_t price,
    int32_t agr_oid,
    int32_t time_s,
    int32_t time_ns,
    int32_t agr_tid,
    int32_t side,
    int32_t is_bid,
    int32_t n_orders,
    int32_t n_trades
) {
    for (int iter = 0; iter < n_orders; iter++) {
        if (qtm <= 0) break;

        // ── Parallel best-price scan ──
        // Each thread scans its stripe of the order array
        int my_best_idx = -1;
        int32_t my_best_price = is_bid ? MAXINT : -1;
        int32_t my_best_ts = MAXINT, my_best_tns = MAXINT;

        for (int i = tid; i < n_orders; i += WARP_SIZE) {
            int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
            int32_t q = side_arr[i * ORDER_COLS + O_QTY];
            if (p == EMPTY || q <= 0) continue;

            if (is_bid) {
                if (p > price) continue;
                int32_t ts = side_arr[i * ORDER_COLS + O_TIME];
                int32_t tns = side_arr[i * ORDER_COLS + O_TNS];
                if (p < my_best_price ||
                    (p == my_best_price && ts < my_best_ts) ||
                    (p == my_best_price && ts == my_best_ts && tns < my_best_tns)) {
                    my_best_price = p; my_best_ts = ts; my_best_tns = tns; my_best_idx = i;
                }
            } else {
                if (p < price) continue;
                int32_t ts = side_arr[i * ORDER_COLS + O_TIME];
                int32_t tns = side_arr[i * ORDER_COLS + O_TNS];
                if (p > my_best_price ||
                    (p == my_best_price && ts < my_best_ts) ||
                    (p == my_best_price && ts == my_best_ts && tns < my_best_tns)) {
                    my_best_price = p; my_best_ts = ts; my_best_tns = tns; my_best_idx = i;
                }
            }
        }

        // Warp reduce to find global best
        int best_idx = warp_reduce_best(my_best_idx, my_best_price, my_best_ts, my_best_tns, is_bid);

        if (best_idx == -1) break;

        // ── Execute match (thread 0 reads values, broadcasts to all) ──
        int32_t sq, s_oid, s_tid_val, best_p;
        if (tid == 0) {
            sq = side_arr[best_idx * ORDER_COLS + O_QTY];
            s_oid = side_arr[best_idx * ORDER_COLS + O_OID];
            s_tid_val = side_arr[best_idx * ORDER_COLS + O_TID];
            best_p = side_arr[best_idx * ORDER_COLS + O_PRICE];

            int32_t diff = sq - qtm;
            int32_t new_qty = (diff > 0) ? diff : 0;
            int32_t matched = sq - new_qty;

            if (new_qty <= 0) {
                for (int c = 0; c < ORDER_COLS; c++)
                    side_arr[best_idx * ORDER_COLS + c] = EMPTY;
            } else {
                side_arr[best_idx * ORDER_COLS + O_QTY] = new_qty;
            }

            // Record trade — find first empty slot
            int trade_idx = n_trades - 1;
            for (int t = 0; t < n_trades; t++) {
                if (trade_arr[t * TRADE_COLS + T_PRICE] == EMPTY) {
                    trade_idx = t;
                    break;
                }
            }
            trade_arr[trade_idx * TRADE_COLS + T_PRICE] = best_p;
            trade_arr[trade_idx * TRADE_COLS + T_QTY]   = -side * matched;
            trade_arr[trade_idx * TRADE_COLS + T_POID]  = s_oid;
            trade_arr[trade_idx * TRADE_COLS + T_AOID]  = agr_oid;
            trade_arr[trade_idx * TRADE_COLS + T_TIME]  = time_s;
            trade_arr[trade_idx * TRADE_COLS + T_TNS]   = time_ns;
            trade_arr[trade_idx * TRADE_COLS + T_PTID]  = s_tid_val;
            trade_arr[trade_idx * TRADE_COLS + T_ATID]  = agr_tid;

            qtm -= sq;
        }
        // Broadcast updated qtm from lane 0 to all lanes
        qtm = __shfl_sync(FULL_MASK, qtm, 0);
    }
    return qtm;
}


// ─── Warp-parallel add_order ─────────────────────────────────
__device__ void add_order_warp(
    int tid,
    int32_t* side_arr,
    int32_t price, int32_t qty, int32_t oid, int32_t tid_val,
    int32_t time_s, int32_t time_ns,
    int32_t n_orders
) {
    // Parallel scan: each thread checks its stripe for ANY -1 column
    int my_found = 1000000;  // sentinel = not found
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        for (int c = 0; c < ORDER_COLS; c++) {
            if (side_arr[i * ORDER_COLS + c] == EMPTY) {
                if (i < my_found) my_found = i;
                break;
            }
        }
    }

    int empty_idx = warp_find_first(my_found, n_orders - 1);

    // Thread 0 writes the order
    if (tid == 0) {
        int32_t safe_qty = (qty > 0) ? qty : 0;
        side_arr[empty_idx * ORDER_COLS + O_PRICE] = price;
        side_arr[empty_idx * ORDER_COLS + O_QTY]   = safe_qty;
        side_arr[empty_idx * ORDER_COLS + O_OID]   = oid;
        side_arr[empty_idx * ORDER_COLS + O_TID]   = tid_val;
        side_arr[empty_idx * ORDER_COLS + O_TIME]  = time_s;
        side_arr[empty_idx * ORDER_COLS + O_TNS]   = time_ns;
    }
}


// ─── Warp-parallel removeZeroNegQuant ────────────────────────
__device__ void removeZeroNegQuant_warp(int tid, int32_t* side_arr, int32_t n_orders) {
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        if (side_arr[i * ORDER_COLS + O_QTY] <= 0) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[i * ORDER_COLS + c] = EMPTY;
        }
    }
    __syncwarp(FULL_MASK);
}


// ─── Warp-parallel check_book_fill ───────────────────────────
__device__ void check_book_fill_bids_warp(int tid, int32_t* side_arr, int32_t n_orders) {
    // Parallel check: is book full? (all prices >= 0)
    int my_empty = 0;
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        if (side_arr[i * ORDER_COLS + O_PRICE] < 0) my_empty = 1;
    }
    // OR-reduce: if any thread found empty, book is not full
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        my_empty |= __shfl_down_sync(FULL_MASK, my_empty, offset);
    int not_full = __shfl_sync(FULL_MASK, my_empty, 0);
    if (not_full) return;

    // Parallel find worst bid (min price)
    int32_t my_min = MAXINT;
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
        if (p < my_min) my_min = p;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int32_t other = __shfl_down_sync(FULL_MASK, my_min, offset);
        if (other < my_min) my_min = other;
    }
    int32_t worst = __shfl_sync(FULL_MASK, my_min, 0);

    // Parallel remove all at worst price
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        if (side_arr[i * ORDER_COLS + O_PRICE] == worst) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[i * ORDER_COLS + c] = EMPTY;
        }
    }
    __syncwarp(FULL_MASK);
}

__device__ void check_book_fill_asks_warp(int tid, int32_t* side_arr, int32_t n_orders) {
    int my_empty = 0;
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        if (side_arr[i * ORDER_COLS + O_PRICE] < 0) my_empty = 1;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        my_empty |= __shfl_down_sync(FULL_MASK, my_empty, offset);
    int not_full = __shfl_sync(FULL_MASK, my_empty, 0);
    if (not_full) return;

    int32_t my_max = -1;
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
        if (p > my_max) my_max = p;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int32_t other = __shfl_down_sync(FULL_MASK, my_max, offset);
        if (other > my_max) my_max = other;
    }
    int32_t worst = __shfl_sync(FULL_MASK, my_max, 0);

    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        if (side_arr[i * ORDER_COLS + O_PRICE] == worst) {
            for (int c = 0; c < ORDER_COLS; c++)
                side_arr[i * ORDER_COLS + c] = EMPTY;
        }
    }
    __syncwarp(FULL_MASK);
}


// ─── Warp-parallel cancel_order ──────────────────────────────
__device__ void cancel_order_warp(
    int tid,
    int32_t* side_arr,
    int32_t price, int32_t qty, int32_t oid,
    int32_t n_orders,
    int32_t init_id
) {
    // Parallel OID search
    int my_found = 1000000;
    for (int i = tid; i < n_orders; i += WARP_SIZE) {
        if (side_arr[i * ORDER_COLS + O_OID] == oid) {
            if (i < my_found) my_found = i;
        }
    }
    int idx = warp_find_first(my_found, -1);

    // Fallback: init order match (parallel)
    if (idx == -1) {
        int book_depth = 10;
        my_found = 1000000;
        for (int i = tid; i < n_orders; i += WARP_SIZE) {
            int32_t p = side_arr[i * ORDER_COLS + O_PRICE];
            int32_t o = side_arr[i * ORDER_COLS + O_OID];
            int32_t q = side_arr[i * ORDER_COLS + O_QTY];
            if (p == price && o <= init_id && o >= init_id - (book_depth * 2) && q >= qty) {
                if (i < my_found) my_found = i;
            }
        }
        idx = warp_find_first(my_found, -1);
    }

    // JAX fallback: not found → last row
    if (idx == -1) idx = n_orders - 1;

    // Thread 0 modifies
    if (tid == 0) {
        side_arr[idx * ORDER_COLS + O_QTY] -= qty;
    }
    __syncwarp(FULL_MASK);

    // Parallel removeZeroNegQuant
    removeZeroNegQuant_warp(tid, side_arr, n_orders);
}


// ─── Main kernel: all 32 threads participate ─────────────────
__global__ void batch_process_messages_kernel(
    const int32_t* __restrict__ asks_in,
    const int32_t* __restrict__ bids_in,
    const int32_t* __restrict__ trades_in,
    const int32_t* __restrict__ msgs,
    int32_t* __restrict__ asks_out,
    int32_t* __restrict__ bids_out,
    int32_t* __restrict__ trades_out,
    int32_t n_orders,
    int32_t n_trades,
    int32_t n_msgs,
    int32_t init_id
) {
    int env_id = blockIdx.x;
    int tid = threadIdx.x;

    int ask_offset = env_id * n_orders * ORDER_COLS;
    int bid_offset = env_id * n_orders * ORDER_COLS;
    int trade_offset = env_id * n_trades * TRADE_COLS;
    int msg_offset = env_id * n_msgs * MSG_COLS;

    int ask_size = n_orders * ORDER_COLS;
    int bid_size = n_orders * ORDER_COLS;
    int trade_size = n_trades * TRADE_COLS;

    // Parallel copy (all 32 threads)
    for (int i = tid; i < ask_size; i += WARP_SIZE)
        asks_out[ask_offset + i] = asks_in[ask_offset + i];
    for (int i = tid; i < bid_size; i += WARP_SIZE)
        bids_out[bid_offset + i] = bids_in[bid_offset + i];
    for (int i = tid; i < trade_size; i += WARP_SIZE)
        trades_out[trade_offset + i] = trades_in[trade_offset + i];
    __syncwarp(FULL_MASK);

    int32_t* asks = asks_out + ask_offset;
    int32_t* bids = bids_out + bid_offset;
    int32_t* trades = trades_out + trade_offset;
    const int32_t* msg_base = msgs + msg_offset;

    for (int m = 0; m < n_msgs; m++) {
        // All threads read message (broadcast via L1 cache)
        const int32_t* msg = msg_base + m * MSG_COLS;
        int32_t type    = msg[M_TYPE];
        int32_t side    = msg[M_SIDE];
        int32_t qty     = msg[M_QTY];
        int32_t price   = msg[M_PRICE];
        int32_t oid     = msg[M_OID];
        int32_t tid_val = msg[M_TID];
        int32_t time_s  = msg[M_TIME];
        int32_t time_ns = msg[M_TNS];

        int32_t eff_side = (type == 4) ? -side : side;

        if ((type == 0 && side == 0) || type == 5 || type == 6 || type == 7) {
            continue;
        }

        if ((type == 1 || type == 4) && eff_side == -1) {
            int32_t qtm = match_against_orders_warp(
                tid, bids, trades, qty, price, oid, time_s, time_ns, tid_val,
                eff_side, 0, n_orders, n_trades
            );
            if (type != 4) {
                check_book_fill_asks_warp(tid, asks, n_orders);
                add_order_warp(tid, asks, price, qtm, oid, tid_val, time_s, time_ns, n_orders);
                removeZeroNegQuant_warp(tid, asks, n_orders);
            }
        }
        else if ((type == 1 || type == 4) && eff_side == 1) {
            int32_t qtm = match_against_orders_warp(
                tid, asks, trades, qty, price, oid, time_s, time_ns, tid_val,
                eff_side, 1, n_orders, n_trades
            );
            if (type != 4) {
                check_book_fill_bids_warp(tid, bids, n_orders);
                add_order_warp(tid, bids, price, qtm, oid, tid_val, time_s, time_ns, n_orders);
                removeZeroNegQuant_warp(tid, bids, n_orders);
            }
        }
        else if ((type == 2 || type == 3) && side == -1) {
            cancel_order_warp(tid, asks, price, qty, oid, n_orders, init_id);
        }
        else if ((type == 2 || type == 3) && side == 1) {
            cancel_order_warp(tid, bids, price, qty, oid, n_orders, init_id);
        }
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
    batch_process_messages_kernel<<<n_envs, WARP_SIZE, 0, stream>>>(
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
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
        .Attr<int32_t>("n_envs")
        .Attr<int32_t>("n_orders")
        .Attr<int32_t>("n_trades")
        .Attr<int32_t>("n_msgs")
        .Attr<int32_t>("init_id")
);
