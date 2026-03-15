# PIYUSH C++20 Matching Engine - Source Code Analysis for CUDA Port

**Date**: 2026-03-15
**Source**: https://github.com/PIYUSH-KUMAR1809/order-matching-engine
**Performance**: ~160M orders/sec on Apple M1 Pro (multi-threaded, sharded)

---

## 1. Architecture Overview

```
                    Multi-threaded C++ Architecture
                    ================================

  Producer Threads          RingBuffer (per-shard)         Worker Threads
  ┌─────────┐              ┌─────────────────┐            ┌──────────────────┐
  │ Thread 0 │──batch 256──>│ Shard 0 Queue   │──pop 256──>│ Worker 0         │
  │ Thread 1 │──batch 256──>│ Shard 1 Queue   │──pop 256──>│ Worker 1         │
  │ Thread N │──batch 256──>│ Shard N Queue   │──pop 256──>│ Worker N         │
  └─────────┘              └─────────────────┘            │  ├─ OrderBook[]   │
                                                           │  ├─ MatchStrategy │
                                                           │  └─ TradeBuffer  │
                                                           └──────────────────┘

  For CUDA port: We ONLY need the single-threaded core:
  OrderBook + MatchingStrategy + PriceBitset
  (One thread processes one symbol sequentially)
```

## 2. Data Structures

### 2.1 Order (Order.hpp)

```cpp
// 40 bytes per order (packed)
struct Order {
  OrderId id;            // uint64_t - 8 bytes
  Price price;           // int64_t  - 8 bytes
  uint64_t clientOrderId;// uint64_t - 8 bytes
  int32_t symbolId;      // int32_t  - 4 bytes
  Quantity quantity;      // uint32_t - 4 bytes
  OrderSide side;        // uint8_t  - 1 byte (Buy=0, Sell=1)
  OrderType type;        // uint8_t  - 1 byte (Limit=0, Market=1)
  bool active;           // 1 byte   - tombstone flag
};
```

**CUDA mapping**: This fits in registers or shared memory. For a Triton/CUDA kernel processing
one message at a time, the incoming order is just a few registers.

### 2.2 PriceLevel (OrderBook.hpp)

```cpp
struct PriceLevel {
  std::pmr::vector<Order> orders;  // Flat contiguous vector of orders at this price
  int32_t activeCount = 0;         // Number of non-cancelled orders
  int32_t headIndex = 0;           // First potentially active order (skip tombstones)
};
```

**Key insight**: Orders at a price level are stored in a **flat contiguous vector** (not a linked list).
- Insertion: O(1) amortized - `push_back`
- Time priority: Maintained by insertion order (FIFO within the vector)
- Cancellation: Tombstone (`active = false`), no removal/reallocation
- `headIndex` optimization: Skips leading tombstones without scanning

**CUDA mapping**: Each price level maps to a contiguous segment of shared memory or global memory.
The `headIndex` skip is trivially implementable.

### 2.3 PriceBitset (Bitset.hpp) - THE KEY DATA STRUCTURE

```cpp
class PriceBitset {
  std::vector<uint64_t> data_;  // ceil(MAX_PRICE/64) = ceil(100000/64) = 1563 words
  size_t size_;                 // MAX_PRICE = 100000

  // Set bit for a price level that has orders
  void set(size_t index) {
    data_[index / 64] |= (1ULL << (index % 64));
  }

  // Clear bit when price level becomes empty
  void clear(size_t index) {
    data_[index / 64] &= ~(1ULL << (index % 64));
  }

  // Find lowest set bit >= start (for asks: find best ask)
  size_t findFirstSet(size_t start) const {
    size_t idx = start / 64;
    size_t bit = start % 64;
    // Mask out bits below 'start' in the first word
    uint64_t word = data_[idx] & (~0ULL << bit);
    if (word != 0) return idx * 64 + __builtin_ctzll(word);
    // Scan subsequent words
    for (idx++; idx < data_.size(); idx++)
      if (data_[idx] != 0) return idx * 64 + __builtin_ctzll(data_[idx]);
    return size_;  // No set bit found
  }

  // Find highest set bit <= start (for bids: find best bid)
  size_t findFirstSetDown(size_t start) const {
    size_t idx = start / 64;
    size_t bit = start % 64;
    uint64_t mask = (bit == 63) ? ~0ULL : ((1ULL << (bit + 1)) - 1);
    uint64_t word = data_[idx] & mask;
    if (word != 0) return idx * 64 + (63 - __builtin_clzll(word));
    for (size_t i = idx; i-- > 0;)
      if (data_[i] != 0) return i * 64 + (63 - __builtin_clzll(data_[i]));
    return size_;
  }
};
```

**Algorithm**:
- `findFirstSet` (used for asks): Scans forward through 64-bit words using `__builtin_ctzll`
  (count trailing zeros). Finds the lowest price with orders in O(MAX_PRICE/64) worst case.
- `findFirstSetDown` (used for bids): Scans backward using `__builtin_clzll`
  (count leading zeros). Finds the highest price with orders.

**Why it's fast on CPU**:
- `__builtin_ctzll` compiles to a single instruction (`TZCNT` on x86, `RBIT+CLZ` on ARM)
- Each 64-bit word covers 64 price levels → only ~1563 words for 100K prices
- In practice, best bid/ask is found in 1-2 word scans (prices cluster near spread)

### 2.4 OrderBook (OrderBook.hpp/cpp)

```cpp
class OrderBook {
  static constexpr int MAX_PRICE = 100000;  // Price range [0, 100000)

  // FLAT ARRAY indexed by price - THE critical design choice
  std::vector<PriceLevel> bids;  // bids[price] = PriceLevel at that price
  std::vector<PriceLevel> asks;  // asks[price] = PriceLevel at that price

  PriceBitset bidMask;  // Bit i set ↔ bids[i].activeCount > 0
  PriceBitset askMask;  // Bit i set ↔ asks[i].activeCount > 0

  Price bestBid = 0;    // Cached best bid price
  Price bestAsk = -1;   // Cached best ask price (-1 = no asks)

  // O(1) order lookup by ID for cancellation
  std::vector<OrderLocation> idToLocation;  // idToLocation[orderId] → {price, index}

  // PMR memory pool - 512MB pre-allocated
  std::vector<std::byte> buffer;                    // 512 * 1024 * 1024 bytes
  std::pmr::monotonic_buffer_resource pool;         // Arena allocator
};
```

**Architecture diagram**:
```
  Price:   0    1    2   ...  10000  10001  ...  99999
  bids: [ PL ] [PL ] [PL]   [ PL ]  [ PL ]     [ PL ]   ← 100K PriceLevels
  asks: [ PL ] [PL ] [PL]   [ PL ]  [ PL ]     [ PL ]   ← 100K PriceLevels

  bidMask: [word0: bits 0-63] [word1: bits 64-127] ... [word1562: bits 99968-99999]
  askMask: [word0: bits 0-63] [word1: bits 64-127] ... [word1562: bits 99968-99999]

  Lookup:  bids[price]  →  O(1) array index, no hash, no tree traversal
  Find best: bidMask.findFirstSetDown(MAX_PRICE) → scan ~1-3 words
```

**Why flat array is critical**:
- `bids[price]` is O(1) with perfect cache locality (price → array offset)
- No tree rebalancing (unlike `std::map`), no hash collisions
- 100K entries × ~sizeof(PriceLevel) is manageable in L2/L3 cache
- For LOBSTER data, price range is much narrower (~200 ticks), so only ~4 words in bitset matter

## 3. The Matching Algorithm Hot Path

### 3.1 StandardMatchingStrategy::match() - BUY side (simplified)

```cpp
void match(OrderBook& book, Order& incoming, std::vector<Trade>& trades) {
    // Market orders: set price to MAX to match anything
    if (incoming.type == OrderType::Market) {
        if (incoming.side == OrderSide::Buy)
            incoming.price = OrderBook::MAX_PRICE;
        else
            incoming.price = 0;
    }

    // === BUY ORDER: Match against asks (lowest first) ===
    if (incoming.side == OrderSide::Buy) {
        if (book.bestAsk == -1) {
            // No asks exist → place as resting order
            if (incoming.quantity > 0 && incoming.type != OrderType::Market)
                book.addOrder(incoming);
            return;
        }

        Price p = book.getBestAsk();  // Start at best (lowest) ask

        // OUTER LOOP: Walk price levels from best ask upward
        while (p < OrderBook::MAX_PRICE) {

            // BITSET SCAN: Skip empty price levels efficiently
            if (!book.askMask.test(p)) {
                p = (Price)book.askMask.findFirstSet(p);  // Jump to next non-empty
                if (p >= OrderBook::MAX_PRICE) break;
            }

            // LIMIT CHECK: Stop if ask price exceeds our limit
            if (p > incoming.price && incoming.type == OrderType::Limit) break;

            // INNER LOOP: Walk orders at this price level (time priority)
            auto& level = book.asks[p];
            if (level.activeCount > 0) {
                size_t size = level.orders.size();
                for (size_t i = level.headIndex; i < size; ++i) {
                    Order& bookOrder = level.orders[i];

                    // Skip tombstoned orders
                    if (!bookOrder.active) {
                        if (i == level.headIndex) level.headIndex++;
                        continue;
                    }

                    // FILL: min(incoming_qty, resting_qty)
                    Quantity qty = std::min(incoming.quantity, bookOrder.quantity);
                    trades.emplace_back(bookOrder.id, incoming.id,
                                       incoming.symbolId, bookOrder.price, qty);

                    bookOrder.quantity -= qty;
                    incoming.quantity -= qty;

                    // Resting order fully filled → tombstone it
                    if (bookOrder.quantity == 0) {
                        bookOrder.active = false;
                        level.activeCount--;
                        if (i == level.headIndex) level.headIndex++;

                        // Level exhausted → clear bitset, free memory
                        if (level.activeCount == 0) {
                            book.askMask.clear(p);
                            level.orders.clear();
                            level.headIndex = 0;
                            break;  // Exit inner loop
                        }
                    }
                    // Incoming fully filled → done
                    if (incoming.quantity == 0) break;
                }
            }

            if (incoming.quantity == 0) break;
            p++;  // Move to next price level

            // Update bestAsk cache
            if (p > book.bestAsk)
                book.bestAsk = (p < OrderBook::MAX_PRICE) ? p : -1;
        }

        // Refresh bestAsk from bitset
        if (book.askMask.findFirstSet(book.bestAsk) >= OrderBook::MAX_PRICE)
            book.bestAsk = -1;
    }

    // Unfilled remainder → place as resting order
    if (incoming.quantity > 0 && incoming.type != OrderType::Market)
        book.addOrder(incoming);
}
```

### 3.2 SELL side matching (mirrors buy, scans bids from highest down)

```cpp
    // === SELL ORDER: Match against bids (highest first) ===
    } else {
        if (book.bestBid == 0 && !book.bidMask.test(0)) {
            // No bids → place as resting
            book.addOrder(incoming);
            return;
        }

        Price p = book.getBestBid();  // Start at best (highest) bid

        while (p >= 0) {
            // BITSET SCAN: Skip empty price levels (scanning downward)
            if (!book.bidMask.test(p)) {
                if (p == 0) break;
                size_t next = book.bidMask.findFirstSetDown(p - 1);
                if (next >= OrderBook::MAX_PRICE) {
                    if (book.bidMask.test(0)) p = 0;
                    else break;
                } else {
                    p = (Price)next;
                }
                if (!book.bidMask.test(p)) break;
            }

            if (p < incoming.price && incoming.type == OrderType::Limit) break;

            // Same inner loop as buy side...
            auto& level = book.bids[p];
            // ... (identical matching logic)

            if (incoming.quantity == 0) break;
            if (p == 0) break;
            p--;
            book.bestBid = p;
        }

        // Refresh bestBid from bitset
        if (book.bestBid > 0 && !book.bidMask.test(book.bestBid)) {
            size_t next = book.bidMask.findFirstSetDown(book.bestBid);
            book.bestBid = (next >= OrderBook::MAX_PRICE) ? 0 : (Price)next;
        }
    }
```

### 3.3 addOrder (OrderBook.cpp) - Resting order placement

```cpp
void OrderBook::addOrder(const Order& order) {
    if (order.price < 0 || order.price >= MAX_PRICE) return;

    // Resize ID lookup if needed
    if (order.id >= idToLocation.size())
        idToLocation.resize(order.id * 2);

    bool isBid = (order.side == OrderSide::Buy);
    auto& levels = isBid ? bids : asks;
    auto& level = levels[order.price];

    // Record location for O(1) cancellation
    idToLocation[order.id] = {order.price, (int32_t)level.orders.size()};

    // Append to end (preserves time priority)
    level.orders.push_back(order);
    level.activeCount++;

    // Update bitset and best price cache
    if (isBid) {
        bidMask.set(order.price);
        if (order.price > bestBid) bestBid = order.price;
    } else {
        askMask.set(order.price);
        if (bestAsk == -1 || order.price < bestAsk) bestAsk = order.price;
    }
}
```

### 3.4 cancelOrder (OrderBook.cpp)

```cpp
void OrderBook::cancelOrder(OrderId orderId) {
    OrderLocation loc = idToLocation[orderId];    // O(1) lookup
    if (loc.price == -1) return;

    // Try bids first, then asks
    // Set active = false (tombstone), decrement activeCount
    // If level becomes empty: clear bitset, update bestBid/bestAsk
}
```

## 4. PMR Allocator Setup

```cpp
OrderBook::OrderBook()
    : buffer(static_cast<size_t>(512 * 1024 * 1024)),     // 512MB raw buffer
      pool(buffer.data(), buffer.size(),
           std::pmr::new_delete_resource()),                // Arena over the buffer
      bidMask(MAX_PRICE),
      askMask(MAX_PRICE) {

    idToLocation.resize(10000000);   // Pre-allocate 10M order lookup slots

    bids.reserve(MAX_PRICE);         // Pre-allocate 100K price level slots
    asks.reserve(MAX_PRICE);

    for (int i = 0; i < MAX_PRICE; ++i) {
        bids.emplace_back(&pool);    // Each PriceLevel's vector uses the arena
        asks.emplace_back(&pool);
    }
}
```

**How it works**:
- `std::pmr::monotonic_buffer_resource` is a bump allocator
- All `PriceLevel::orders` vectors allocate from this 512MB pool
- No `malloc`/`free` on the hot path — just pointer bumps
- `pool.release()` frees everything at once during `reset()`

**CUDA mapping**: Irrelevant. In CUDA, all memory is pre-allocated as arrays.
The PMR pattern is solving a C++ problem that doesn't exist in CUDA/Triton.

## 5. CUDA Port Mapping

### 5.1 Bitset → CUDA Warp-Level Operations

```
  CPU:  __builtin_ctzll(word)     →  CUDA: __ffsll(word) - 1   (find first set, 0-indexed)
  CPU:  __builtin_clzll(word)     →  CUDA: __clzll(word)       (count leading zeros)

  CPU bitset scan (1563 words):
    for (idx++; idx < data_.size(); idx++)
      if (data_[idx] != 0) return idx * 64 + __builtin_ctzll(data_[idx]);

  CUDA single-thread equivalent (identical logic):
    for (int idx = start_word + 1; idx < num_words; idx++)
      if (bitset[idx] != 0) return idx * 64 + (__ffsll(bitset[idx]) - 1);
```

**Warp-level acceleration** (32 threads cooperatively scan 32 words at once):
```
  // Each lane loads one word
  uint64_t my_word = bitset[warp_base + lane_id];
  // Ballot: which lanes have non-zero words?
  uint32_t nonempty = __ballot_sync(0xFFFFFFFF, my_word != 0);
  // Find first lane with data
  int first_lane = __ffs(nonempty) - 1;  // 0-31
  // That lane broadcasts its word
  uint64_t winner_word = __shfl_sync(0xFFFFFFFF, my_word, first_lane);
  // Compute the price
  int price = (warp_base + first_lane) * 64 + (__ffsll(winner_word) - 1);
```

This scans 32×64 = **2048 price levels per warp instruction**, vs 64 per CPU instruction.
For 100K prices, that's ~49 warp iterations worst case, ~1-2 in practice.

**BUT**: For your use case (Triton kernel, single-thread-per-book), the simple sequential
scan is actually better. The warp trick is only useful if you have spare threads.

### 5.2 Flat Vector → Shared Memory / Global Memory

```
  CPU layout:
    bids[100000] × PriceLevel { orders[dynamic], activeCount, headIndex }

  CUDA layout (fixed-size, no dynamic allocation):
    // Option A: Global memory (large books)
    int32_t  bid_active_count[MAX_PRICE];       // 400KB for 100K prices
    int32_t  bid_head_index[MAX_PRICE];          // 400KB
    Order    bid_orders[MAX_PRICE][MAX_DEPTH];   // Fixed max depth per level
    uint64_t bid_mask[NUM_WORDS];                // ~12.5KB for 100K prices

    // Option B: Shared memory (small price range, e.g. 200 ticks)
    __shared__ int32_t  bid_active[200];
    __shared__ int32_t  bid_head[200];
    __shared__ Order    bid_orders[200][32];      // Max 32 orders per level
    __shared__ uint64_t bid_mask[4];              // 200/64 = 4 words
```

**For LOBSTER data** (typical spread ~5-20 ticks, active levels ~50-200):
- Price range can be compressed to ~500 ticks around mid-price
- This fits in shared memory (~48KB per SM)
- Bitset is only ~8 words → trivial

### 5.3 Sequential Message Processing in CUDA

```
  // Triton/CUDA kernel: one thread block processes one symbol's message stream
  __global__ void match_kernel(
      Message* messages,   // Input: N messages in time order
      int N,
      Trade* trades,       // Output: matched trades
      int* trade_count,
      // Order book state (global memory, persistent across messages)
      int32_t* bid_active, int32_t* ask_active,
      int32_t* bid_head,   int32_t* ask_head,
      Order*   bid_orders, Order*   ask_orders,  // [MAX_PRICE * MAX_DEPTH]
      uint64_t* bid_mask,  uint64_t* ask_mask,
      Price*   best_bid,   Price*   best_ask
  ) {
      // Single thread processes messages sequentially
      for (int msg = 0; msg < N; msg++) {
          Message& m = messages[msg];

          if (m.type == ADD) {
              // Identical to OrderBook::addOrder
              int p = m.price;
              int side = m.side;
              // ... append to bid/ask_orders[p * MAX_DEPTH + count]
              // ... set bitset bit, update best
          }
          else if (m.type == CANCEL) {
              // Identical to OrderBook::cancelOrder
              // ... lookup by ID, tombstone, maybe clear bitset
          }
          else if (m.type == EXECUTE) {
              // Matching already done by exchange in LOBSTER data
              // Just update quantities
          }
      }
  }
```

**Critical insight for LOBSTER data**: LOBSTER messages are already the *result* of matching.
You don't need to run the matching algorithm — you need to *replay* the messages to reconstruct
order book state. This is simpler than matching:

```
  for each message:
    ADD    → insert order at price level
    CANCEL → tombstone order
    EXECUTE/PARTIAL_FILL → reduce quantity, maybe tombstone
    DELETE → tombstone order
```

The matching loop from PIYUSH is relevant if you want to simulate *new* orders against the book,
e.g., for RL agent actions.

### 5.4 Complexity Comparison

| Operation              | PIYUSH CPU              | CUDA Single-Thread       | Notes                    |
|------------------------|-------------------------|--------------------------|--------------------------|
| Add order              | O(1)                    | O(1)                     | Array index + push_back  |
| Cancel order           | O(1)                    | O(1)                     | ID lookup + tombstone    |
| Find best bid          | O(W) W=num_words        | O(W) same                | Bitset scan, W≈1-3      |
| Find best ask          | O(W)                    | O(W) same                | Bitset scan              |
| Match at one level     | O(D) D=orders_at_level  | O(D) same                | Linear scan, skip tombs  |
| Full match             | O(L×D) L=crossed levels | O(L×D) same              | Usually L=1, D=1-5       |

## 6. Key Takeaways for CUDA Port

1. **The bitset is the most important optimization**. It replaces O(log N) tree lookups
   (std::map) with O(1) amortized bit scans. In CUDA, `__ffsll()` is a single PTX instruction.

2. **The flat array eliminates pointer chasing**. `bids[price]` is a single offset calculation.
   No hash tables, no tree nodes scattered in memory. This is ideal for GPU memory access patterns.

3. **Tombstone cancellation avoids reallocation**. Setting `active = false` is a single store.
   No vector erasure, no node deletion. Perfect for GPU where memory management is expensive.

4. **The PMR allocator is irrelevant for CUDA**. It solves C++ heap allocation overhead.
   In CUDA, you pre-allocate all buffers and use index arithmetic.

5. **The sharding/threading is irrelevant for your use case**. Each CUDA thread block handles
   one symbol independently. The RingBuffer, SpinLock, and Exchange orchestration don't port.

6. **For LOBSTER replay, you don't need the matching loop**. LOBSTER gives you the executed
   trades. You just replay ADD/CANCEL/EXECUTE to maintain book state. The matching algorithm
   is only needed for simulating agent actions.

7. **Price range compression is critical for shared memory**. LOBSTER prices cluster in a
   narrow range (~200 ticks). Map prices to a compact range and the entire bitset + active
   counts fit in shared memory.
