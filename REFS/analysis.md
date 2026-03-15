# Order Matching Engine Architecture Analysis
## Reference: PIYUSH-KUMAR1809/order-matching-engine vs JAXOB

Date: 2026-03-15

---

## 1. C++ Engine 架构总览

### 核心设计: Shard-per-Core + Flat Array + Bitset

```
Producer Threads ──> hash(symbolId) ──> Shard[N]
                                          │
                                    RingBuffer<Cmd, 65536>
                                    (SpinLock + TTAS)
                                    (batch push/pop 256)
                                          │
                                    Worker Thread (pinned)
                                          │
                                    OrderBook
                                    ├── bids[100000]  (flat array by price)
                                    ├── asks[100000]
                                    ├── bidMask/askMask (PriceBitset)
                                    ├── idToLocation[10M]
                                    └── PMR pool (512MB monotonic)
```

### 性能优化汇总 (从 1M → 160M ops/sec, 160x 提升)

| 优化类别 | 具体技术 | 效果 |
|----------|----------|------|
| **O(1) 价格查找** | `bids[price]` / `asks[price]` 直接下标 | 消除红黑树 O(log N) |
| **Bitset + CTZ/CLZ** | `__builtin_ctzll` 单指令跳过 64 个空价位 | 稀疏 book 高效扫描 |
| **PMR Arena** | 512MB monotonic_buffer, 分配=指针递增 | 零 malloc/free |
| **POD Order** | int32 symbolId 替代 string, memcpy 安全 | 零堆分配 |
| **Shard 隔离** | 每 symbol 一个独立 OrderBook + Worker | 无 mutex 竞争 |
| **批量传输** | 256-cmd batch push/pop via memcpy | 摊薄锁+cache成本 |
| **Tombstone 删除** | `active=false` 标记, headIndex 跳过 | 避免 vector erase |
| **绑核+对齐** | pthread_setaffinity_np, alignas(128) | 无 migration, 无 false sharing |
| **LTO + -march=native** | 编译器跨TU内联 + CPU 特定指令 | 全链路优化 |

### 关键演进数据

| 阶段 | ops/sec | 瓶颈 |
|------|---------|------|
| v1 (mutex + std::map) | ~1M | OS context switch + tree traversal |
| + Ring Buffer | ~9M | 消除 OS 调度 |
| + Memory Pool | ~17.5M | 消除 malloc 竞争 |
| + POD Zero-Copy | ~27.7M | 消除 string 拷贝 |
| + PMR Monotonic | ~156M | 消除所有堆操作 |
| **Final** | **~160M avg, 171M peak** | — |

---

## 2. JAXOB 架构总览

### 数据结构

```
asks/bids: (nOrders, 6) int32   [Price, Qty, OID, TID, Sec, NSec]
trades:    (nTrades, 8) int32   [Price, SignedQty, PassOID, AgrOID, Sec, NSec, PassTID, AgrTID]
msgs:      (N, 8)      int32   [Type, Side, Qty, Price, OID, TID, Sec, NSec]
```

### 处理流程

```
lax.scan(msgs) ──> cond_type_side() ──> {bid_lim, ask_lim, bid_cancel, ask_cancel}
                                              │
                                        _match_against_*_orders()
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                              JAX while_loop      Triton kernel
                              (dynamic iter)    (static_range, 1 block)
                                    │                   │
                              O(K*N) per match    O(N) fixed iterations
```

### 瓶颈

| 瓶颈 | 复杂度 | 说明 |
|------|--------|------|
| `_get_top_ask/bid_order_idx` | O(N) per call | 全扫描找最优, 无排序/堆/Bitset |
| `while_loop` | O(K*N) | K 次匹配 × N 次扫描, vmap 下统一为最大 K |
| `jnp.where(size=1)` | O(N) | 每次 add/cancel/match 找空槽 |
| `_removeZeroNegQuant` | O(N) | 每次操作后全表扫描清理 |
| `lax.scan` 顺序 | O(M) serial | M 条消息严格顺序 (不可并行) |
| Triton vmap | 退化为 map | 非真正并行, 顺序执行各环境 |

---

## 3. 架构对比: 为什么 C++ 引擎快

### 3.1 数据结构对比

| 维度 | C++ Engine | JAXOB |
|------|-----------|-------|
| **价格查找** | O(1) 数组下标 `asks[price]` | O(N) 全扫描 `jnp.where` |
| **最优价格** | O(1) `bestBid`/`bestAsk` 缓存 + Bitset | O(N) `jnp.min`/`jnp.max` |
| **空槽管理** | `headIndex` 指针 + activeCount | O(N) `jnp.where(==-1, size=1)` |
| **删除** | O(1) tombstone `active=false` | O(N) `jnp.where` 条件覆盖 |
| **Cancel 查找** | O(1) `idToLocation[orderId]` | O(N) 全表扫描 |
| **内存** | PMR arena, 零分配热路径 | JAX 数组 (XLA 管理) |

### 3.2 算法复杂度对比 (单次匹配操作)

| 操作 | C++ | JAXOB |
|------|-----|-------|
| 找最优价格 | O(1) cached | O(N) scan |
| 匹配 K 个 orders | O(K) + O(K * Bitset) ≈ O(K) | O(K * N) |
| 添加 order | O(1) amortized | O(N) find slot |
| 取消 order | O(1) lookup + tombstone | O(N) search |
| 处理 M 条消息 | O(M) with O(1) per msg | O(M * N) |

---

## 4. 对 JAXOB 的可借鉴点

### 4.1 直接可用的优化 (不破坏 JAX 兼容性)

| 优化 | 方法 | 预期收益 |
|------|------|----------|
| **Bitset 价格掩码** | 维护 `(MAX_PRICE/32,)` int32 数组作为 bitmask | 最优价格查找从 O(N) → O(MAX_PRICE/32) |
| **Best price 缓存** | LobState 中增加 bestBid/bestAsk 字段 | 匹配检查从 O(N) → O(1) |
| **排序数组** | 保持 asks/bids 按 (price, time) 排序 | 最优在 index 0, 查找 O(1) |
| **空槽指针** | 维护 next_free_slot 计数器 | add_order 从 O(N) → O(1) |
| **Tombstone + headIndex** | 配合排序数组使用 | 减少无效遍历 |

### 4.2 需要 Triton/自定义 kernel 的优化

| 优化 | 说明 |
|------|------|
| **Bitset intrinsics** | Triton 中 `tl.where` 可模拟 CTZ/CLZ, 但不如 CPU 原生 |
| **批量匹配** | 将多个消息融合为一个 kernel, 减少 kernel launch |
| **per-price-level 数组** | 二维 (MAX_PRICE, orders_per_level) 布局 |

### 4.3 不适用的优化

| 优化 | 原因 |
|------|------|
| Shard-per-Core | JAXOB 是单 symbol, 不需要分片 |
| RingBuffer | JAXOB 无生产者/消费者模型 |
| SpinLock/TTAS | GPU 无自旋锁概念 |
| PMR Arena | JAX 自管内存 |
| 绑核 | GPU thread 不可绑定 |

---

## 5. 数千并行环境的架构思考

### 5.1 当前方案: JAX vmap

```
vmap(env.step)(batch_of_states)
  → XLA 自动向量化, while_loop 统一为 max iterations
  → 问题: 一个环境匹配 50 orders, 其他都匹配 0, 全部跑 50 次迭代
```

### 5.2 方案对比

| 方案 | 单环境性能 | 多环境扩展性 | Kernel Fusion | 开发难度 |
|------|-----------|-------------|---------------|----------|
| **JAX while_loop + vmap** | 差 (O(K*N)) | 好 (XLA vectorize) | 好 (XLA autotuner) | 低 |
| **Triton 单 kernel** | 中 (O(N) fixed) | 差 (vmap→map) | 无 | 中 |
| **Triton batched kernel** | 中 | 好 (显式 batch dim) | 自定义 | 高 |
| **排序数组 JAX** | 好 (O(1) best) | 好 (XLA vectorize) | 好 | 中 |
| **C++/CUDA 完全自写** | 极好 | 需自己并行化 | 无 (黑盒) | 极高 |

### 5.3 推荐方向: 排序数组 + JAX, 保留 XLA fusion

核心思路: 不离开 JAX 生态, 但重新设计数据结构:

```
# 现在:  unsorted (nOrders, 6), 全扫描
# 改为:  sorted by (price, time), best = array[0]

# asks 按 price 升序: asks[0] = best ask
# bids 按 price 降序: bids[0] = best bid

# 匹配: 直接取 array[0], O(1)
# 添加: jnp.searchsorted + array insert (shift), O(N) 但 memory-friendly
# 删除: shift down, O(N) 但连续内存
```

这样保留了 XLA 的 kernel fusion 能力, 同时大幅降低匹配的计算量。
在 vmap 场景下, XLA 可以将多个环境的 sorted array 操作融合为高效的向量化 kernel。

---

## 6. JAX vs 自写 Kernel 的权衡

### XLA Kernel Fusion 的价值

```
# JAX 写法 (多个操作):
best_price = jnp.min(asks[:, 0])
mask = asks[:, 0] == best_price
best_idx = jnp.argmin(jnp.where(mask, asks[:, 4], jnp.iinfo(jnp.int32).max))
trade_qty = jnp.minimum(incoming_qty, asks[best_idx, 1])

# XLA 编译后: 可能融合为单个 kernel, 一次遍历完成
# 自写 Triton: 需要手动管理每个步骤, 但可以做得更优
```

### 决策矩阵

| 因素 | JAX + XLA | 自写 Triton/CUDA |
|------|-----------|-----------------|
| **开发速度** | 快 (Python) | 慢 (低级 GPU 编程) |
| **Kernel Fusion** | XLA autotuner 自动优化 | 需要手动 fuse |
| **vmap 兼容** | 原生支持 | 需 custom_vmap 规则 |
| **可调试性** | 好 (jit disabled 可 print) | 差 (GPU kernel 调试困难) |
| **数据结构灵活性** | 受限 (静态 shape) | 灵活 (动态 alloc) |
| **峰值性能** | 受 XLA 编译器限制 | 可达理论峰值 |
| **维护成本** | 低 | 高 |

### 结论

对于数千并行环境场景, **JAX + 更好的数据结构** 是最优路径:
1. XLA 的 kernel fusion 在大 batch (数千环境) 下价值巨大
2. 自写 kernel 在单环境可能更快, 但 vmap/fusion 能力损失更大
3. C++ engine 的核心洞察 (O(1) 查找、排序、Bitset) 可以在 JAX 数组语义下实现
