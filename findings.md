# Findings: GPU-Parallel High-Performance Matching Engine

## Requirements
- 千个 OrderBook GPU 并行运行
- 最大化 orders/sec 吞吐量
- 不限定必须用 JAX，核心价值是并行能力
- 参考 PIYUSH C++20 matching engine 的算法和数据结构
- 必须能集成到现有 RL 训练 pipeline 中

## Benchmark Baseline (Job 2873056, 2026-03-15)

### 单消息延迟
| 消息类型 | Median | 吞吐量 |
|----------|--------|--------|
| noop | 140.9 μs | 7,095 /s |
| limit_passive | 171.0 μs | 5,847 /s |
| limit_crossing | 205.8 μs | 4,858 /s |
| cancel | 171.3 μs | 5,837 /s |

### 批量 scan (500 msgs)
| 场景 | Per Msg | 吞吐量 |
|------|---------|--------|
| Passive | 60.4 μs | 16,543 /s |
| Crossing | 76.9 μs | 13,002 /s |
| Mixed | 63.1 μs | 15,840 /s |

### VMAP 扩展
| Books | Per Msg | 吞吐量 | 扩展效率 |
|-------|---------|--------|----------|
| 1 | 135.4 μs | 7,386 /s | baseline |
| 10 | 17.9 μs | 55,760 /s | ~7.5x |
| 100 | 2.5 μs | 408,051 /s | ~55x |
| 1000 | 398.2 ns | 2,511,547 /s | ~340x |

## Research Findings

### PIYUSH C++20 Engine 核心技术
- Flat `std::vector` (不用 `std::map` 红黑树)
- Bitset scanning (`__builtin_ctzll`) O(1) 跳过空价格层
- SPSC Ring Buffer (lock-free, `alignas(128)` cache-line 对齐)
- PMR Vectors (`std::pmr::vector` + `monotonic_buffer_resource`)
- 512MB 栈分配 monotonic buffer，热路径零内存分配
- 峰值: 171M orders/sec (Apple M1 Pro), 132M (Binance L3 replay)

### GPU 匹配引擎的根本矛盾
| 问题 | 说明 |
|------|------|
| 顺序依赖 | 第 N+1 条消息的结果取决于第 N 条的状态 |
| 分支发散 | limit/cancel/match 不同路径，CUDA warp 内发散 |
| 随机访问 | scatter/gather 模式，非 coalesced access |
| Kernel launch overhead | 每条消息一次 launch ~2-5 μs |

### 但 GPU 并行化是可行的
- 关键洞察：虽然单 book 内消息是顺序的，但多个 book 之间完全独立
- CUDA 方案：每个 thread block 跑一个 order book
- 1000 个 thread blocks = 1000 个并行 book
- VMAP 已证明这个并行模型有效（1000 books 仅 2.94x 时间增长）

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 量化现有基线性能 | 精确数据驱动决策 |
| Triton 路径因 CUDA 版本不兼容暂时搁置 | PTX 8.7 vs ptxas 8.5 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| conda.sh hardcoded broken paths | 改用直接 PATH=/projects/s5e/quant/miniforge3/bin |
| Triton PTX version mismatch | 需要 CUDA 12.7+ 或降级 Triton 的 PTX target |

## Resources
- PIYUSH C++20 engine: https://github.com/PIYUSH-KUMAR1809/order-matching-engine
- C++ optimization blog: https://medium.com/@kpiyush8826/how-i-optimized-a-c-matching-engine-from-100k-to-150-million-orders-per-second-35b2065fa4c0
- JAX custom calls: https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html
- XLA custom call: https://openxla.org/xla/custom_call
- LMAX Disruptor: https://lmax-exchange.github.io/disruptor/disruptor.html
