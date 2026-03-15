# Task Plan: GPU-Parallel High-Performance Matching Engine

## Goal
将 order book matching engine 从纯 JAX 实现替换为高性能 CUDA C++ kernel，保留 GPU 上千 book 并行能力，目标：单 book 吞吐量提升 100x+（从 ~16K → ~1M+ msgs/sec），1000 books 并行时接近 C++ 单线程水平（~100M msgs/sec）。

## Current Phase
Phase 1

## Context & Motivation

### 当前性能 (benchmark job 2873056, 2026-03-15)

| 场景 | Per Msg | 吞吐量 |
|------|---------|--------|
| JAX single msg (crossing) | 205.8 μs | 4,858 /s |
| JAX batch scan (mixed) | 63.1 μs | 15,840 /s |
| JAX vmap 1000 books | 398.2 ns | 2,511,547 /s |

### 对比目标

| Engine | 吞吐量 | 技术 |
|--------|--------|------|
| C++20 PIYUSH | 132M /s | Flat vector + bitset + SPSC + PMR |
| Rust matching-engine-rs | 11.3M /s | Price-time priority |
| LMAX (Java) | 5M /s | Disruptor pattern |

### 瓶颈分析

JAX matching engine 慢的根因：
1. `jax.lax.switch` — vmap 下 5 个分支全执行，只取 1 个结果
2. `jax.lax.while_loop` — 每次循环需要 D2H 同步
3. O(N) 线性扫描 — 找最优价格/空 slot 都是 O(100) 全扫
4. noop 就要 141 μs — dispatch overhead 占 68%

### 用户核心需求
- 千个 OrderBook GPU 并行
- 最大化 orders/sec
- 不关心是否用 JAX，关心的是并行能力
- 参考: PIYUSH C++20 matching engine

## Phases

### Phase 1: Research & Architecture Design
- [ ] 分析 PIYUSH C++20 engine 的核心算法和数据结构
- [ ] 调研 CUDA C++ custom kernel 集成 JAX 的路径 (XLA custom call vs pybind)
- [ ] 调研纯 CUDA 方案 (不依赖 JAX, 用 CUDA thread block 并行)
- [ ] 调研 Triton kernel 方案 (扩展现有 triton_matching.py)
- [ ] 对比 4 种方案的可行性/性能上限/开发量
- **Status:** in_progress

### Phase 2: Architecture Decision & Detailed Design
- [ ] 选定方案并写 design spec
- [ ] 定义 kernel API (输入/输出/内存布局)
- [ ] 定义与训练 pipeline 的集成接口
- [ ] 用户确认设计
- **Status:** pending

### Phase 3: Prototype Implementation
- [ ] 实现核心 matching kernel
- [ ] 实现 JAX/Python 绑定
- [ ] 单 book 功能验证 (对比 JAX 结果)
- **Status:** pending

### Phase 4: Benchmark & Optimization
- [ ] 单 book 性能对比
- [ ] VMAP/parallel 扩展测试
- [ ] Profiling (occupancy, memory bandwidth)
- [ ] 优化热路径
- **Status:** pending

### Phase 5: Integration & Training Validation
- [ ] 替换训练 pipeline 中的 matching engine
- [ ] 端到端训练验证 (loss 一致性)
- [ ] 多节点测试
- **Status:** pending

## Key Questions
1. CUDA custom kernel 如何集成到 JAX 的 vmap/scan 中？→ XLA custom call 支持 batching rule 吗？
2. PIYUSH 的 bitset scanning 能否直接映射到 CUDA warp-level 操作？
3. 内存布局：AoS vs SoA 哪个对 GPU coalesced access 更友好？
4. 单 thread block 跑一个 book vs 多 thread 跑一个 book？
5. 是否需要保留 JAX fallback 路径？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 用 benchmark 数据量化现有性能 | 需要精确基线才能评估改进效果 |
| 测试 JAX 和 Triton 两条路径 | Triton 因 PTX 版本问题失败 (8.7 vs 8.5) |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| conda.sh 路径错误 (job 2873047) | 1 | 改用直接 PATH 设置，跟 node_wrapper.sh 一致 |
| Triton PTX 8.7 vs ptxas 8.5 | 1 | CUDA 12.6 不支持，需升级或降级 Triton target |

## Notes
- GH200 GPU, ARM aarch64 平台
- CUDA 12.6 (module load cuda/12.6)
- 计算节点 4x GPU, NV6 互联
- 现有 Triton matching kernel 已部分替换 while_loop (但因 CUDA 版本无法运行)
