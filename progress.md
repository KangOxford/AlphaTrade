# Progress Log: GPU-Parallel High-Performance Matching Engine

## Session: 2026-03-15

### Phase 1: Benchmark & Research
- **Status:** in_progress
- **Started:** 2026-03-15

- Actions taken:
  - 写了 `benchmark_matching_engine.py` 微基准测试脚本
  - 写了 `benchmark_matching_engine.batch` sbatch 提交脚本
  - Job 2873047 失败 (conda 路径), 修复后 Job 2873056 成功
  - 并行 subagent 研究: JAX 代码架构 / PIYUSH C++20 引擎 / 现有 profiling 数据
  - 完成基线性能数据收集
  - 创建 task_plan.md, findings.md, progress.md

- Files created/modified:
  - benchmark_matching_engine.py (created)
  - benchmark_matching_engine.batch (created, 修复 conda 路径)
  - logs/benchmark_mengine_2873056.out (benchmark 结果)
  - task_plan.md (created)
  - findings.md (created)
  - progress.md (created)

- Key findings:
  - JAX single book: ~16K msgs/sec
  - JAX 1000 books vmap: ~2.5M msgs/sec
  - C++20 reference: 132M msgs/sec (单线程)
  - Triton kernel 因 CUDA 12.6 PTX 不兼容无法运行

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| JAX benchmark (job 2873056) | 500 msgs, 100 orders | Complete results | 11 test points, all complete | ✓ |
| Triton benchmark | Same config | Complete results | PTX version crash | ✗ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-03-15 16:26 | conda.sh broken path (job 2873047) | 1 | Changed to direct PATH setup |
| 2026-03-15 16:26 | Triton PTX 8.7 vs ptxas 8.5 | 1 | Needs CUDA 12.7+ or Triton downgrade |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1: Research & Architecture Design |
| Where am I going? | Need to decide: CUDA custom kernel vs extended Triton vs pure CUDA |
| What's the goal? | 替换 matching engine, 保留 GPU 千 book 并行, 100x+ 单 book 提升 |
| What have I learned? | See findings.md — 基线数据完整, C++技术路线明确 |
| What have I done? | Benchmark 完成, 3 个 subagent 研究完成, 基线数据已量化 |
