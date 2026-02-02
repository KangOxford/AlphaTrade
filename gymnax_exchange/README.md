# Gymnax Exchange - Core Implementation

This folder contains all the critical code for the Gymnax Exchange project. It implements a high-performance order book matching engine with JAX and provides reinforcement learning (MARL) environments for multi-agent trading simulations. The codebase is designed for both market microstructure research and algorithmic trading strategy development.

## Directory Structure

### 📊 **`jaxob/`** - Order Book Engine
The core functional order book implementation using pure JAX operations for maximum performance and JIT compilation.

**Key Files:**
- **`JaxOrderBookArrays.py`** - Functional order book operations
  - `add_order()` / `cancel_order()` - Order management
  - `match_order()` - Single match operation
  - `_match_against_bid_orders()` / `_match_against_ask_orders()` - Iterative matching loops
  - `bid_lim()` / `ask_lim()` / `bid_cancel()` / `ask_cancel()` - Order type handling
  - `scan_through_entire_array()` - Batch message processing with `jax.lax.scan`
  - `get_L2_state()` - Extract market microstructure snapshots
  
- **`jaxob_config.py`** - Configuration dataclasses
  - `JAXLOB_Configuration` - Order book behavior (matching rules, book depth, etc.)
  - `MarketMaking_EnvironmentConfig` - Market making specific parameters
  - `Execution_EnvironmentConfig` - Trade execution parameters
  - `World_EnvironmentConfig` - Multi-agent world configuration

- **`jaxob_constants.py`** - Constants and enumerations
- **`jorderbook.py`** - Additional orderbook utilities
- **`config_io.py`** - Configuration file I/O
- **`jaxob_cuda_kernels.py`** - (Optional) CUDA acceleration for matching loops

---

### 📥 **`jaxlobster/`** - LOBSTER Data Loading
Loads and preprocesses real market data from LOBSTER for realistic simulations.

**Key Files:**
- **`lobster_loader.py`** - Main data loading class
  - `LoadLOBSTER_resample` - Resamples data into fixed-size windows
  - Handles message slicing, orderbook initialization, and data curation
  
- **`data_loading.py`** - Additional data utilities

**Functionality:**
- Load real LOBSTER CSV files (messages + snapshots)
- Initialize orderbooks from snapshots

---

### 🎮 **`jaxen/`** - Environments & Agent Types
Defines the MARL environments and agent types for different trading tasks.

**Key Files:**
- **`base_env.py`** - `BaseLOBEnv` class
  - Base environment inheriting from Gymnax
  - Handles data loading, message processing, orderbook state management
  - Episode initialization and stepping logic
  - Configurable via `LoadedEnvParams` and `LoadedEnvState`

- **`mm_env.py`** - Market Making Agent (`MarketMakingAgent`)
  - Extends `BaseLOBEnv` for market making tasks
  - Observation: L2 order book data, inventory, positions
  - Actions: bid/ask prices and quantities
  - Reward: PnL-based with inventory penalties
  - Supports multiple reward functions (spooner, pnl-based, etc.)

- **`exec_env.py`** - Execution Agent (`ExecutionAgent`)
  - Extends `BaseLOBEnv` for optimal execution tasks
  - Observation: remaining inventory, time, market state
  - Actions: order size and aggressiveness
  - Reward: minimizes market impact and execution cost

- **`marl_env.py`** - Multi-Agent Environment (`MARLEnv`)
  - Combines multiple agent types (e.g., multiple market makers + executor)
  - Handles concurrent action processing and simultaneous state updates
  - Message priority and order interaction handling
  - Returns observations/rewards/dones for all agents

- **`StatesandParams.py`** - State and config dataclasses
  - `LoadedEnvState` / `LoadedEnvParams` - Base environment state
  - `MultiAgentState` / `MultiAgentParams` - Multi-agent state
  - `WorldState` - Global world state for MARL

---

### 🤖 **`jaxrl/`** - Multi-Agent RL Training
End-to-end training scripts for multi-agent RL using the environments above.

**Key Files:**
- **`MARL/ippo_rnn_JAXMARL.py`** - Independent PPO with RNN
  - Multi-agent PPO (IPPO) implementation
  - GRU-based RNN for recurrent policies
  - Multi-action output heads (independent actions per agent)
  - JAX JIT-compiled training loops
  - Supports variable number of agents

- **`MARL/ippo_rnn_JAXMARL_pmap.py`** - Distributed training variant
  - Uses `jax.pmap` for data parallelism across devices
  - Scales training to multiple GPUs

- **`market_making_baselines/`** - Baseline strategies
  - Reference implementations for comparison

**Training Pipeline:**
1. Load LOBSTER data via `jaxlobster`
2. Initialize `MARLEnv` with agents (market makers, executors, etc.)
3. Run IPPO training with learned policies
4. Evaluate on test data windows
5. Log metrics to W&B, save checkpoints with Orbax

---

## Data Flow

```
LOBSTER CSV Files
       ↓
jaxlobster.LoadLOBSTER
(messages, snapshots)
       ↓
jaxen.base_env.BaseLOBEnv.__init__
(preprocessed data cubes)
       ↓
jaxen.marl_env.MARLEnv (multi-agent)
  ├─ mm_env.MarketMakingAgent
  ├─ exec_env.ExecutionAgent
  └─ ... (more agents)
       ↓
jaxob.JaxOrderBookArrays.scan_through_entire_array
(process action + market messages)
       ↓
jaxrl.MARL.ippo_rnn_JAXMARL
(train policies with gradient descent)
```

---
