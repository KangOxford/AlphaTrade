# JAXMARL-HFT: GPU-Accelerated Multi-Agent Reinforcement Learning for High-Frequency Trading

A JAX-based framework for multi-agent reinforcement learning in high-frequency trading environments, featuring GPU-accelerated order book simulation and market making algorithms.

## Key Features

- **GPU-Accelerated**: Built on JAX for high-performance parallel computation with JIT compilation
- **Dual-Level Parallelization**: Parallel processing across episodes and agent types using `vmap`
- **Multi-Agent RL**: Supports market making, execution, and directional trading agents
- **LOBSTER Data Integration**: Real market data support with efficient GPU memory usage
- **Scalable**: Handles thousands of parallel environments 
- **Heterogeneous Agents**: Supports different observation/action spaces per agent type

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up data directory
mkdir -p ~/data
```

### Docker Setup

```bash
# Build and run with Docker
make build
make run
```

### Training

```bash
# Run IPPO training
python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py
```

## Performance

JAXMARL-HFT achieves significant speed improvements over existing frameworks:

- **Environment Rollouts**: Up to 351,119 steps/second (vs 4,896 for CPU-MARL)
- **RL Training**: 5x-240x speedup depending on agent count
- **Memory Efficiency**: Full year of AMZN data in 4GB GPU memory
- **Multi-GPU Support**: 50x speedup with 8 GPUs vs single GPU

## Agent Types

### Market Making Agents
- **Action Spaces**: Spread-Skew, Fixed Quantity, Avellaneda-Stoikov
- **Reward Functions**: Spooner, Buy-Sell PnL with configurable inventory penalties
- **Observation Spaces**: Flexible feature sets from simple statistics to complex LOB states

### Execution Agents  
- **Action Spaces**: Discrete quantity selection at reference prices
- **Task Types**: Large order execution with minimal market impact
- **Reward Functions**: Slippage-based with end-of-episode penalties

### Directional Trading
- **Action Spaces**: Bid/ask at best prices or no action
- **Strategy**: Reuses market making infrastructure with specialized actions

## Repository Structure

```
gymnax_exchange/
├── jaxen/           # Environment implementations
│   ├── marl_env.py  # Multi-agent RL environment
│   ├── mm_env.py    # Market making environment  
│   ├── exec_env.py  # Execution environment
│   └── Speed_test.py # Performance benchmarking
├── jaxrl/           # Reinforcement learning algorithms
│   └── MARL/        # IPPO implementation
├── jaxob/           # Order book implementation
└── jaxlobster/      # LOBSTER data integration
```

## Configuration

Edit configuration files in `gymnax_exchange/jaxrl/MARL/config/` to customize:
- Number of parallel environments (default: 4096)
- Training parameters (steps, learning rates, etc.)
- Agent configurations (action spaces, reward functions)
- Market data settings (resolution, episode length)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- JAX, Flax, and related dependencies (see `requirements.txt`)

