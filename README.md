# JAXMARL-HFT: GPU-Accelerated Multi-Agent Reinforcement Learning for High-Frequency Trading

A JAX-based framework for multi-agent reinforcement learning for high-frequency trading.

## Key Features

- **GPU-Accelerated**: Built on JAX for high-performance parallel computation with JIT compilation
- **Dual-Level Parallelization**: Parallel processing across episodes and agent types using `vmap`
- **Multi-Agent RL**: Supports market making, execution, and directional trading agents
- **LOBSTER Data Integration**: Real market data support with efficient GPU memory usage
- **Scalable**: Handles thousands of parallel environments 
- **Heterogeneous Agents**: Supports different observation/action spaces per agent type

## Quick Start

### Docker Setup

```bash
# Set up data directory
mkdir -p ~/data
```

**Note**: Configure the Makefile for your specific environment (GPU device, data directory path, etc.)

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

## Agent Types

### Market Making Agents
- **Purpose**: Provide liquidity by posting bid/ask orders
- **Action Spaces**: Multiple discrete action spaces (spread_skew, fixed_quants, AvSt, directional_trading, simple)
- **Reward Functions**: Various PnL-based rewards with configurable inventory penalties
- **Observation Spaces**: Engineered features, message-based, or basic LOB statistics

### Execution Agents  
- **Purpose**: Execute large orders with minimal market impact
- **Action Spaces**: Discrete quantity selection at reference prices (fixed_quants, fixed_prices, complex variants)
- **Task Types**: Random, buy, or sell execution tasks
- **Reward Functions**: Slippage-based with configurable end-of-episode penalties

### Directional Trading
- **Purpose**: Simple directional trading strategy
- **Action Spaces**: Bid/ask at best prices or no action
- **Strategy**: Reuses market making infrastructure with specialized actions

## Repository Structure

```
gymnax_exchange/
├── jaxen/            # Environment implementations
│   ├── marl_env.py  # Multi-agent RL environment
│   ├── mm_env.py    # Market making (and directional trading) environment  
│   └── exec_env.py  # Execution environment
├── jaxrl/           # Reinforcement learning algorithms
│   └── MARL/        # IPPO implementation
├── jaxob/           # Order book implementation
└── jaxlobster/      # LOBSTER data integration
```

## Configuration

The framework uses a comprehensive configuration system with dataclasses for different components:

### Core Configuration Classes

- **`MultiAgentConfig`**: Main configuration combining world and agent settings
- **`World_EnvironmentConfig`**: Global environment parameters (data paths, episode settings, market hours)
- **`MarketMaking_EnvironmentConfig`**: Market making (and directional trading) agent configuration (action spaces, reward functions, observation spaces)
- **`Execution_EnvironmentConfig`**: Execution agent configuration (task types, action spaces, reward parameters)

### Training Configuration

Edit YAML files in `gymnax_exchange/jaxrl/MARL/config/` to customize:
- Number of parallel environments (default: 4096)
- Training parameters (steps, learning rates, etc.)
- Agent configurations (action spaces, reward functions)
- Market data settings (resolution, episode length)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- JAX, Flax, and related dependencies (see `requirements.txt`)

