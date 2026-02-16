#!/usr/bin/env python3
"""
Benchmark script for JAX Order Book with VMAP parallelization.

This script loads LOBSTER sample data, initializes orderbooks, and processes
messages in a vmapped setting across multiple parallel environments.

Usage:
    python benchmark_vmap_orderbook.py [--n-envs 32] [--n-msgs 500] [--n-runs 5]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
from functools import partial

# Add path for gymnax_exchange module
sys.path.insert(0, '/home/myuser')

import gymnax_exchange.jaxob.JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_sample_data(data_path: str = "/home/myuser/data/rawLOBSTER/AMZN/Sample"):
    """
    Load LOBSTER sample message and orderbook data.
    
    LOBSTER Format:
    - Message: [time, type, order_id, quantity, price, direction]
        - type: 1=limit, 2=cancel, 3=delete, 4=execute
        - direction: 1=bid (buy), -1=ask (sell)
    - Orderbook: [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...]
    """
    message_file = os.path.join(data_path, "AMZN_2012-06-21_34200000_57600000_message_10.csv")
    orderbook_file = os.path.join(data_path, "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv")
    
    # Load message data
    messages_df = pd.read_csv(message_file, header=None)
    messages_df.columns = ['time', 'type', 'order_id', 'quantity', 'price', 'direction']
    
    # Load orderbook data (10 levels = 40 columns)
    orderbook_df = pd.read_csv(orderbook_file, header=None)
    n_levels = orderbook_df.shape[1] // 4
    columns = []
    for i in range(n_levels):
        columns.extend([f'ask_price_{i+1}', f'ask_size_{i+1}', f'bid_price_{i+1}', f'bid_size_{i+1}'])
    orderbook_df.columns = columns
    
    return messages_df, orderbook_df


def preprocess_messages_to_jax(messages_df: pd.DataFrame, n_msgs: int = None) -> jnp.ndarray:
    """
    Convert LOBSTER messages to JAX format for cond_type_side.
    
    JAX message format: [type, side, quantity, price, trader_id, order_id, time_s, time_ns]
    """
    # Filter to valid message types
    valid_types = [1, 2, 3, 4]
    messages_df = messages_df[messages_df['type'].isin(valid_types)].copy()
    
    if n_msgs is not None:
        messages_df = messages_df.head(n_msgs)
    
    # Extract time components
    time_s = messages_df['time'].astype(int)
    time_ns = ((messages_df['time'] - time_s) * 1_000_000_000).astype(int)
    
    # Create JAX message array
    msg_array = np.zeros((len(messages_df), 8), dtype=np.int32)
    msg_array[:, 0] = messages_df['type'].values
    msg_array[:, 1] = messages_df['direction'].values
    msg_array[:, 2] = messages_df['quantity'].values
    msg_array[:, 3] = messages_df['price'].values
    msg_array[:, 4] = messages_df['order_id'].values
    msg_array[:, 5] = messages_df['order_id'].values
    msg_array[:, 6] = time_s.values
    msg_array[:, 7] = time_ns.values
    
    return jnp.array(msg_array)


def get_l2_from_orderbook_row(orderbook_row, n_levels: int = 10) -> jnp.ndarray:
    """Convert a LOBSTER orderbook row to L2 format."""
    return jnp.array(orderbook_row[:n_levels * 4].values, dtype=jnp.int32)


# ============================================================================
# VMAP Processing Functions
# ============================================================================

def create_vmapped_scan(cfg: JAXLOB_Configuration):
    """Create a vmapped version of scan_through_entire_array."""
    @jax.jit
    def vmapped_scan(vasks, vbids, vtrades, vkeys, msgs):
        """
        Process messages across multiple environments in parallel.
        
        Args:
            vasks: Batched ask sides (n_envs, n_orders, 6)
            vbids: Batched bid sides (n_envs, n_orders, 6)
            vtrades: Batched trades (n_envs, n_trades, 8)
            vkeys: Batched random keys (n_envs, 2)
            msgs: Messages to process (n_msgs, 8) - same for all envs
        
        Returns:
            Final state tuple for all environments
        """
        def scan_single_env(asks, bids, trades, key):
            book_state = (asks, bids, trades)
            return job.scan_through_entire_array(cfg, key, msgs, book_state)
        
        return jax.vmap(scan_single_env)(vasks, vbids, vtrades, vkeys)
    
    return vmapped_scan


def run_benchmark(n_envs: int = 32, n_msgs: int = 500, n_runs: int = 5, verbose: bool = True):
    """
    Run the VMAP orderbook benchmark.
    
    Args:
        n_envs: Number of parallel environments
        n_msgs: Number of messages to process
        n_runs: Number of timing runs
        verbose: Print detailed output
    
    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        print("=" * 70)
        print("JAX Order Book VMAP Benchmark")
        print("=" * 70)
        print(f"JAX version: {jax.__version__}")
        print(f"Available devices: {jax.devices()}")
        print(f"Number of environments: {n_envs}")
        print(f"Number of messages: {n_msgs}")
        print(f"Number of timing runs: {n_runs}")
        print()

    # Configuration
    cfg = JAXLOB_Configuration(
        nOrders=100,
        nTrades=100,
        book_depth=10,
    )

    # Load data
    if verbose:
        print("Loading LOBSTER sample data...")
    messages_df, orderbook_df = load_sample_data()
    
    # Get initial L2 state
    initial_l2 = get_l2_from_orderbook_row(orderbook_df.iloc[0], n_levels=cfg.book_depth)
    
    # Convert messages to JAX format
    jax_messages = preprocess_messages_to_jax(messages_df, n_msgs=n_msgs)
    if verbose:
        print(f"Prepared {len(jax_messages)} messages for processing")

    # Initialize orderbook
    ob = OrderBook(cfg=cfg)
    initial_time = jnp.array([34200, 0])
    state = ob.reset(l2_book=initial_l2, time=initial_time)

    # Create batched states for all environments
    vasks = jnp.stack([state.asks] * n_envs)
    vbids = jnp.stack([state.bids] * n_envs)
    vtrades = jnp.stack([state.trades] * n_envs)
    
    # Create batched random keys
    master_key = jax.random.PRNGKey(cfg.seed)
    vkeys = jax.random.split(master_key, n_envs)

    if verbose:
        print(f"\nInitialized {n_envs} parallel orderbooks")
        print(f"  Asks shape: {vasks.shape}")
        print(f"  Bids shape: {vbids.shape}")
        print(f"  Trades shape: {vtrades.shape}")

    # Get initial L2 state for verification
    l2_before = job.get_L2_state(state.asks, state.bids, cfg.book_depth, cfg)
    if verbose:
        print(f"\nInitial L2 state:")
        print(f"  Best Ask: {l2_before[0]} @ {l2_before[1]} shares")
        print(f"  Best Bid: {l2_before[2]} @ {l2_before[3]} shares")

    # Create vmapped scan function
    vmapped_scan = create_vmapped_scan(cfg)

    # Warm up JIT compilation
    if verbose:
        print("\nWarming up JIT compilation...")
    _ = vmapped_scan(vasks, vbids, vtrades, vkeys, jax_messages)
    jax.block_until_ready(_)

    # Run benchmark
    if verbose:
        print(f"\nRunning {n_runs} timing iterations...")
    
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        if i == 0:
            jax.profiler.start_trace(f"MinimalReproTrace")
        final_asks, final_bids, final_trades = vmapped_scan(
            vasks, vbids, vtrades, vkeys, jax_messages
        )
        jax.block_until_ready(final_asks)
        if i == 0:
            jax.profiler.stop_trace()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if verbose:
            print(f"  Run {i+1}: {elapsed*1000:.2f}ms")

    # Compute statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    # Per-environment and per-message metrics
    time_per_env = mean_time / n_envs
    time_per_msg = mean_time / n_msgs
    time_per_env_msg = mean_time / (n_envs * n_msgs)
    throughput_msgs = (n_envs * n_msgs) / mean_time

    results = {
        'n_envs': n_envs,
        'n_msgs': n_msgs,
        'n_runs': n_runs,
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'time_per_env_ms': time_per_env * 1000,
        'time_per_msg_us': time_per_msg * 1e6,
        'time_per_env_msg_us': time_per_env_msg * 1e6,
        'throughput_msgs_per_sec': throughput_msgs,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)
        print(f"Total time: mean={mean_time*1000:.2f}ms, std={std_time*1000:.2f}ms, min={min_time*1000:.2f}ms")
        print(f"Time per environment: {time_per_env*1000:.2f}ms")
        print(f"Time per message (total): {time_per_msg*1e6:.2f}µs")
        print(f"Time per message per env: {time_per_env_msg*1e6:.2f}µs")
        print(f"Throughput: {throughput_msgs:.0f} messages/sec")

    # Verify final state
    l2_after = job.get_L2_state(final_asks[0], final_bids[0], cfg.book_depth, cfg)
    n_trades = jnp.sum(final_trades[0][:, 0] != -1)
    
    if verbose:
        print(f"\nFinal L2 state (env 0):")
        print(f"  Best Ask: {l2_after[0]} @ {l2_after[1]} shares")
        print(f"  Best Bid: {l2_after[2]} @ {l2_after[3]} shares")
        print(f"  Trades recorded: {n_trades}")
        print("\n" + "=" * 70)
        print("Benchmark completed successfully!")
        print("=" * 70)

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark JAX Order Book with VMAP parallelization"
    )
    parser.add_argument(
        '--n-envs', type=int, default=32,
        help='Number of parallel environments (default: 32)'
    )
    parser.add_argument(
        '--n-msgs', type=int, default=500,
        help='Number of messages to process (default: 500)'
    )
    parser.add_argument(
        '--n-runs', type=int, default=5,
        help='Number of timing runs (default: 5)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    results = run_benchmark(
        n_envs=args.n_envs,
        n_msgs=args.n_msgs,
        n_runs=args.n_runs,
        verbose=not args.quiet
    )
    
    return results


if __name__ == '__main__':
    main()
