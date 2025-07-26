"""
Focused test runner for core JaxOrderBookArrays operations
"""

import sys
import os
sys.path.append('/home/myuser/gymnax_exchange')

import jax
import jax.numpy as jnp
from jax import random

# Import functions we know exist
from jaxob.JaxOrderBookArrays import (
    add_order, _removeZeroNegQuant, cancel_order, init_orderside,
    get_best_bid, get_best_ask, get_volume_at_price,
    doNothing, bid_lim, ask_lim, bid_cancel, ask_cancel,
    cond_type_side, scan_through_entire_array
)
from jaxob.jaxob_config import JAXLOB_Configuration

def run_core_tests():
    """Run core functionality tests"""
    cfg = JAXLOB_Configuration()
    key = random.PRNGKey(42)
    
    print("Testing Core Operations...")
    
    # Test 1: Basic order addition
    try:
        empty_orderside = init_orderside(100)
        sample_msg = {
            'type': 1, 'side': 1, 'price': 15000, 'quantity': 100,
            'orderid': 12345, 'traderid': 999, 'time': 34200, 'time_ns': 500000000
        }
        
        result = add_order(empty_orderside, sample_msg)
        assert result[0, 0] == sample_msg['price']
        assert result[0, 1] == sample_msg['quantity']
        print("✅ test_add_order_basic")
    except Exception as e:
        print(f"❌ test_add_order_basic: {e}")
    
    # Test 2: Negative quantity handling
    try:
        msg_negative = sample_msg.copy()
        msg_negative['quantity'] = -50
        result = add_order(empty_orderside, msg_negative)
        assert jnp.all(result == empty_orderside)
        print("✅ test_add_order_negative_quantity")
    except Exception as e:
        print(f"❌ test_add_order_negative_quantity: {e}")
    
    # Test 3: Zero quantity removal
    try:
        orderside = jnp.array([
            [15000, 100, 101, 501, 34200, 500000000],
            [15001, 0, 102, 502, 34200, 600000000],    # Zero quantity
            [15002, -50, 103, 503, 34200, 700000000],  # Negative quantity
            [15003, 200, 104, 504, 34200, 800000000],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        result = _removeZeroNegQuant(orderside)
        assert result[0, 1] == 100  # First order remains
        assert jnp.all(result[1] == -1)  # Zero quantity order removed
        assert jnp.all(result[2] == -1)  # Negative quantity order removed
        assert result[3, 1] == 200  # Last order remains
        print("✅ test_removeZeroNegQuant")
    except Exception as e:
        print(f"❌ test_removeZeroNegQuant: {e}")
    
    # Test 4: Order cancellation
    try:
        filled_orderside = jnp.array([
            [15050, 100, 101, 501, 34200, 500000000],
            [15049, 200, 102, 502, 34200, 600000000],
            [15048, 150, 103, 503, 34200, 700000000],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        msg_cancel = {'orderid': 101, 'quantity': 50, 'price': 15050}
        result = cancel_order(cfg, key, filled_orderside, msg_cancel)
        assert result[0, 1] == 50  # 100 - 50 = 50
        print("✅ test_cancel_order_by_id")
    except Exception as e:
        print(f"❌ test_cancel_order_by_id: {e}")
    
    # Test 5: Best bid/ask
    try:
        # Create filled_orderside with 100 rows
        filled_orderside = jnp.ones((100, 6), dtype=jnp.int32) * -1
        filled_orderside = filled_orderside.at[0].set(jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32))
        filled_orderside = filled_orderside.at[1].set(jnp.array([15049, 200, 102, 502, 34200, 600000000], dtype=jnp.int32))
        
        # Create filled_askside with 100 rows
        filled_askside = jnp.ones((100, 6), dtype=jnp.int32) * -1
        filled_askside = filled_askside.at[0].set(jnp.array([15051, 80, 201, 601, 34200, 500000000], dtype=jnp.int32))
        filled_askside = filled_askside.at[1].set(jnp.array([15052, 120, 202, 602, 34200, 600000000], dtype=jnp.int32))
        
        best_bid = get_best_bid(cfg, filled_orderside)
        best_ask = get_best_ask(cfg, filled_askside)
        assert best_bid == 15050
        assert best_ask == 15051
        print("✅ test_get_best_bid_and_ask")
    except Exception as e:
        print(f"❌ test_get_best_bid_and_ask: {e}")
    
    # Test 6: Volume at price
    try:
        volume = get_volume_at_price(filled_orderside, 15050)
        assert volume == 100
        
        volume_zero = get_volume_at_price(filled_orderside, 14000)
        assert volume_zero == 0
        print("✅ test_get_volume_at_price")
    except Exception as e:
        print(f"❌ test_get_volume_at_price: {e}")
    
    # Test 7: doNothing function
    try:
        empty_trades = jnp.ones((10, 8), dtype=jnp.int32) * -1
        result = doNothing(sample_msg, filled_askside, filled_orderside, empty_trades)
        askside, bidside, trades = result
        assert jnp.array_equal(askside, filled_askside)
        assert jnp.array_equal(bidside, filled_orderside)
        assert jnp.array_equal(trades, empty_trades)
        print("✅ test_doNothing")
    except Exception as e:
        print(f"❌ test_doNothing: {e}")
    
    # Test 8: Bid limit order
    try:
        msg = sample_msg.copy()
        msg['side'] = 1
        msg['price'] = 15040  # Below market
        empty_trades = jnp.ones((100, 8), dtype=jnp.int32) * -1
        result = bid_lim(cfg, msg, filled_askside, empty_orderside, empty_trades)
        result = bid_lim(cfg, msg, filled_askside, empty_orderside, empty_trades)
        # Device put all arguments except config for profiling
        msg_device = jax.device_put(msg)
        filled_askside_device = jax.device_put(filled_askside)
        empty_orderside_device = jax.device_put(empty_orderside)
        empty_trades_device = jax.device_put(empty_trades)
        result = bid_lim(cfg, msg_device, filled_askside_device, empty_orderside_device, empty_trades_device)


        jax.profiler.start_trace("/tmp/profile-data")


        # Now run the function with device-resident arguments
        result1 = bid_lim(cfg, msg_device, filled_askside_device, empty_orderside_device, empty_trades_device)

        result2 = bid_lim(cfg, msg, filled_askside, empty_orderside, empty_trades)
        jax.block_until_ready(result1)
        jax.block_until_ready(result2)
        jax.profiler.stop_trace()
        askside, bidside, trades = result
        assert bidside[0, 0] == msg['price']
        assert bidside[0, 1] == msg['quantity']
        print("✅ test_bid_lim_basic")
    except Exception as e:
        print(f"❌ test_bid_lim_basic: {e}")
    
    # Test 9: Ask limit order
    try:
        msg = sample_msg.copy()
        msg['side'] = -1
        msg['price'] = 15060  # Above market
        
        result = ask_lim(cfg, msg, empty_orderside, filled_orderside, empty_trades)
        askside, bidside, trades = result
        assert askside[0, 0] == msg['price']
        assert askside[0, 1] == msg['quantity']
        print("✅ test_ask_lim_basic")
    except Exception as e:
        print(f"❌ test_ask_lim_basic: {e}")
    
    # Test 10: Conditional type/side routing
    try:
        data = jnp.array([1, 1, 100, 15040, 12345, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (filled_askside, empty_orderside, empty_trades)
        
        result = cond_type_side(cfg, book_state, it_data)
        new_book_state, saved_state = result
        askside, bidside, trades = new_book_state
        assert bidside[0, 0] == 15040
        print("✅ test_cond_type_side_bid_limit")
    except Exception as e:
        print(f"❌ test_cond_type_side_bid_limit: {e}")
    
    # Test 11: Scan through message array
    try:
        msg_array = jnp.array([
            [1, 1, 100, 15040, 12345, 999, 34200, 500000000],   # Bid limit
            [1, -1, 80, 15055, 12346, 998, 34200, 600000000],   # Ask limit
        ], dtype=jnp.int32)
        
        book_state = (empty_orderside, empty_orderside, empty_trades)
        result = scan_through_entire_array(cfg, key, msg_array, book_state)
        askside, bidside, trades = result
        assert askside[0, 0] == 15055
        assert bidside[0, 0] == 15040
        print("✅ test_scan_through_entire_array")
    except Exception as e:
        print(f"❌ test_scan_through_entire_array: {e}")

if __name__ == "__main__":
    print("Running focused core functionality tests...")
    run_core_tests()
    print("\nFocused testing complete!")
