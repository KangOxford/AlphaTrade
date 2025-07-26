"""
Comprehensive test suite for advanced JaxOrderBookArrays functionality
Tests matching functions, helper functions, and edge cases
"""

import sys
import os
sys.path.append('/home/myuser/gymnax_exchange')

import jax
import jax.numpy as jnp
from jax import random

from gymnax_exchange.jaxob.JaxOrderBookArrays import (
    # Advanced matching functions
    match_order, _match_bid_order, _match_ask_order,
    _get_top_bid_order_idx, _get_top_ask_order_idx,
    _check_before_matching_bid, _check_before_matching_ask,
    _match_against_bid_orders, _match_against_ask_orders,
    
    # Helper functions
    add_trade, create_trade, get_agent_trades,
    get_best_bid_and_ask, get_best_bid_and_ask_inclQuants,
    init_msgs_from_l2, get_init_volume_at_price,
    get_order_by_id, get_order_by_id_and_price, get_order_by_time,
    get_order_ids, get_next_executable_order, get_L2_state,
    
    # Cancel message functions
    getCancelMsgs,
    
    # Scan functions with state saving
    scan_through_entire_array_save_states, scan_through_entire_array_save_bidask,
    cond_type_side_save_states, cond_type_side_save_bidask,
    
    # Basic functions needed for setup
    init_orderside, add_order
)
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration

def test_matching_functions():
    """Test order matching functionality"""
    print("\nTesting Matching Functions...")
    cfg = JAXLOB_Configuration()
    key = random.PRNGKey(42)
    
    # Setup test data
    filled_orderside = jnp.array([
        [15050, 100, 101, 501, 34200, 500000000],  # Best bid
        [15049, 200, 102, 502, 34200, 600000000],
        [15048, 150, 103, 503, 34200, 700000000],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=jnp.int32)
    
    filled_askside = jnp.array([
        [15051, 80, 201, 601, 34200, 500000000],   # Best ask
        [15052, 120, 202, 602, 34200, 600000000],
        [15053, 90, 203, 603, 34200, 700000000],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=jnp.int32)
    
    empty_trades = jnp.ones((10, 8), dtype=jnp.int32) * -1
    
    # Test 1: Get top bid order index
    try:
        idx = _get_top_bid_order_idx(cfg, filled_orderside)
        assert idx[0] == 0  # Should return index 0 (highest price)
        print("✅ test_get_top_bid_order_idx")
    except Exception as e:
        print(f"❌ test_get_top_bid_order_idx: {e}")
    
    # Test 2: Get top ask order index
    try:
        idx = _get_top_ask_order_idx(cfg, filled_askside)
        assert idx[0] == 0  # Should return index 0 (lowest price)
        print("✅ test_get_top_ask_order_idx")
    except Exception as e:
        print(f"❌ test_get_top_ask_order_idx: {e}")
    
    # Test 3: Time priority for bids
    try:
        orderside_time_priority = jnp.array([
            [15050, 100, 101, 501, 34201, 0],      # Later time
            [15050, 200, 102, 502, 34200, 0],      # Earlier time - should win
            [15049, 150, 103, 503, 34200, 0],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        idx = _get_top_bid_order_idx(cfg, orderside_time_priority)
        assert idx[0] == 1  # Should return index 1 (earlier time)
        print("✅ test_get_top_bid_order_idx_time_priority")
    except Exception as e:
        print(f"❌ test_get_top_bid_order_idx_time_priority: {e}")
    
    # Test 4: Nanosecond priority
    try:
        orderside_ns_priority = jnp.array([
            [15050, 100, 101, 501, 34200, 600000000],  # Later nanosecond
            [15050, 200, 102, 502, 34200, 500000000],  # Earlier nanosecond - should win
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        idx = _get_top_bid_order_idx(cfg, orderside_ns_priority)
        assert idx[0] == 1  # Should return index 1 (earlier nanosecond)
        print("✅ test_get_top_bid_order_idx_nanosecond_priority")
    except Exception as e:
        print(f"❌ test_get_top_bid_order_idx_nanosecond_priority: {e}")
    
    # Test 5: Check before matching bid - should match
    try:
        top_order_idx = 0
        qtm = 50
        price = 15049  # Below best bid price (15050)
        dummy_vals = [None] * 6
        data_tuple = (top_order_idx, filled_orderside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_bid(data_tuple)
        assert result == True  # Should match
        print("✅ test_check_before_matching_bid_should_match")
    except Exception as e:
        print(f"❌ test_check_before_matching_bid_should_match: {e}")
    
    # Test 6: Check before matching bid - should not match
    try:
        price = 15051  # Above best bid price (15050)
        data_tuple = (top_order_idx, filled_orderside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_bid(data_tuple)
        assert result == False  # Should not match
        print("✅ test_check_before_matching_bid_no_match")
    except Exception as e:
        print(f"❌ test_check_before_matching_bid_no_match: {e}")
    
    # Test 7: Check before matching ask - should match
    try:
        top_order_idx = 0
        qtm = 50
        price = 15052  # Above best ask price (15051)
        data_tuple = (top_order_idx, filled_askside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_ask(data_tuple)
        assert result == True  # Should match
        print("✅ test_check_before_matching_ask_should_match")
    except Exception as e:
        print(f"❌ test_check_before_matching_ask_should_match: {e}")
    
    # Test 8: Basic match order functionality
    try:
        top_order_idx = 0
        orderside = filled_orderside.copy()
        qtm = 50  # Quantity to match
        price = 15000
        trade = empty_trades.copy()
        agrOID = 999
        time = 34201
        time_ns = 0
        agrTID = 888
        side = -1  # Incoming ask
        
        data_tuple = (top_order_idx, orderside, qtm, price, trade, 
                      agrOID, time, time_ns, agrTID, side)
        
        result = match_order(data_tuple)
        new_orderside, new_qtm, _, new_trade, _, _, _, _, _ = result
        
        # Check basic matching worked
        assert new_orderside[0, 1] == 50  # 100 - 50 = 50
        assert new_trade[0, 0] == 15050  # Trade price from standing order
        print("✅ test_match_order_basic")
    except Exception as e:
        print(f"❌ test_match_order_basic: {e}")


def test_helper_functions():
    """Test helper and utility functions"""
    print("\nTesting Helper Functions...")
    cfg = JAXLOB_Configuration()
    
    # Setup test data
    filled_orderside = jnp.array([
        [15050, 100, 101, 501, 34200, 500000000],
        [15049, 200, 102, 502, 34200, 600000000],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=jnp.int32)
    
    filled_askside = jnp.array([
        [15051, 80, 201, 601, 34200, 500000000],
        [15052, 120, 202, 602, 34200, 600000000],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=jnp.int32)
    
    # Test 1: Create trade
    try:
        trade = create_trade(15050, 100, 101, 999, 34200, 500000000, 501, 888)
        expected = jnp.array([15050, 100, 101, 999, 34200, 500000000, 501, 888], dtype=jnp.int32)
        assert jnp.array_equal(trade, expected)
        print("✅ test_create_trade")
    except Exception as e:
        print(f"❌ test_create_trade: {e}")
    
    # Test 2: Add trade to array
    try:
        empty_trades = jnp.ones((5, 8), dtype=jnp.int32) * -1
        new_trade = jnp.array([15050, 100, 101, 999, 34200, 500000000, 501, 888], dtype=jnp.int32)
        
        result = add_trade(empty_trades, new_trade)
        assert jnp.array_equal(result[0], new_trade)
        assert jnp.all(result[1] == -1)  # Rest should remain empty
        print("✅ test_add_trade")
    except Exception as e:
        print(f"❌ test_add_trade: {e}")
    
    # Test 3: Get agent trades
    try:
        trades = jnp.array([
            [15050, 100, 101, 999, 34200, 500000000, 501, 888],  # Agent 501 involved
            [15051, 50, 102, 998, 34200, 600000000, 502, 889],   # Agent 501 not involved
            [15049, 75, 103, 997, 34200, 700000000, 503, 501],   # Agent 501 involved
            [-1, -1, -1, -1, -1, -1, -1, -1],  # Empty
        ], dtype=jnp.int32)
        
        agent_id = 501
        result = get_agent_trades(trades, agent_id)
        assert result[0, 6] == 501  # First trade (passive)
        assert result[2, 7] == 501  # Third trade (aggressive)
        print("✅ test_get_agent_trades")
    except Exception as e:
        print(f"❌ test_get_agent_trades: {e}")
    
    # Test 4: Get best bid and ask with quantities
    try:
        best_ask, best_bid = get_best_bid_and_ask_inclQuants(cfg, filled_askside, filled_orderside)
        assert best_ask[0] == 15051  # Price
        assert best_ask[1] == 80     # Quantity
        assert best_bid[0] == 15050  # Price
        assert best_bid[1] == 100    # Quantity
        print("✅ test_get_best_bid_and_ask_inclQuants")
    except Exception as e:
        print(f"❌ test_get_best_bid_and_ask_inclQuants: {e}")
    
    # Test 5: Initialize messages from L2 data
    try:
        book_l2 = jnp.array([15052, 100, 15053, 150, 15050, 200, 15049, 120], dtype=jnp.int32)
        time = jnp.array([34200, 0], dtype=jnp.int32)
        
        result = init_msgs_from_l2(cfg, book_l2, time)
        assert result.shape[0] == 4  # 2 ask + 2 bid orders
        assert result.shape[1] == 8  # Message format
        assert jnp.all(result[:, 0] == 1)  # All limit orders
        assert result[0, 3] == 15052  # First ask price
        assert result[0, 1] == -1     # Ask side
        print("✅ test_init_msgs_from_l2")
    except Exception as e:
        print(f"❌ test_init_msgs_from_l2: {e}")
    
    # Test 6: Get initial volume at price
    try:
        orderside = jnp.array([
            [15050, 100, cfg.init_id, 501, 34200, 0],        # Init order
            [15050, 50, cfg.init_id-1, 502, 34200, 0],       # Init order
            [15050, 75, 999, 503, 34200, 0],                 # Non-init order
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        volume = get_init_volume_at_price(orderside, 15050, cfg)
        assert volume == 150  # 100 + 50 = 150 (75 excluded)
        print("✅ test_get_init_volume_at_price")
    except Exception as e:
        print(f"❌ test_get_init_volume_at_price: {e}")
    
    # Test 7: Get order by ID
    try:
        order = get_order_by_id(filled_orderside, 101)
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
        print("✅ test_get_order_by_id")
    except Exception as e:
        print(f"❌ test_get_order_by_id: {e}")
    
    # Test 8: Get order by ID (not found)
    try:
        order = get_order_by_id(filled_orderside, 99999)
        expected = jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
        print("✅ test_get_order_by_id_not_found")
    except Exception as e:
        print(f"❌ test_get_order_by_id_not_found: {e}")
    
    # Test 9: Get order by ID and price
    try:
        order = get_order_by_id_and_price(filled_orderside, 101, 15050)
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
        print("✅ test_get_order_by_id_and_price")
    except Exception as e:
        print(f"❌ test_get_order_by_id_and_price: {e}")
    
    # Test 10: Get order by time
    try:
        order = get_order_by_time(filled_orderside, 34200, 500000000)
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
        print("✅ test_get_order_by_time")
    except Exception as e:
        print(f"❌ test_get_order_by_time: {e}")
    
    # Test 11: Get order IDs
    try:
        order_ids = get_order_ids(filled_orderside)
        assert 101 in order_ids
        assert 102 in order_ids
        print("✅ test_get_order_ids")
    except Exception as e:
        print(f"❌ test_get_order_ids: {e}")
    
    # Test 12: Get next executable order (ask)
    try:
        order = get_next_executable_order(cfg, 0, filled_askside)
        expected = jnp.array([15051, 80, 201, 601, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
        print("✅ test_get_next_executable_order_ask")
    except Exception as e:
        print(f"❌ test_get_next_executable_order_ask: {e}")
    
    # Test 13: Get next executable order (bid)
    try:
        order = get_next_executable_order(cfg, 1, filled_orderside)
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
        print("✅ test_get_next_executable_order_bid")
    except Exception as e:
        print(f"❌ test_get_next_executable_order_bid: {e}")
    
    # Test 14: Get L2 state
    try:
        n_levels = 2
        result = get_L2_state(filled_askside, filled_orderside, n_levels, cfg)
        expected_length = n_levels * 4  # 2 levels * (ask_p, ask_q, bid_p, bid_q)
        assert result.shape[0] == expected_length
        assert result[0] == 15051  # Best ask price
        assert result[1] == 80     # Best ask quantity
        print("✅ test_get_L2_state")
    except Exception as e:
        print(f"❌ test_get_L2_state: {e}")


def test_cancel_message_functions():
    """Test cancel message generation"""
    print("\nTesting Cancel Message Functions...")
    
    filled_orderside = jnp.array([
        [15050, 100, 101, 501, 34200, 500000000],
        [15049, 200, 102, 502, 34200, 600000000],
        [15048, 150, 103, 503, 34200, 700000000],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=jnp.int32)
    
    # Test 1: Get cancel messages for agent
    try:
        agentID = 501
        size = 3
        side = 1  # Bid side
        cancel_time = 34300
        cancel_time_ns = 0
        
        result = getCancelMsgs(filled_orderside, agentID, size, side, cancel_time, cancel_time_ns)
        assert result.shape == (size, 8)
        assert result[0, 0] == 2  # Cancel message type
        assert result[0, 1] == side  # Correct side
        assert result[0, 6] == cancel_time  # Cancel time
        print("✅ test_getCancelMsgs_basic")
    except Exception as e:
        print(f"❌ test_getCancelMsgs_basic: {e}")


def test_scan_functions_advanced():
    """Test advanced scan functions with state saving"""
    print("\nTesting Advanced Scan Functions...")
    cfg = JAXLOB_Configuration()
    key = random.PRNGKey(42)
    
    empty_orderside = init_orderside(100)
    empty_trades = jnp.ones((100, 8), dtype=jnp.int32) * -1
    
    # Test 1: Scan with state saving
    try:
        # Generate 100 messages for testing
        # Mix of limit orders (type 1), cancellations (type 2) for both sides
        base_messages = [
            [1, 1, 100, 15040, 12345, 999, 34200, 500000000],   # Bid limit
            [1, -1, 80, 15055, 12346, 998, 34200, 600000000],   # Ask limit
            [1, 1, 150, 15045, 12347, 999, 34201, 500000000],   # Another bid limit
            [1, -1, 120, 15060, 12348, 998, 34201, 600000000],  # Another ask limit
            [2, 1, 0, 15040, 12345, 999, 34202, 0],             # Cancel the first bid
            [1, 1, 200, 15042, 12349, 999, 34203, 500000000],   # New bid after cancel
            [2, -1, 0, 15055, 12346, 998, 34204, 0],            # Cancel the first ask
            [1, -1, 50, 15058, 12350, 998, 34205, 600000000],   # New ask after cancel
        ]
        
        # Create a larger array with 100 messages by extending the pattern
        extended_messages = []
        order_id = 12351
        trade_id_bid = 999
        trade_id_ask = 998
        time = 34206
        
        # Add the base messages first
        extended_messages.extend(base_messages)
        
        # Fill the rest with a mix of orders and cancellations
        while len(extended_messages) < 100:
            # Add a bid limit order
            if len(extended_messages) < 100:
                extended_messages.append([1, 1, 100 + (order_id % 10) * 5, 15040 - (order_id % 7), 
                            order_id, trade_id_bid, time, 500000000 + (order_id % 10) * 10000000])
                order_id += 1
                time += 1
            
            # Add an ask limit order
            if len(extended_messages) < 100:
                extended_messages.append([1, -1, 80 + (order_id % 8) * 5, 15055 + (order_id % 6), 
                            order_id, trade_id_ask, time, 600000000 + (order_id % 10) * 10000000])
                order_id += 1
                time += 1
            
            # Occasionally add a cancellation
            if len(extended_messages) < 100 and order_id % 5 == 0:
                # Cancel a bid (using an earlier order id)
                cancel_id = 12345 + (order_id % 30)
                extended_messages.append([2, 1, 0, 15040, cancel_id, trade_id_bid, time, 0])
                time += 1
            
            if len(extended_messages) < 100 and order_id % 6 == 0:
                # Cancel an ask (using an earlier order id)
                cancel_id = 12346 + (order_id % 30)
                extended_messages.append([2, -1, 0, 15055, cancel_id, trade_id_ask, time, 0])
                time += 1
        
        # Trim to exactly 100 messages
        msg_array = jnp.array(extended_messages[:100], dtype=jnp.int32)
        book_state = (empty_orderside.copy(), empty_orderside.copy(), empty_trades.copy())
        N_steps = 100
        
        result = scan_through_entire_array_save_states(cfg, key, msg_array, book_state, N_steps)

        jax.block_until_ready(result)  # Ensure all computations are complete

        (saved_askside, saved_bidside, final_trades) = result
        
        assert saved_askside.shape[0] == N_steps
        assert saved_bidside.shape[0] == N_steps
        # assert saved_askside[-1, 0, 0] == 15055  # Final ask order
        # assert saved_bidside[-1, 0, 0] == 15040  # Final bid order
        print("✅ test_scan_through_entire_array_save_states")
    except Exception as e:
        print(f"❌ test_scan_through_entire_array_save_states: {e}")
    
    # Test 2: Scan with bid/ask saving
    try:
        result = scan_through_entire_array_save_bidask(cfg, key, msg_array, book_state, N_steps)
        jax.block_until_ready(result)  # Ensure all computations are complete
    
        final_book_state, (saved_best_asks, saved_best_bids) = result
        
        assert saved_best_asks.shape[0] == N_steps
        assert saved_best_bids.shape[0] == N_steps
        # assert saved_best_asks[-1, 0] == 15055  # Best ask price
        # assert saved_best_bids[-1, 0] == 15040  # Best bid price
        print("✅ test_scan_through_entire_array_save_bidask")
    except Exception as e:
        print(f"❌ test_scan_through_entire_array_save_bidask: {e}")
    
    # Test 3: Conditional type/side with state saving
    try:
        data = jnp.array([1, 1, 100, 15040, 12345, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (empty_orderside.copy(), empty_orderside.copy(), empty_trades.copy())

        jax.profiler.start_trace("/tmp/lowleveltraces")
        res = cond_type_side_save_states(cfg, book_state, it_data)
        res = cond_type_side_save_states(cfg, book_state, it_data)

        
        result = cond_type_side_save_states(cfg, book_state, it_data)
        jax.block_until_ready(result)  # Ensure all computations are complete
        jax.profiler.stop_trace()

        new_book_state, saved_book_state = result
        askside, bidside, trades = new_book_state
        saved_askside, saved_bidside, saved_trades = saved_book_state
        
        assert bidside[0, 0] == 15040
        assert jnp.array_equal(saved_askside, askside)
        assert jnp.array_equal(saved_bidside, bidside)
        print("✅ test_cond_type_side_save_states")
    except Exception as e:
        print(f"❌ test_cond_type_side_save_states: {e}")
    
    # Test 4: Conditional type/side with bid/ask saving
    try:
        result = cond_type_side_save_bidask(cfg, book_state, it_data)
        new_book_state, saved_bidask = result
        askside, bidside, trades = new_book_state
        best_ask, best_bid = saved_bidask
        
        assert bidside[0, 0] == 15040
        assert best_bid[0] == 15040  # Best bid price
        print("✅ test_cond_type_side_save_bidask")
        jax.block_until_ready(result)  # Ensure all computations are complete

    except Exception as e:
        print(f"❌ test_cond_type_side_save_bidask: {e}")


def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\nTesting Edge Cases...")
    cfg = JAXLOB_Configuration()
    
    # Test 1: Empty book operations
    try:
        empty_book = init_orderside(5)
        
        best_bid = get_best_bid(cfg, empty_book)
        best_ask = get_best_ask(cfg, empty_book)
        
        assert best_bid == -1
        assert best_ask == -1
        print("✅ test_empty_book_operations")
    except Exception as e:
        print(f"❌ test_empty_book_operations: {e}")
    
    # Test 2: Single order book
    try:
        single_order_book = init_orderside(5)
        single_order_book = single_order_book.at[0, :].set([15050, 100, 101, 501, 34200, 0])
        
        best_bid = get_best_bid(cfg, single_order_book)
        assert best_bid == 15050
        print("✅ test_single_order_book")
    except Exception as e:
        print(f"❌ test_single_order_book: {e}")
    
    # Test 3: Maximum values
    try:
        max_val = cfg.maxint - 1000  # Stay well below max to avoid overflow
        
        order_msg = {
            'type': 1, 'side': 1, 'price': max_val, 'quantity': 1000,
            'orderid': max_val, 'traderid': max_val, 'time': max_val, 'time_ns': max_val
        }
        
        empty_orderside = init_orderside(10)
        result = add_order(empty_orderside, order_msg)
        assert result[0, 0] == max_val
        print("✅ test_maximum_values")
    except Exception as e:
        print(f"❌ test_maximum_values: {e}")
    
    # Test 4: Zero values
    try:
        zero_msg = {
            'type': 1, 'side': 1, 'price': 0, 'quantity': 0,
            'orderid': 0, 'traderid': 0, 'time': 0, 'time_ns': 0
        }
        
        empty_orderside = init_orderside(10)
        result = add_order(empty_orderside, zero_msg)
        assert jnp.all(result == empty_orderside)  # Should not add zero quantity
        print("✅ test_zero_values")
    except Exception as e:
        print(f"❌ test_zero_values: {e}")


if __name__ == "__main__":
    print("Running advanced JaxOrderBookArrays tests...")
    # test_matching_functions()
    # test_helper_functions()
    # test_cancel_message_functions()
    test_scan_functions_advanced()
    test_scan_functions_advanced()

    jax.profiler.start_trace("/tmp/profile-data")
    test_scan_functions_advanced()
    jax.profiler.stop_trace()
    # test_edge_cases()
    print("\n🎉 Advanced testing complete!")
