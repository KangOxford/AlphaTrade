"""
Comprehensive Test Suite for JaxOrderBookArrays.py

This test suite covers all functions in the JaxOrderBookArrays module with extensive edge cases.
Tests are designed to be run independently and verify correctness, edge cases, and JAX compatibility.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import chex
from typing import Dict, Tuple, Any

# Import the module under test
import sys
import os
sys.path.append('/home/myuser/gymnax_exchange')

# Import all available functions explicitly
from jaxob.JaxOrderBookArrays import (
    # Core operations
    add_order, _removeZeroNegQuant, cancel_order, 
    get_init_id_match, get_random_id_match, get_random_large_id_match,
    
    # Matching functions
    match_order, _match_bid_order, _match_ask_order,
    _get_top_bid_order_idx, _get_top_ask_order_idx,
    _check_before_matching_bid, _check_before_matching_ask,
    _match_against_bid_orders, _match_against_ask_orders,
    
    # Type and side functions
    doNothing, bid_lim, bid_cancel, ask_lim, ask_cancel,
    match_top_order_if_pricematch,
    
    # Branching functions
    cond_type_side, cond_type_side_save_states, cond_type_side_save_bidask,
    
    # Scan functions
    scan_through_entire_array, scan_through_entire_array_save_states,
    scan_through_entire_array_save_bidask,
    
    # Cancel message functions
    getCancelMsgs, getCancelMsgs_smart, remove_cnl_if_renewed,
    
    # Helper functions
    add_trade, create_trade, get_agent_trades, get_volume_at_price,
    get_best_ask, get_best_bid, get_best_bid_and_ask, get_best_bid_and_ask_inclQuants,
    init_orderside, init_msgs_from_l2, get_init_volume_at_price,
    get_order_by_id, get_order_by_id_and_price, get_order_by_time,
    get_order_ids, get_next_executable_order, get_L2_state
)

from jaxob.jaxob_config import JAXLOB_Configuration
import jaxob.jaxob_constants as cst


class TestJaxOrderBookArrays:
    """Test class for all JaxOrderBookArrays functions"""
    
    def setup_method(self):
        """Setup test fixtures before each test"""
        self.cfg = JAXLOB_Configuration()
        self.key = random.PRNGKey(42)
        
        # Standard test data
        self.empty_orderside = init_orderside(10)
        self.sample_msg = {
            'type': 1,
            'side': 1,
            'price': 15000,
            'quantity': 100,
            'orderid': 12345,
            'traderid': 999,
            'time': 34200,
            'time_ns': 500000000
        }
        
        # Pre-filled orderside for testing
        self.filled_orderside = jnp.array([
            [15050, 100, 101, 501, 34200, 500000000],  # Best bid
            [15049, 200, 102, 502, 34200, 600000000],
            [15048, 150, 103, 503, 34200, 700000000],
            [-1, -1, -1, -1, -1, -1],  # Empty slots
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=jnp.int32)
        
        self.filled_askside = jnp.array([
            [15051, 80, 201, 601, 34200, 500000000],   # Best ask
            [15052, 120, 202, 602, 34200, 600000000],
            [15053, 90, 203, 603, 34200, 700000000],
            [-1, -1, -1, -1, -1, -1],  # Empty slots
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=jnp.int32)
        
        # Empty trades array
        self.empty_trades = jnp.ones((10, 8), dtype=jnp.int32) * -1


class TestCoreOperations(TestJaxOrderBookArrays):
    """Test core order book operations"""
    
    def test_add_order_basic(self):
        """Test basic order addition"""
        result = add_order(self.empty_orderside, self.sample_msg)
        
        # Check that order was added to first empty slot
        assert result[0, 0] == self.sample_msg['price']
        assert result[0, 1] == self.sample_msg['quantity']
        assert result[0, 2] == self.sample_msg['orderid']
        assert result[0, 3] == self.sample_msg['traderid']
        assert result[0, 4] == self.sample_msg['time']
        assert result[0, 5] == self.sample_msg['time_ns']
    
    def test_add_order_negative_quantity(self):
        """Test that negative quantities are clamped to 0 and removed"""
        msg_negative = self.sample_msg.copy()
        msg_negative['quantity'] = -50
        
        result = add_order(self.empty_orderside, msg_negative)
        
        # Should not add order with negative quantity
        assert jnp.all(result == self.empty_orderside)
    
    def test_add_order_zero_quantity(self):
        """Test that zero quantities are not added"""
        msg_zero = self.sample_msg.copy()
        msg_zero['quantity'] = 0
        
        result = add_order(self.empty_orderside, msg_zero)
        
        # Should not add order with zero quantity
        assert jnp.all(result == self.empty_orderside)
    
    def test_add_order_full_book(self):
        """Test adding order to full book"""
        # Fill all slots
        full_book = jnp.ones((10, 6), dtype=jnp.int32) * 100
        full_book = full_book.at[:, 0].set(15000)  # Set valid prices
        
        result = add_order(full_book, self.sample_msg)
        
        # Should not change when book is full (no -1 slots available)
        assert jnp.array_equal(result, full_book)
    
    def test_add_order_multiple_orders(self):
        """Test adding multiple orders"""
        orderside = self.empty_orderside.copy()
        
        # Add first order
        orderside = add_order(orderside, self.sample_msg)
        
        # Add second order
        msg2 = self.sample_msg.copy()
        msg2['orderid'] = 54321
        msg2['price'] = 15001
        orderside = add_order(orderside, msg2)
        
        # Check both orders are present
        assert orderside[0, 2] == self.sample_msg['orderid']
        assert orderside[1, 2] == msg2['orderid']
    
    def test_removeZeroNegQuant(self):
        """Test removal of zero/negative quantity orders"""
        orderside = jnp.array([
            [15000, 100, 101, 501, 34200, 500000000],
            [15001, 0, 102, 502, 34200, 600000000],    # Zero quantity
            [15002, -50, 103, 503, 34200, 700000000],  # Negative quantity
            [15003, 200, 104, 504, 34200, 800000000],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        result = _removeZeroNegQuant(orderside)
        
        # Check that zero and negative quantity orders are removed
        assert result[0, 1] == 100  # First order remains
        assert jnp.all(result[1] == -1)  # Zero quantity order removed
        assert jnp.all(result[2] == -1)  # Negative quantity order removed
        assert result[3, 1] == 200  # Last order remains
    
    def test_cancel_order_by_id(self):
        """Test canceling order by order ID"""
        msg_cancel = {
            'orderid': 101,
            'quantity': 50,
            'price': 15050
        }
        
        result = cancel_order(self.cfg, self.key, self.filled_orderside, msg_cancel)
        
        # Check that quantity was reduced
        assert result[0, 1] == 50  # 100 - 50 = 50
        assert result[0, 2] == 101  # Order ID unchanged
    
    def test_cancel_order_complete_removal(self):
        """Test complete order cancellation"""
        msg_cancel = {
            'orderid': 101,
            'quantity': 100,  # Cancel entire quantity
            'price': 15050
        }
        
        result = cancel_order(self.cfg, self.key, self.filled_orderside, msg_cancel)
        
        # Check that order was completely removed
        assert jnp.all(result[0] == -1)
    
    def test_cancel_order_over_quantity(self):
        """Test canceling more than available quantity"""
        msg_cancel = {
            'orderid': 101,
            'quantity': 150,  # More than the 100 available
            'price': 15050
        }
        
        result = cancel_order(self.cfg, self.key, self.filled_orderside, msg_cancel)
        
        # Should remove the entire order
        assert jnp.all(result[0] == -1)
    
    def test_cancel_order_nonexistent_id(self):
        """Test canceling with non-existent order ID"""
        msg_cancel = {
            'orderid': 99999,  # Non-existent ID
            'quantity': 50,
            'price': 15050
        }
        
        result = cancel_order(self.cfg, self.key, self.filled_orderside, msg_cancel)
        
        # Should look for init_id match
        # Since no init_id orders, book should remain largely unchanged
        # Just check the test doesn't crash
        assert result.shape == self.filled_orderside.shape


class TestMatchingFunctions(TestJaxOrderBookArrays):
    """Test order matching functionality"""
    
    def test_match_order_basic(self):
        """Test basic order matching"""
        # Setup: standing bid order and incoming ask
        top_order_idx = 0
        orderside = self.filled_orderside.copy()
        qtm = 50  # Quantity to match
        price = 15000
        trade = self.empty_trades.copy()
        agrOID = 999
        time = 34201
        time_ns = 0
        agrTID = 888
        side = -1  # Incoming ask
        
        data_tuple = (top_order_idx, orderside, qtm, price, trade, 
                      agrOID, time, time_ns, agrTID, side)
        
        result = match_order(data_tuple)
        
        # Unpack result
        new_orderside, new_qtm, _, new_trade, _, _, _, _, _ = result
        
        # Check that standing order quantity was reduced
        assert new_orderside[0, 1] == 50  # 100 - 50 = 50
        
        # Check that incoming quantity was satisfied
        assert new_qtm == 0  # 50 - 100 (clamped to 0) but should be negative handling
        
        # Check trade was recorded
        assert new_trade[0, 0] == 15050  # Trade price from standing order
        assert new_trade[0, 2] == 101    # Standing order ID
        assert new_trade[0, 3] == agrOID # Aggressive order ID
    
    def test_match_order_partial_fill(self):
        """Test partial fill of incoming order"""
        top_order_idx = 0
        orderside = self.filled_orderside.copy()
        qtm = 150  # More than standing order quantity
        price = 15000
        trade = self.empty_trades.copy()
        agrOID = 999
        time = 34201
        time_ns = 0
        agrTID = 888
        side = -1
        
        data_tuple = (top_order_idx, orderside, qtm, price, trade, 
                      agrOID, time, time_ns, agrTID, side)
        
        result = match_order(data_tuple)
        new_orderside, new_qtm, _, new_trade, _, _, _, _, _ = result
        
        # Standing order should be completely filled
        assert jnp.all(new_orderside[0] == -1)  # Order removed
        
        # Incoming order should have remaining quantity
        assert new_qtm == 50  # 150 - 100 = 50
        
        # Trade should record full standing order quantity
        assert new_trade[0, 1] == 100  # Full quantity traded
    
    def test_get_top_bid_order_idx(self):
        """Test finding best bid order index"""
        # Best bid should be highest price, earliest time
        idx = _get_top_bid_order_idx(self.cfg, self.filled_orderside)
        
        # Should return index 0 (highest price: 15050)
        assert idx[0] == 0
    
    def test_get_top_bid_order_idx_time_priority(self):
        """Test bid order priority with same price"""
        # Create orderside with same prices but different times
        orderside = jnp.array([
            [15050, 100, 101, 501, 34201, 0],      # Later time
            [15050, 200, 102, 502, 34200, 0],      # Earlier time - should win
            [15049, 150, 103, 503, 34200, 0],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        idx = _get_top_bid_order_idx(self.cfg, orderside)
        
        # Should return index 1 (same price but earlier time)
        assert idx[0] == 1
    
    def test_get_top_bid_order_idx_nanosecond_priority(self):
        """Test bid order priority with nanosecond precision"""
        orderside = jnp.array([
            [15050, 100, 101, 501, 34200, 600000000],  # Later nanosecond
            [15050, 200, 102, 502, 34200, 500000000],  # Earlier nanosecond - should win
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        idx = _get_top_bid_order_idx(self.cfg, orderside)
        assert idx[0] == 1
    
    def test_get_top_ask_order_idx(self):
        """Test finding best ask order index"""
        idx = _get_top_ask_order_idx(self.cfg, self.filled_askside)
        
        # Should return index 0 (lowest price: 15051)
        assert idx[0] == 0
    
    def test_get_top_ask_order_idx_empty_book(self):
        """Test ask order index with empty book"""
        empty_asks = init_orderside(5)
        idx = _get_top_ask_order_idx(self.cfg, empty_asks)
        
        # Should return -1 for empty book
        assert idx[0] == -1
    
    def test_check_before_matching_bid(self):
        """Test bid matching condition"""
        top_order_idx = 0
        orderside = self.filled_orderside
        qtm = 50
        price = 15049  # Below best bid price
        dummy_vals = [None] * 6
        
        data_tuple = (top_order_idx, orderside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_bid(data_tuple)
        
        # Should match because bid price (15050) >= incoming price (15049)
        assert result == True
    
    def test_check_before_matching_bid_no_overlap(self):
        """Test bid matching condition with no price overlap"""
        top_order_idx = 0
        orderside = self.filled_orderside
        qtm = 50
        price = 15051  # Above best bid price
        dummy_vals = [None] * 6
        
        data_tuple = (top_order_idx, orderside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_bid(data_tuple)
        
        # Should not match because bid price (15050) < incoming price (15051)
        assert result == False
    
    def test_check_before_matching_bid_zero_quantity(self):
        """Test bid matching condition with zero quantity"""
        top_order_idx = 0
        orderside = self.filled_orderside
        qtm = 0  # No quantity to match
        price = 15049
        dummy_vals = [None] * 6
        
        data_tuple = (top_order_idx, orderside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_bid(data_tuple)
        
        # Should not match because qtm = 0
        assert result == False
    
    def test_check_before_matching_ask(self):
        """Test ask matching condition"""
        top_order_idx = 0
        orderside = self.filled_askside
        qtm = 50
        price = 15052  # Above best ask price
        dummy_vals = [None] * 6
        
        data_tuple = (top_order_idx, orderside, qtm, price, *dummy_vals)
        
        result = _check_before_matching_ask(data_tuple)
        
        # Should match because ask price (15051) <= incoming price (15052)
        assert result == True
    
    def test_match_against_bid_orders_full_match(self):
        """Test matching against bid orders with full satisfaction"""
        qtm = 50
        price = 15050
        trade = self.empty_trades.copy()
        agrOID = 999
        time = 34201
        time_ns = 0
        agrTID = 888
        side = -1
        
        result = _match_against_bid_orders(
            self.cfg, self.filled_orderside, qtm, price, trade,
            agrOID, time, time_ns, agrTID, side
        )
        
        new_orderside, remaining_qtm, _, new_trade = result
        
        # Should fully match
        assert remaining_qtm == 0
        assert new_trade[0, 0] == 15050  # Trade at bid price


class TestTypeAndSideFunctions(TestJaxOrderBookArrays):
    """Test order type and side processing functions"""
    
    def test_doNothing(self):
        """Test doNothing function"""
        result = doNothing(
            self.sample_msg, 
            self.filled_askside, 
            self.filled_orderside, 
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Should return everything unchanged
        assert jnp.array_equal(askside, self.filled_askside)
        assert jnp.array_equal(bidside, self.filled_orderside)
        assert jnp.array_equal(trades, self.empty_trades)
    
    def test_bid_lim_basic(self):
        """Test bid limit order processing"""
        msg = self.sample_msg.copy()
        msg['side'] = 1  # Buy side
        msg['price'] = 15040  # Below market, should not match
        
        result = bid_lim(
            self.cfg, msg,
            self.filled_askside,
            self.empty_orderside,
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Should add to bid side without matching
        assert bidside[0, 0] == msg['price']
        assert bidside[0, 1] == msg['quantity']
        
        # Ask side should be unchanged
        assert jnp.array_equal(askside, self.filled_askside)
    
    def test_bid_lim_with_matching(self):
        """Test bid limit order that matches against asks"""
        msg = self.sample_msg.copy()
        msg['side'] = 1  # Buy side
        msg['price'] = 15052  # Above best ask, should match
        msg['quantity'] = 50
        
        result = bid_lim(
            self.cfg, msg,
            self.filled_askside,
            self.empty_orderside,
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Should match against ask side
        assert askside[0, 1] == 30  # 80 - 50 = 30 remaining
        
        # Should record trade
        assert trades[0, 0] == 15051  # Trade at ask price
    
    def test_ask_lim_basic(self):
        """Test ask limit order processing"""
        msg = self.sample_msg.copy()
        msg['side'] = -1  # Sell side
        msg['price'] = 15060  # Above market, should not match
        
        result = ask_lim(
            self.cfg, msg,
            self.empty_orderside,  # Empty ask side
            self.filled_orderside,
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Should add to ask side without matching
        assert askside[0, 0] == msg['price']
        assert askside[0, 1] == msg['quantity']
        
        # Bid side should be unchanged
        assert jnp.array_equal(bidside, self.filled_orderside)
    
    def test_bid_cancel(self):
        """Test bid cancellation"""
        msg_cancel = {
            'orderid': 101,
            'quantity': 50,
            'price': 15050
        }
        
        result = bid_cancel(
            self.cfg, self.key, msg_cancel,
            self.filled_askside,
            self.filled_orderside,
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Ask side should be unchanged
        assert jnp.array_equal(askside, self.filled_askside)
        
        # Bid side should have reduced quantity
        assert bidside[0, 1] == 50  # 100 - 50 = 50
        
        # Trades should be unchanged
        assert jnp.array_equal(trades, self.empty_trades)
    
    def test_ask_cancel(self):
        """Test ask cancellation"""
        msg_cancel = {
            'orderid': 201,
            'quantity': 30,
            'price': 15051
        }
        
        result = ask_cancel(
            self.cfg, self.key, msg_cancel,
            self.filled_askside,
            self.filled_orderside,
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Bid side should be unchanged
        assert jnp.array_equal(bidside, self.filled_orderside)
        
        # Ask side should have reduced quantity
        assert askside[0, 1] == 50  # 80 - 30 = 50
        
        # Trades should be unchanged
        assert jnp.array_equal(trades, self.empty_trades)


class TestBranchingFunctions(TestJaxOrderBookArrays):
    """Test conditional branching functions"""
    
    def test_cond_type_side_bid_limit(self):
        """Test conditional routing for bid limit order"""
        key = self.key
        data = jnp.array([1, 1, 100, 15040, 12345, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.filled_askside, self.empty_orderside, self.empty_trades)
        
        result = cond_type_side(self.cfg, book_state, it_data)
        
        new_book_state, saved_state = result
        askside, bidside, trades = new_book_state
        
        # Should add bid order
        assert bidside[0, 0] == 15040
        assert bidside[0, 1] == 100
        assert saved_state == 0  # Nothing saved in basic version
    
    def test_cond_type_side_ask_limit(self):
        """Test conditional routing for ask limit order"""
        key = self.key
        data = jnp.array([1, -1, 100, 15060, 12345, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.empty_orderside, self.filled_orderside, self.empty_trades)
        
        result = cond_type_side(self.cfg, book_state, it_data)
        
        new_book_state, saved_state = result
        askside, bidside, trades = new_book_state
        
        # Should add ask order
        assert askside[0, 0] == 15060
        assert askside[0, 1] == 100
    
    def test_cond_type_side_bid_cancel(self):
        """Test conditional routing for bid cancel order"""
        key = self.key
        data = jnp.array([2, 1, 50, 15050, 101, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.filled_askside, self.filled_orderside, self.empty_trades)
        
        result = cond_type_side(self.cfg, book_state, it_data)
        
        new_book_state, saved_state = result
        askside, bidside, trades = new_book_state
        
        # Should cancel bid order
        assert bidside[0, 1] == 50  # 100 - 50 = 50
    
    def test_cond_type_side_ask_cancel(self):
        """Test conditional routing for ask cancel order"""
        key = self.key
        data = jnp.array([2, -1, 30, 15051, 201, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.filled_askside, self.filled_orderside, self.empty_trades)
        
        result = cond_type_side(self.cfg, book_state, it_data)
        
        new_book_state, saved_state = result
        askside, bidside, trades = new_book_state
        
        # Should cancel ask order
        assert askside[0, 1] == 50  # 80 - 30 = 50
    
    def test_cond_type_side_do_nothing(self):
        """Test conditional routing for do nothing case"""
        key = self.key
        data = jnp.array([0, 0, 0, 0, 0, 0, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.filled_askside, self.filled_orderside, self.empty_trades)
        
        result = cond_type_side(self.cfg, book_state, it_data)
        
        new_book_state, saved_state = result
        askside, bidside, trades = new_book_state
        
        # Should do nothing - everything unchanged
        assert jnp.array_equal(askside, self.filled_askside)
        assert jnp.array_equal(bidside, self.filled_orderside)
        assert jnp.array_equal(trades, self.empty_trades)
    
    def test_cond_type_side_save_states(self):
        """Test conditional routing with state saving"""
        key = self.key
        data = jnp.array([1, 1, 100, 15040, 12345, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.filled_askside, self.empty_orderside, self.empty_trades)
        
        result = cond_type_side_save_states(self.cfg, book_state, it_data)
        
        new_book_state, saved_book_state = result
        askside, bidside, trades = new_book_state
        saved_askside, saved_bidside, saved_trades = saved_book_state
        
        # Should add bid order
        assert bidside[0, 0] == 15040
        assert bidside[0, 1] == 100
        
        # Saved state should match new state
        assert jnp.array_equal(saved_askside, askside)
        assert jnp.array_equal(saved_bidside, bidside)
        assert jnp.array_equal(saved_trades, trades)
    
    def test_cond_type_side_save_bidask(self):
        """Test conditional routing with bid/ask saving"""
        key = self.key
        data = jnp.array([1, 1, 100, 15040, 12345, 999, 34200, 500000000], dtype=jnp.int32)
        it_data = (key, data)
        book_state = (self.filled_askside, self.filled_orderside, self.empty_trades)
        
        result = cond_type_side_save_bidask(self.cfg, book_state, it_data)
        
        new_book_state, saved_bidask = result
        askside, bidside, trades = new_book_state
        best_ask, best_bid = saved_bidask
        
        # Should add bid order
        assert bidside[0, 0] == 15040
        
        # Should save best bid/ask with quantities
        assert best_ask[0] == 15051  # Best ask price
        assert best_ask[1] == 80     # Best ask quantity
        assert best_bid[0] == 15050  # Best bid price (from original filled side)
        assert best_bid[1] == 100    # Best bid quantity


class TestScanFunctions(TestJaxOrderBookArrays):
    """Test scan wrapper functions"""
    
    def test_scan_through_entire_array(self):
        """Test scanning through message array"""
        # Create a small message array
        msg_array = jnp.array([
            [1, 1, 100, 15040, 12345, 999, 34200, 500000000],   # Bid limit
            [1, -1, 80, 15055, 12346, 998, 34200, 600000000],   # Ask limit
            [2, 1, 50, 15040, 12345, 999, 34200, 700000000],    # Bid cancel
        ], dtype=jnp.int32)
        
        book_state = (self.empty_orderside, self.empty_orderside, self.empty_trades)
        
        result = scan_through_entire_array(self.cfg, self.key, msg_array, book_state)
        
        askside, bidside, trades = result
        
        # Should have processed all messages
        assert askside[0, 0] == 15055   # Ask order added
        assert bidside[0, 1] == 50      # Bid order added then partially cancelled
    
    def test_scan_through_entire_array_save_states(self):
        """Test scanning with state saving"""
        msg_array = jnp.array([
            [1, 1, 100, 15040, 12345, 999, 34200, 500000000],   # Bid limit
            [1, -1, 80, 15055, 12346, 998, 34200, 600000000],   # Ask limit
        ], dtype=jnp.int32)
        
        book_state = (self.empty_orderside, self.empty_orderside, self.empty_trades)
        N_steps = 2
        
        result = scan_through_entire_array_save_states(
            self.cfg, self.key, msg_array, book_state, N_steps
        )
        
        (saved_askside, saved_bidside, final_trades) = result
        
        # Should have saved states for last N_steps
        assert saved_askside.shape[0] == N_steps
        assert saved_bidside.shape[0] == N_steps
        
        # Final state should have both orders
        assert saved_askside[-1, 0, 0] == 15055  # Final ask order
        assert saved_bidside[-1, 0, 0] == 15040  # Final bid order
    
    def test_scan_through_entire_array_save_bidask(self):
        """Test scanning with bid/ask saving"""
        msg_array = jnp.array([
            [1, 1, 100, 15040, 12345, 999, 34200, 500000000],   # Bid limit
            [1, -1, 80, 15055, 12346, 998, 34200, 600000000],   # Ask limit
        ], dtype=jnp.int32)
        
        book_state = (self.empty_orderside, self.empty_orderside, self.empty_trades)
        N_steps = 2
        
        result = scan_through_entire_array_save_bidask(
            self.cfg, self.key, msg_array, book_state, N_steps
        )
        
        final_book_state, (saved_best_asks, saved_best_bids) = result
        
        # Should have saved bid/ask for last N_steps
        assert saved_best_asks.shape[0] == N_steps
        assert saved_best_bids.shape[0] == N_steps
        
        # Final best prices should match
        assert saved_best_asks[-1, 0] == 15055  # Best ask price
        assert saved_best_bids[-1, 0] == 15040  # Best bid price


class TestCancelMessageFunctions(TestJaxOrderBookArrays):
    """Test cancel message generation functions"""
    
    def test_getCancelMsgs_basic(self):
        """Test basic cancel message generation"""
        agentID = 501
        size = 3
        side = 1  # Bid side
        cancel_time = 34300
        cancel_time_ns = 0
        
        result = getCancelMsgs(
            self.filled_orderside, agentID, size, side, cancel_time, cancel_time_ns
        )
        
        # Should return cancel messages for agent orders
        assert result.shape == (size, 8)
        assert result[0, 0] == 2  # Cancel message type
        assert result[0, 1] == side  # Correct side
        assert result[0, 4] == 101  # Order ID from agent
        assert result[0, 6] == cancel_time  # Cancel time
    
    def test_getCancelMsgs_no_agent_orders(self):
        """Test cancel message generation with no agent orders"""
        agentID = 99999  # Non-existent agent
        size = 3
        side = 1
        cancel_time = 34300
        cancel_time_ns = 0
        
        result = getCancelMsgs(
            self.filled_orderside, agentID, size, side, cancel_time, cancel_time_ns
        )
        
        # Should return empty cancel messages
        assert result.shape == (size, 8)
        # Messages should be zero/empty since no matching agent orders


class TestHelperFunctions(TestJaxOrderBookArrays):
    """Test helper and utility functions"""
    
    def test_add_trade(self):
        """Test adding trade to trades array"""
        new_trade = jnp.array([15050, 100, 101, 999, 34200, 500000000, 501, 888], dtype=jnp.int32)
        
        result = add_trade(self.empty_trades, new_trade)
        
        # Should add trade to first empty slot
        assert jnp.array_equal(result[0], new_trade)
        assert jnp.all(result[1] == -1)  # Rest should remain empty
    
    def test_create_trade(self):
        """Test trade creation"""
        price = 15050
        quant = 100
        passOID = 101
        agrOID = 999
        time = 34200
        time_ns = 500000000
        passTID = 501
        agrTID = 888
        
        result = create_trade(price, quant, passOID, agrOID, time, time_ns, passTID, agrTID)
        
        expected = jnp.array([15050, 100, 101, 999, 34200, 500000000, 501, 888], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)
    
    def test_get_agent_trades(self):
        """Test filtering trades by agent ID"""
        # Create some trades
        trades = jnp.array([
            [15050, 100, 101, 999, 34200, 500000000, 501, 888],  # Agent 501 involved
            [15051, 50, 102, 998, 34200, 600000000, 502, 889],   # Agent 501 not involved
            [15049, 75, 103, 997, 34200, 700000000, 503, 501],   # Agent 501 involved (as aggressor)
            [-1, -1, -1, -1, -1, -1, -1, -1],  # Empty
        ], dtype=jnp.int32)
        
        agent_id = 501
        result = get_agent_trades(trades, agent_id)
        
        # Should return only trades involving agent 501
        assert result[0, 6] == 501  # First trade (passive)
        assert jnp.all(result[1] == 0)  # Second trade zeroed out
        assert result[2, 7] == 501  # Third trade (aggressive)
        assert jnp.all(result[3] == 0)  # Empty trade zeroed out
    
    def test_get_volume_at_price(self):
        """Test volume calculation at specific price"""
        # Add another order at same price
        orderside = self.filled_orderside.copy()
        orderside = orderside.at[3, :].set([15050, 50, 104, 504, 34200, 800000000])
        
        volume = get_volume_at_price(orderside, 15050)
        
        # Should sum quantities at price 15050
        assert volume == 150  # 100 + 50 = 150
    
    def test_get_volume_at_price_no_orders(self):
        """Test volume calculation with no orders at price"""
        volume = get_volume_at_price(self.filled_orderside, 14000)
        
        # Should return 0 for non-existent price
        assert volume == 0
    
    def test_get_best_ask(self):
        """Test best ask price retrieval"""
        best_ask = get_best_ask(self.cfg, self.filled_askside)
        
        # Should return lowest ask price
        assert best_ask == 15051
    
    def test_get_best_ask_empty_book(self):
        """Test best ask with empty book"""
        empty_asks = init_orderside(5)
        best_ask = get_best_ask(self.cfg, empty_asks)
        
        # Should return -1 for empty book
        assert best_ask == -1
    
    def test_get_best_bid(self):
        """Test best bid price retrieval"""
        best_bid = get_best_bid(self.cfg, self.filled_orderside)
        
        # Should return highest bid price
        assert best_bid == 15050
    
    def test_get_best_bid_empty_book(self):
        """Test best bid with empty book"""
        empty_bids = init_orderside(5)
        best_bid = get_best_bid(self.cfg, empty_bids)
        
        # Should return -1 for empty book
        assert best_bid == -1
    
    def test_get_best_bid_and_ask(self):
        """Test getting both best bid and ask"""
        best_ask, best_bid = get_best_bid_and_ask(self.cfg, self.filled_askside, self.filled_orderside)
        
        assert best_ask == 15051
        assert best_bid == 15050
    
    def test_get_best_bid_and_ask_inclQuants(self):
        """Test getting best bid/ask with quantities"""
        best_ask, best_bid = get_best_bid_and_ask_inclQuants(
            self.cfg, self.filled_askside, self.filled_orderside
        )
        
        # Should return price and quantity arrays
        assert best_ask[0] == 15051  # Price
        assert best_ask[1] == 80     # Quantity
        assert best_bid[0] == 15050  # Price
        assert best_bid[1] == 100    # Quantity
    
    def test_init_orderside(self):
        """Test orderside initialization"""
        n_orders = 5
        result = init_orderside(n_orders)
        
        # Should create array filled with -1
        expected_shape = (n_orders, 6)
        assert result.shape == expected_shape
        assert jnp.all(result == -1)
        assert result.dtype == jnp.int32
    
    def test_init_msgs_from_l2(self):
        """Test message initialization from L2 data"""
        # Create sample L2 book state: [ask_p1, ask_q1, ask_p2, ask_q2, bid_p1, bid_q1, bid_p2, bid_q2]
        book_l2 = jnp.array([15052, 100, 15053, 150, 15050, 200, 15049, 120], dtype=jnp.int32)
        time = jnp.array([34200, 0], dtype=jnp.int32)
        
        result = init_msgs_from_l2(self.cfg, book_l2, time)
        
        # Should create limit order messages to recreate book state
        assert result.shape[0] == 4  # 2 ask + 2 bid orders
        assert result.shape[1] == 8  # Message format
        
        # Check message types are all limit orders (type 1)
        assert jnp.all(result[:, 0] == 1)
        
        # Check prices are correctly assigned
        assert result[0, 3] == 15052  # First ask price
        assert result[1, 3] == 15050  # First bid price
        
        # Check sides are correctly assigned
        assert result[0, 1] == -1  # Ask side
        assert result[1, 1] == 1   # Bid side
    
    def test_get_init_volume_at_price(self):
        """Test getting initial volume at price"""
        # Create orderside with init_id orders
        orderside = jnp.array([
            [15050, 100, self.cfg.init_id, 501, 34200, 0],      # Init order
            [15050, 50, self.cfg.init_id-1, 502, 34200, 0],     # Init order
            [15050, 75, 999, 503, 34200, 0],                    # Non-init order
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        volume = get_init_volume_at_price(orderside, 15050, self.cfg)
        
        # Should sum only init orders at price
        assert volume == 150  # 100 + 50 = 150 (75 excluded as non-init)
    
    def test_get_order_by_id(self):
        """Test getting order by ID"""
        order = get_order_by_id(self.filled_orderside, 101)
        
        # Should return the order with ID 101
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_order_by_id_not_found(self):
        """Test getting order by non-existent ID"""
        order = get_order_by_id(self.filled_orderside, 99999)
        
        # Should return array of -1s
        expected = jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_order_by_id_and_price(self):
        """Test getting order by ID and price"""
        order = get_order_by_id_and_price(self.filled_orderside, 101, 15050)
        
        # Should return the order with ID 101 at price 15050
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_order_by_id_and_price_wrong_price(self):
        """Test getting order by ID with wrong price"""
        order = get_order_by_id_and_price(self.filled_orderside, 101, 15000)
        
        # Should return array of -1s since price doesn't match
        expected = jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_order_by_time(self):
        """Test getting order by timestamp"""
        order = get_order_by_time(self.filled_orderside, 34200, 500000000)
        
        # Should return the order with matching timestamp
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_order_by_time_not_found(self):
        """Test getting order by non-existent timestamp"""
        order = get_order_by_time(self.filled_orderside, 99999, 99999)
        
        # Should return array of -2s (specific to this function)
        expected = jnp.array([-2, -2, -2, -2, -2, -2], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_order_ids(self):
        """Test getting all order IDs"""
        order_ids = get_order_ids(self.filled_orderside)
        
        # Should return unique order IDs
        assert 101 in order_ids
        assert 102 in order_ids
        assert 103 in order_ids
    
    def test_get_next_executable_order_ask(self):
        """Test getting next executable ask order"""
        order = get_next_executable_order(self.cfg, 0, self.filled_askside)
        
        # Should return best ask order
        expected = jnp.array([15051, 80, 201, 601, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_next_executable_order_bid(self):
        """Test getting next executable bid order"""
        order = get_next_executable_order(self.cfg, 1, self.filled_orderside)
        
        # Should return best bid order
        expected = jnp.array([15050, 100, 101, 501, 34200, 500000000], dtype=jnp.int32)
        assert jnp.array_equal(order, expected)
    
    def test_get_L2_state(self):
        """Test L2 state generation"""
        n_levels = 2
        
        result = get_L2_state(self.filled_askside, self.filled_orderside, n_levels, self.cfg)
        
        # Should return flattened array: [ask_p1, ask_q1, ask_p2, ask_q2, bid_p1, bid_q1, bid_p2, bid_q2]
        expected_length = n_levels * 4  # 2 levels * (ask_p, ask_q, bid_p, bid_q)
        assert result.shape[0] == expected_length
        
        # Check prices are in correct order
        assert result[0] == 15051  # Best ask price
        assert result[1] == 80     # Best ask quantity
        assert result[4] == 15050  # Best bid price
        assert result[5] == 100    # Best bid quantity


class TestEdgeCases(TestJaxOrderBookArrays):
    """Test extreme edge cases and error conditions"""
    
    def test_empty_book_operations(self):
        """Test operations on completely empty book"""
        empty_book = init_orderside(5)
        
        # Test best bid/ask on empty book
        best_bid = get_best_bid(self.cfg, empty_book)
        best_ask = get_best_ask(self.cfg, empty_book)
        
        assert best_bid == -1
        assert best_ask == -1
        
        # Test volume on empty book
        volume = get_volume_at_price(empty_book, 15000)
        assert volume == 0
    
    def test_single_order_book(self):
        """Test operations with single order"""
        single_order_book = init_orderside(5)
        single_order_book = single_order_book.at[0, :].set([15050, 100, 101, 501, 34200, 0])
        
        best_bid = get_best_bid(self.cfg, single_order_book)
        assert best_bid == 15050
        
        volume = get_volume_at_price(single_order_book, 15050)
        assert volume == 100
    
    def test_maximum_values(self):
        """Test with maximum integer values"""
        max_val = self.cfg.maxint
        
        order_msg = {
            'type': 1,
            'side': 1,
            'price': max_val - 1,  # Just below max to avoid overflow
            'quantity': max_val - 1,
            'orderid': max_val - 1,
            'traderid': max_val - 1,
            'time': max_val - 1,
            'time_ns': max_val - 1
        }
        
        result = add_order(self.empty_orderside, order_msg)
        
        # Should handle maximum values correctly
        assert result[0, 0] == max_val - 1
        assert result[0, 1] == max_val - 1
    
    def test_zero_values(self):
        """Test with zero values"""
        zero_msg = {
            'type': 1,
            'side': 1,
            'price': 0,
            'quantity': 0,
            'orderid': 0,
            'traderid': 0,
            'time': 0,
            'time_ns': 0
        }
        
        result = add_order(self.empty_orderside, zero_msg)
        
        # Zero quantity should not be added
        assert jnp.all(result == self.empty_orderside)
    
    def test_negative_values_handling(self):
        """Test handling of negative values"""
        negative_msg = {
            'type': 1,
            'side': 1,
            'price': -100,
            'quantity': -50,
            'orderid': -123,
            'traderid': -456,
            'time': -789,
            'time_ns': -999
        }
        
        result = add_order(self.empty_orderside, negative_msg)
        
        # Negative quantity should be clamped and order not added
        assert jnp.all(result == self.empty_orderside)
    
    def test_duplicate_order_ids(self):
        """Test handling of duplicate order IDs"""
        # Add first order
        orderside = add_order(self.empty_orderside, self.sample_msg)
        
        # Add second order with same ID but different price
        duplicate_msg = self.sample_msg.copy()
        duplicate_msg['price'] = 15001
        orderside = add_order(orderside, duplicate_msg)
        
        # Both orders should be present (system allows duplicate IDs)
        assert orderside[0, 2] == self.sample_msg['orderid']
        assert orderside[1, 2] == duplicate_msg['orderid']
        assert orderside[0, 0] != orderside[1, 0]  # Different prices
    
    def test_cancel_more_than_available(self):
        """Test canceling more quantity than available"""
        cancel_msg = {
            'orderid': 101,
            'quantity': 1000,  # Much more than available
            'price': 15050
        }
        
        result = cancel_order(self.cfg, self.key, self.filled_orderside, cancel_msg)
        
        # Should remove entire order
        assert jnp.all(result[0] == -1)
    
    def test_match_zero_quantity_order(self):
        """Test matching with zero quantity"""
        # Create data tuple with zero quantity to match
        top_order_idx = 0
        orderside = self.filled_orderside.copy()
        qtm = 0  # Zero quantity
        price = 15050
        trade = self.empty_trades.copy()
        
        data_tuple = (top_order_idx, orderside, qtm, price, trade, 999, 34200, 0, 888, 1)
        
        result = match_order(data_tuple)
        new_orderside, new_qtm, _, new_trade, _, _, _, _, _ = result
        
        # Should not affect orderside with zero quantity
        assert jnp.array_equal(new_orderside, orderside)
        assert new_qtm == 0
    
    def test_time_priority_edge_cases(self):
        """Test time priority with edge cases"""
        # Create orders with same price but extreme time differences
        orderside = jnp.array([
            [15050, 100, 101, 501, 0, 0],              # Earliest possible time
            [15050, 100, 102, 502, self.cfg.maxint, 999999999],  # Latest time
            [15050, 100, 103, 503, 34200, 500000000],  # Middle time
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=jnp.int32)
        
        idx = _get_top_bid_order_idx(self.cfg, orderside)
        
        # Should return earliest time (index 0)
        assert idx[0] == 0
    
    def test_cross_price_scenarios(self):
        """Test scenarios where bid >= ask (crossed market)"""
        # Create crossed market scenario
        high_bid_msg = {
            'type': 1,
            'side': 1,
            'price': 15055,  # Higher than best ask (15051)
            'quantity': 100,
            'orderid': 12345,
            'traderid': 999,
            'time': 34201,
            'time_ns': 0
        }
        
        result = bid_lim(
            self.cfg, high_bid_msg,
            self.filled_askside,
            self.empty_orderside,
            self.empty_trades
        )
        
        askside, bidside, trades = result
        
        # Should match against ask orders
        # The exact behavior depends on matching logic implementation
        # At minimum, should create some trades
        assert not jnp.all(trades == -1)  # Some trades should occur
    
    def test_large_message_arrays(self):
        """Test with large message arrays"""
        # Create a large array of messages
        n_messages = 1000
        msg_array = jnp.zeros((n_messages, 8), dtype=jnp.int32)
        
        # Fill with alternating bid/ask limit orders
        for i in range(n_messages):
            side = 1 if i % 2 == 0 else -1
            price = 15050 + (i % 10) - 5  # Vary prices around 15050
            msg_array = msg_array.at[i, :].set([1, side, 100, price, 10000+i, 999, 34200, i])
        
        book_state = (self.empty_orderside.copy(), self.empty_orderside.copy(), self.empty_trades.copy())
        
        # This should not crash and should handle the large array
        result = scan_through_entire_array(self.cfg, self.key, msg_array, book_state)
        
        askside, bidside, trades = result
        
        # Should have some orders in the book
        assert not jnp.all(askside == -1)
        assert not jnp.all(bidside == -1)


def run_all_tests():
    """Run all test classes"""
    test_classes = [
        TestCoreOperations,
        TestMatchingFunctions, 
        TestTypeAndSideFunctions,
        TestBranchingFunctions,
        TestScanFunctions,
        TestCancelMessageFunctions,
        TestHelperFunctions,
        TestEdgeCases
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*50}")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                # Create instance and run setup
                instance = test_class()
                instance.setup_method()
                
                # Run the test method
                method = getattr(instance, method_name)
                method()
                
                print(f"✅ {method_name}")
                passed += 1
                
            except Exception as e:
                print(f"❌ {method_name}: {str(e)}")
                failed += 1
                errors.append(f"{test_class.__name__}.{method_name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if errors:
        print(f"\nFAILED TESTS:")
        for error in errors:
            print(f"  - {error}")
    
    return passed, failed, errors


if __name__ == "__main__":
    print("Starting comprehensive tests for JaxOrderBookArrays...")
    run_all_tests()
