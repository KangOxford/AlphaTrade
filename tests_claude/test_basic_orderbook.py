"""
Simple test runner for JaxOrderBookArrays basic functionality
"""

import sys
import os
sys.path.append('/home/myuser/gymnax_exchange')

import jax
import jax.numpy as jnp
from jax import random

# Test basic imports
try:
    from jaxob.JaxOrderBookArrays import add_order, init_orderside, get_best_bid, get_best_ask
    from jaxob.jaxob_config import JAXLOB_Configuration
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test basic functionality
def test_basic_functionality():
    """Test basic order book operations"""
    cfg = JAXLOB_Configuration()
    
    # Test init_orderside
    orderside = init_orderside(5)
    assert orderside.shape == (5, 6)
    assert jnp.all(orderside == -1)
    print("✅ init_orderside works")
    
    # Test add_order
    msg = {
        'type': 1,
        'side': 1,
        'price': 15000,
        'quantity': 100,
        'orderid': 12345,
        'traderid': 999,
        'time': 34200,
        'time_ns': 500000000
    }
    
    result = add_order(orderside, msg)
    assert result[0, 0] == 15000
    assert result[0, 1] == 100
    print("✅ add_order works")
    
    # Test get_best_bid
    best_bid = get_best_bid(cfg, result)
    assert best_bid == 15000
    print("✅ get_best_bid works")
    
    # Test get_best_ask with empty book
    empty_asks = init_orderside(5)
    best_ask = get_best_ask(cfg, empty_asks)
    assert best_ask == -1
    print("✅ get_best_ask works")

if __name__ == "__main__":
    print("Testing basic JaxOrderBookArrays functionality...")
    test_basic_functionality()
    print("✅ All basic tests passed!")
