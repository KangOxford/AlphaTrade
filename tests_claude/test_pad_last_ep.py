#!/usr/bin/env python3
"""
Test script for the _pad_last_ep method from LoadLOBSTER_resample class.

This script tests the padding functionality that ensures the last episode
has a length that's a multiple of n_data_msg_per_step.
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock

# Add the project directory to path
sys.path.append('/home/myuser/gymnax_exchange')

# Mock the dependencies that might not be available
class MockModule:
    def __getattr__(self, name):
        return MagicMock()

sys.modules['jax'] = MockModule()
sys.modules['pandas'] = MockModule()
sys.modules['warnings'] = MockModule()

from jaxlobster.lobster_loader import LoadLOBSTER_resample


class TestPadLastEp(unittest.TestCase):
    """Test cases for the _pad_last_ep method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock loader instance with minimal config
        self.loader = LoadLOBSTER_resample(
            datapath="/tmp",  # dummy path
            n_data_msg_per_step=100,  # 100 messages per step
            window_length=1800,
            window_resolution=60
        )
        
    def test_pad_last_ep_basic(self):
        """Test basic padding functionality."""
        print("\n=== Test 1: Basic Padding ===")
        
        # Create sample messages (type, direction, qty, price, trader_id, order_id, time_s, time_ns)
        # 8 columns as per the message format
        n_messages = 250  # Not a multiple of 100
        messages = np.random.randint(1, 1000, size=(n_messages, 8), dtype=np.int32)
        
        # Set realistic time values in the last two columns
        messages[:, -2] = np.arange(34200, 34200 + n_messages)  # time_s (seconds since midnight)
        messages[:, -1] = np.random.randint(0, 1000000000, n_messages)  # time_ns (nanoseconds)
        
        # Create max_msgs_in_windows_arr with the last window having 250 messages
        max_msgs_in_windows_arr = np.array([100, 200, 150, 250], dtype=np.int32)
        
        print(f"Original messages shape: {messages.shape}")
        print(f"Original max_msgs_in_windows_arr: {max_msgs_in_windows_arr}")
        print(f"Last episode length: {max_msgs_in_windows_arr[-1]}")
        print(f"n_data_msg_per_step: {self.loader.n_data_msg_per_step}")
        
        # Call the method
        padded_messages, padded_max_msgs = self.loader._pad_last_ep(
            messages, max_msgs_in_windows_arr
        )
        
        print(f"\nAfter padding:")
        print(f"Padded messages shape: {padded_messages.shape}")
        print(f"Padded max_msgs_in_windows_arr: {padded_max_msgs}")
        print(f"New last episode length: {padded_max_msgs[-1]}")
        
        # Verify the results
        expected_new_length = ((250 // 100) + 1) * 100  # Should be 300
        self.assertEqual(padded_max_msgs[-1], expected_new_length)
        self.assertEqual(padded_messages.shape[0], messages.shape[0] + (expected_new_length - 250))
        
        # Check that the padded rows have correct time values
        original_last_time = messages[-1, -2:]
        expected_new_time = np.array([original_last_time[0] + 1, 0])
        
        print(f"Original last message time: {original_last_time}")
        print(f"Expected padded time: {expected_new_time}")
        
        # Check the first few padded messages
        padding_start_idx = messages.shape[0]
        for i in range(min(5, padded_messages.shape[0] - padding_start_idx)):
            padded_row = padded_messages[padding_start_idx + i]
            print(f"Padded row {i}: {padded_row}")
            # Check that time columns are set correctly
            np.testing.assert_array_equal(padded_row[-2:], expected_new_time)
            # Check that other columns are zero
            np.testing.assert_array_equal(padded_row[:-2], np.zeros(6))
    
    def test_pad_last_ep_already_multiple(self):
        """Test when last episode is already a multiple of n_data_msg_per_step."""
        print("\n=== Test 2: Already Multiple ===")
        
        # Create messages where last episode is exactly 200 (multiple of 100)
        n_messages = 200
        messages = np.random.randint(1, 1000, size=(n_messages, 8), dtype=np.int32)
        messages[:, -2] = np.arange(34200, 34200 + n_messages)
        messages[:, -1] = np.random.randint(0, 1000000000, n_messages)
        
        max_msgs_in_windows_arr = np.array([100, 100], dtype=np.int32)
        
        print(f"Original messages shape: {messages.shape}")
        print(f"Last episode length: {max_msgs_in_windows_arr[-1]} (already multiple of {self.loader.n_data_msg_per_step})")
        
        padded_messages, padded_max_msgs = self.loader._pad_last_ep(
            messages, max_msgs_in_windows_arr
        )
        
        print(f"After padding:")
        print(f"Padded messages shape: {padded_messages.shape}")
        print(f"New last episode length: {padded_max_msgs[-1]}")
        
        # Should still add one full step worth of padding
        expected_new_length = 100 + 100  # Original + one full step
        self.assertEqual(padded_max_msgs[-1], expected_new_length)
    
    def test_pad_last_ep_edge_cases(self):
        """Test edge cases."""
        print("\n=== Test 3: Edge Cases ===")
        
        # Test with very small message count
        messages = np.array([[1, 1, 1, 100, 1, 1, 34200, 0]], dtype=np.int32)
        max_msgs_in_windows_arr = np.array([1], dtype=np.int32)
        
        print(f"Edge case - single message:")
        print(f"Original: {messages.shape}, last episode: {max_msgs_in_windows_arr[-1]}")
        
        padded_messages, padded_max_msgs = self.loader._pad_last_ep(
            messages, max_msgs_in_windows_arr
        )
        
        print(f"After padding: {padded_messages.shape}, new last episode: {padded_max_msgs[-1]}")
        
        expected_new_length = ((1 // 100) + 1) * 100  # Should be 100
        self.assertEqual(padded_max_msgs[-1], expected_new_length)
        self.assertEqual(padded_messages.shape[0], 100)
    
    def test_pad_last_ep_disable_padding(self):
        """Test when n_data_msg_per_step is 0 (padding disabled)."""
        print("\n=== Test 4: Padding Disabled ===")
        
        # Create a loader with padding disabled
        loader_no_pad = LoadLOBSTER_resample(
            datapath="/tmp",
            n_data_msg_per_step=0  # Disable padding
        )
        
        messages = np.random.randint(1, 1000, size=(250, 8), dtype=np.int32)
        max_msgs_in_windows_arr = np.array([100, 150], dtype=np.int32)
        
        # This should trigger the condition where padding is skipped
        print(f"n_data_msg_per_step = {loader_no_pad.n_data_msg_per_step}")
        
        # The run_loading method checks if n_data_msg_per_step != 0 before calling _pad_last_ep
        # So we simulate that condition here
        if loader_no_pad.n_data_msg_per_step != 0:
            padded_messages, padded_max_msgs = loader_no_pad._pad_last_ep(
                messages, max_msgs_in_windows_arr
            )
        else:
            print("Padding skipped as expected when n_data_msg_per_step = 0")
            padded_messages, padded_max_msgs = messages, max_msgs_in_windows_arr
        
        print(f"No padding applied: {padded_messages.shape}, {padded_max_msgs}")


def demonstrate_message_structure():
    """Demonstrate the message structure and what padding does."""
    print("\n" + "="*60)
    print("MESSAGE STRUCTURE DEMONSTRATION")
    print("="*60)
    
    loader = LoadLOBSTER_resample(datapath="/tmp", n_data_msg_per_step=5)  # Small step size for demo
    
    # Create realistic message data
    # Message format: [type, direction, qty, price, trader_id, order_id, time_s, time_ns]
    original_messages = np.array([
        [1, 1, 100, 15050, 123, 123, 34200, 500000000],  # Limit order
        [1, -1, 50, 15049, 124, 124, 34200, 600000000],  # Limit order
        [2, 1, 0, 0, 123, 123, 34200, 700000000],        # Cancel order
        [4, 1, 25, 15050, 125, 125, 34200, 800000000],   # Execution
        [1, -1, 75, 15048, 126, 126, 34200, 900000000],  # Limit order
        [1, 1, 200, 15051, 127, 127, 34201, 100000000],  # Limit order (next second)
        [2, -1, 0, 0, 124, 124, 34201, 200000000],       # Cancel
    ], dtype=np.int32)
    
    # Window array indicating 7 messages in the last (only) window
    max_msgs_array = np.array([7], dtype=np.int32)
    
    print("Original Messages:")
    print("Format: [type, direction, qty, price, trader_id, order_id, time_s, time_ns]")
    print("Types: 1=Limit, 2=Cancel, 4=Execution")
    print("Direction: 1=Buy, -1=Sell")
    for i, msg in enumerate(original_messages):
        print(f"  {i:2d}: {msg}")
    
    print(f"\nOriginal max_msgs_array: {max_msgs_array}")
    print(f"Messages need to be padded to multiple of {loader.n_data_msg_per_step}")
    print(f"Current length: {max_msgs_array[-1]}, needs padding to: {((7//5)+1)*5} = 10")
    
    # Apply padding
    padded_messages, padded_max_msgs = loader._pad_last_ep(original_messages, max_msgs_array)
    
    print(f"\nAfter padding:")
    print(f"New max_msgs_array: {padded_max_msgs}")
    print(f"New message count: {padded_messages.shape[0]}")
    
    print("\nPadded Messages:")
    for i, msg in enumerate(padded_messages):
        marker = " (PADDED)" if i >= len(original_messages) else ""
        print(f"  {i:2d}: {msg}{marker}")
    
    print(f"\nKey observations:")
    print(f"- Padding adds {padded_messages.shape[0] - original_messages.shape[0]} zero messages")
    print(f"- Padded messages have time = last_time + 1 second = {padded_messages[-1, -2:]}")
    print(f"- All other fields in padded messages are zero")
    print(f"- This ensures message array can be reshaped into steps of size {loader.n_data_msg_per_step}")


if __name__ == "__main__":
    print("Testing _pad_last_ep method from LoadLOBSTER_resample")
    print("="*60)
    
    # Run the demonstration first
    demonstrate_message_structure()
    
    # Run the unit tests
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The _pad_last_ep method:")
    print("1. Takes the last episode length from max_msgs_in_windows_arr")
    print("2. Calculates how many messages needed to make it divisible by n_data_msg_per_step")
    print("3. Adds zero-filled messages with incremented timestamp")
    print("4. Updates the last episode length in max_msgs_in_windows_arr")
    print("5. This ensures all episodes can be reshaped into uniform step sizes")
