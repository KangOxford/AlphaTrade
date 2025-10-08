#!/usr/bin/env python3
"""
Quick NPZ Inspector

A simple utility to quickly load and inspect NPZ arrays with the LOBSTER format.
Usage: python quick_npz_inspector.py <path_to_npz_file>
"""

import numpy as np
import sys
import os


def quick_inspect_npz(file_path):
    """Quick inspection of an NPZ file with LOBSTER format."""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n🔍 Quick NPZ Inspector")
    print(f"📁 File: {os.path.basename(file_path)}")
    print(f"📏 Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print("=" * 60)
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        print(f"🗂️  Arrays found: {list(data.keys())}")
        print("-" * 60)
        
        # Quick stats for each array
        for name in data.keys():
            arr = data[name]
            print(f"📊 {name}")
            print(f"   Shape: {arr.shape}")
            print(f"   Type:  {arr.dtype}")
            print(f"   Size:  {arr.nbytes / (1024*1024):.2f} MB")
            
            if name == 'msgs' and len(arr) > 0:
                print(f"   First message: {arr[0]}")
                print(f"   Last message:  {arr[-1]}")
                
            elif name in ['starts', 'ends'] and len(arr) > 0:
                print(f"   Range: {arr[0]} to {arr[-1]}")
                print(f"   Count: {len(arr)} windows")
                
            elif name == 'max_msgs_in_windows_arr' and len(arr) > 0:
                print(f"   Min/Max window size: {np.min(arr)} / {np.max(arr)}")
                print(f"   Average window size: {np.mean(arr):.1f}")
                
            elif name == 'obs' and len(arr) > 0:
                print(f"   First observation shape: {arr[0].shape if len(arr.shape) > 1 else 'scalar'}")
                
            print()
        
        # Window analysis for LOBSTER format
        if 'starts' in data and 'ends' in data:
            starts = data['starts']
            ends = data['ends']
            window_lengths = ends - starts
            
            print("🪟 Window Analysis:")
            print(f"   Total windows: {len(starts)}")
            print(f"   Window lengths: {np.min(window_lengths)} - {np.max(window_lengths)} messages")
            print(f"   Average length: {np.mean(window_lengths):.1f} ± {np.std(window_lengths):.1f}")
            print()
        
        # Message format analysis
        if 'msgs' in data:
            msgs = data['msgs']
            print("📝 Message Format Analysis (LOBSTER):")
            print("   Columns: [type, direction, qty, price, trader_id, order_id, time_s, time_ns]")
            if len(msgs) > 0:
                print(f"   Sample message: {msgs[0]}")
                
                # Message type distribution
                if len(msgs.shape) > 1 and msgs.shape[1] >= 1:
                    types = msgs[:, 0]
                    unique_types, counts = np.unique(types, return_counts=True)
                    print("   Message type distribution:")
                    for t, c in zip(unique_types, counts):
                        pct = 100 * c / len(msgs)
                        type_name = {1: "Limit Order", 2: "Cancel", 3: "Delete", 4: "Execution"}.get(int(t), f"Type {int(t)}")
                        print(f"     {type_name}: {c:,} ({pct:.1f}%)")
        
        data.close()
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_npz_inspector.py <path_to_npz_file>")
        print("\nExample:")
        print("python quick_npz_inspector.py /home/myuser/data/rawLOBSTER/AMZN/2017Jan/lobster_AMZN_2017Jan_10_fixed_time_600_50_100_34200_57600.npz")
        return
    
    file_path = sys.argv[1]
    quick_inspect_npz(file_path)


if __name__ == "__main__":
    main()
