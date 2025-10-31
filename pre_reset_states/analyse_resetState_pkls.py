import os
import pickle
import sys
from pathlib import Path
import io

#!/usr/bin/env python3
"""
Helper script to analyze .pkl files and get their uncompressed size.
"""


def get_pickle_size(filepath):
    """Get the uncompressed size of a pickle file by loading it."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Calculate size by re-pickling without compression in memory only
        with io.BytesIO() as buffer:
            pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            uncompressed_size = buffer.tell()
        
        return uncompressed_size, None
    except Exception as e:
        return None, str(e)

def bytes_to_mb(bytes_size):
    """Convert bytes to MB."""
    return bytes_size / (1024 * 1024)

def analyze_pickle_files(directory="."):
    """Analyze all .pkl files in the given directory."""
    directory = Path(directory)
    pkl_files = list(directory.glob("*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found in the directory.")
        return
    
    print(f"Found {len(pkl_files)} .pkl file(s):")
    print("-" * 80)
    print(f"{'Filename':<30} {'File Size':<12} {'Uncompressed':<15} {'Status'}")
    print("-" * 80)
    
    total_file_size = 0
    total_uncompressed = 0
    
    for pkl_file in sorted(pkl_files):
        file_size = pkl_file.stat().st_size
        total_file_size += file_size
        
        uncompressed_size, error = get_pickle_size(pkl_file)
        
        if uncompressed_size is not None:
            total_uncompressed += uncompressed_size
            status = "OK"
            uncomp_str = f"{bytes_to_mb(uncompressed_size):.2f} MB"
        else:
            status = f"ERROR: {error}"
            uncomp_str = "N/A"
        
        print(f"{pkl_file.name:<30} {bytes_to_mb(file_size):>8.2f} MB {uncomp_str:<15} {status}")
    
    print("-" * 80)
    print(f"{'TOTAL':<30} {bytes_to_mb(total_file_size):>8.2f} MB {bytes_to_mb(total_uncompressed):>11.2f} MB")
    
    if total_file_size > 0 and total_uncompressed > 0:
        compression_ratio = (1 - total_file_size / total_uncompressed) * 100
        print(f"Compression ratio: {compression_ratio:.1f}%")

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_pickle_files(directory)