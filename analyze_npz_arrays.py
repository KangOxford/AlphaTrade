#!/usr/bin/env python3
"""
NPZ Array Analyzer Script

This script loads and analyzes NPZ arrays saved in the LOBSTER format,
examining the structure and contents of:
- msgs: message data arrays  
- starts: window start indices
- ends: window end indices
- obs: observation/orderbook data
- max_msgs_in_windows_arr: maximum messages per window

The script generates a comprehensive readable report of the data structure,
statistics, and sample data for analysis.
"""

import numpy as np
import os
import glob
from datetime import datetime
import argparse


def analyze_npz_file(file_path):
    """
    Analyze a single NPZ file and extract comprehensive information.
    
    Args:
        file_path (str): Path to the NPZ file
    
    Returns:
        dict: Analysis results containing structure, stats, and samples
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        # Load the NPZ file
        data = np.load(file_path, allow_pickle=True)
        
        analysis = {
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'arrays': {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Expected arrays in LOBSTER format
        expected_arrays = ['msgs', 'starts', 'ends', 'obs', 'max_msgs_in_windows_arr']
        
        print(f"File size: {analysis['file_size_mb']:.2f} MB")
        print(f"Arrays found: {list(data.keys())}")
        
        # Analyze each array
        for array_name in data.keys():
            arr = data[array_name]
            
            array_info = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'size': arr.size,
                'memory_mb': arr.nbytes / (1024 * 1024),
                'min_val': None,
                'max_val': None,
                'mean_val': None,
                'std_val': None,
                'sample_data': None
            }
            
            # Statistical analysis for numeric arrays
            if np.issubdtype(arr.dtype, np.number):
                try:
                    array_info['min_val'] = float(np.min(arr))
                    array_info['max_val'] = float(np.max(arr))
                    array_info['mean_val'] = float(np.mean(arr))
                    array_info['std_val'] = float(np.std(arr))
                except:
                    pass
            
            # Sample data extraction
            if array_name == 'msgs':
                # Messages array: show first few messages
                if len(arr.shape) >= 2:
                    array_info['sample_data'] = arr[:min(5, arr.shape[0])]
                else:
                    array_info['sample_data'] = arr[:min(10, len(arr))]
                    
            elif array_name in ['starts', 'ends']:
                # Index arrays: show first 10 values
                array_info['sample_data'] = arr[:min(10, len(arr))]
                
            elif array_name == 'obs':
                # Observation data: show first few observations
                if len(arr.shape) >= 2:
                    array_info['sample_data'] = arr[:min(3, arr.shape[0])]
                else:
                    array_info['sample_data'] = arr[:min(10, len(arr))]
                    
            elif array_name == 'max_msgs_in_windows_arr':
                # Max messages: show distribution statistics
                array_info['sample_data'] = arr[:min(10, len(arr))]
                if len(arr) > 0:
                    array_info['unique_values'] = len(np.unique(arr))
                    array_info['most_common'] = float(np.bincount(arr.flatten()).argmax()) if arr.dtype == int else None
            
            analysis['arrays'][array_name] = array_info
            
        # Cross-array analysis
        if 'starts' in data and 'ends' in data:
            starts = data['starts']
            ends = data['ends']
            
            if len(starts) == len(ends):
                window_lengths = ends - starts
                analysis['window_analysis'] = {
                    'num_windows': len(starts),
                    'min_window_length': int(np.min(window_lengths)),
                    'max_window_length': int(np.max(window_lengths)),
                    'mean_window_length': float(np.mean(window_lengths)),
                    'std_window_length': float(np.std(window_lengths))
                }
        
        data.close()
        return analysis
        
    except Exception as e:
        print(f"ERROR analyzing {file_path}: {e}")
        return None


def print_analysis_report(analysis, output_file=None):
    """
    Print a comprehensive analysis report.
    
    Args:
        analysis (dict): Analysis results from analyze_npz_file
        output_file (file): Optional file handle to write to
    """
    def print_both(text="", file=None):
        print(text)
        if file:
            file.write(text + "\n")
    
    print_both(f"\n📊 ANALYSIS REPORT", output_file)
    print_both(f"File: {os.path.basename(analysis['file_path'])}", output_file)
    print_both(f"Size: {analysis['file_size_mb']:.2f} MB", output_file)
    print_both(f"Analyzed: {analysis['timestamp']}", output_file)
    print_both("-" * 60, output_file)
    
    # Array structure overview
    print_both(f"\n🗂️  ARRAY STRUCTURE", output_file)
    for name, info in analysis['arrays'].items():
        print_both(f"  {name:25} {str(info['shape']):20} {info['dtype']:12} {info['memory_mb']:6.2f}MB", output_file)
    
    # Detailed array analysis
    for name, info in analysis['arrays'].items():
        print_both(f"\n📋 {name.upper()}", output_file)
        print_both(f"  Shape: {info['shape']}", output_file)
        print_both(f"  Data type: {info['dtype']}", output_file)
        print_both(f"  Memory: {info['memory_mb']:.2f} MB", output_file)
        
        if info['min_val'] is not None:
            print_both(f"  Range: [{info['min_val']:.2e}, {info['max_val']:.2e}]", output_file)
            print_both(f"  Mean: {info['mean_val']:.2e} ± {info['std_val']:.2e}", output_file)
        
        if 'unique_values' in info:
            print_both(f"  Unique values: {info['unique_values']}", output_file)
            if info['most_common'] is not None:
                print_both(f"  Most common: {info['most_common']}", output_file)
        
        if info['sample_data'] is not None:
            print_both(f"  Sample data:", output_file)
            sample = info['sample_data']
            if len(sample.shape) == 1:
                print_both(f"    {sample}", output_file)
            else:
                for i, row in enumerate(sample):
                    if i < 3:  # Limit to first 3 rows for readability
                        print_both(f"    [{i}] {row}", output_file)
                    elif i == 3:
                        print_both(f"    ... ({len(sample)-3} more rows)", output_file)
                        break
    
    # Window analysis if available
    if 'window_analysis' in analysis:
        wa = analysis['window_analysis']
        print_both(f"\n🪟 WINDOW ANALYSIS", output_file)
        print_both(f"  Number of windows: {wa['num_windows']}", output_file)
        print_both(f"  Window lengths: {wa['min_window_length']} - {wa['max_window_length']} messages", output_file)
        print_both(f"  Average length: {wa['mean_window_length']:.1f} ± {wa['std_window_length']:.1f}", output_file)


def main():
    parser = argparse.ArgumentParser(description='Analyze NPZ arrays in LOBSTER format')
    parser.add_argument('--pattern', '-p', 
                       default='/home/myuser/data/**/*.npz',
                       help='Glob pattern for NPZ files to analyze')
    parser.add_argument('--output', '-o',
                       default='/home/myuser/npz_analysis_report.txt',
                       help='Output file for the analysis report')
    parser.add_argument('--limit', '-l', type=int, default=10,
                       help='Maximum number of files to analyze')
    
    args = parser.parse_args()
    
    # Find NPZ files
    npz_files = glob.glob(args.pattern, recursive=True)
    
    if not npz_files:
        print(f"No NPZ files found matching pattern: {args.pattern}")
        return
    
    # Sort by file size (largest first)
    npz_files = sorted(npz_files, key=lambda x: os.path.getsize(x), reverse=True)
    npz_files = npz_files[:args.limit]  # Limit number of files
    
    print(f"Found {len(npz_files)} NPZ files to analyze")
    
    all_analyses = []
    
    # Analyze each file
    for file_path in npz_files:
        analysis = analyze_npz_file(file_path)
        if analysis:
            all_analyses.append(analysis)
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print(f"GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*80}")
    
    with open(args.output, 'w') as f:
        f.write("NPZ ARRAY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files analyzed: {len(all_analyses)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("📋 SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'File':40} {'Size (MB)':>10} {'Arrays':>8} {'Windows':>8}\n")
        f.write("-" * 80 + "\n")
        
        for analysis in all_analyses:
            filename = os.path.basename(analysis['file_path'])[:39]
            size_mb = analysis['file_size_mb']
            num_arrays = len(analysis['arrays'])
            num_windows = analysis.get('window_analysis', {}).get('num_windows', 'N/A')
            f.write(f"{filename:40} {size_mb:>10.2f} {num_arrays:>8} {num_windows:>8}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # Detailed analysis for each file
        for analysis in all_analyses:
            print_analysis_report(analysis, f)
            f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Report saved to: {args.output}")
    print(f"📊 Files analyzed: {len(all_analyses)}")
    
    # Print summary to console
    if all_analyses:
        total_size = sum(a['file_size_mb'] for a in all_analyses)
        total_windows = sum(a.get('window_analysis', {}).get('num_windows', 0) for a in all_analyses)
        print(f"💾 Total data size: {total_size:.2f} MB")
        print(f"🪟 Total windows: {total_windows}")


if __name__ == "__main__":
    main()
