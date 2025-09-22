#!/usr/bin/env python3
"""
Quick plotting script to visualize action distributions from action_dist.csv
Creates a bar chart with actions grouped by Type and averaged across seeds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_process_data(csv_file='action_dist.csv'):
    """Load CSV and process data for plotting"""
    df = pd.read_csv(csv_file)
    
    # Convert action columns to numeric (they appear to be strings in CSV)
    action_cols = ['0-DoNothing', '6-AskSkew', '7-BidSkew']
    for col in action_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Group by Type and calculate mean across seeds
    grouped = df.groupby('Type')[action_cols].mean()
    
    return grouped, action_cols

def create_bar_chart(data, action_cols, output_file='action_distribution_plot.png'):
    """Create grouped bar chart with actions on x-axis and types as grouped bars"""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Number of groups (actions) and bars per group (types)
    n_actions = len(action_cols)
    n_types = len(data.index)
    
    # Set width of bars and positions
    bar_width = 0.2
    r = np.arange(n_actions)
    
    # Create bars for each type
    colors = ['#4D93E2', '#002147', '#EB3557', '#A01A34']
    
    for i, type_name in enumerate(data.index):
        positions = r + i * bar_width
        values = [data.loc[type_name, action] for action in action_cols]
        bars = ax.bar(positions, values, bar_width, 
                     label=type_name, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # Customize the plot
    ax.set_xlabel('Actions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Action Distribution by Type (Averaged Across Three Seeds)', 
                fontsize=18, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(r + bar_width * (n_types - 1) / 2)
    ax.set_xticklabels(action_cols, fontsize=12)
    
    # Add legend
    ax.legend(title='Types', loc='upper right', fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Show the plot
    plt.show()

def print_summary_stats(data):
    """Print summary statistics"""
    print("\nSummary Statistics (Averaged Across Three Seeds):")
    print("=" * 50)
    for type_name in data.index:
        print(f"\n{type_name}:")
        for action in data.columns:
            value = data.loc[type_name, action]
            print(f"  {action}: {value:.2f}%")

def main():
    """Main function"""
    csv_file = 'action_dist.csv'
    
    # Check if CSV file exists
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found in current directory")
        return
    
    try:
        # Load and process data
        print(f"Loading data from {csv_file}...")
        data, action_cols = load_and_process_data(csv_file)
        
        # Print summary statistics
        print_summary_stats(data)
        
        # Create bar chart
        print("\nCreating bar chart...")
        create_bar_chart(data, action_cols)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
