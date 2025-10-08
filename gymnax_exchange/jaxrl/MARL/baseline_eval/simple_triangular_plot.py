#!/usr/bin/env python3
"""
Simple script to create triangular heatmap showing MarketMaker and ExecutionAgent rewards
for the 4 combinations: BB, BL, LB, LL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import argparse
import sys

def load_data(csv_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print("Loaded data:")
        print(df)
        return df
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def create_triangular_plot(df, reward_label='Reward', save_path=None):
    """Create triangular plot with clear mapping"""
    
    # Use mean aggregation by default for multiple episodes
    agg_data = df.groupby('Configuration')[['MarketMaker', 'ExecutionAgent']].mean()
    
    print(f"Aggregated data (using mean):")
    print(agg_data)
    
    # Create a 2x2 grid
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define the grid positions for each configuration
    # We want: BB bottom-left, BL bottom-right, LB top-left, LL top-right
    positions = {
        'Config_BB': (1, 0),  # bottom-left (row 1, col 0)
        'Config_BL': (1, 1),  # bottom-right (row 1, col 1) 
        'Config_LB': (0, 0),  # top-left (row 0, col 0)
        'Config_LL': (0, 1)   # top-right (row 0, col 1)
    }
    
    # Color settings
    vmin, vmax = -40, 0
    cmap = mcolors.LinearSegmentedColormap.from_list('RdYlGn', ['red', 'yellow', 'green'])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Process each configuration
    for config in agg_data.index:
        mm_reward = agg_data.loc[config, 'MarketMaker']
        ea_reward = agg_data.loc[config, 'ExecutionAgent']
        
        print(f"Processing {config}: MM={mm_reward:.1f}, EA={ea_reward:.1f}")
        
        if config in positions:
            grid_row, grid_col = positions[config]
            
            # Clamp values for color mapping
            mm_color = np.clip(mm_reward, vmin, vmax)
            ea_color = np.clip(ea_reward, vmin, vmax)
            
            # Create triangular patches
            # Upper triangle for MarketMaker
            upper_triangle = mpatches.Polygon(
                [(grid_col, grid_row+1), (grid_col+1, grid_row+1), (grid_col+1, grid_row)], 
                closed=True, 
                facecolor=cmap(norm(mm_color)), 
                edgecolor='black', 
                linewidth=1
            )
            ax.add_patch(upper_triangle)
            
            # Lower triangle for ExecutionAgent
            lower_triangle = mpatches.Polygon(
                [(grid_col, grid_row), (grid_col, grid_row+1), (grid_col+1, grid_row)],
                closed=True, 
                facecolor=cmap(norm(ea_color)),
                edgecolor='black', 
                linewidth=1
            )
            ax.add_patch(lower_triangle)
            
            # Add labels and values for both triangles
            # Upper triangle: MM label and value
            ax.text(grid_col+0.75, grid_row+0.85, 'MM', 
                   ha='center', va='center', fontsize=18, fontweight='bold', color='black')
            ax.text(grid_col+0.75, grid_row+0.65, f'{mm_reward:.1f}', 
                   ha='center', va='center', fontsize=18, fontweight='bold', color='black')
            
            # Lower triangle: EXEC label and value  
            ax.text(grid_col+0.25, grid_row+0.35, 'EXEC', 
                   ha='center', va='center', fontsize=18, fontweight='bold', color='black')
            ax.text(grid_col+0.25, grid_row+0.15, f'{ea_reward:.1f}', 
                   ha='center', va='center', fontsize=18, fontweight='bold', color='black')
            
            print(f"  -> Placed at grid position ({grid_row}, {grid_col})")
    
    # Configure the plot
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    
    # Set ticks and labels
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Baseline', 'Learned'], fontsize=18)
    ax.set_xlabel('ExecutionAgent (Slippage)', fontsize=24)
    
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Learned', 'Baseline'],fontsize=18,rotation='vertical')  # Note: y-axis is inverted in matplotlib
    ax.set_ylabel('MarketMaker (Portfolio Value)', fontsize=24)
    
    # Invert y-axis to make bottom-left = BB
    ax.invert_yaxis()
    
    # Add title
    ax.set_title(f'{reward_label}: Baseline vs Learned', 
                fontsize=24, pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(f'{reward_label}', rotation=270, labelpad=15, fontsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create triangular heatmap for agent rewards')
    parser.add_argument('csv_file', help='Path to CSV file containing the data')
    parser.add_argument('--reward-label', default='Rewards', 
                       help='Label for the reward type being displayed (default: Rewards)')
    parser.add_argument('--save-path', help='Path to save the plot (optional)')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.csv_file)
    
    # Create the plot
    save_path = args.save_path
    if save_path is None:
        # Generate default save path
        import os
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        save_path = f'{base_name}_triangular.png'
    
    fig = create_triangular_plot(df, reward_label=args.reward_label, save_path=save_path)
    
    print("Plot created and saved successfully!")

if __name__ == "__main__":
    main()
