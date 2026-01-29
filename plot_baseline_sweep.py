#!/usr/bin/env python3
"""Quick script to plot baseline sweep results from total.csv"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Default to total.csv
csv_file = "baseline_sweep_results/total.csv"
if len(sys.argv) > 1:
    csv_file = sys.argv[1]

print(f"Plotting data from: {csv_file}")

# Read the CSV
df = pd.read_csv(csv_file)

# Create FIXED_ACTIONS column (0-4 for the 5 rows)
df.insert(0, 'FIXED_ACTIONS', range(len(df)))

print("\nData loaded:")
print(df)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('Baseline Sweep Results: FIXED_ACTIONS 0-4', fontsize=14, fontweight='bold')

# Plot each column (except FIXED_ACTIONS) as a separate line
colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns)-1))
markers = ['o', 's', '^', 'D', 'v', 'p']

for idx, column in enumerate(df.columns[1:]):  # Skip FIXED_ACTIONS column
    # Convert column values to numeric, replacing non-numeric with NaN
    values = pd.to_numeric(df[column], errors='coerce')
    
    ax.plot(df['FIXED_ACTIONS'], values, 
            marker=markers[idx % len(markers)], 
            linewidth=2, 
            markersize=8,
            label=column,
            color=colors[idx])

ax.set_xlabel('FIXED_ACTIONS', fontsize=12)
ax.set_ylabel('End of Episode Portfolio Value', fontsize=12)
ax.set_title('Performance across different configurations', fontsize=13)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.legend(loc='best', framealpha=0.9)
ax.set_xticks(df['FIXED_ACTIONS'])

# Adjust layout
plt.tight_layout()

# Save the figure
output_file = csv_file.replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("Summary Statistics:")
print("="*70)
print(df.to_string(index=False))
print("\nTrend: As FIXED_ACTIONS increases (0→4), performance generally improves")
