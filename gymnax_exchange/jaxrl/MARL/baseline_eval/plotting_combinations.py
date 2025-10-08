#!/usr/bin/env python3
"""
Plotting script for visualizing agent rewards in Baseline vs Learned configurations.
Creates an n×n heatmap where n is the number of agents, showing rewards for each agent
across all possible combinations of Baseline (B) and Learned (L) agent types.

For example, with 2 agents:
- Config (0,0): Baseline vs Baseline
- Config (0,1): Baseline vs Learned  
- Config (1,0): Learned vs Baseline
- Config (1,1): Learned vs Learned

Higher values (green) indicate better performance, lower values (red) indicate worse performance.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd


class AgentRewardPlotter:
    """Class for creating heatmap visualizations of agent rewards in Baseline vs Learned configurations."""
    
    def __init__(self, n_agents: int = 2, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize the plotter.
        
        Args:
            n_agents: Number of agents (creates 2^n_agents configurations)
            figsize: Figure size for the plot
        """
        self.n_agents = n_agents
        self.n_configs = 2 ** n_agents  # 2^n possible configurations
        self.figsize = figsize
        self.data = None
        
    def generate_sample_data(self) -> Dict[str, np.ndarray]:
        """
        Generate sample reward data for testing.
        
        Returns:
            Dictionary containing reward data for each configuration
        """
        np.random.seed(42)  # For reproducibility
        
        data = {}
        
        # Generate all possible configurations (Baseline=0, Learned=1)
        for config_id in range(2 ** self.n_agents):
            # Convert config_id to binary representation for agent types
            agent_types = []
            temp_id = config_id
            for _ in range(self.n_agents):
                agent_types.append(temp_id % 2)  # 0=Baseline, 1=Learned
                temp_id //= 2
            
            # Create configuration name
            type_names = ['B' if t == 0 else 'L' for t in agent_types]
            config_name = f"Config_{''.join(type_names)}"
            
            # Generate sample reward data
            # Shape: (num_episodes, n_agents)
            num_episodes = np.random.randint(50, 200)
            base_reward = np.random.uniform(0.1, 0.5, size=(num_episodes, self.n_agents))
            
            # Different reward patterns for Baseline vs Learned agents
            for agent_idx in range(self.n_agents):
                if agent_types[agent_idx] == 0:  # Baseline agent
                    # Baseline: steady but lower performance
                    improvement = np.linspace(1.0, 1.5, num_episodes)
                    base_reward[:, agent_idx] = base_reward[:, agent_idx] * improvement
                else:  # Learned agent
                    # Learned: higher performance with learning curve
                    improvement = np.linspace(1.0, 3.0, num_episodes)
                    base_reward[:, agent_idx] = base_reward[:, agent_idx] * improvement
                    
                    # Add some volatility during learning
                    volatility = 1 + 0.2 * np.sin(np.linspace(0, 8, num_episodes))
                    base_reward[:, agent_idx] = base_reward[:, agent_idx] * volatility
            
            # Add some noise and ensure positive rewards
            reward_data = base_reward + np.random.normal(0, 0.05, base_reward.shape)
            reward_data = np.maximum(reward_data, 0.01)  # Ensure positive rewards
            
            data[config_name] = reward_data
        
        self.data = data
        return data
    
    def load_data(self, data_path: str) -> Dict[str, np.ndarray]:
        """
        Load reward data from file.
        
        Args:
            data_path: Path to the data file (.pkl, .json, .npz, or .csv)
            
        Returns:
            Dictionary containing reward data for each configuration
        """
        data_path_obj = Path(data_path)
        
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {data_path_obj}")
        
        if data_path_obj.suffix == '.pkl':
            with open(data_path_obj, 'rb') as f:
                data = pickle.load(f)
        elif data_path_obj.suffix == '.json':
            with open(data_path_obj, 'r') as f:
                json_data = json.load(f)
            # Convert lists back to numpy arrays
            data = {k: np.array(v) for k, v in json_data.items()}
        elif data_path_obj.suffix == '.npz':
            npz_data = np.load(data_path_obj)
            data = {key: npz_data[key] for key in npz_data.files}
        elif data_path_obj.suffix == '.csv':
            # Load CSV data
            df = pd.read_csv(data_path_obj)
            data = self._parse_csv_data(df)
        else:
            raise ValueError(f"Unsupported file format: {data_path_obj.suffix}")
        
        self.data = data
        return data
    
    def _parse_csv_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Parse CSV data into the expected format.
        
        Expected CSV format:
        - Column 'Configuration': Configuration names (e.g., Config_BB, Config_BL, Config_LB, Config_LL)
        - Column 'Episode': Episode numbers
        - Columns 'MarketMaker', 'ExecutionAgent', etc.: Reward values for each agent
        
        Args:
            df: DataFrame loaded from CSV
            
        Returns:
            Dictionary containing reward data for each configuration
        """
        data = {}
        
        # Group by configuration
        for config_name, group in df.groupby('Configuration'):
            # Sort by episode to ensure correct order
            group = group.sort_values('Episode')
            
            # Extract agent columns - try both naming conventions
            agent_cols = []
            
            # First try MarketMaker, ExecutionAgent naming
            if 'MarketMaker' in group.columns:
                agent_cols.append('MarketMaker')
            if 'ExecutionAgent' in group.columns:
                agent_cols.append('ExecutionAgent')
            
            # If we don't have the named columns, fall back to Agent_0, Agent_1, etc.
            if not agent_cols:
                agent_cols = [col for col in group.columns if col.startswith('Agent_')]
            
            if not agent_cols:
                raise ValueError("No agent columns found. Expected columns like 'MarketMaker', 'ExecutionAgent' or 'Agent_0', 'Agent_1', etc.")
            
            # Convert to numpy array (episodes x agents)
            reward_data = group[agent_cols].values
            data[config_name] = reward_data
        
        return data
    
    def _create_sample_csv_data(self) -> pd.DataFrame:
        """
        Create sample data in CSV format for easy editing.
        
        Returns:
            DataFrame with sample reward data
        """
        # Ensure we have data to work with
        if self.data is None:
            # Generate sample data if none exists
            self.generate_sample_data()
        
        if self.data is None:
            raise ValueError("Failed to generate sample data")
        
        # Define agent column names based on number of agents
        if self.n_agents == 2:
            agent_names = ['MarketMaker', 'ExecutionAgent']
        else:
            # For more than 2 agents, use Agent_0, Agent_1, etc.
            agent_names = [f'Agent_{i}' for i in range(self.n_agents)]
        
        rows = []
        for config_name, reward_data in self.data.items():
            num_episodes, num_agents = reward_data.shape
            for episode in range(num_episodes):
                row = {
                    'Configuration': config_name,
                    'Episode': episode
                }
                for agent_idx in range(num_agents):
                    if agent_idx < len(agent_names):
                        row[agent_names[agent_idx]] = reward_data[episode, agent_idx]
                    else:
                        row[f'Agent_{agent_idx}'] = reward_data[episode, agent_idx]
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_data(self, data: Dict[str, np.ndarray], save_path: str):
        """
        Save reward data to file.
        
        Args:
            data: Dictionary containing reward data
            save_path: Path where to save the data
        """
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path_obj.suffix == '.pkl':
            with open(save_path_obj, 'wb') as f:
                pickle.dump(data, f)
        elif save_path_obj.suffix == '.json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {k: v.tolist() for k, v in data.items()}
            with open(save_path_obj, 'w') as f:
                json.dump(json_data, f, indent=2)
        elif save_path_obj.suffix == '.npz':
            np.savez(save_path_obj, **data)
        elif save_path_obj.suffix == '.csv':
            # Save as CSV for easy editing
            df = self._create_sample_csv_data()
            df.to_csv(save_path_obj, index=False)
        else:
            raise ValueError(f"Unsupported file format: {save_path_obj.suffix}")
    
    def compute_overall_reward(self, reward_data: np.ndarray, method: str = 'mean') -> Union[float, np.floating]:
        """
        Compute overall reward from agent reward data.
        
        Args:
            reward_data: Array of shape (episodes, agents)
            method: Method to compute overall reward ('mean', 'sum', 'max', 'final')
            
        Returns:
            Overall reward value
        """
        if method == 'mean':
            return np.mean(reward_data)
        elif method == 'sum':
            return np.sum(reward_data, axis=1).mean()  # Sum across agents, then average across episodes
        elif method == 'max':
            return np.max(reward_data)
        elif method == 'final':
            return np.mean(reward_data[-1, :])  # Final episode average
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_heatmap(self, method: str = 'mean', cmap: str = 'RdYlGn', 
                      title: Optional[str] = None, save_path: Optional[str] = None) -> mpl_figure.Figure:
        """
        Create heatmap visualization with triangular splits showing each agent's rewards.
        
        Args:
            method: Method to compute overall reward
            cmap: Colormap for the heatmap (default: RdYlGn for red-green)
            title: Custom title for the plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data loaded. Call generate_sample_data() or load_data() first.")
        
        # Create grid size based on number of agents (2^n_agents configurations arranged in square)
        grid_size = int(np.ceil(np.sqrt(self.n_configs)))
        
        # Separate matrices for each agent
        agent_rewards = []
        for agent_idx in range(self.n_agents):
            agent_rewards.append(np.full((grid_size, grid_size), np.nan))
        
        config_labels = []
        
        # Fill matrices with configuration data
        # For 2x2 grid, we want: BB bottom-left, BL bottom-right, LB top-left, LL top-right
        config_positions = {
            'Config_BB': (1, 0),  # bottom-left
            'Config_BL': (1, 1),  # bottom-right  
            'Config_LB': (0, 0),  # top-left
            'Config_LL': (0, 1)   # top-right
        }
        
        for config_name, reward_data in self.data.items():
            if config_name in config_positions:
                row, col = config_positions[config_name]
                config_labels.append(config_name)
                
                print(f"Debug: {config_name} -> grid position ({row}, {col})")
                print(f"Debug: {config_name} shape: {reward_data.shape}, sample data: {reward_data[:2, :]}")
                
                # Compute rewards for each agent separately
                for agent_idx in range(min(self.n_agents, reward_data.shape[1])):
                    if method == 'mean':
                        agent_reward = np.mean(reward_data[:, agent_idx])
                    elif method == 'sum':
                        agent_reward = np.sum(reward_data[:, agent_idx])
                    elif method == 'max':
                        agent_reward = np.max(reward_data[:, agent_idx])
                    elif method == 'final':
                        agent_reward = reward_data[-1, agent_idx]
                    else:
                        agent_reward = np.mean(reward_data[:, agent_idx])
                    
                    print(f"Debug: {config_name} Agent {agent_idx} reward: {agent_reward}")
                    agent_rewards[agent_idx][row, col] = agent_reward
            else:
                print(f"Warning: Configuration {config_name} not found in position mapping")
        
        # Create the figure with triangular split visualization
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Agent names for legend
        if self.n_agents == 2:
            agent_names = ['MarketMaker', 'ExecutionAgent']
        else:
            agent_names = [f'Agent_{i}' for i in range(self.n_agents)]
        
        # Create triangular heatmap for 2 agents
        if self.n_agents == 2:
            # Normalize the data for consistent color scaling
            all_values = []
            for agent_idx in range(2):
                valid_values = agent_rewards[agent_idx][~np.isnan(agent_rewards[agent_idx])]
                all_values.extend(valid_values)
            
            # Fixed color scale from -50 to 0
            vmin, vmax = -25, 0
            
            # Create triangular plots using the fixed position mapping
            config_positions = {
                'Config_BB': (1, 0),  # bottom-left
                'Config_BL': (1, 1),  # bottom-right  
                'Config_LB': (0, 0),  # top-left
                'Config_LL': (0, 1)   # top-right
            }
            
            for config_name in config_positions:
                if config_name in self.data:
                    row, col = config_positions[config_name]
                    
                    # Get values for both agents
                    mm_val = agent_rewards[0][row, col]  # MarketMaker
                    ea_val = agent_rewards[1][row, col]  # ExecutionAgent
                    
                    if not (np.isnan(mm_val) or np.isnan(ea_val)):
                        # Create triangular patches
                        # Upper triangle for MarketMaker
                        triangle_upper = mpatches.Polygon([(col, row+1), (col+1, row+1), (col+1, row)], 
                                                   closed=True)
                        # Lower triangle for ExecutionAgent  
                        triangle_lower = mpatches.Polygon([(col, row), (col, row+1), (col+1, row)], 
                                                   closed=True)
                            
                    # Fixed color scale from -50 to 0 with saturation for out-of-range values
                    color_min, color_max = -50, 0
                    
                    # Clip values to color range for color mapping
                    mm_val_clipped = np.clip(mm_val, color_min, color_max)
                    ea_val_clipped = np.clip(ea_val, color_min, color_max)
                    
                    # Normalize colors
                    colormap = plt.colormaps.get_cmap(cmap)
                    mm_color = colormap((mm_val_clipped - color_min) / (color_max - color_min))
                    ea_color = colormap((ea_val_clipped - color_min) / (color_max - color_min))
                    
                    triangle_upper.set_facecolor(mm_color)
                    triangle_lower.set_facecolor(ea_color)
                    triangle_upper.set_edgecolor('white')
                    triangle_lower.set_edgecolor('white')
                    triangle_upper.set_linewidth(1)
                    triangle_lower.set_linewidth(1)
                    
                    ax.add_patch(triangle_upper)
                    ax.add_patch(triangle_lower)
                    
                    # Add text annotations with original values
                    ax.text(col+0.75, row+0.75, f'{mm_val:.1f}', ha='center', va='center', 
                           fontsize=8, fontweight='bold')
                    ax.text(col+0.25, row+0.25, f'{ea_val:.1f}', ha='center', va='center', 
                           fontsize=8, fontweight='bold')
            
            # Set up the plot
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')
            # Orientation is now handled by flipped grid mapping
            
            # Create custom labels for individual agents
            if self.n_agents == 2:
                # For 2 agents, create a 2x2 grid where:
                # X-axis represents ExecutionAgent (Agent 1): Baseline/Learned
                # Y-axis represents MarketMaker (Agent 0): Baseline/Learned
                config_labels_x = ['Baseline', 'Learned']  # ExecutionAgent
                config_labels_y = ['Baseline', 'Learned']  # MarketMaker
            else:
                # For more agents, use the combined configuration labels
                config_labels_x = []
                config_labels_y = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        config_id = i * grid_size + j
                        if config_id < self.n_configs:
                            agent_types = []
                            temp_id = config_id
                            for _ in range(self.n_agents):
                                agent_types.append(temp_id % 2)
                                temp_id //= 2
                            type_names = ['B' if t == 0 else 'L' for t in agent_types]
                            label = ''.join(type_names)
                        else:
                            label = ''
                        
                        if i == 0:
                            config_labels_x.append(label)
                        if j == 0:
                            config_labels_y.append(label)
            
            # Set ticks and labels
            ax.set_xticks([i + 0.5 for i in range(len(config_labels_x))])
            ax.set_xticklabels(config_labels_x)
            ax.set_yticks([i + 0.5 for i in range(len(config_labels_y))])
            ax.set_yticklabels(config_labels_y)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(f'Reward ({method}) [Range: -50 to 0, saturated beyond]', rotation=270, labelpad=15)
            
            # Add legend for triangles
            legend_elements = [
                mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                          markersize=10, label=f'{agent_names[0]} (upper triangle)'),
                mlines.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', 
                          markersize=10, label=f'{agent_names[1]} (lower triangle)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))
            
        else:
            # For more than 2 agents, fall back to regular heatmap with average
            combined_rewards = np.full((grid_size, grid_size), np.nan)
            for i in range(grid_size):
                for j in range(grid_size):
                    valid_rewards = []
                    for agent_idx in range(self.n_agents):
                        if not np.isnan(agent_rewards[agent_idx][i, j]):
                            valid_rewards.append(agent_rewards[agent_idx][i, j])
                    if valid_rewards:
                        combined_rewards[i, j] = np.mean(valid_rewards)
            
            mask = np.isnan(combined_rewards)
            sns.heatmap(combined_rewards, annot=~mask, fmt='.3f', cmap=cmap, ax=ax,
                       mask=mask, cbar_kws={'label': f'Average Reward ({method})'})
        
        # Set title
        if title is None:
            if self.n_agents == 2:
                title = f'Agent Rewards: Baseline vs Learned\n(Upper: {agent_names[0]}, Lower: {agent_names[1]})'
            else:
                title = f'Agent Rewards: Baseline vs Learned ({self.n_agents} Agents, {self.n_configs} Configurations)'
        ax.set_title(title, fontsize=14, pad=20)
        
        # Labels
        if self.n_agents == 2:
            ax.set_xlabel('ExecutionAgent Type', fontsize=12)
            ax.set_ylabel('MarketMaker Type', fontsize=12)
        else:
            ax.set_xlabel('Agent Configuration (B=Baseline, L=Learned)', fontsize=12)
            ax.set_ylabel('Agent Configuration (B=Baseline, L=Learned)', fontsize=12)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path_obj}")
        
        return fig
    
    def create_detailed_plot(self, save_path: Optional[str] = None) -> mpl_figure.Figure:
        """
        Create a detailed plot showing individual agent rewards and overall reward heatmap.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data loaded. Call generate_sample_data() or load_data() first.")
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 12))
        
        # Main heatmap (top half)
        ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Create grid size based on number of agents
        grid_size = int(np.ceil(np.sqrt(self.n_configs)))
        reward_matrix = np.full((grid_size, grid_size), np.nan)
        
        for config_id in range(self.n_configs):
            # Use flipped row mapping for consistency
            row = (grid_size - 1) - (config_id // grid_size)  # Flip row0
            col = config_id % grid_size
            
            # Convert config_id to agent types
            agent_types = []
            temp_id = config_id
            for _ in range(self.n_agents):
                agent_types.append(temp_id % 2)
                temp_id //= 2
            
            type_names = ['B' if t == 0 else 'L' for t in agent_types]
            config_name = f"Config_{''.join(type_names)}"
            
            if config_name in self.data:
                reward_data = self.data[config_name]
                reward_matrix[row, col] = self.compute_overall_reward(reward_data, 'mean')
        
        # Create labels for the heatmap
        row_labels = []
        col_labels = []
        for i in range(grid_size):
            for j in range(grid_size):
                config_id = i * grid_size + j
                if config_id < self.n_configs:
                    agent_types = []
                    temp_id = config_id
                    for _ in range(self.n_agents):
                        agent_types.append(temp_id % 2)
                        temp_id //= 2
                    type_names = ['B' if t == 0 else 'L' for t in agent_types]
                    label = ''.join(type_names)
                else:
                    label = ''
                
                if j == 0:
                    row_labels.append(label)
                if i == 0:
                    col_labels.append(label)
        
        # Create main heatmap
        mask = np.isnan(reward_matrix)
        sns.heatmap(reward_matrix, annot=~mask, fmt='.3f', cmap='RdYlGn', ax=ax_main,
                   mask=mask, cbar_kws={'label': 'Overall Reward (mean)'},
                   xticklabels=col_labels, yticklabels=row_labels)
        ax_main.set_title(f'Baseline vs Learned Agent Rewards ({self.n_agents} Agents, {self.n_configs} Configurations)', 
                         fontsize=16, pad=20)
        
        # Example individual agent rewards (bottom left)
        ax_agents = plt.subplot2grid((3, 2), (2, 0))
        # Show first configuration as example
        first_config = list(self.data.keys())[0]
        if first_config in self.data:
            reward_data = self.data[first_config]
            # Define agent names
            if self.n_agents == 2:
                agent_names = ['MarketMaker', 'ExecutionAgent']
            else:
                agent_names = [f'Agent {i}' for i in range(self.n_agents)]
            
            for agent_idx in range(self.n_agents):
                agent_name = agent_names[agent_idx] if agent_idx < len(agent_names) else f'Agent {agent_idx}'
                ax_agents.plot(reward_data[:, agent_idx], label=agent_name, alpha=0.7)
            ax_agents.set_xlabel('Episode')
            ax_agents.set_ylabel('Reward')
            ax_agents.set_title(f'Individual Agent Rewards - {first_config}')
            ax_agents.legend()
            ax_agents.grid(True, alpha=0.3)
        
        # Reward distribution (bottom right)
        ax_dist = plt.subplot2grid((3, 2), (2, 1))
        all_rewards = []
        for config_data in self.data.values():
            all_rewards.extend(config_data.flatten())
        ax_dist.hist(all_rewards, bins=30, alpha=0.7, edgecolor='black')
        ax_dist.set_xlabel('Reward Value')
        ax_dist.set_ylabel('Frequency')
        ax_dist.set_title('Reward Distribution Across All Configurations')
        ax_dist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path_obj, dpi=300, bbox_inches='tight')
            print(f"Detailed plot saved to: {save_path_obj}")
        
        return fig


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Plot agent rewards for Baseline vs Learned configurations')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, help='Path to load data from (.pkl, .json, .npz, or .csv)')
    parser.add_argument('--save-data', type=str, help='Path to save generated sample data')
    parser.add_argument('--output-dir', type=str, default='./plots', 
                       help='Directory to save plots (default: ./plots)')
    
    # Configuration arguments
    parser.add_argument('--n-agents', type=int, default=2, help='Number of agents (creates 2^n configurations)')
    parser.add_argument('--method', type=str, default='mean', 
                       choices=['mean', 'sum', 'max', 'final'],
                       help='Method to compute overall reward')
    
    # Plot arguments
    parser.add_argument('--cmap', type=str, default='RdYlGn', help='Colormap for heatmap (default: RdYlGn)')
    parser.add_argument('--title', type=str, help='Custom title for the plot')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 10], 
                       help='Figure size (width height)')
    parser.add_argument('--detailed', action='store_true', 
                       help='Create detailed plot with individual agent rewards')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create plotter
    plotter = AgentRewardPlotter(n_agents=args.n_agents, figsize=tuple(args.figsize))
    
    # Load or generate data
    if args.data_path:
        print(f"Loading data from: {args.data_path}")
        data = plotter.load_data(args.data_path)
        print(f"Loaded data for {len(data)} configurations")
    else:
        print("Generating sample data...")
        data = plotter.generate_sample_data()
        print(f"Generated sample data for {len(data)} configurations")
    
    # Save data if requested
    if args.save_data:
        plotter.save_data(data, args.save_data)
        print(f"Data saved to: {args.save_data}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save plots
    if args.detailed:
        # Create detailed plot
        print("Creating detailed plot...")
        fig = plotter.create_detailed_plot(
            save_path=str(output_dir / f'detailed_baseline_vs_learned_n{args.n_agents}.png')
        )
    else:
        # Create simple heatmap
        print("Creating heatmap...")
        fig = plotter.create_heatmap(
            method=args.method,
            cmap=args.cmap,
            title=args.title,
            save_path=str(output_dir / f'baseline_vs_learned_heatmap_n{args.n_agents}_{args.method}.png')
        )
    
    # Show plot
    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
