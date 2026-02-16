import os
from re import L
import time
import numpy as np
import pandas as pd
import pickle
import glob
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig
from gymnax_exchange.jaxrl.MARL.baseline_eval.baseline_JAXMARL import Transition
import argparse

def get_latest_pickle_file(directory="trajectories", combo_desc="default", datetime_str=None, datetime_range=None):
    """
    Find a pickle file in the specified directory.
    
    Args:
        directory: Directory to search for pickle files
        combo_desc: Combo description to filter files
        datetime_str: Optional datetime string to match in filename (e.g., "20260123-143022"). 
                     If None, returns the most recently created file.
        datetime_range: Optional tuple of (start_datetime_str, end_datetime_str) to filter files within a datetime range.
                       Format: ("20260123-140000", "20260123-150000"). Takes precedence over datetime_str.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    pickle_files = glob.glob(os.path.join(directory, f"*{combo_desc}*.pkl"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in directory '{directory}'")
    
    # If datetime range provided, filter files within that range
    if datetime_range is not None:
        start_dt_str, end_dt_str = datetime_range
        # Accept both formats: YYYYMMDD-HHMMSS or YYYYMMDD_HHMMSS
        try:
            # Try parsing with hyphens first
            if '-' in start_dt_str:
                start_dt = datetime.strptime(start_dt_str, "%Y%m%d-%H%M%S")
                end_dt = datetime.strptime(end_dt_str, "%Y%m%d-%H%M%S")
            else:
                start_dt = datetime.strptime(start_dt_str, "%Y%m%d_%H%M%S")
                end_dt = datetime.strptime(end_dt_str, "%Y%m%d_%H%M%S")
        except ValueError:
            raise ValueError(f"Invalid datetime format in range. Expected format: YYYYMMDD-HHMMSS or YYYYMMDD_HHMMSS")
        
        matching_files = []
        for f in pickle_files:
            # Extract datetime from filename (supports both hyphen and underscore formats)
            import re
            match = re.search(r'(\d{8}[-_]\d{6})', f)
            if match:
                file_dt_str = match.group(1)
                try:
                    # Try parsing with the format found in the filename
                    if '-' in file_dt_str:
                        file_dt = datetime.strptime(file_dt_str, "%Y%m%d-%H%M%S")
                    else:
                        file_dt = datetime.strptime(file_dt_str, "%Y%m%d_%H%M%S")
                    if start_dt <= file_dt <= end_dt:
                        matching_files.append(f)
                except ValueError:
                    continue
        
        if not matching_files:
            raise FileNotFoundError(f"No pickle files found within datetime range {datetime_range} in directory '{directory}'")
        latest_file = max(matching_files, key=os.path.getctime)
    
    # If datetime string provided, filter files containing that datetime
    elif datetime_str is not None:
        matching_files = [f for f in pickle_files if datetime_str in f]
        if not matching_files:
            raise FileNotFoundError(f"No pickle files found matching datetime '{datetime_str}' in directory '{directory}'")
        # If multiple files match, return the most recent one
        latest_file = max(matching_files, key=os.path.getctime)
    else:
        # Sort files by creation time (most recent last)
        latest_file = max(pickle_files, key=os.path.getctime)
    
    return latest_file

def main():
    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Plot episode features from trajectory data")
        parser.add_argument("--directory", "-d", type=str, default="trajectories", 
                            help="Directory containing trajectory pickle files")
        parser.add_argument("--combo", "-c", type=str, nargs='+', default=["BB,LL"], 
                    help="Combo description(s) to filter pickle files (can provide multiple)")
        parser.add_argument("--save", type=str, default="intra-episode-figs", 
                    help="Save Directory for plots (can provide multiple)")
        return parser.parse_args()

    args = parse_args()
    """Load trajectory batch from latest pickle file and generate plots."""

    # plot_same_axis([2],["quant_left"],args.combo,input_dir=args.directory, output_dir=args.save+f"/single_plots")

    for combo in args.combo:
        try:
            # Find and load the latest pickle file for each combo
            latest_file = get_latest_pickle_file(directory=args.directory, combo_desc=combo)
            print(f"Loading trajectory data from: {latest_file}")
            
            with open(latest_file, "rb") as f:
                traj_batch = pickle.load(f)
            print(f"Loaded trajectory batch with {len(traj_batch)} agents.")
            # Plot episode features
            plot_episode_features(traj_batch, output_dir=args.save+f"/{combo}",)
            # plot_specific(traj_batch,[0],["quant_left"], output_dir=args.save+f"/{combo}")
            print(f"Plotting complete. Check the 'intra-episode-figs' directory for output.")
            
        except Exception as e:
            print(f"Error processing combo '{combo}': {e}")

def plot_same_axis(env_indices, features,combos,input_dir="", output_dir="intra-episode-figs", feature_names=None, obs_features=None, 
                   custom_titles=None, custom_ylabels=None, custom_legends=None, subplot_layout=None, datetime_str=None, datetime_range=None,colour_map=None):
    """
    Plot features from multiple combos on the same axes.
    
    Args:
        custom_titles: Dict mapping feature names to custom titles, e.g., {"action": "My Custom Title"}
        custom_ylabels: Dict mapping feature names to custom y-labels, e.g., {"action": "Custom Y Label"}
        custom_legends: Dict mapping combo names to custom legend labels, e.g., {"B": "Buyer", "L": "Liquidity Provider"}
        subplot_layout: Tuple (vertical, horizontal) specifying subplot grid layout. Must satisfy vertical*horizontal == len(env_indices).
                       Defaults to (len(env_indices), 1) if not provided.
        datetime_str: Optional datetime string to match in filename (e.g., "20260123-143022")
        datetime_range: Optional tuple of (start_datetime_str, end_datetime_str) to filter files within a datetime range.
    """
    num_agent_types = 2
    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.cm.get_cmap('seismic', num_agent_types) if colour_map is None else mcolors.ListedColormap(colour_map)
    num_envs= len(env_indices)
    
    # Set up subplot layout
    if subplot_layout is None:
        subplot_layout = (num_envs, 1)
    
    nrows, ncols = subplot_layout
    if nrows * ncols != num_envs:
        raise ValueError(f"subplot_layout {subplot_layout} does not match number of environments {num_envs}. Must satisfy nrows*ncols == num_envs")
    
    info_fig, info_axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), sharex=True)
    
    # Flatten axes array for consistent indexing
    if num_envs == 1:
        info_axes = [info_axes]
    else:
        info_axes = info_axes.flatten()
    
    # Track handles and labels for shared legend
    legend_handles = []
    legend_labels = []
    
    for c_indx, c in enumerate(combos):
        latest_file = get_latest_pickle_file(directory=input_dir, combo_desc=c, datetime_str=datetime_str, datetime_range=datetime_range)
        print(f"Loading trajectory data from: {latest_file}")
        
        with open(latest_file, "rb") as f:
            traj_batch = pickle.load(f)
        print(f"Loaded trajectory batch with {len(traj_batch)} agents.")
        for key in features:

            steps = np.arange(traj_batch[0].action.shape[0])
            
            # Track all y-values for action plots to ensure consistent y-axis
            all_action_values = [] if key == "action" else None
            print(info_axes)

            # Plot this metric for each agent type on the same subplot
            for env_idx,env_actual in enumerate(env_indices):
                # Check if this is a world feature (plot once, not per agent)
                is_world_feature = False
                if 'world' in traj_batch[0].info and key in traj_batch[0].info['world']:
                    is_world_feature = True
                    values = traj_batch[0].info['world'][key]
                    
                    if len(values.shape) <= 2:  # Only plot simple scalar features
                        if len(values.shape) == 3:
                            env_values = values[:, env_actual, 0] + values[:, env_actual, 1] / 1e9
                        elif len(values.shape) == 2:
                            env_values = values[:, env_actual]
                        else:
                            env_values = values
                        
                        # Get custom legend label if provided (use first combo for world features)
                        legend_label = custom_legends.get(c, f"World - {c}") if custom_legends else f"World - {c}"
                        
                        if key == "action":
                            line = info_axes[env_idx].scatter(steps, env_values, color=cmap(c_indx), label=legend_label, alpha=0.6, s=20, zorder=3)
                            all_action_values.extend(env_values)
                        else:
                            line = info_axes[env_idx].plot(steps, env_values, color=cmap(c_indx), label=legend_label)[0]
                        
                        # Collect legend handles and labels only once (from first environment)
                        if env_idx == 0 and legend_label not in legend_labels:
                            legend_handles.append(line)
                            legend_labels.append(legend_label)
                        
                        # Set title, xlabel, ylabel with custom values if provided
                        title = custom_titles.get(key, f"Trajectory plot for {key} measure") if custom_titles else f"Trajectory plot for {key} measure"
                        ylabel = custom_ylabels.get(key, key) if custom_ylabels else key
                        
                        # Format title with environment index if placeholder exists
                        title = title.format(env_idx=env_actual)
                        ylabel = ylabel.format(env_idx=env_actual)
                        
                        # Determine position in grid
                        row = env_idx // ncols
                        col = env_idx % ncols
                        
                        info_axes[env_idx].set_title(title, fontsize=18)
                        
                        # Only show xlabel on bottom row
                        if row == nrows - 1:
                            info_axes[env_idx].set_xlabel("Steps",fontsize=16)
                        
                        # Only show ylabel on first column
                        if col == 0:
                            info_axes[env_idx].set_ylabel(ylabel,fontsize=16)
                        
                        # Remove individual legends from subplots
                        info_axes[env_idx].grid(True)
                    else:
                        print(f"Skipping plotting for world {key} as it has more than 2 dimensions. {values.shape}")
                
                # If not a world feature, plot per-agent features
                if not is_world_feature:
                    for agent_idx, traj in enumerate(traj_batch):
                        # Handle special cases for action and reward
                        if key == "action":
                            values = traj.action
                        elif key == "reward":
                            values = traj.reward
                        elif 'agent' in traj.info and key in traj.info['agent']:
                            values = traj.info['agent'][key]
                        else:
                            continue
                    
                        if len(values.shape) <= 2:  # Only plot simple scalar features
                            env_values = values[:, env_actual] if len(values.shape) > 1 else values
                            
                            # Get custom legend label if provided
                            legend_label = custom_legends.get(c, f"Execution Agent - {c}") if custom_legends else f"Execution Agent - {c}"
                            
                            if key == "action":
                                line = info_axes[env_idx].scatter(steps, env_values, color=cmap(c_indx), label=legend_label, alpha=0.6, s=20, zorder=3)
                                all_action_values.extend(env_values)
                            else:
                                line = info_axes[env_idx].plot(steps, env_values, color=cmap(c_indx), label=legend_label)[0]
                            
                            # Collect legend handles and labels only once (from first environment)
                            if env_idx == 0 and legend_label not in legend_labels:
                                legend_handles.append(line)
                                legend_labels.append(legend_label)
                            
                            # Set title, xlabel, ylabel with custom values if provided
                            title = custom_titles.get(key, f"Trajectory plot for {key} measure") if custom_titles else f"Trajectory plot for {key} measure"
                            ylabel = custom_ylabels.get(key, key) if custom_ylabels else key
                            
                            # Format title with environment index if placeholder exists
                            title = title.format(env_idx=env_actual)
                            ylabel = ylabel.format(env_idx=env_actual)
                            
                            # Determine position in grid
                            row = env_idx // ncols
                            col = env_idx % ncols
                            
                            info_axes[env_idx].set_title(title, fontsize=18)
                            
                            # Only show xlabel on bottom row
                            if row == nrows - 1:
                                info_axes[env_idx].set_xlabel("Steps",fontsize=16)
                            
                            # Only show ylabel on first column
                            if col == 0:
                                info_axes[env_idx].set_ylabel(ylabel,fontsize=16)
                            
                            # Remove individual legends from subplots
                            info_axes[env_idx].grid(True)
                        else:
                            print(f"Skipping plotting for {key} as it has more than 2 dimensions. {values.shape}")
    
    # Apply consistent y-axis limits for action plots across all environments
    if all_action_values is not None and len(all_action_values) > 0:
        y_min = np.min(all_action_values)
        y_max = np.max(all_action_values)
        y_range = y_max - y_min
        # Add 5% padding
        y_min_padded = y_min - 0.05 * y_range
        y_max_padded = y_max + 0.05 * y_range
        
        # Apply to all subplots
        for env_idx in range(num_envs):
            info_axes[env_idx].set_ylim(y_min_padded, y_max_padded)
    
    # Add a single legend outside the subplots (after all plotting is done)
    if legend_handles:
        info_fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                       ncol=len(legend_labels), fontsize=14, frameon=True)
    
    # Save the figure for this metric
    info_fig.tight_layout()
    # Adjust layout to make room for legend at bottom
    if legend_handles:
        info_fig.subplots_adjust(bottom=0.15)
    
    metric_path = os.path.join(output_dir, f"Mega_plot_{time.strftime('%Y%m%d-%H%M%S')}.png")
    print(f"Saving plot to: {metric_path}")
    info_fig.savefig(metric_path, bbox_inches='tight')
    plt.close(info_fig)



def plot_specific(traj_batch,env_indices,features, output_dir="intra-episode-figs", feature_names=None, obs_features=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    num_agent_types = len(traj_batch)
    # Create a colormap that can handle all agent types
    cmap = plt.cm.get_cmap('viridis', num_agent_types)
    
    # Get the number of environments
    num_envs = len(env_indices)
    
    # Create figure with subplots for each environment
    fig, axes = plt.subplots(num_envs, 1, figsize=(12, 5*num_envs), sharex=True)
    if num_envs == 1:
        axes = [axes]
    
    # Plot rewards - one subplot per environment on a single figure
    fig_rewards = plt.figure(figsize=(12, 5*num_envs))
    for env_idx in env_indices:
        ax_reward = fig_rewards.add_subplot(num_envs, 1, env_idx+1)
        steps = np.arange(traj_batch[0].reward.shape[0])
        
        # For each agent type, plot a line with different color
        for agent_idx, traj in enumerate(traj_batch):
            rewards = traj.reward
            env_rewards = rewards[:, env_idx] if len(rewards.shape) > 1 else rewards
            ax_reward.plot(steps, env_rewards, color=cmap(agent_idx), label=f"Agent Type {agent_idx}")
        
        ax_reward.set_title(f"Environment {env_idx} Rewards")
        ax_reward.set_xlabel("Steps")
        ax_reward.set_ylabel("Reward")
        ax_reward.legend()
        ax_reward.grid(True)
    
    fig_rewards.tight_layout()
    output_path = os.path.join(output_dir, f"rewards_all_envs_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig_rewards.savefig(output_path)
    plt.close(fig_rewards)
    
    # Plot actions - one subplot per environment on a single figure
    fig_actions = plt.figure(figsize=(12, 5*num_envs))
    for env_idx in env_indices:
        ax_action = fig_actions.add_subplot(num_envs, 1, env_idx+1)
        steps = np.arange(traj_batch[0].action.shape[0])
        
        # For each agent type, plot a line with different color
        for agent_idx, traj in enumerate(traj_batch):
            actions = traj.action
            env_actions = actions[:, env_idx] if len(actions.shape) > 1 else actions
            ax_action.plot(steps, env_actions, color=cmap(agent_idx), label=f"Agent Type {agent_idx}")
        
        ax_action.set_title(f"Environment {env_idx} Actions")
        ax_action.set_xlabel("Steps")
        ax_action.set_ylabel("Action")
        ax_action.legend()
        ax_action.grid(True)
    
    fig_actions.tight_layout()
    output_path = os.path.join(output_dir, f"actions_all_envs_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig_actions.savefig(output_path)
    plt.close(fig_actions)
    
    # Plot agent-specific info
    if hasattr(traj_batch[0], 'info') and traj_batch[0].info is not None:
        # Collect all unique keys from agent info across all agent types
        all_keys = set()
        for agent_idx, traj in enumerate(traj_batch):
            if 'agent' in traj.info:
                agent_info = traj.info['agent']
                all_keys.update(agent_info.keys())
        print(all_keys)
        
        # For each metric, create a figure with subplots for each environment
        for key in all_keys:
            if key not in features:
                continue
            for agent_idx, traj in enumerate(traj_batch):
                if 'agent' in traj.info and key in traj.info['agent']:
                    values = traj.info['agent'][key]
            # If values is a dict, create a subplot for each key in the dict
            if isinstance(values, dict):
                # Create a figure with subplots for each environment
                all_sub_keys = set()
                for agent_idx, traj in enumerate(traj_batch):
                    if 'agent' in traj.info:
                        agent_info = traj.info['agent'][key]
                        all_sub_keys.update(agent_info.keys())
                print(all_sub_keys)
                subkey_to_index = {subkey: i for i, subkey in enumerate(all_sub_keys)}
                info_fig, info_axes = plt.subplots(num_envs, len(all_sub_keys), figsize=(6*len(all_sub_keys), 5*num_envs), sharex=True)
                if num_envs == 1:
                    info_axes = [info_axes]
                for env_idx in range(num_envs):
                    for agent_idx, traj in enumerate(traj_batch):
                        if 'agent' in traj.info and key in traj.info['agent']:
                            values = traj.info['agent'][key]      
                            print(values.keys())
                            for subkey in all_sub_keys:
                                if subkey not in values:
                                    # print(f"Skipping plotting for {key}.{subkey} as it is not present in agent info.")
                                    continue
                                subvalues = values[subkey]
                                if len(subvalues.shape) <= 2:  # Only plot simple scalar features
                                    # print(f"Plotting {key}.{subkey} for agent type {agent_idx} in environment {env_idx}")
                                    env_values = subvalues[:, env_idx] if len(subvalues.shape) > 1 else subvalues
                                    info_axes[env_idx,subkey_to_index[subkey]].plot(steps, env_values, color=cmap(agent_idx), 
                                                               label=f"Agent Type {agent_idx} - {subkey}")
                                    info_axes[env_idx,subkey_to_index[subkey]].set_title(f"Environment {env_idx} - {key}.{subkey}")
                                    info_axes[env_idx,subkey_to_index[subkey]].set_xlabel("Steps")
                                    info_axes[env_idx,subkey_to_index[subkey]].set_ylabel(f"{key}.{subkey}")
                                    info_axes[env_idx,subkey_to_index[subkey]].legend()
                                    info_axes[env_idx,subkey_to_index[subkey]].grid(True)
                                else:
                                    print(f"Skipping plotting for {key}.{subkey} as it has more than 2 dimensions. {subvalues.shape}")
                # Save the figure for this metric
                metric_path = os.path.join(output_dir, f"{key}_all_agents_{time.strftime('%Y%m%d-%H%M%S')}.png")
                info_fig.tight_layout()
                info_fig.savefig(metric_path)
                plt.close(info_fig)
            elif hasattr(values, 'shape'):  # Handle normal array case
                # Create a figure with subplots for each environment
                info_fig, info_axes = plt.subplots(num_envs, 1, figsize=(10, 5*num_envs), sharex=True)
                if num_envs == 1:
                    info_axes = [info_axes]
                # Plot this metric for each agent type on the same subplot
                for env_idx in range(num_envs):
                    for agent_idx, traj in enumerate(traj_batch):
                        if 'agent' in traj.info and key in traj.info['agent']:
                            values = traj.info['agent'][key]      
                            if len(values.shape) <= 2:  # Only plot simple scalar features
                                env_values = values[:, env_idx] if len(values.shape) > 1 else values
                                info_axes[env_idx].plot(steps, env_values, color=cmap(agent_idx), label=f"Agent Type {agent_idx}")
                                info_axes[env_idx].set_title(f"Environment {env_idx} - {key}")
                                info_axes[env_idx].set_xlabel("Steps")
                                info_axes[env_idx].set_ylabel(key)
                                info_axes[env_idx].legend()
                                info_axes[env_idx].grid(True)
                            else:
                                print(f"Skipping plotting for {key} as it has more than 2 dimensions. {values.shape}")
                # Save the figure for this metric
                metric_path = os.path.join(output_dir, f"{key}_all_agents_{time.strftime('%Y%m%d-%H%M%S')}.png")
                info_fig.tight_layout()
                info_fig.savefig(metric_path)
                plt.close(info_fig)
            else:
                print(f"Skipping plotting for {key} as it is neither a dict nor has a shape attribute.")
    
    # Plot world info
    if hasattr(traj_batch[0], 'info') and 'world' in traj_batch[0].info:
        world_info = traj_batch[0].info['world']
        steps = np.arange(traj_batch[0].reward.shape[0])
        
        # Collect all unique keys from world info
        world_keys = set()
        for key in world_info:
            world_keys.add(key)
        
        # For each world metric, create a figure with subplots for each environment
        for key in world_keys:
            if key not in features:
                continue
            # Skip if the key doesn't contain plottable data
            if key not in world_info or not hasattr(world_info[key], 'shape'):
                continue
                
            values = world_info[key]
            
            # Skip complex data structures
            if len(values.shape) > 3:
                print(f"Skipping plotting for world info {key} as it has more than 3 dimensions. {values.shape}")
                continue
                
            # Create a figure with subplots for each environment
            world_fig, world_axes = plt.subplots(num_envs, 1, figsize=(10, 5*num_envs), sharex=True)
            if num_envs == 1:
                world_axes = [world_axes]
            
            # Plot this world metric for each environment
            for env_idx in range(num_envs):
                if len(values.shape) == 2:
                    env_values = values[:, env_idx]
                    world_axes[env_idx].plot(steps, env_values, color='blue')  # World info uses blue
                elif len(values.shape) == 3:
                    env_values = values[:, env_idx,0]+ values[:, env_idx,1]/1e9
                    world_axes[env_idx].plot(steps, env_values, color='blue')  # World info uses blue
                else:
                    world_axes[env_idx].plot(steps, values, color='blue')  # World info uses blue
                    
                world_axes[env_idx].set_title(f"Environment {env_idx} - World {key}")
                world_axes[env_idx].set_xlabel("Steps")
                world_axes[env_idx].set_ylabel(key)
                world_axes[env_idx].grid(True)
            
            # Save the figure for this metric
            world_metric_path = os.path.join(output_dir, f"world_{key}_{time.strftime('%Y%m%d-%H%M%S')}.png")
            world_fig.tight_layout()
            world_fig.savefig(world_metric_path)
            plt.close(world_fig)



def plot_episode_features(traj_batch, output_dir="intra-episode-figs", feature_names=None, obs_features=None):
    """
    Plot features from trajectory batch for each environment.
    
    Args:
        traj_batch: List of trajectory objects containing episode data
        feature_names: Optional list of feature names to plot. If None, will try to infer from data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create csvs subdirectory for CSV output
    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)
    
    num_agent_types = len(traj_batch)
    # Create a colormap that can handle all agent types
    cmap = plt.cm.get_cmap('viridis', num_agent_types)
    
    # Get the number of environments
    num_envs = traj_batch[0].reward.shape[1] if len(traj_batch[0].reward.shape) > 1 else 1
    
    # Initialize a dictionary to collect data for each environment's CSV
    # Structure: {env_idx: {'step': [...], 'feature_name': [...], ...}}
    env_csv_data = {env_idx: {'step': np.arange(traj_batch[0].reward.shape[0]).tolist()} for env_idx in range(num_envs)}
    
    # Create figure with subplots for each environment
    fig, axes = plt.subplots(num_envs, 1, figsize=(12, 5*num_envs), sharex=True)
    if num_envs == 1:
        axes = [axes]
    
    # Plot rewards - one subplot per environment on a single figure
    fig_rewards = plt.figure(figsize=(12, 5*num_envs))
    for env_idx in range(num_envs):
        ax_reward = fig_rewards.add_subplot(num_envs, 1, env_idx+1)
        steps = np.arange(traj_batch[0].reward.shape[0])
        
        # For each agent type, plot a line with different color
        for agent_idx, traj in enumerate(traj_batch):
            rewards = traj.reward
            env_rewards = rewards[:, env_idx] if len(rewards.shape) > 1 else rewards
            ax_reward.plot(steps, env_rewards, color=cmap(agent_idx), label=f"Agent Type {agent_idx}")
            # Save to CSV data
            env_csv_data[env_idx][f'reward_agent_{agent_idx}'] = env_rewards.tolist() if hasattr(env_rewards, 'tolist') else list(env_rewards)
        
        ax_reward.set_title(f"Environment {env_idx} Rewards")
        ax_reward.set_xlabel("Steps")
        ax_reward.set_ylabel("Reward")
        ax_reward.legend()
        ax_reward.grid(True)
    
    fig_rewards.tight_layout()
    output_path = os.path.join(output_dir, f"rewards_all_envs_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig_rewards.savefig(output_path)
    plt.close(fig_rewards)
    
    # Plot actions - one subplot per environment on a single figure
    fig_actions = plt.figure(figsize=(12, 5*num_envs))
    for env_idx in range(num_envs):
        ax_action = fig_actions.add_subplot(num_envs, 1, env_idx+1)
        steps = np.arange(traj_batch[0].action.shape[0])
        
        # For each agent type, plot a line with different color
        for agent_idx, traj in enumerate(traj_batch):
            actions = traj.action
            env_actions = actions[:, env_idx] if len(actions.shape) > 1 else actions
            ax_action.plot(steps, env_actions, color=cmap(agent_idx), label=f"Agent Type {agent_idx}")
            # Save to CSV data
            env_csv_data[env_idx][f'action_agent_{agent_idx}'] = env_actions.tolist() if hasattr(env_actions, 'tolist') else list(env_actions)
        
        ax_action.set_title(f"Environment {env_idx} Actions")
        ax_action.set_xlabel("Steps")
        ax_action.set_ylabel("Action")
        ax_action.legend()
        ax_action.grid(True)
    
    fig_actions.tight_layout()
    output_path = os.path.join(output_dir, f"actions_all_envs_{time.strftime('%Y%m%d-%H%M%S')}.png")
    fig_actions.savefig(output_path)
    plt.close(fig_actions)
    
    # Plot agent-specific info
    if hasattr(traj_batch[0], 'info') and traj_batch[0].info is not None:
        # Collect all unique keys from agent info across all agent types
        all_keys = set()
        for agent_idx, traj in enumerate(traj_batch):
            if 'agent' in traj.info:
                agent_info = traj.info['agent']
                all_keys.update(agent_info.keys())
        print(all_keys)
        
        # For each metric, create a figure with subplots for each environment
        for key in all_keys:
            for agent_idx, traj in enumerate(traj_batch):
                if 'agent' in traj.info and key in traj.info['agent']:
                    values = traj.info['agent'][key]
            # If values is a dict, create a subplot for each key in the dict
            if isinstance(values, dict):
                # Create a figure with subplots for each environment
                all_sub_keys = set()
                for agent_idx, traj in enumerate(traj_batch):
                    if 'agent' in traj.info:
                        agent_info = traj.info['agent'][key]
                        all_sub_keys.update(agent_info.keys())
                print(all_sub_keys)
                subkey_to_index = {subkey: i for i, subkey in enumerate(all_sub_keys)}
                info_fig, info_axes = plt.subplots(num_envs, len(all_sub_keys), figsize=(6*len(all_sub_keys), 5*num_envs), sharex=True)
                if num_envs == 1:
                    info_axes = [info_axes]
                for env_idx in range(num_envs):
                    for agent_idx, traj in enumerate(traj_batch):
                        if 'agent' in traj.info and key in traj.info['agent']:
                            values = traj.info['agent'][key]      
                            print(values.keys())
                            for subkey in all_sub_keys:
                                if subkey not in values:
                                    # print(f"Skipping plotting for {key}.{subkey} as it is not present in agent info.")
                                    continue
                                subvalues = values[subkey]
                                if len(subvalues.shape) <= 2:  # Only plot simple scalar features
                                    # print(f"Plotting {key}.{subkey} for agent type {agent_idx} in environment {env_idx}")
                                    env_values = subvalues[:, env_idx] if len(subvalues.shape) > 1 else subvalues
                                    info_axes[env_idx,subkey_to_index[subkey]].plot(steps, env_values, color=cmap(agent_idx), 
                                                               label=f"Agent Type {agent_idx} - {subkey}")
                                    info_axes[env_idx,subkey_to_index[subkey]].set_title(f"Environment {env_idx} - {key}.{subkey}")
                                    info_axes[env_idx,subkey_to_index[subkey]].set_xlabel("Steps")
                                    info_axes[env_idx,subkey_to_index[subkey]].set_ylabel(f"{key}.{subkey}")
                                    info_axes[env_idx,subkey_to_index[subkey]].legend()
                                    info_axes[env_idx,subkey_to_index[subkey]].grid(True)
                                    # Save to CSV data
                                    csv_col_name = f'{key}_{subkey}_agent_{agent_idx}'
                                    env_csv_data[env_idx][csv_col_name] = env_values.tolist() if hasattr(env_values, 'tolist') else list(env_values)
                                else:
                                    print(f"Skipping plotting for {key}.{subkey} as it has more than 2 dimensions. {subvalues.shape}")
                # Save the figure for this metric
                metric_path = os.path.join(output_dir, f"{key}_all_agents_{time.strftime('%Y%m%d-%H%M%S')}.png")
                info_fig.tight_layout()
                info_fig.savefig(metric_path)
                plt.close(info_fig)
            elif hasattr(values, 'shape'):  # Handle normal array case
                # Create a figure with subplots for each environment
                info_fig, info_axes = plt.subplots(num_envs, 1, figsize=(10, 5*num_envs), sharex=True)
                if num_envs == 1:
                    info_axes = [info_axes]
                # Plot this metric for each agent type on the same subplot
                for env_idx in range(num_envs):
                    for agent_idx, traj in enumerate(traj_batch):
                        if 'agent' in traj.info and key in traj.info['agent']:
                            values = traj.info['agent'][key]      
                            if len(values.shape) <= 2:  # Only plot simple scalar features
                                env_values = values[:, env_idx] if len(values.shape) > 1 else values
                                info_axes[env_idx].plot(steps, env_values, color=cmap(agent_idx), label=f"Agent Type {agent_idx}")
                                info_axes[env_idx].set_title(f"Environment {env_idx} - {key}")
                                info_axes[env_idx].set_xlabel("Steps")
                                info_axes[env_idx].set_ylabel(key)
                                info_axes[env_idx].legend()
                                info_axes[env_idx].grid(True)
                                # Save to CSV data
                                csv_col_name = f'{key}_agent_{agent_idx}'
                                env_csv_data[env_idx][csv_col_name] = env_values.tolist() if hasattr(env_values, 'tolist') else list(env_values)
                            else:
                                print(f"Skipping plotting for {key} as it has more than 2 dimensions. {values.shape}")
                # Save the figure for this metric
                metric_path = os.path.join(output_dir, f"{key}_all_agents_{time.strftime('%Y%m%d-%H%M%S')}.png")
                info_fig.tight_layout()
                info_fig.savefig(metric_path)
                plt.close(info_fig)
            else:
                print(f"Skipping plotting for {key} as it is neither a dict nor has a shape attribute.")
    
    # Plot world info
    if hasattr(traj_batch[0], 'info') and 'world' in traj_batch[0].info:
        world_info = traj_batch[0].info['world']
        steps = np.arange(traj_batch[0].reward.shape[0])
        
        # Collect all unique keys from world info
        world_keys = set()
        for key in world_info:
            world_keys.add(key)
        
        # For each world metric, create a figure with subplots for each environment
        for key in world_keys:
            # Skip if the key doesn't contain plottable data
            if key not in world_info or not hasattr(world_info[key], 'shape'):
                continue
                
            values = world_info[key]
            
            # Skip complex data structures
            if len(values.shape) > 3:
                print(f"Skipping plotting for world info {key} as it has more than 3 dimensions. {values.shape}")
                continue
                
            # Create a figure with subplots for each environment
            world_fig, world_axes = plt.subplots(num_envs, 1, figsize=(10, 5*num_envs), sharex=True)
            if num_envs == 1:
                world_axes = [world_axes]
            
            # Plot this world metric for each environment
            for env_idx in range(num_envs):
                if len(values.shape) == 2:
                    env_values = values[:, env_idx]
                    world_axes[env_idx].plot(steps, env_values, color='blue')  # World info uses blue
                    # Save to CSV data
                    env_csv_data[env_idx][f'world_{key}'] = env_values.tolist() if hasattr(env_values, 'tolist') else list(env_values)
                elif len(values.shape) == 3:
                    env_values = values[:, env_idx,0]+ values[:, env_idx,1]/1e9
                    world_axes[env_idx].plot(steps, env_values, color='blue')  # World info uses blue
                    # Save to CSV data
                    env_csv_data[env_idx][f'world_{key}'] = env_values.tolist() if hasattr(env_values, 'tolist') else list(env_values)
                else:
                    world_axes[env_idx].plot(steps, values, color='blue')  # World info uses blue
                    # Save to CSV data
                    env_csv_data[env_idx][f'world_{key}'] = values.tolist() if hasattr(values, 'tolist') else list(values)
                    
                world_axes[env_idx].set_title(f"Environment {env_idx} - World {key}")
                world_axes[env_idx].set_xlabel("Steps")
                world_axes[env_idx].set_ylabel(key)
                world_axes[env_idx].grid(True)
            
            # Save the figure for this metric
            world_metric_path = os.path.join(output_dir, f"world_{key}_{time.strftime('%Y%m%d-%H%M%S')}.png")
            world_fig.tight_layout()
            world_fig.savefig(world_metric_path)
            plt.close(world_fig)
    
    # Save CSV files for each environment
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    for env_idx in range(num_envs):
        csv_path = os.path.join(csv_dir, f"env_{env_idx}_{timestamp}.csv")
        df = pd.DataFrame(env_csv_data[env_idx])
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV for environment {env_idx} to: {csv_path}")


if __name__ == "__main__":
    main()
