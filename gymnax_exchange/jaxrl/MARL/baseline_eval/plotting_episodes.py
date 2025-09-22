import os
from re import L
import time
import numpy as np
import pickle
import glob

import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"


from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig
from baseline_JAXMARL import Transition
import argparse

def get_latest_pickle_file(directory="trajectories", combo_desc="default"):
    """Find the most recently created pickle file in the specified directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    pickle_files = glob.glob(os.path.join(directory, f"*{combo_desc}*.pkl"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle files found in directory '{directory}'")
    
    # Sort files by creation time (most recent last)
    latest_file = max(pickle_files, key=os.path.getctime)
    return latest_file

def main():
    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Plot episode features from trajectory data")
        parser.add_argument("--directory", "-d", type=str, default="trajectories", 
                            help="Directory containing trajectory pickle files")
        parser.add_argument("--combo", "-c", type=str, nargs='+', default=["BB"], 
                    help="Combo description(s) to filter pickle files (can provide multiple)")
        parser.add_argument("--save", type=str, default="intra-episode-figs", 
                    help="Save Directory for plots (can provide multiple)")
        return parser.parse_args()

    args = parse_args()
    """Load trajectory batch from latest pickle file and generate plots."""

    plot_same_axis([2],["quant_left"],args.combo,input_dir=args.directory, output_dir=args.save+f"/single_plots")

    # for combo in args.combo:
    #     try:
    #         # Find and load the latest pickle file for each combo
    #         latest_file = get_latest_pickle_file(directory=args.directory, combo_desc=combo)
    #         print(f"Loading trajectory data from: {latest_file}")
            
    #         with open(latest_file, "rb") as f:
    #             traj_batch = pickle.load(f)
    #         print(f"Loaded trajectory batch with {len(traj_batch)} agents.")
    #         # Plot episode features
    #         # plot_episode_features(traj_batch, output_dir=args.save+f"/{combo}",)
    #         # plot_specific(traj_batch,[0],["quant_left"], output_dir=args.save+f"/{combo}")
    #         print(f"Plotting complete. Check the 'intra-episode-figs' directory for output.")
            
    #     except Exception as e:
    #         print(f"Error processing combo '{combo}': {e}")

def plot_same_axis(env_indices, features,combos,input_dir="", output_dir="intra-episode-figs", feature_names=None, obs_features=None):
    num_agent_types = 2
    os.makedirs(output_dir, exist_ok=True)
    cmap = plt.cm.get_cmap('seismic', num_agent_types)
    num_envs= len(env_indices)
    info_fig, info_axes = plt.subplots(num_envs, 1, figsize=(7, 5*num_envs), sharex=True)
    if num_envs == 1:
        info_axes = [info_axes] 
    for c_indx, c in enumerate(combos):
        latest_file = get_latest_pickle_file(directory=input_dir, combo_desc=c)
        print(f"Loading trajectory data from: {latest_file}")
        
        with open(latest_file, "rb") as f:
            traj_batch = pickle.load(f)
        print(f"Loaded trajectory batch with {len(traj_batch)} agents.")
        for key in features:

            steps = np.arange(traj_batch[0].action.shape[0])
            print(info_axes)

            # Plot this metric for each agent type on the same subplot
            for env_idx,env_actual in enumerate(env_indices):
                for agent_idx, traj in enumerate(traj_batch):
                    if 'agent' in traj.info and key in traj.info['agent']:
                        values = traj.info['agent'][key]      
                        if len(values.shape) <= 2:  # Only plot simple scalar features
                            env_values = values[:, env_actual] if len(values.shape) > 1 else values
                            info_axes[env_idx].plot(steps, env_values, color=cmap(c_indx), label=f"Execution Agent - {c}")
                            info_axes[env_idx].set_title(f"Trajectory plot for {key} measure")
                            info_axes[env_idx].set_xlabel("Steps")
                            info_axes[env_idx].set_ylabel(key)
                            info_axes[env_idx].legend()
                            info_axes[env_idx].grid(True)
                        else:
                            print(f"Skipping plotting for {key} as it has more than 2 dimensions. {values.shape}")
            # Save the figure for this metric
            metric_path = os.path.join(output_dir, f"Mega_plot_{time.strftime('%Y%m%d-%H%M%S')}.png")
            info_fig.tight_layout()
            print(f"Saving plot to: {metric_path}")
            info_fig.savefig(metric_path)
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
    
    num_agent_types = len(traj_batch)
    # Create a colormap that can handle all agent types
    cmap = plt.cm.get_cmap('viridis', num_agent_types)
    
    # Get the number of environments
    num_envs = traj_batch[0].reward.shape[1] if len(traj_batch[0].reward.shape) > 1 else 1
    
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


if __name__ == "__main__":
    main()
