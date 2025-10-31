"""
Helper functions for saving and loading JAXOB configuration objects to/from files.
"""

import json
import os
from typing import Dict, Any, Union
from dataclasses import asdict, fields
from gymnax_exchange.jaxob.jaxob_config import (
    MultiAgentConfig,
    World_EnvironmentConfig, 
    MarketMaking_EnvironmentConfig,
    Execution_EnvironmentConfig,
    JAXLOB_Configuration
)


def save_config_to_file(config: MultiAgentConfig, filepath: str) -> None:
    """
    Save a MultiAgentConfig instance to a JSON file.
    
    Args:
        config: MultiAgentConfig instance to save
        filepath: Path where to save the configuration file
    """
    # Convert the dataclass to a dictionary
    print(f"##################### \n CConfig  is {config}")

    config_dict = asdict(config)
    print(f"##################### \n CConfig dict is {config_dict}")

    
    # Ensure the directory exists (only if filepath contains a directory)
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_file(filepath: str) -> MultiAgentConfig:
    """
    Load a MultiAgentConfig instance from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        MultiAgentConfig instance loaded from file
    """
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return _dict_to_multiagent_config(config_dict)


def _dict_to_multiagent_config(config_dict: Dict[str, Any]) -> MultiAgentConfig:
    """
    Convert a dictionary to a MultiAgentConfig instance, handling nested configurations.
    
    Args:
        config_dict: Dictionary containing configuration data
        
    Returns:
        MultiAgentConfig instance
    """
    # Handle world_config
    world_config_dict = config_dict.get('world_config', {})
    world_config = _dict_to_world_config(world_config_dict)
    
    # Handle dict_of_agents_configs
    agents_configs_dict = config_dict.get('dict_of_agents_configs', {})
    dict_of_agents_configs = {}
    
    for agent_type, agent_config_dict in agents_configs_dict.items():
        if agent_type == "MarketMaking":
            dict_of_agents_configs[agent_type] = _dict_to_marketmaking_config(agent_config_dict)
        elif agent_type == "Execution":
            dict_of_agents_configs[agent_type] = _dict_to_execution_config(agent_config_dict)
        else:
            # For unknown agent types, try to determine based on fields
            dict_of_agents_configs[agent_type] = _auto_detect_agent_config(agent_config_dict)
    
    # Handle number_of_agents_per_type
    number_of_agents_per_type = config_dict.get('number_of_agents_per_type', [1])
    
    return MultiAgentConfig(
        world_config=world_config,
        dict_of_agents_configs=dict_of_agents_configs,
        number_of_agents_per_type=number_of_agents_per_type
    )


def _dict_to_world_config(config_dict: Dict[str, Any]) -> World_EnvironmentConfig:
    """
    Convert a dictionary to a World_EnvironmentConfig instance.
    Uses default values for any missing parameters.
    """
    # Get default instance to fill in missing values
    default_config = World_EnvironmentConfig()
    
    # Create kwargs dict with defaults and update with provided values
    kwargs = {}
    for field in fields(World_EnvironmentConfig):
        kwargs[field.name] = config_dict.get(field.name, getattr(default_config, field.name))
    
    return World_EnvironmentConfig(**kwargs)


def _dict_to_marketmaking_config(config_dict: Dict[str, Any]) -> MarketMaking_EnvironmentConfig:
    """
    Convert a dictionary to a MarketMaking_EnvironmentConfig instance.
    Uses default values for any missing parameters.
    """
    # Get default instance to fill in missing values
    default_config = MarketMaking_EnvironmentConfig()
    
    # Create kwargs dict with defaults and update with provided values
    kwargs = {}
    for field in fields(MarketMaking_EnvironmentConfig):
        kwargs[field.name] = config_dict.get(field.name, getattr(default_config, field.name))
    
    return MarketMaking_EnvironmentConfig(**kwargs)


def _dict_to_execution_config(config_dict: Dict[str, Any]) -> Execution_EnvironmentConfig:
    """
    Convert a dictionary to an Execution_EnvironmentConfig instance.
    Uses default values for any missing parameters.
    """
    # Get default instance to fill in missing values
    default_config = Execution_EnvironmentConfig()
    
    # Create kwargs dict with defaults and update with provided values
    kwargs = {}
    for field in fields(Execution_EnvironmentConfig):
        kwargs[field.name] = config_dict.get(field.name, getattr(default_config, field.name))
    
    return Execution_EnvironmentConfig(**kwargs)


def _auto_detect_agent_config(config_dict: Dict[str, Any]) -> Union[MarketMaking_EnvironmentConfig, Execution_EnvironmentConfig]:
    """
    Auto-detect the agent configuration type based on the fields present in the dictionary.
    """
    # Get field names for each config type
    mm_fields = set(field.name for field in fields(MarketMaking_EnvironmentConfig))
    exec_fields = set(field.name for field in fields(Execution_EnvironmentConfig))
    
    config_keys = set(config_dict.keys())
    
    # Calculate overlap with each config type
    mm_overlap = len(config_keys.intersection(mm_fields))
    exec_overlap = len(config_keys.intersection(exec_fields))
    
    # Choose the config type with more field overlap
    if mm_overlap >= exec_overlap:
        return _dict_to_marketmaking_config(config_dict)
    else:
        return _dict_to_execution_config(config_dict)


def save_config_to_yaml(config: MultiAgentConfig, filepath: str) -> None:
    """
    Save a MultiAgentConfig instance to a YAML file.
    
    Args:
        config: MultiAgentConfig instance to save
        filepath: Path where to save the configuration file
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install PyYAML")
    
    # Convert the dataclass to a dictionary
    config_dict = asdict(config)
    
    # Ensure the directory exists (only if filepath contains a directory)
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save to YAML file
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config_from_yaml(filepath: str) -> MultiAgentConfig:
    """
    Load a MultiAgentConfig instance from a YAML file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        MultiAgentConfig instance loaded from file
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install PyYAML")
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return _dict_to_multiagent_config(config_dict)


def get_config_summary(config: MultiAgentConfig) -> str:
    """
    Get a human-readable summary of the configuration.
    
    Args:
        config: MultiAgentConfig instance
        
    Returns:
        String summary of the configuration
    """
    summary_lines = []
    summary_lines.append("=== MultiAgent Configuration Summary ===")
    summary_lines.append("")
    
    # World config summary
    summary_lines.append("World Configuration:")
    summary_lines.append(f"  Episode type: {config.world_config.ep_type}")
    summary_lines.append(f"  Episode time: {config.world_config.episode_time}")
    summary_lines.append(f"  Stock: {config.world_config.stock}")
    summary_lines.append(f"  Data messages per step: {config.world_config.n_data_msg_per_step}")
    summary_lines.append("")
    
    # Agent configs summary
    summary_lines.append("Agent Configurations:")
    for agent_type, agent_config in config.dict_of_agents_configs.items():
        summary_lines.append(f"  {agent_type}:")
        if hasattr(agent_config, 'action_space'):
            summary_lines.append(f"    Action space: {agent_config.action_space}")
        if hasattr(agent_config, 'observation_space'):
            summary_lines.append(f"    Observation space: {agent_config.observation_space}")
        if hasattr(agent_config, 'reward_space'):
            summary_lines.append(f"    Reward space: {agent_config.reward_space}")
        if hasattr(agent_config, 'n_actions'):
            summary_lines.append(f"    Number of actions: {agent_config.n_actions}")
    
    summary_lines.append("")
    summary_lines.append(f"Number of agents per type: {config.number_of_agents_per_type}")
    
    return "\n".join(summary_lines)


if __name__ == "__main__":
    """
    Simple example demonstrating config save/load with default parameters.
    """
    print("=== JAXOB Configuration I/O Example ===\n")
    
    # Create a default MultiAgentConfig (uses all default parameters)
    default_config = MultiAgentConfig()
    
    print("1. Created default configuration:")
    print(get_config_summary(default_config))
    print("\n" + "-"*50 + "\n")
    
    # Save the default configuration to JSON
    config_file = "default_config_example.json"
    save_config_to_file(default_config, config_file)
    print(f"2. Saved default configuration to '{config_file}'")
    
    # Load the configuration back from file
    loaded_config = load_config_from_file(config_file)
    print(f"3. Loaded configuration from '{config_file}'")
    print("\nLoaded configuration summary:")
    print(get_config_summary(loaded_config))
    print("\n" + "-"*50 + "\n")
    
    # Verify they are equivalent by comparing their dictionary representations
    original_dict = asdict(default_config)
    loaded_dict = asdict(loaded_config)
    
    if original_dict == loaded_dict:
        print("✅ SUCCESS: Original and loaded configurations are identical!")
    else:
        print("❌ ERROR: Configurations differ after save/load cycle")
    
    print(f"\n4. Example JSON file '{config_file}' has been created in the current directory.")
    print("   You can inspect or modify it, then load it back using load_config_from_file()")
    

    
    

    # # Clean up example file
    # import os
    # if os.path.exists(config_file):
    #     print(f"\n5. Cleaning up example file '{config_file}'")
    #     os.remove(config_file)