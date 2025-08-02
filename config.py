#!/usr/bin/env python
"""
Configuration Loading for ML Pipeline

This module provides simple functions to load configurations from YAML files.
Users can create multiple configuration files and select them by name.
"""

import os
import logging
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


def load_config(config_name: Optional[str] = None, config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of the configuration to load (without .yaml extension).
                    If None, loads default_config.yaml
        config_dir: Directory containing configuration files

    Returns:
        Configuration dictionary

    Raises:
        ImportError: If PyYAML is not available
        FileNotFoundError: If config file doesn't exist
    """
    # Determine config file name
    if config_name is None:
        config_file = "default_config.yaml"
    else:
        config_file = f"{config_name}.yaml"

    config_path = os.path.join(config_dir, config_file)

    # Check if file exists, fallback to default if needed
    if not os.path.exists(config_path):
        if config_name is not None:
            # Try falling back to default config
            default_path = os.path.join(config_dir, "default_config.yaml")
            if os.path.exists(default_path):
                logger.warning(
                    f"Config file {config_path} not found. Using default configuration.")
                config_path = default_path
            else:
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}")
        else:
            raise FileNotFoundError(
                f"Default configuration file not found: {config_path}")

    # Load the YAML file
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise ValueError(
            f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(
            f"Error loading configuration file {config_path}: {e}")


def get_model_definitions(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model definitions from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of model definitions

    Raises:
        KeyError: If model_definitions not found in config
    """
    if "model_definitions" not in config:
        raise KeyError("model_definitions not found in configuration")

    return config["model_definitions"]


def get_preprocessing_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract preprocessing options from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of preprocessing options
    """
    return config.get("preprocessing_options", {})


def get_sampling_methods(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract sampling methods from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of sampling methods
    """
    return config.get("sampling_methods", {})


def get_feature_selection_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract feature selection options from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of feature selection options
    """
    return config.get("feature_selection", {})


def list_available_configs(config_dir: str = "configs") -> list:
    """
    List all available configuration files.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        List of available configuration names (without _config.yaml suffix)
    """
    if not os.path.exists(config_dir):
        return []

    config_files = []
    for file in os.listdir(config_dir):
        if file.endswith("_config.yaml"):
            # Remove the _config.yaml suffix
            config_name = file.replace("_config.yaml", "")
            config_files.append(config_name)
        elif file == "default_config.yaml":
            config_files.append("default")

    return sorted(config_files)


def load_data_from_source(config: Dict[str, Any]) -> 'pandas.DataFrame':
    """
    Load data based on the configured source type.

    Args:
        config: Configuration dictionary

    Returns:
        Pandas DataFrame with the loaded data

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If unsupported source type
    """
    source_type = config["data"]["source_type"]
    source_path = config["data"]["source_path"]

    if source_type == "local":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for data loading. Install with: pip install pandas")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Data file not found: {source_path}")

        # Determine file type and load accordingly
        if source_path.lower().endswith('.csv'):
            df = pd.read_csv(source_path)
        elif source_path.lower().endswith('.parquet'):
            df = pd.read_parquet(source_path)
        elif source_path.lower().endswith('.json'):
            df = pd.read_json(source_path)
        else:
            # Default to CSV
            df = pd.read_csv(source_path)

        logger.info(f"Loaded data from {source_path}: {df.shape}")
        return df

    else:
        raise ValueError(
            f"Unsupported data source type: {source_type}. Only 'local' is supported.")


# Legacy function names for backward compatibility
def get_model_configs() -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Loads default config and returns model definitions.
    """
    config = load_config()
    return {"model_definitions": get_model_definitions(config)}
