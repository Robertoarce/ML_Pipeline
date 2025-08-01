#!/usr/bin/env python
"""
ML Pipeline Builder

This module provides functions to build scikit-learn pipelines based on configuration.
It supports various preprocessing steps, feature engineering, and multiple model types.
All configurations are loaded from JSON files.
"""

import logging
import importlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import f_classif

# Import configuration functions
from config import get_model_definitions, get_preprocessing_options, get_sampling_methods, get_feature_selection_options

# Imbalanced-learn imports (optional)
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    ImbPipeline = Pipeline
    IMBALANCED_LEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


def _import_class(module_name: str, class_name: str):
    """
    Dynamically import a class from a module.

    Args:
        module_name: Name of the module to import from
        class_name: Name of the class to import

    Returns:
        The imported class
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(
            f"Could not import {class_name} from {module_name}: {e}")
        return None


def get_model_definitions_with_instances(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model definitions from configuration with dynamic imports.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of model definitions with instantiated models
    """
    random_state = config["data"]["random_state"]
    model_definitions = get_model_definitions(config)

    instantiated_models = {}

    for model_name, model_config in model_definitions.items():
        try:
            # Import the model class
            model_class = _import_class(
                model_config["module"], model_config["class_name"])

            if model_class is None:
                logger.warning(
                    f"Skipping {model_name} - could not import class")
                continue

            # Prepare initialization parameters
            init_params = model_config.get("init_params", {}).copy()
            init_params["random_state"] = random_state

            # Create model instance
            model_instance = model_class(**init_params)

            instantiated_models[model_name] = {
                "model": model_instance,
                "param_grid": model_config.get("param_grid", {})
            }

        except Exception as e:
            logger.warning(f"Failed to instantiate {model_name}: {e}")
            continue

    return instantiated_models


def build_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """
    Build the preprocessing pipeline based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        ColumnTransformer for preprocessing
    """
    preprocessing_options = get_preprocessing_options(config)

    transformers = []

    # Categorical features pipeline
    categorical_features = config["features"]["categorical"]
    if categorical_features:
        categorical_steps = []

        # Handle missing values
        if config["preprocessing"]["handle_missing"]["enabled"]:
            imputer_config = preprocessing_options["imputers"]["SimpleImputer"]
            imputer_class = _import_class(
                imputer_config["module"], imputer_config["class_name"])

            strategy = config["preprocessing"]["handle_missing"]["strategy"]
            fill_value = config["preprocessing"]["handle_missing"]["fill_value"]

            if strategy == "constant" and fill_value is not None:
                imputer = imputer_class(
                    strategy=strategy, fill_value=fill_value)
            else:
                imputer = imputer_class(
                    strategy="constant", fill_value="missing")

            categorical_steps.append(("imputer", imputer))

        # One-hot encoding
        encoder_config = preprocessing_options["encoders"]["OneHotEncoder"]
        encoder_class = _import_class(
            encoder_config["module"], encoder_config["class_name"])
        encoder = encoder_class(**encoder_config["init_params"])
        categorical_steps.append(("encoder", encoder))

        categorical_pipeline = Pipeline(categorical_steps)
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features))

    # Numeric features pipeline
    numeric_features = config["features"]["numeric"]
    if numeric_features:
        numeric_steps = []

        # Handle missing values
        if config["preprocessing"]["handle_missing"]["enabled"]:
            imputer_config = preprocessing_options["imputers"]["SimpleImputer"]
            imputer_class = _import_class(
                imputer_config["module"], imputer_config["class_name"])

            strategy = config["preprocessing"]["handle_missing"]["strategy"]
            fill_value = config["preprocessing"]["handle_missing"]["fill_value"]

            if strategy == "constant" and fill_value is not None:
                imputer = imputer_class(
                    strategy=strategy, fill_value=fill_value)
            else:
                imputer = imputer_class(strategy=strategy)

            numeric_steps.append(("imputer", imputer))

        # Scaling
        if config["preprocessing"]["numeric_scaling"]["enabled"]:
            scaler_name = config["preprocessing"]["numeric_scaling"]["method"]
            scaler_config = preprocessing_options["scalers"].get(scaler_name)

            if scaler_config:
                scaler_class = _import_class(
                    scaler_config["module"], scaler_config["class_name"])
                scaler = scaler_class(**scaler_config["init_params"])
                numeric_steps.append(("scaler", scaler))
            else:
                logger.warning(
                    f"Unknown scaler '{scaler_name}', skipping scaling")

        if numeric_steps:
            numeric_pipeline = Pipeline(numeric_steps)
        else:
            numeric_pipeline = "passthrough"

        transformers.append(("numeric", numeric_pipeline, numeric_features))

    # Boolean features pipeline
    boolean_features = config["features"]["boolean"]
    if boolean_features:
        boolean_pipeline = Pipeline([
            ("caster", FunctionTransformer(
                lambda x: x.astype(int),
                feature_names_out="one-to-one"
            ))
        ])
        transformers.append(("boolean", boolean_pipeline, boolean_features))

    # Create the column transformer
    if not transformers:
        raise ValueError(
            "No feature transformers defined. Check your feature configuration.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"  # Drop any features not specified
    )

    return preprocessor


def build_feature_selector(config: Dict[str, Any]) -> Optional[Any]:
    """
    Build feature selection step if enabled.

    Args:
        config: Configuration dictionary

    Returns:
        Feature selector object or None
    """
    if not config["feature_engineering"]["feature_selection"]["enabled"]:
        return None

    method = config["feature_engineering"]["feature_selection"]["method"]
    k = config["feature_engineering"]["feature_selection"]["k"]

    feature_selection_options = get_feature_selection_options(config)
    feature_selection_config = feature_selection_options.get(method)

    if feature_selection_config:
        selector_class = _import_class(
            feature_selection_config["module"],
            feature_selection_config["class_name"]
        )
        if selector_class:
            return selector_class(f_classif, k=k)

    logger.warning(
        f"Unknown feature selection method '{method}', skipping feature selection")
    return None


def build_sampler(config: Dict[str, Any]) -> Optional[Any]:
    """
    Build sampling step for class imbalance handling if enabled.

    Args:
        config: Configuration dictionary

    Returns:
        Sampler object or None
    """
    if not config["class_imbalance"]["enabled"] or not IMBALANCED_LEARN_AVAILABLE:
        if config["class_imbalance"]["enabled"] and not IMBALANCED_LEARN_AVAILABLE:
            logger.warning(
                "Class imbalance handling requested but imbalanced-learn not available")
        return None

    method = config["class_imbalance"]["method"]
    sampling_strategy = config["class_imbalance"]["sampling_strategy"]
    random_state = config["data"]["random_state"]

    sampling_methods = get_sampling_methods(config)
    sampler_config = sampling_methods.get(method)

    if sampler_config:
        sampler_class = _import_class(
            sampler_config["module"], sampler_config["class_name"])
        if sampler_class:
            init_params = sampler_config["init_params"].copy()
            init_params["sampling_strategy"] = sampling_strategy

            # Add random_state if the sampler supports it
            try:
                sampler = sampler_class(
                    random_state=random_state, **init_params)
            except TypeError:
                # Some samplers don't support random_state
                sampler = sampler_class(**init_params)

            return sampler

    logger.warning(
        f"Unknown sampling method '{method}', skipping class imbalance handling")
    return None


def build_pipeline(config: Dict[str, Any], model_name: str) -> Pipeline:
    """
    Build a complete ML pipeline based on configuration.

    Args:
        config: Configuration dictionary
        model_name: Name of the model to use

    Returns:
        Complete ML pipeline
    """
    logger.info(f"Building pipeline for {model_name}...")

    # Get model definitions with instantiated models
    model_definitions = get_model_definitions_with_instances(config)

    if model_name not in model_definitions:
        available_models = list(model_definitions.keys())
        raise ValueError(
            f"Model '{model_name}' not available. Available models: {available_models}")

    # Build pipeline steps
    steps = []

    # 1. Preprocessing
    preprocessor = build_preprocessor(config)
    steps.append(("preprocessor", preprocessor))

    # 2. Feature selection (optional)
    feature_selector = build_feature_selector(config)
    if feature_selector is not None:
        steps.append(("feature_selection", feature_selector))

    # 3. Sampling for class imbalance (optional)
    sampler = build_sampler(config)
    if sampler is not None:
        steps.append(("sampler", sampler))

    # 4. Classifier
    classifier = model_definitions[model_name]["model"]
    steps.append(("classifier", classifier))

    # Create pipeline (use ImbPipeline if sampler is present)
    if sampler is not None and IMBALANCED_LEARN_AVAILABLE:
        pipeline = ImbPipeline(steps)
    else:
        pipeline = Pipeline(steps)

    logger.info(f"Pipeline built successfully for {model_name}")
    logger.info(f"Pipeline steps: {[step[0] for step in steps]}")

    return pipeline


def get_hyperparameter_grid(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get hyperparameter grid for a specific model.

    Args:
        config: Configuration dictionary
        model_name: Name of the model

    Returns:
        Parameter grid for hyperparameter tuning
    """
    model_definitions = get_model_definitions(config)

    if model_name not in model_definitions:
        raise ValueError(f"Model '{model_name}' not found")

    return model_definitions[model_name]["param_grid"]


def get_feature_names_after_preprocessing(
    pipeline: Pipeline,
    config: Dict[str, Any]
) -> List[str]:
    """
    Get feature names after preprocessing transformations.

    Args:
        pipeline: Fitted pipeline
        config: Configuration dictionary

    Returns:
        List of feature names after preprocessing
    """
    try:
        # Get preprocessor from pipeline
        if "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()

            # If feature selection was applied, filter the names
            if "feature_selection" in pipeline.named_steps:
                selector = pipeline.named_steps["feature_selection"]
                if hasattr(selector, "get_support"):
                    mask = selector.get_support()
                    feature_names = [
                        name for name, selected in zip(feature_names, mask) if selected
                    ]

            return list(feature_names)

    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")

    # Fallback to generic names
    n_features = len(pipeline.named_steps["classifier"].feature_importances_) if hasattr(
        pipeline.named_steps["classifier"], "feature_importances_"
    ) else len(config["features"]["categorical"] + config["features"]["numeric"] + config["features"]["boolean"])

    return [f"feature_{i}" for i in range(n_features)]


def create_pipeline_summary(pipeline: Pipeline, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a summary of the pipeline configuration.

    Args:
        pipeline: Built pipeline
        config: Configuration dictionary

    Returns:
        Dictionary with pipeline summary
    """
    summary = {
        "pipeline_steps": [step[0] for step in pipeline.steps],
        "preprocessing": {
            "categorical_features": len(config["features"]["categorical"]),
            "numeric_features": len(config["features"]["numeric"]),
            "boolean_features": len(config["features"]["boolean"]),
            "scaling_enabled": config["preprocessing"]["numeric_scaling"]["enabled"],
            "scaling_method": config["preprocessing"]["numeric_scaling"]["method"] if config["preprocessing"]["numeric_scaling"]["enabled"] else None,
            "missing_value_handling": config["preprocessing"]["handle_missing"]["enabled"],
        },
        "feature_engineering": {
            "feature_selection_enabled": config["feature_engineering"]["feature_selection"]["enabled"],
            "feature_selection_k": config["feature_engineering"]["feature_selection"]["k"] if config["feature_engineering"]["feature_selection"]["enabled"] else None,
        },
        "class_imbalance": {
            "sampling_enabled": config["class_imbalance"]["enabled"],
            "sampling_method": config["class_imbalance"]["method"] if config["class_imbalance"]["enabled"] else None,
        },
        "model": {
            "type": pipeline.named_steps["classifier"].__class__.__name__,
        }
    }

    return summary


if __name__ == "__main__":
    print("Pipeline Builder Module")

    # Example configuration (this would normally be loaded from JSON)
    sample_config = {
        "data": {"random_state": 42},
        "features": {
            "categorical": ["cat1", "cat2"],
            "numeric": ["num1", "num2", "num3"],
            "boolean": ["bool1"]
        },
        "preprocessing": {
            "numeric_scaling": {"enabled": True, "method": "StandardScaler"},
            "handle_missing": {"enabled": True, "strategy": "mean", "fill_value": None}
        },
        "feature_engineering": {
            "feature_selection": {"enabled": False, "method": "SelectKBest", "k": 5}
        },
        "class_imbalance": {
            "enabled": False, "method": "SMOTE", "sampling_strategy": "auto"
        }
    }

    try:
        # Test model definitions
        models = get_model_definitions(sample_config)
        print(f"Available models: {list(models.keys())}")

        if models:
            # Test pipeline building with first available model
            first_model = list(models.keys())[0]
            pipeline = build_pipeline(sample_config, first_model)
            print(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")

            # Test pipeline summary
            summary = create_pipeline_summary(pipeline, sample_config)
            print(f"Pipeline summary: {summary}")

        print("Pipeline builder module loaded successfully")

    except Exception as e:
        print(f"Error testing pipeline builder: {e}")
        print("Make sure the JSON configuration files exist in the 'configs/' directory")
