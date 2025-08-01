#!/usr/bin/env python
"""
Model Utilities for ML Pipeline

This module provides utilities for:
- Model persistence (save/load locally)
- Model validation and checks
- Model metadata management
- Generic model operations
"""

import os
import datetime
import pickle
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def persist_model(
    model: Pipeline,
    config: Dict[str, Any],
    model_metadata: Dict[str, Any],
    experiment_tracker=None  # wandb tracker or other tracking platform
) -> None:
    """
    Save the model locally and log to experiment tracking platform.

    Args:
        model: Trained scikit-learn pipeline
        config: Configuration dictionary
        model_metadata: Model metadata dictionary
        experiment_tracker: Experiment tracking platform object (e.g., WandBTracker)
    """
    try:
        # Determine save directory
        local_dir = config["model_persistence"]["local_dir"]

        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}.pkl"
        local_model_path = os.path.join(local_dir, model_filename)

        # Save model locally using cloudpickle for better compatibility
        try:
            import cloudpickle
            with open(local_model_path, "wb") as f:
                cloudpickle.dump(model, f, protocol=4)
        except ImportError:
            # Fallback to standard pickle
            logger.warning("cloudpickle not available, using standard pickle")
            with open(local_model_path, "wb") as f:
                pickle.dump(model, f, protocol=4)

        logger.info(f"Model saved locally at {local_model_path}")

        # Save metadata
        metadata_path = local_model_path.replace(".pkl", "_metadata.json")
        import json
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved to {metadata_path}")

        # Log to experiment tracking platform
        if experiment_tracker is not None and experiment_tracker.is_active:
            try:
                experiment_tracker.log_artifact(
                    artifact_path=local_model_path,
                    artifact_name="trained_model",
                    artifact_type="model",
                    metadata={
                        "framework": "scikit-learn",
                        **model_metadata,
                        "config_summary": {
                            "project_name": config["project_name"],
                            "target_label": config["target_label"],
                            "models_enabled": config["models"]["enabled"]
                        }
                    }
                )
                logger.info("Model logged to experiment tracking platform")
            except Exception as e:
                logger.error(
                    f"Failed to log model to experiment tracking: {e}")

    except Exception as e:
        logger.error(f"Failed to persist model: {e}")
        raise


def load_model_from_file(model_path: str) -> Pipeline:
    """
    Load a model from a local file.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded scikit-learn pipeline

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Try cloudpickle first
        try:
            import cloudpickle
            with open(model_path, "rb") as f:
                model = cloudpickle.load(f)
        except ImportError:
            # Fallback to standard pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        logger.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def add_missing_categories(
    model: Pipeline,
    supported_categories: Dict[str, list],
    preprocessor_name: str = "preprocessor"
) -> Pipeline:
    """
    Add missing categories to one-hot encoders to handle unseen categories during inference.

    This is a production safety feature that prevents model crashes when new category values
    appear during inference that weren't present in the training data.

    Args:
        model: Trained scikit-learn pipeline
        supported_categories: Dictionary mapping feature names to supported category lists
        preprocessor_name: Name of the preprocessor step in the pipeline

    Returns:
        Updated pipeline with complete category support
    """
    try:
        # Find the preprocessor step
        if preprocessor_name not in model.named_steps:
            logger.warning(
                f"Preprocessor '{preprocessor_name}' not found in pipeline")
            return model

        preprocessor = model.named_steps[preprocessor_name]

        # Find categorical transformer
        if hasattr(preprocessor, "named_transformers_") and "categorical" in preprocessor.named_transformers_:
            cat_transformer = preprocessor.named_transformers_["categorical"]

            # Find the encoder
            if hasattr(cat_transformer, "named_steps") and "encoder" in cat_transformer.named_steps:
                encoder = cat_transformer.named_steps["encoder"]

                if hasattr(encoder, "categories_"):
                    # Update categories
                    new_categories = []
                    for i, current_cats in enumerate(encoder.categories_):
                        if i < len(supported_categories):
                            feature_name = list(supported_categories.keys())[i]
                            supported_cats = supported_categories[feature_name]
                            missing_cats = np.setdiff1d(
                                supported_cats, current_cats)
                            new_cats = np.concatenate(
                                [current_cats, missing_cats])
                            new_categories.append(new_cats)
                        else:
                            new_categories.append(current_cats)

                    encoder.categories_ = new_categories
                    logger.info(
                        "Updated encoder categories to handle missing categories")

        return model

    except Exception as e:
        logger.error(f"Failed to add missing categories: {e}")
        return model


def validate_model_inputs(
    model: Pipeline,
    X: pd.DataFrame,
    expected_features: list
) -> Tuple[bool, list]:
    """
    Validate that input features match what the model expects.

    Args:
        model: Trained model pipeline
        X: Input features DataFrame
        expected_features: List of expected feature names

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check if all expected features are present
    missing_features = set(expected_features) - set(X.columns)
    if missing_features:
        issues.append(f"Missing features: {list(missing_features)}")

    # Check for unexpected features
    extra_features = set(X.columns) - set(expected_features)
    if extra_features:
        issues.append(f"Unexpected features: {list(extra_features)}")

    # Check data types
    try:
        # Try to make a prediction with a small sample to check compatibility
        sample = X.head(1)
        model.predict(sample)
    except Exception as e:
        issues.append(f"Model prediction failed: {str(e)}")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("Model input validation passed")
    else:
        logger.warning(f"Model input validation failed: {issues}")

    return is_valid, issues


def get_model_feature_importance(
    model: Pipeline,
    feature_names: list,
    top_n: int = 20
) -> Optional[pd.DataFrame]:
    """
    Extract feature importance from tree-based models.

    Args:
        model: Trained model pipeline
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance or None if not available
    """
    try:
        # Get the classifier step
        if "classifier" not in model.named_steps:
            logger.warning("No classifier step found in pipeline")
            return None

        classifier = model.named_steps["classifier"]

        # Check if model has feature importance
        if not hasattr(classifier, "feature_importances_"):
            logger.info("Model does not have feature_importances_ attribute")
            return None

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(classifier.feature_importances_)],
            "importance": classifier.feature_importances_
        })

        # Sort by importance
        importance_df = importance_df.sort_values(
            "importance", ascending=False)

        # Return top N features
        return importance_df.head(top_n)

    except Exception as e:
        logger.error(f"Failed to extract feature importance: {e}")
        return None


def create_model_metadata(
    model_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for a trained model.

    Args:
        model_name: Name of the model
        config: Configuration dictionary
        metrics: Performance metrics dictionary
        X_train: Training features
        X_test: Test features

    Returns:
        Dictionary containing model metadata
    """
    metadata = {
        # Model information
        "model_name": model_name,
        "model_type": "binary_classification",
        "framework": "scikit-learn",

        # Project information
        "project_name": config["project_name"],
        "target_label": config["target_label"],
        "experiment_description": config.get("experiment_description", ""),

        # Data information
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(X_train.columns),
        "feature_names": list(X_train.columns),

        # Feature types
        "categorical_features": config["features"]["categorical"],
        "numeric_features": config["features"]["numeric"],
        "boolean_features": config["features"]["boolean"],

        # Performance metrics
        "performance_metrics": metrics,

        # Configuration summary
        "configuration": {
            "hyperparameter_tuning_enabled": config["hyperparameter_tuning"]["enabled"],
            "cross_validation_enabled": config["cross_validation"]["enabled"],
            "feature_selection_enabled": config["feature_engineering"]["feature_selection"]["enabled"],
            "class_imbalance_handling": config["class_imbalance"]["enabled"],
            "preprocessing_scaling": config["preprocessing"]["numeric_scaling"]["enabled"],
        },

        # Timestamps
        "created_at": datetime.datetime.now().isoformat(),
        "training_duration": None,  # Can be updated if training time is tracked

        # Version info
        "pipeline_version": "1.0.0",
    }

    return metadata


def compare_model_versions(
    model_metadata_list: list,
    comparison_metric: str = "test_roc_auc"
) -> pd.DataFrame:
    """
    Compare different model versions based on performance metrics.

    Args:
        model_metadata_list: List of model metadata dictionaries
        comparison_metric: Metric to use for comparison

    Returns:
        DataFrame with model comparison results
    """
    comparison_data = []

    for metadata in model_metadata_list:
        try:
            comparison_data.append({
                "model_name": metadata["model_name"],
                "created_at": metadata["created_at"],
                "train_samples": metadata["train_samples"],
                "test_samples": metadata["test_samples"],
                "n_features": metadata["n_features"],
                comparison_metric: metadata["performance_metrics"].get(comparison_metric, None),
                "hyperparameter_tuning": metadata["configuration"]["hyperparameter_tuning_enabled"],
                "cross_validation": metadata["configuration"]["cross_validation_enabled"],
            })
        except KeyError as e:
            logger.warning(f"Missing key in metadata: {e}")
            continue

    if not comparison_data:
        logger.warning("No valid model metadata for comparison")
        return pd.DataFrame()

    df = pd.DataFrame(comparison_data)

    # Sort by comparison metric (descending)
    if comparison_metric in df.columns and df[comparison_metric].notna().any():
        df = df.sort_values(comparison_metric, ascending=False, na_last=True)

    return df


def generate_model_summary_report(
    model: Pipeline,
    metadata: Dict[str, Any],
    metrics: Dict[str, float]
) -> str:
    """
    Generate a human-readable summary report for a trained model.

    Args:
        model: Trained model pipeline
        metadata: Model metadata
        metrics: Performance metrics

    Returns:
        Formatted summary report string
    """
    report = []
    report.append("=" * 60)
    report.append("ML PIPELINE MODEL SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")

    # Project Information
    report.append("PROJECT INFORMATION:")
    report.append(f"  Project Name: {metadata.get('project_name', 'Unknown')}")
    report.append(f"  Model Name: {metadata.get('model_name', 'Unknown')}")
    report.append(
        f"  Target Variable: {metadata.get('target_label', 'Unknown')}")
    report.append(f"  Created: {metadata.get('created_at', 'Unknown')}")
    report.append("")

    # Data Information
    report.append("DATA INFORMATION:")
    report.append(
        f"  Training Samples: {metadata.get('train_samples', 'Unknown'):,}")
    report.append(
        f"  Test Samples: {metadata.get('test_samples', 'Unknown'):,}")
    report.append(f"  Total Features: {metadata.get('n_features', 'Unknown')}")
    report.append(
        f"  Categorical Features: {len(metadata.get('categorical_features', []))}")
    report.append(
        f"  Numeric Features: {len(metadata.get('numeric_features', []))}")
    report.append(
        f"  Boolean Features: {len(metadata.get('boolean_features', []))}")
    report.append("")

    # Performance Metrics
    report.append("PERFORMANCE METRICS:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            report.append(f"  {metric_name}: {value:.4f}")
    report.append("")

    # Configuration
    report.append("CONFIGURATION:")
    config = metadata.get('configuration', {})
    report.append(
        f"  Hyperparameter Tuning: {config.get('hyperparameter_tuning_enabled', 'Unknown')}")
    report.append(
        f"  Cross Validation: {config.get('cross_validation_enabled', 'Unknown')}")
    report.append(
        f"  Feature Selection: {config.get('feature_selection_enabled', 'Unknown')}")
    report.append(
        f"  Class Imbalance Handling: {config.get('class_imbalance_handling', 'Unknown')}")
    report.append(
        f"  Numeric Scaling: {config.get('preprocessing_scaling', 'Unknown')}")
    report.append("")

    report.append("=" * 60)

    return "\n".join(report)


def find_latest_model(model_dir: str) -> Optional[str]:
    """
    Find the most recently saved model in a directory.

    Args:
        model_dir: Directory containing model files

    Returns:
        Path to the latest model file or None if no models found
    """
    if not os.path.exists(model_dir):
        return None

    model_files = [f for f in os.listdir(model_dir) if f.endswith(
        '.pkl') and f.startswith('model_')]

    if not model_files:
        return None

    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: os.path.getmtime(
        os.path.join(model_dir, x)), reverse=True)

    return os.path.join(model_dir, model_files[0])


def list_saved_models(model_dir: str) -> List[Dict[str, Any]]:
    """
    List all saved models in a directory with their metadata.

    Args:
        model_dir: Directory containing model files

    Returns:
        List of dictionaries containing model information
    """
    if not os.path.exists(model_dir):
        return []

    models = []
    model_files = [f for f in os.listdir(model_dir) if f.endswith(
        '.pkl') and f.startswith('model_')]

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        metadata_path = model_path.replace('.pkl', '_metadata.json')

        model_info = {
            "model_file": model_file,
            "model_path": model_path,
            "created": datetime.datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
            "size_mb": os.path.getsize(model_path) / 1024 / 1024,
        }

        # Load metadata if available
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_info.update({
                    "model_name": metadata.get("model_name", "Unknown"),
                    "project_name": metadata.get("project_name", "Unknown"),
                    "performance_metrics": metadata.get("performance_metrics", {}),
                })
            except Exception as e:
                logger.warning(
                    f"Could not load metadata for {model_file}: {e}")

        models.append(model_info)

    # Sort by creation time (most recent first)
    models.sort(key=lambda x: x["created"], reverse=True)

    return models


if __name__ == "__main__":
    print("Model Utilities Module")
    print("This module provides utilities for model management and persistence.")

    # Example of creating metadata
    sample_config = {
        "project_name": "Test Project",
        "target_label": "target",
        "experiment_description": "Test experiment",
        "features": {
            "categorical": ["cat1", "cat2"],
            "numeric": ["num1", "num2", "num3"],
            "boolean": ["bool1"]
        },
        "hyperparameter_tuning": {"enabled": True},
        "cross_validation": {"enabled": True},
        "feature_engineering": {"feature_selection": {"enabled": False}},
        "class_imbalance": {"enabled": False},
        "preprocessing": {"numeric_scaling": {"enabled": True}}
    }

    sample_metrics = {
        "test_roc_auc": 0.85,
        "test_accuracy": 0.82,
        "test_f1_score": 0.78
    }

    # Create dummy dataframes
    X_train = pd.DataFrame(np.random.randn(100, 6), columns=[
                           "cat1", "cat2", "num1", "num2", "num3", "bool1"])
    X_test = pd.DataFrame(np.random.randn(20, 6), columns=[
                          "cat1", "cat2", "num1", "num2", "num3", "bool1"])

    metadata = create_model_metadata(
        "TestModel", sample_config, sample_metrics, X_train, X_test)
    print("\nSample metadata created successfully")
    print(f"Metadata keys: {list(metadata.keys())}")

    # Test model directory listing
    models_dir = "./models/"
    if os.path.exists(models_dir):
        saved_models = list_saved_models(models_dir)
        print(f"\nFound {len(saved_models)} saved models in {models_dir}")
    else:
        print(f"\nModels directory {models_dir} does not exist")
