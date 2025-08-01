#!/usr/bin/env python
"""
Generic ML Pipeline Main Script

This is a generic, modular machine learning pipeline for binary classification.
It can be configured for different use cases through YAML configuration files.

Features:
- Configurable preprocessing and feature engineering
- Multiple model support with hyperparameter tuning
- Cross-validation and model evaluation
- Experiment tracking (W&B integration)
- Model persistence and deployment readiness
- Data validation and monitoring

Usage:
    python main.py [--config CONFIG_NAME] [--help]

Examples:
    python main.py                          # Use default configuration
    python main.py --config customer_churn  # Use customer churn configuration
"""

import os
import sys
import argparse
import datetime
import logging
import traceback
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Local imports
from config import load_config, load_data_from_source
from utils.model_utils import persist_model, create_model_metadata
from utils.evaluation_utils import (
    evaluate_model_on_dataset,
    perform_cross_validation,
    check_model_performance,
    compare_model_performance,
    log_model_evaluation_summary
)
from utils.wandb_tracker import create_wandb_tracker
from src.pipeline_builder import (
    build_pipeline,
    get_hyperparameter_grid,
    get_feature_names_after_preprocessing
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(config: Dict[str, Any], tracker) -> pd.DataFrame:
    """
    Load the training data.

    Args:
        config: Configuration dictionary
        tracker: Experiment tracker (WandB)

    Returns:
        Loaded DataFrame
    """
    logger.info("Loading training data...")

    # Load data
    df = load_data_from_source(config)

    logger.info(f"Data loaded successfully: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Log data summary
    target_column = config["target_label"]
    if target_column in df.columns:
        target_dist = df[target_column].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")

        # Log to experiment tracking
        if tracker.is_active:
            tracker.log_dataset_info(df, "training")

    return df


def split_data(
    df: pd.DataFrame,
    config: Dict[str, Any],
    tracker
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets.

    Args:
        df: Complete dataset
        config: Configuration dictionary
        tracker: Experiment tracker

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Splitting data...")

    # Separate features and target
    target_column = config["target_label"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_set_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    # Second split: separate train and validation
    val_size_from_temp = config["data"]["validation_set_size"] / \
        (1 - config["data"]["test_set_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_from_temp,
        random_state=config["data"]["random_state"],
        stratify=y_temp
    )

    logger.info(
        f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Log split information
    if tracker.is_active:
        tracker.log_data_split_info(len(X_train), len(X_val), len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any],
    model_name: str,
    tracker
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a single model and return it with metrics.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary
        model_name: Name of the model to train
        tracker: Experiment tracker

    Returns:
        Tuple of (trained_model, metrics)
    """
    logger.info(f"Training {model_name}...")

    # Build pipeline
    pipeline = build_pipeline(config, model_name)

    # Hyperparameter tuning if enabled
    if config["hyperparameter_tuning"]["enabled"]:
        trained_model = train_with_hyperparameter_tuning(
            pipeline, X_train, y_train, config, model_name, tracker
        )
    else:
        trained_model = pipeline.fit(X_train, y_train)

    # Cross-validation
    cv_metrics = perform_cross_validation(
        trained_model, X_train, y_train, config, model_name)

    # Validation evaluation
    val_metrics = evaluate_model_on_dataset(
        trained_model, X_val, y_val, "validation")

    # Log to tracker
    if tracker.is_active:
        tracker.log_model_performance(val_metrics, model_name, "validation")
        if cv_metrics:
            tracker.log_metrics(cv_metrics)

    # Combine metrics
    all_metrics = {**cv_metrics, **val_metrics}

    logger.info(f"{model_name} training completed")
    return trained_model, all_metrics


def train_with_hyperparameter_tuning(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    model_name: str,
    tracker
) -> Pipeline:
    """
    Train model with hyperparameter tuning.

    Args:
        pipeline: Base pipeline to tune
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        model_name: Name of the model
        tracker: Experiment tracker

    Returns:
        Best trained model
    """
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

    logger.info(f"Starting hyperparameter tuning for {model_name}...")

    # Get parameter grid
    param_grid = get_hyperparameter_grid(config, model_name)

    # Create cross-validation object
    cv = StratifiedKFold(
        n_splits=config["hyperparameter_tuning"]["cv_folds"],
        shuffle=True,
        random_state=config["data"]["random_state"]
    )

    # Choose search method
    if config["hyperparameter_tuning"]["method"] == "GridSearchCV":
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=config["hyperparameter_tuning"]["scoring"],
            n_jobs=config["hyperparameter_tuning"]["n_jobs"],
            verbose=1
        )
    else:  # RandomizedSearchCV
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=config["hyperparameter_tuning"]["n_iter"],
            cv=cv,
            scoring=config["hyperparameter_tuning"]["scoring"],
            n_jobs=config["hyperparameter_tuning"]["n_jobs"],
            random_state=config["data"]["random_state"],
            verbose=1
        )

    # Perform search
    search.fit(X_train, y_train)

    # Log results
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")

    if tracker.is_active:
        tracker.log_hyperparameter_results(
            search.best_params_, search.best_score_, model_name)

    return search.best_estimator_


def compare_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
    tracker
) -> Tuple[Pipeline, str, Dict[str, float]]:
    """
    Train and compare multiple models.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
        tracker: Experiment tracker

    Returns:
        Tuple of (best_model, best_model_name, best_metrics)
    """
    logger.info("Comparing multiple models...")

    models = {}
    all_metrics = {}

    for model_name in config["models"]["enabled"]:
        try:
            model, metrics = train_single_model(
                X_train, y_train, X_val, y_val, config, model_name, tracker
            )

            # Evaluate on test set
            test_metrics = evaluate_model_on_dataset(
                model, X_test, y_test, "test")

            # Log test performance
            if tracker.is_active:
                tracker.log_model_performance(test_metrics, model_name, "test")

            # Combine metrics
            combined_metrics = {**metrics, **test_metrics}

            models[model_name] = model
            all_metrics[model_name] = combined_metrics

            logger.info(
                f"{model_name} - Test ROC AUC: {test_metrics.get('test_roc_auc', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue

    if not models:
        raise ValueError("No models were successfully trained")

    # Select best model based on validation performance
    comparison_metric = "validation_roc_auc"
    best_model_name = max(
        models.keys(),
        key=lambda name: all_metrics[name].get(comparison_metric, 0)
    )

    best_model = models[best_model_name]
    best_metrics = all_metrics[best_model_name]

    logger.info(f"Best model: {best_model_name}")

    # Log comparison results
    if tracker.is_active:
        tracker.log_model_comparison(all_metrics, best_model_name)

    return best_model, best_model_name, best_metrics


def perform_final_evaluation(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
    tracker
) -> Dict[str, float]:
    """
    Perform final model evaluation and validation checks.

    Args:
        model: Trained model
        model_name: Name of the model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Configuration dictionary
        tracker: Experiment tracker

    Returns:
        Dictionary of final metrics
    """
    logger.info("Performing final evaluation...")

    # Get predictions and metrics for train and test sets
    train_metrics = evaluate_model_on_dataset(model, X_train, y_train, "train")
    test_metrics = evaluate_model_on_dataset(model, X_test, y_test, "test")

    # Log to tracker
    if tracker.is_active:
        tracker.log_model_performance(train_metrics, model_name, "train")
        tracker.log_model_performance(test_metrics, model_name, "test")

    # Performance checks
    warnings = check_model_performance(train_metrics, test_metrics, config)

    # Log warnings
    if warnings and tracker.is_active:
        tracker.log_metrics({"performance_warnings": warnings})

    # Feature importance (if available)
    if config["interpretability"]["enabled"] and config["interpretability"]["feature_importance"]:
        try:
            feature_names = get_feature_names_after_preprocessing(
                model, config)

            if hasattr(model.named_steps["classifier"], "feature_importances_"):
                importances = model.named_steps["classifier"].feature_importances_

                if tracker.is_active:
                    tracker.log_feature_importance(
                        feature_names, importances, model_name)

                # Show top features in log
                importance_pairs = list(zip(feature_names, importances))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                logger.info(
                    f"Top 5 important features: {importance_pairs[:5]}")

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

    # Combine all metrics
    final_metrics = {**train_metrics, **test_metrics}

    return final_metrics


def save_model_and_results(
    model: Pipeline,
    model_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    tracker
) -> None:
    """
    Save the trained model and results.

    Args:
        model: Trained model
        model_name: Name of the model
        config: Configuration dictionary
        metrics: Performance metrics
        X_train: Training features
        X_test: Test features
        tracker: Experiment tracker
    """
    logger.info("Saving model and results...")

    # Create model metadata
    metadata = create_model_metadata(
        model_name=model_name,
        config=config,
        metrics=metrics,
        X_train=X_train,
        X_test=X_test
    )

    # Save model
    persist_model(
        model=model,
        config=config,
        model_metadata=metadata,
        experiment_tracker=tracker
    )

    logger.info("Model and results saved successfully")


def main(config_name: Optional[str] = None):
    """
    Main training workflow.

    Args:
        config_name: Name of the configuration to use
    """
    try:
        # Load configuration
        config = load_config(config_name)

        logger.info(f"Starting ML pipeline: {config['project_name']}")
        logger.info(f"Target variable: {config['target_label']}")
        logger.info(f"Models to train: {config['models']['enabled']}")

        # Setup experiment tracking
        tracker = create_wandb_tracker(config)
        tracker.initialize()

        # Load data
        df = load_data(config, tracker)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, config, tracker)

        # Train models
        if config["models"]["compare_models"] and len(config["models"]["enabled"]) > 1:
            # Multiple model comparison
            best_model, best_model_name, metrics = compare_multiple_models(
                X_train, y_train, X_val, y_val, X_test, y_test, config, tracker
            )
        else:
            # Single model training
            model_name = config["models"]["enabled"][0]
            model, metrics = train_single_model(
                X_train, y_train, X_val, y_val, config, model_name, tracker
            )

            # Final evaluation on test set
            test_metrics = perform_final_evaluation(
                model, model_name, X_train, y_train, X_test, y_test, config, tracker
            )

            best_model = model
            best_model_name = model_name
            metrics.update(test_metrics)

        # Log training summary
        if tracker.is_active:
            tracker.log_training_summary(metrics, best_model_name)

        # Save model and results
        if config["model_persistence"]["enabled"]:
            save_model_and_results(
                best_model, best_model_name, config, metrics, X_train, X_test, tracker
            )

        # Final summary
        test_score = metrics.get(
            "test_roc_auc", metrics.get("test_accuracy", 0))
        logger.info(f"Training completed successfully!")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Final test score: {test_score:.4f}")

        if tracker.is_active and tracker.run_url:
            logger.info(f"W&B Run URL: {tracker.run_url}")

        # Finish experiment tracking
        tracker.finish(exit_code=0)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Try to log error to tracker
        try:
            if 'tracker' in locals() and tracker.is_active:
                tracker.log_metrics(
                    {"error": str(e), "traceback": traceback.format_exc()})
                tracker.finish(exit_code=1)
        except:
            pass

        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic ML Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration name to use (e.g., 'customer_churn')"
    )
    args = parser.parse_args()

    main(config_name=args.config)
