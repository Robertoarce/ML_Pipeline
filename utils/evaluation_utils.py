#!/usr/bin/env python
"""
Evaluation Utilities for ML Pipeline

This module provides utilities for:
- Model performance evaluation
- Metrics calculation and logging
- Cross-validation operations
- Validation checks and monitoring
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for positive class)
        prefix: Prefix to add to metric names (e.g., "train_", "test_")

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{prefix}precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0)
    metrics[f"{prefix}recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0)
    metrics[f"{prefix}f1_score"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0)

    # Probability-based metrics (if probabilities are provided)
    if y_pred_proba is not None:
        try:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            metrics[f"{prefix}average_precision"] = average_precision_score(
                y_true, y_pred_proba)
        except ValueError as e:
            logger.warning(
                f"Could not calculate probability-based metrics: {e}")

    # Per-class metrics for binary classification
    if len(np.unique(y_true)) == 2:
        # Positive class metrics
        metrics[f"{prefix}precision_pos"] = precision_score(
            y_true, y_pred, pos_label=1, zero_division=0)
        metrics[f"{prefix}recall_pos"] = recall_score(
            y_true, y_pred, pos_label=1, zero_division=0)
        metrics[f"{prefix}f1_pos"] = f1_score(
            y_true, y_pred, pos_label=1, zero_division=0)

        # Negative class metrics
        metrics[f"{prefix}precision_neg"] = precision_score(
            y_true, y_pred, pos_label=0, zero_division=0)
        metrics[f"{prefix}recall_neg"] = recall_score(
            y_true, y_pred, pos_label=0, zero_division=0)
        metrics[f"{prefix}f1_neg"] = f1_score(
            y_true, y_pred, pos_label=0, zero_division=0)

    return metrics


def evaluate_model_on_dataset(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "test"
) -> Dict[str, float]:
    """
    Evaluate a model on a given dataset and return comprehensive metrics.

    Args:
        model: Trained model pipeline
        X: Features
        y: True labels
        dataset_name: Name of the dataset (for metric prefixes)

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating model on {dataset_name} dataset...")

    try:
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(
            model, "predict_proba") else None

        # Calculate metrics
        metrics = calculate_classification_metrics(
            y_true=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            prefix=f"{dataset_name}_"
        )

        # Log metrics summary
        logger.info(
            f"{dataset_name.capitalize()} dataset evaluation completed")
        for metric_name, value in metrics.items():
            if "roc_auc" in metric_name or "accuracy" in metric_name:
                logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    except Exception as e:
        logger.error(
            f"Failed to evaluate model on {dataset_name} dataset: {e}")
        raise


def perform_cross_validation(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Perform cross-validation evaluation.

    Args:
        model: Model pipeline to evaluate
        X: Training features
        y: Training labels
        config: Configuration dictionary
        model_name: Name of the model (for logging)

    Returns:
        Dictionary of cross-validation metrics
    """
    if not config["cross_validation"]["enabled"]:
        logger.info("Cross-validation is disabled, skipping...")
        return {}

    logger.info(f"Performing cross-validation for {model_name}...")

    try:
        # Create cross-validation object
        cv = StratifiedKFold(
            n_splits=config["cross_validation"]["cv_folds"],
            shuffle=True,
            random_state=config["data"]["random_state"],
        )

        # Perform cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=config["cross_validation"]["scoring"],
            return_train_score=True,
            n_jobs=config["hyperparameter_tuning"].get("n_jobs", -1),
        )

        # Calculate mean and std for each metric
        cv_metrics = {}
        for metric in config["cross_validation"]["scoring"]:
            test_scores = cv_results[f"test_{metric}"]
            train_scores = cv_results[f"train_{metric}"]

            cv_metrics[f"cv_{metric}_mean"] = test_scores.mean()
            cv_metrics[f"cv_{metric}_std"] = test_scores.std()
            cv_metrics[f"cv_train_{metric}_mean"] = train_scores.mean()
            cv_metrics[f"cv_train_{metric}_std"] = train_scores.std()

            # Calculate overfitting ratio
            if test_scores.mean() > 0:
                cv_metrics[f"cv_{metric}_overfitting_ratio"] = (
                    train_scores.mean() / test_scores.mean()
                )

        # Log results summary
        logger.info("Cross-validation results:")
        for metric in config["cross_validation"]["scoring"]:
            mean_score = cv_metrics[f"cv_{metric}_mean"]
            std_score = cv_metrics[f"cv_{metric}_std"]
            logger.info(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")

        return cv_metrics

    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return {}


def check_model_performance(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    config: Dict[str, Any]
) -> List[str]:
    """
    Check model performance against configured thresholds and detect issues.

    Args:
        train_metrics: Training dataset metrics
        test_metrics: Test dataset metrics
        config: Configuration dictionary

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Extract key metrics
    train_score = train_metrics.get(
        "train_roc_auc", train_metrics.get("train_accuracy", 0))
    test_score = test_metrics.get(
        "test_roc_auc", test_metrics.get("test_accuracy", 0))

    # Check overfitting
    overfitting_threshold = config["training"]["overfitting_threshold"]
    if train_score > test_score * overfitting_threshold:
        warning = (
            f"Overfitting detected: Train score ({train_score:.4f}) is "
            f"{((train_score/test_score - 1) * 100):.1f}% higher than test score ({test_score:.4f})"
        )
        warnings.append(warning)
        logger.warning(warning)

    # Check minimum performance threshold
    min_test_score = config["performance_thresholds"]["min_test_score"]
    if test_score < min_test_score:
        warning = (
            f"Low performance detected: Test score ({test_score:.4f}) "
            f"is below minimum threshold ({min_test_score:.4f})"
        )
        warnings.append(warning)
        logger.warning(warning)

    # Check performance stability
    if abs(train_score - test_score) > 0.15:  # 15% difference
        warning = (
            f"Large performance gap detected: "
            f"Train-Test difference is {abs(train_score - test_score):.4f}"
        )
        warnings.append(warning)
        logger.warning(warning)

    # Check for extremely low scores
    if test_score < 0.5:
        warning = f"Very low test score detected ({test_score:.4f}), model may be performing worse than random"
        warnings.append(warning)
        logger.warning(warning)

    if not warnings:
        logger.info("All performance checks passed")

    return warnings


def create_classification_report_dict(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a detailed classification report as a dictionary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the classes

    Returns:
        Classification report dictionary
    """
    try:
        if target_names is None:
            unique_labels = sorted(y_true.unique())
            target_names = [f"class_{label}" for label in unique_labels]

        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )

        return report

    except Exception as e:
        logger.error(f"Failed to create classification report: {e}")
        return {}


def calculate_prediction_confidence_stats(y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about prediction confidence.

    Args:
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of confidence statistics
    """
    stats = {}

    try:
        stats["mean_confidence"] = y_pred_proba.mean()
        stats["std_confidence"] = y_pred_proba.std()
        stats["min_confidence"] = y_pred_proba.min()
        stats["max_confidence"] = y_pred_proba.max()
        stats["median_confidence"] = np.median(y_pred_proba)

        # Confidence quartiles
        stats["q25_confidence"] = np.percentile(y_pred_proba, 25)
        stats["q75_confidence"] = np.percentile(y_pred_proba, 75)

        # Predictions with high/low confidence
        high_confidence = (y_pred_proba > 0.8) | (y_pred_proba < 0.2)
        stats["high_confidence_ratio"] = high_confidence.mean()

        # Very uncertain predictions (around 0.5)
        uncertain = (y_pred_proba > 0.4) & (y_pred_proba < 0.6)
        stats["uncertain_predictions_ratio"] = uncertain.mean()

    except Exception as e:
        logger.error(f"Failed to calculate confidence stats: {e}")

    return stats


def compare_model_performance(
    models_metrics: Dict[str, Dict[str, float]],
    comparison_metric: str = "test_roc_auc"
) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Args:
        models_metrics: Dictionary mapping model names to their metrics
        comparison_metric: Primary metric for comparison

    Returns:
        DataFrame with model comparison results
    """
    comparison_data = []

    for model_name, metrics in models_metrics.items():
        row = {"model_name": model_name}

        # Add key metrics
        key_metrics = [
            "test_accuracy", "test_precision", "test_recall", "test_f1_score",
            "test_roc_auc", "test_average_precision"
        ]

        for metric in key_metrics:
            row[metric] = metrics.get(metric, None)

        # Add cross-validation metrics if available
        cv_metrics = ["cv_roc_auc_mean",
                      "cv_accuracy_mean", "cv_f1_weighted_mean"]
        for metric in cv_metrics:
            row[metric] = metrics.get(metric, None)

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Sort by comparison metric if available
    if comparison_metric in df.columns and df[comparison_metric].notna().any():
        df = df.sort_values(comparison_metric, ascending=False, na_last=True)

    return df


def log_model_evaluation_summary(
    model_name: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    cv_metrics: Optional[Dict[str, float]] = None,
    experiment_tracking=None
) -> None:
    """
    Log a comprehensive evaluation summary to experiment tracking platform.

    Args:
        model_name: Name of the model
        train_metrics: Training metrics
        test_metrics: Test metrics  
        cv_metrics: Cross-validation metrics (optional)
        experiment_tracking: Experiment tracking platform object
    """
    # Create summary dictionary
    summary = {
        "model_name": model_name,
        **train_metrics,
        **test_metrics
    }

    if cv_metrics:
        summary.update(cv_metrics)

    # Log to experiment tracking platform
    if experiment_tracking is not None:
        try:
            experiment_tracking.log(summary)
            logger.info(f"Evaluation summary logged for {model_name}")
        except Exception as e:
            logger.error(f"Failed to log evaluation summary: {e}")

    # Print summary to console
    print(f"\n=== {model_name} Evaluation Summary ===")
    print(f"Test ROC AUC: {test_metrics.get('test_roc_auc', 'N/A'):.4f}")
    print(f"Test Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"Test F1 Score: {test_metrics.get('test_f1_score', 'N/A'):.4f}")

    if cv_metrics and "cv_roc_auc_mean" in cv_metrics:
        print(
            f"CV ROC AUC: {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics.get('cv_roc_auc_std', 0):.4f}")


if __name__ == "__main__":
    print("Evaluation Utilities Module")

    # Example usage with dummy data
    np.random.seed(42)
    y_true = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))
    y_pred = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
    y_pred_proba = np.random.random(100)

    # Calculate metrics
    metrics = calculate_classification_metrics(
        y_true, y_pred, y_pred_proba, "test_")
    print("\nSample metrics calculated:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Confidence stats
    confidence_stats = calculate_prediction_confidence_stats(y_pred_proba)
    print(f"\nConfidence stats: {confidence_stats}")

    print("\nEvaluation utilities module loaded successfully")
