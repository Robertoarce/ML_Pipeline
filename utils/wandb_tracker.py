#!/usr/bin/env python
"""
Weights & Biases Experiment Tracker

This module provides a dedicated class for managing Weights & Biases integration
with comprehensive experiment tracking capabilities.
"""

import os
import datetime
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class WandBTracker:
    """
    Weights & Biases experiment tracker for ML pipeline.

    This class encapsulates all W&B functionality and provides a clean interface
    for experiment tracking throughout the ML pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the W&B tracker.

        Args:
            config: Configuration dictionary containing experiment tracking settings
        """
        self.config = config
        self.tracking_config = config.get("experiment_tracking", {})
        self.run = None
        self.enabled = self.tracking_config.get(
            "enabled", False) and WANDB_AVAILABLE

        if self.enabled and not WANDB_AVAILABLE:
            logger.warning(
                "W&B tracking enabled but wandb package not available")
            self.enabled = False

    def initialize(self) -> bool:
        """
        Initialize W&B session.

        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.enabled:
            logger.info("W&B tracking is disabled")
            return False

        try:
            # Generate run name if not provided
            run_name = self.tracking_config.get("run_name")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            if run_name is None:
                project_name = self.config.get(
                    "project_name", "ML-Pipeline").replace(" ", "_")
                run_name = f"{project_name}_{timestamp}"
            else:
                run_name = f"{run_name}_{timestamp}"

            # Initialize run in online mode - wandb will handle authentication automatically
            self.run = wandb.init(
                project=self.tracking_config.get(
                    "project", "ML-Pipeline-Experiments"),
                entity=self.tracking_config.get("entity"),
                name=run_name,
                config=self.config,
                tags=self.tracking_config.get("tags", ["ml-pipeline"]),
                mode="online"  # Force online mode - no fallback to offline
            )

            logger.info(f"W&B tracking initialized online: {run_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize W&B tracking: {e}")
            self.enabled = False
            return False

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if not self.enabled or not self.run:
            return

        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

            logger.debug(f"Logged metrics to W&B: {list(metrics.keys())}")

        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")

    def log_dataset_info(self, df: pd.DataFrame, dataset_name: str = "training") -> None:
        """
        Log dataset information to W&B.

        Args:
            df: Dataset DataFrame
            dataset_name: Name of the dataset
        """
        if not self.enabled or not self.run:
            return

        try:
            target_column = self.config.get("target_label")

            dataset_info = {
                f"{dataset_name}_samples": len(df),
                f"{dataset_name}_features": len(df.columns) - (1 if target_column in df.columns else 0),
                f"{dataset_name}_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            }

            # Log target distribution if available
            if target_column and target_column in df.columns:
                target_dist = df[target_column].value_counts().to_dict()
                dataset_info[f"{dataset_name}_target_distribution"] = target_dist

                # Calculate class imbalance ratio
                if len(target_dist) == 2:
                    values = list(target_dist.values())
                    imbalance_ratio = max(
                        values) / min(values) if min(values) > 0 else float('inf')
                    dataset_info[f"{dataset_name}_imbalance_ratio"] = imbalance_ratio

            self.log_metrics(dataset_info)
            logger.info(f"Logged {dataset_name} dataset info to W&B")

        except Exception as e:
            logger.error(f"Failed to log dataset info to W&B: {e}")

    def log_data_split_info(self, train_size: int, val_size: int, test_size: int) -> None:
        """
        Log data split information.

        Args:
            train_size: Size of training set
            val_size: Size of validation set
            test_size: Size of test set
        """
        if not self.enabled or not self.run:
            return

        try:
            total_size = train_size + val_size + test_size
            split_info = {
                "train_samples": train_size,
                "val_samples": val_size,
                "test_samples": test_size,
                "total_samples": total_size,
                "train_ratio": train_size / total_size,
                "val_ratio": val_size / total_size,
                "test_ratio": test_size / total_size,
            }

            self.log_metrics(split_info)
            logger.info("Logged data split info to W&B")

        except Exception as e:
            logger.error(f"Failed to log data split info to W&B: {e}")

    def log_model_performance(self, metrics: Dict[str, float], model_name: str, dataset: str = "test") -> None:
        """
        Log model performance metrics.

        Args:
            metrics: Performance metrics dictionary
            model_name: Name of the model
            dataset: Dataset name (train, val, test)
        """
        if not self.enabled or not self.run:
            return

        try:
            # Add model and dataset context to metrics
            contextualized_metrics = {
                f"{model_name}_{dataset}_{key}": value
                for key, value in metrics.items()
            }
            contextualized_metrics[f"current_model"] = model_name
            contextualized_metrics[f"current_dataset"] = dataset

            self.log_metrics(contextualized_metrics)
            logger.info(
                f"Logged {model_name} performance on {dataset} set to W&B")

        except Exception as e:
            logger.error(f"Failed to log model performance to W&B: {e}")

    def log_hyperparameter_results(self, best_params: Dict[str, Any], best_score: float, model_name: str) -> None:
        """
        Log hyperparameter tuning results.

        Args:
            best_params: Best parameters found
            best_score: Best score achieved
            model_name: Name of the model
        """
        if not self.enabled or not self.run:
            return

        try:
            hp_results = {
                f"{model_name}_best_score": best_score,
                f"{model_name}_best_params": best_params,
                "tuning_model": model_name,
            }

            # Log individual parameters with model prefix
            for param, value in best_params.items():
                clean_param = param.replace("classifier__", "")
                hp_results[f"{model_name}_{clean_param}"] = value

            self.log_metrics(hp_results)
            logger.info(
                f"Logged hyperparameter results for {model_name} to W&B")

        except Exception as e:
            logger.error(f"Failed to log hyperparameter results to W&B: {e}")

    def log_feature_importance(self, feature_names: List[str], importances: np.ndarray, model_name: str) -> None:
        """
        Log feature importance information.

        Args:
            feature_names: List of feature names
            importances: Array of feature importances
            model_name: Name of the model
        """
        if not self.enabled or not self.run:
            return

        try:
            # Create feature importance table
            importance_data = []
            for name, importance in zip(feature_names, importances):
                importance_data.append([name, float(importance)])

            # Sort by importance
            importance_data.sort(key=lambda x: x[1], reverse=True)

            # Take top 20 features
            top_features = importance_data[:20]

            importance_table = wandb.Table(
                data=top_features,
                columns=["Feature", "Importance"]
            )

            self.log_metrics({
                f"{model_name}_feature_importance": importance_table,
                f"{model_name}_top_feature": top_features[0][0] if top_features else "None",
                f"{model_name}_top_importance": top_features[0][1] if top_features else 0.0,
            })

            logger.info(f"Logged feature importance for {model_name} to W&B")

        except Exception as e:
            logger.error(f"Failed to log feature importance to W&B: {e}")

    def log_model_comparison(self, models_metrics: Dict[str, Dict[str, float]], best_model: str) -> None:
        """
        Log model comparison results.

        Args:
            models_metrics: Dictionary mapping model names to their metrics
            best_model: Name of the best performing model
        """
        if not self.enabled or not self.run:
            return

        try:
            # Create comparison table
            comparison_data = []
            for model_name, metrics in models_metrics.items():
                row = [
                    model_name,
                    metrics.get("test_roc_auc", 0.0),
                    metrics.get("test_accuracy", 0.0),
                    metrics.get("test_f1_score", 0.0),
                    metrics.get("test_precision", 0.0),
                    metrics.get("test_recall", 0.0),
                    "✅" if model_name == best_model else "❌"
                ]
                comparison_data.append(row)

            comparison_table = wandb.Table(
                data=comparison_data,
                columns=["Model", "ROC AUC", "Accuracy",
                         "F1 Score", "Precision", "Recall", "Selected"]
            )

            # Log comparison results
            self.log_metrics({
                "model_comparison_table": comparison_table,
                "best_model_selected": best_model,
                "models_compared": len(models_metrics),
                "best_model_roc_auc": models_metrics[best_model].get("test_roc_auc", 0.0),
            })

            logger.info(
                f"Logged model comparison with {len(models_metrics)} models to W&B")

        except Exception as e:
            logger.error(f"Failed to log model comparison to W&B: {e}")

    def log_training_summary(self, final_metrics: Dict[str, float], model_name: str) -> None:
        """
        Log final training summary.

        Args:
            final_metrics: Final performance metrics
            model_name: Name of the final model
        """
        if not self.enabled or not self.run:
            return

        try:
            summary = {
                "final_model": model_name,
                "training_completed": True,
                "timestamp": datetime.datetime.now().isoformat(),
                **final_metrics
            }

            self.log_metrics(summary)
            logger.info(f"Logged training summary for {model_name} to W&B")

        except Exception as e:
            logger.error(f"Failed to log training summary to W&B: {e}")

    def log_artifact(self, artifact_path: str, artifact_name: str, artifact_type: str = "model",
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an artifact to W&B.

        Args:
            artifact_path: Path to the artifact file
            artifact_name: Name of the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            metadata: Optional metadata dictionary
        """
        if not self.enabled or not self.run:
            return

        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata or {}
            )

            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)

            logger.info(f"Logged artifact {artifact_name} to W&B")

        except Exception as e:
            logger.error(f"Failed to log artifact to W&B: {e}")

    def log_config_changes(self, config_updates: Dict[str, Any]) -> None:
        """
        Log configuration changes during the run.

        Args:
            config_updates: Dictionary of configuration updates
        """
        if not self.enabled or not self.run:
            return

        try:
            # Update run config
            for key, value in config_updates.items():
                wandb.config.update({key: value})

            logger.info("Updated W&B run configuration")

        except Exception as e:
            logger.error(f"Failed to update W&B configuration: {e}")

    def add_tags(self, tags: List[str]) -> None:
        """
        Add tags to the current run.

        Args:
            tags: List of tags to add
        """
        if not self.enabled or not self.run:
            return

        try:
            for tag in tags:
                wandb.run.tags = wandb.run.tags + (tag,)

            logger.info(f"Added tags to W&B run: {tags}")

        except Exception as e:
            logger.error(f"Failed to add tags to W&B: {e}")

    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the W&B run.

        Args:
            exit_code: Exit code for the run (0 = success, 1 = failure)
        """
        if not self.enabled or not self.run:
            return

        try:
            wandb.finish(exit_code=exit_code)
            logger.info(f"W&B run finished with exit code {exit_code}")

        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")
        finally:
            self.run = None

    @property
    def is_active(self) -> bool:
        """Check if W&B tracking is active."""
        return self.enabled and self.run is not None

    @property
    def run_url(self) -> Optional[str]:
        """Get the URL of the current run."""
        if self.is_active:
            return wandb.run.get_url()
        return None


def create_wandb_tracker(config: Dict[str, Any]) -> WandBTracker:
    """
    Factory function to create a W&B tracker.

    Args:
        config: Configuration dictionary

    Returns:
        WandBTracker instance
    """
    return WandBTracker(config)


if __name__ == "__main__":
    # Example usage
    print("=== W&B Tracker Demo ===")

    # Sample configuration
    sample_config = {
        "project_name": "Test Project",
        "target_label": "target",
        "experiment_tracking": {
            "enabled": True,
            "project": "test-project",
            "tags": ["demo", "test"]
        }
    }

    # Create tracker
    tracker = create_wandb_tracker(sample_config)

    if tracker.enabled:
        print("W&B available - you can test initialization")
        print("Set WANDB_API_KEY environment variable to test")
    else:
        print("W&B not available or disabled")

    print(f"Tracker enabled: {tracker.enabled}")
    print(f"Tracker active: {tracker.is_active}")
