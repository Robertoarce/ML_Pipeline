# Configurable ML Pipeline for Model Deployment

_A practical approach to creating reusable machine learning workflows that adapt to different use cases through configuration rather than code changes._

---

## The Problem

Most machine learning projects follow similar patterns: load data, preprocess features, train models, evaluate results, and persist the best performers. Yet teams often rebuild these workflows from scratch for each project, leading to inconsistent approaches and repeated work.

This system addresses that inefficiency by providing a configurable ML pipeline that handles the entire workflow through YAML configuration files. Change the dataset, target variable, or model types without touching code. It also allows you to generate synthetic data to test the pipeline. By default it is connected to Weights & Biases for experiment tracking.

## What This System Provides

### Core Components

**Configuration Management**: Single YAML files define everything—data sources, feature types, model selections, preprocessing steps, and evaluation metrics.

**Model Pipeline**: Automated preprocessing, feature engineering, hyperparameter tuning, and model comparison.

**Experiment Tracking**: Integration with Weights & Biases for monitoring experiments and results.

**Model Persistence (Optional)**: Automatic model saving with metadata for deployment readiness.

### Supported Algorithms

- **Tree-based**: RandomForest, GradientBoosting, XGBoost, LightGBM
- **Linear**: LogisticRegression with L1/L2 regularization
- **Non-linear**: Support Vector Machines with RBF/linear kernels

### Preprocessing Capabilities

- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoding**: One-hot encoding for categorical variables
- **Imputation**: Mean, median, most frequent, or constant strategies
- **Sampling**: SMOTE, ADASYN, undersampling for class imbalance
- **Feature Selection**: Statistical tests for dimensionality reduction

## How It Works

### 1. Configuration First

Everything starts with a YAML configuration. Here's a minimal example:

```yaml
project_name: "Customer Churn Prediction"
target_label: "will_churn"

features:
  categorical: ["subscription_type", "payment_method"]
  numeric: ["monthly_charges", "account_age_months"]
  boolean: ["has_premium_support"]

data:
  source_path: "data/customer_data.csv"
  test_set_size: 0.2
  validation_set_size: 0.1

models:
  enabled: ["RandomForest", "XGBoost", "LogisticRegression"]
  compare_models: true

hyperparameter_tuning:
  enabled: true
  method: "RandomizedSearchCV"
  n_iter: 30
```

### 2. Training Pipeline

Run the complete training pipeline:

```bash
python main.py --config customer_churn
```

The system automatically:

- Loads and validates data
- Splits into train/validation/test sets
- Builds preprocessing pipelines
- Trains multiple models with hyperparameter tuning
- Performs cross-validation
- Evaluates on test data
- Saves the best model with metadata

### 3. Results

```
2025-01-01 12:00:00 - INFO - Training completed successfully!
2025-01-01 12:00:00 - INFO - Best model: XGBoost
2025-01-01 12:00:00 - INFO - Final test score: 0.8756
2025-01-01 12:00:00 - INFO - Model saved to ./models/model_20250101_120000.pkl
```

## Design Decisions

### No Configuration Validation by Design

Traditional ML pipelines often include extensive configuration validation. This system deliberately omits it. Invalid configurations surface as runtime errors, which provides faster feedback than pre-flight checks and allows for more flexible usage patterns.

### Single-File Configurations

Each configuration file contains all necessary information—model definitions, preprocessing options, and training parameters. This eliminates the complexity of managing multiple interdependent config files.

### Direct YAML Loading

Rather than building a complex configuration management class, the system reads YAML files directly with simple functions. This reduces code complexity and makes the system easier to understand and modify.

## Example Configurations

### Generic Binary Classification

```yaml
project_name: "Generic ML Pipeline"
target_label: "target"

features:
  categorical: ["category_A", "category_B", "category_C"]
  numeric: ["feature_1", "feature_2", "feature_3"]
  boolean: ["flag_1", "flag_2"]

models:
  enabled: ["RandomForest"]
  compare_models: false

preprocessing:
  numeric_scaling:
    enabled: true
    method: "StandardScaler"
```

### Customer Churn with Advanced Features

```yaml
project_name: "Customer Churn Prediction Pipeline"
target_label: "will_churn"

features:
  categorical: ["subscription_type", "payment_method", "customer_segment"]
  numeric: ["monthly_charges", "total_charges", "account_age_months"]
  boolean: ["has_premium_support", "is_auto_pay_enabled"]

models:
  enabled: ["RandomForest", "XGBoost", "LogisticRegression"]
  compare_models: true

hyperparameter_tuning:
  enabled: true
  method: "RandomizedSearchCV"
  n_iter: 50

class_imbalance:
  enabled: true
  method: "SMOTE"

feature_engineering:
  feature_selection:
    enabled: true
    method: "SelectKBest"
    k: 15
```

## Usage Patterns

### Development Workflow

1. **Create configuration** for your use case
2. **Iterate on feature definitions** and model selection
3. **Train with your data**
4. **Deploy the trained model**

### Production Pipeline

```bash
# Train models
python main.py --config production

# Models automatically saved to ./models/ directory
```

### Experiment Tracking

The system integrates with Weights & Biases automatically. Set the `WANDB_API_KEY` environment variable to enable experiment logging.

## File Structure

```
06 ML_Pipeline/
├── main.py                      # Training pipeline entry point
├── config.py                    # Configuration loading
├── src/
│   └── pipeline_builder.py      # Model and preprocessing pipeline construction
├── configs/
│   ├── default_config.yaml      # Base configuration
│   └── customer_churn_config.yaml  # Domain-specific example
├── models/                      # Output directory for trained models
├── utils/
│   ├── evaluation_utils.py      # Model evaluation and cross-validation
│   ├── model_utils.py          # Model persistence and metadata
│   └── wandb_tracker.py        # Experiment tracking
└── requirements.txt             # Python dependencies
```

## Implementation Notes

The system uses scikit-learn as its foundation, with optional dependencies for XGBoost, LightGBM, and imbalanced-learn. Missing dependencies are handled gracefully—unavailable models are skipped with warnings.

Preprocessing pipelines use `ColumnTransformer` to handle different feature types appropriately. Boolean features are converted to integers, categorical features are one-hot encoded, and numeric features are scaled.

Model selection occurs automatically when multiple models are enabled. The system evaluates each model using cross-validation and selects the best performer based on ROC AUC for the final evaluation.

## What This Approach Enables

**Consistency**: Every project follows the same structure and evaluation methodology.

**Reproducibility**: Configurations capture all necessary information to recreate experiments.

**Flexibility**: New use cases require only configuration changes, not code modifications.

**Experimentation**: Rapid testing of different model combinations and preprocessing approaches.

**Deployment**: Models save with all metadata needed for production usage.

---

This system demonstrates that complex ML workflows can be made both powerful and accessible through thoughtful design choices. By prioritizing configuration over code changes, it becomes a practical tool for teams working on multiple machine learning projects.

The complete implementation is available as a working system that handles everything from synthetic data generation to model deployment, ready for adaptation to specific use cases.
