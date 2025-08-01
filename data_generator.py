#!/usr/bin/env python
"""
Standalone Data Generator for ML Pipeline

This script generates synthetic datasets for binary classification problems.
It can create generic datasets as well as domain-specific examples like customer churn prediction.
The data generation follows the configuration specified in YAML files.

Usage:
    python data_generator.py [--config CONFIG_NAME] [--samples N] [--help]

Examples:
    python data_generator.py                          # Generate with default config
    python data_generator.py --config customer_churn  # Generate customer churn data
    python data_generator.py --samples 10000          # Generate 10k samples
"""

import argparse
import sys
import os
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Optional
from config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_classification_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 8,
    n_redundant: int = 2,
    n_clusters_per_class: int = 1,
    class_sep: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a base binary classification dataset using scikit-learn.

    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_clusters_per_class: Number of clusters per class
        class_sep: Larger values spread out the classes
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with features and target
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state
    )

    # Create feature names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def add_categorical_features(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """
    Add categorical features to the dataset.

    Args:
        df: Input DataFrame
        categories: Dictionary mapping feature names to list of possible values

    Returns:
        DataFrame with added categorical features
    """
    np.random.seed(42)

    for feature_name, possible_values in categories.items():
        df[feature_name] = np.random.choice(
            possible_values,
            size=len(df),
            p=None  # Uniform distribution
        )

    return df


def add_boolean_features(df: pd.DataFrame, boolean_features: list) -> pd.DataFrame:
    """
    Add boolean features to the dataset.

    Args:
        df: Input DataFrame
        boolean_features: List of boolean feature names to add

    Returns:
        DataFrame with added boolean features
    """
    np.random.seed(42)

    for feature_name in boolean_features:
        # Generate boolean values with some bias towards False (70% False, 30% True)
        df[feature_name] = np.random.choice(
            [True, False], size=len(df), p=[0.3, 0.7])

    return df


def generate_time_series_features(df: pd.DataFrame, time_features: list) -> pd.DataFrame:
    """
    Add time-based features to the dataset.

    Args:
        df: Input DataFrame
        time_features: List of time-based feature names

    Returns:
        DataFrame with added time features
    """
    np.random.seed(42)

    for feature_name in time_features:
        if "days" in feature_name.lower():
            # Generate days (0-365)
            df[feature_name] = np.random.randint(0, 366, size=len(df))
        elif "months" in feature_name.lower():
            # Generate months (1-60 for up to 5 years)
            df[feature_name] = np.random.randint(1, 61, size=len(df))
        elif "minutes" in feature_name.lower():
            # Generate usage minutes (0-10000)
            df[feature_name] = np.random.randint(0, 10001, size=len(df))
        elif "gb" in feature_name.lower():
            # Generate GB usage (0-100)
            df[feature_name] = np.random.uniform(0, 100, size=len(df)).round(2)
        elif "charges" in feature_name.lower():
            # Generate monetary amounts
            if "monthly" in feature_name.lower():
                df[feature_name] = np.random.uniform(
                    10, 200, size=len(df)).round(2)
            else:  # total charges
                df[feature_name] = np.random.uniform(
                    100, 5000, size=len(df)).round(2)
        elif "tickets" in feature_name.lower():
            # Generate support ticket counts (0-20)
            df[feature_name] = np.random.randint(0, 21, size=len(df))
        elif "percentage" in feature_name.lower():
            # Generate percentage values (0-50%)
            df[feature_name] = np.random.uniform(0, 50, size=len(df)).round(1)
        else:
            # Default: generate random integers
            df[feature_name] = np.random.randint(0, 100, size=len(df))

    return df


def create_correlated_features(df: pd.DataFrame, target_column: str, correlation_strength: float = 0.3) -> pd.DataFrame:
    """
    Modify some features to be correlated with the target variable.

    Args:
        df: Input DataFrame
        target_column: Name of the target column
        correlation_strength: Strength of correlation to introduce

    Returns:
        DataFrame with correlated features
    """
    if target_column not in df.columns:
        return df

    target = df[target_column]

    # Make numeric features correlated with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]

    # Correlate first 3 numeric features
    for i, col in enumerate(numeric_cols[:3]):
        noise = np.random.normal(0, 1, len(df))
        if i % 2 == 0:
            # Positive correlation
            df[col] = df[col] + correlation_strength * \
                target * df[col].std() + 0.1 * noise
        else:
            # Negative correlation
            df[col] = df[col] - correlation_strength * \
                target * df[col].std() + 0.1 * noise

    return df


def generate_customer_churn_dataset(
    n_samples: int = 5000,
    output_path: str = "data/customer_churn_dataset.csv",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a customer churn prediction dataset with realistic features.

    Args:
        n_samples: Number of customer records to generate
        output_path: Path where to save the generated dataset
        random_state: Random seed for reproducibility

    Returns:
        Generated customer churn DataFrame
    """
    logger.info(
        f"Generating customer churn dataset with {n_samples} samples...")

    # Load configuration to get feature names
    try:
        config = load_config("customer_churn")
        feature_config = config["features"]
        target_name = config["target_label"]
    except:
        # Fallback if config loading fails
        feature_config = {
            "categorical": ["subscription_type", "payment_method", "customer_segment", "acquisition_channel"],
            "numeric": ["account_age_months", "monthly_charges", "total_charges", "support_tickets_count",
                        "last_login_days_ago", "usage_minutes_last_month", "data_usage_gb_last_month",
                        "contract_length_months", "days_since_last_payment", "discount_percentage"],
            "boolean": ["has_premium_support", "is_auto_pay_enabled", "has_multiple_services", "received_promotion_last_month"]
        }
        target_name = "will_churn"

    # Generate base dataset
    n_features = len(feature_config["categorical"]) + len(
        feature_config["numeric"]) + len(feature_config["boolean"])
    df = generate_classification_dataset(
        n_samples=n_samples,
        n_features=max(n_features, 10),
        n_informative=8,
        n_redundant=2,
        class_sep=0.8,  # Make it reasonably challenging
        random_state=random_state
    )

    # Remove the default features and target
    df = df.drop([col for col in df.columns if col.startswith(
        "feature_") or col == "target"], axis=1)

    # Add categorical features
    categorical_values = {
        "subscription_type": ["basic", "premium", "enterprise", "starter"],
        "payment_method": ["credit_card", "bank_transfer", "paypal", "debit_card"],
        "customer_segment": ["individual", "small_business", "enterprise", "non_profit"],
        "acquisition_channel": ["organic", "paid_ads", "referral", "direct", "social_media"]
    }

    for feature in feature_config["categorical"]:
        if feature in categorical_values:
            df = add_categorical_features(
                df, {feature: categorical_values[feature]})

    # Add numeric features with realistic distributions
    df = generate_time_series_features(df, feature_config["numeric"])

    # Add boolean features
    df = add_boolean_features(df, feature_config["boolean"])

    # Generate target variable (will_churn)
    np.random.seed(random_state)

    # Create churn probability based on some logical rules
    churn_prob = np.random.uniform(
        0.1, 0.4, n_samples)  # Base churn rate 10-40%

    # Adjust probabilities based on realistic factors
    if "monthly_charges" in df.columns:
        # Higher charges slightly increase churn probability
        high_charges = df["monthly_charges"] > df["monthly_charges"].quantile(
            0.8)
        churn_prob[high_charges] += 0.1

    if "support_tickets_count" in df.columns:
        # More support tickets increase churn probability
        many_tickets = df["support_tickets_count"] > 5
        churn_prob[many_tickets] += 0.2

    if "last_login_days_ago" in df.columns:
        # Long time since last login increases churn
        inactive = df["last_login_days_ago"] > 30
        churn_prob[inactive] += 0.3

    if "is_auto_pay_enabled" in df.columns:
        # Auto-pay customers less likely to churn
        autopay = df["is_auto_pay_enabled"]
        churn_prob[autopay] -= 0.1

    if "has_multiple_services" in df.columns:
        # Customers with multiple services less likely to churn
        multi_service = df["has_multiple_services"]
        churn_prob[multi_service] -= 0.15

    # Ensure probabilities are between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 1)

    # Generate binary target
    df[target_name] = np.random.binomial(1, churn_prob)

    # Add some correlation to make the problem realistic but not too easy
    df = create_correlated_features(df, target_name, correlation_strength=0.2)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Customer churn dataset saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Churn rate: {df[target_name].mean():.2%}")

    return df


def generate_generic_dataset(
    n_samples: int = 1000,
    n_features: int = 15,
    target_name: str = "target",
    output_path: str = "data/training_data.csv",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a generic binary classification dataset with mixed feature types.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        target_name: Name for the target column
        output_path: Path where to save the generated dataset
        random_state: Random seed for reproducibility

    Returns:
        Generated DataFrame
    """
    logger.info(
        f"Generating generic dataset with {n_samples} samples and {n_features} features...")

    # Generate base dataset
    df = generate_classification_dataset(
        n_samples=n_samples,
        # Reserve 5 features for categorical/boolean
        n_features=max(n_features - 5, 5),
        n_informative=max(n_features - 7, 3),
        n_redundant=2,
        class_sep=1.0,
        random_state=random_state
    )

    # Rename target column
    df = df.rename(columns={"target": target_name})

    # Add some categorical features
    categorical_features = {
        "category_A": ["type1", "type2", "type3", "type4"],
        "category_B": ["red", "blue", "green", "yellow"],
        "category_C": ["small", "medium", "large"]
    }
    df = add_categorical_features(df, categorical_features)

    # Add some boolean features
    boolean_features = ["flag_1", "flag_2"]
    df = add_boolean_features(df, boolean_features)

    # Make features somewhat correlated with target
    df = create_correlated_features(df, target_name, correlation_strength=0.25)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Generic dataset saved to {output_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(
        f"Target distribution: {df[target_name].value_counts().to_dict()}")

    return df


def generate_data_from_config(config_name: Optional[str] = None, n_samples: Optional[int] = None):
    """
    Generate synthetic data based on configuration.

    Args:
        config_name: Name of the configuration to use
        n_samples: Override number of samples to generate
    """
    try:
        # Load configuration
        logger.info(f"Loading configuration: {config_name or 'default'}")
        config = load_config(config_name)

        # Determine output path
        output_path = config["data"]["source_path"]

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")

        # Determine number of samples
        if n_samples is None:
            # Use default values based on config type
            if config_name == "customer_churn":
                n_samples = 5000
            else:
                n_samples = 1000

        logger.info(
            f"Generating {n_samples} samples for '{config.get('project_name', 'ML Pipeline')}'")
        logger.info(f"Target variable: {config['target_label']}")
        logger.info(f"Output path: {output_path}")

        # Generate data based on configuration type
        if config_name == "customer_churn":
            logger.info("Generating customer churn dataset...")
            df = generate_customer_churn_dataset(
                n_samples=n_samples,
                output_path=output_path
            )
        else:
            logger.info("Generating generic dataset...")
            df = generate_generic_dataset(
                n_samples=n_samples,
                target_name=config["target_label"],
                output_path=output_path
            )

        # Log summary
        logger.info(f"✅ Dataset generated successfully!")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"   Target distribution:")
        logger.info(
            f"   {df[config['target_label']].value_counts().to_dict()}")
        logger.info(f"   Saved to: {output_path}")

        return df

    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise


def main():
    """Main function for standalone data generation script."""
    parser = argparse.ArgumentParser(
        description="Standalone Data Generator for ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_generator.py                          # Generate with default config
  python data_generator.py --config customer_churn  # Generate customer churn data  
  python data_generator.py --samples 10000          # Generate 10k samples
  python data_generator.py --config customer_churn --samples 2000  # Custom config + samples
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration name to use (e.g., 'customer_churn')"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to generate (overrides config defaults)"
    )

    args = parser.parse_args()

    logger.info("🔧 ML Pipeline - Standalone Data Generator")
    logger.info("=" * 50)

    try:
        generate_data_from_config(
            config_name=args.config,
            n_samples=args.samples
        )

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
