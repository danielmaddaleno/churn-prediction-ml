"""Shared pytest fixtures."""

import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_data():
    """A minimal churn dataframe with a valid schema (data_loader tests)."""
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "tenure": [0, 12, 24],
            "monthly_charges": [50.0, 70.0, 30.0],
            "churn": [0, 1, 0],
        }
    )


@pytest.fixture
def tiny_config(tmp_path):
    """A training config sized for a fast test run, not real training."""
    config = {
        "model": {
            "name": "xgboost_churn_test",
            "type": "xgboost",
            "params": {
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
            },
        },
        "features": {
            "numerical": [
                "tenure",
                "monthly_charges",
                "total_charges",
                "num_support_tickets",
                "avg_monthly_usage",
            ],
            "categorical": ["contract_type", "payment_method"],
            "target": "churn",
        },
        "training": {
            "test_size": 0.25,
            "random_state": 42,
            "cv_folds": 3,
            "optimize": False,
            "n_trials": 1,
        },
        "mlflow": {
            "experiment_name": "test_churn",
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path
