"""Shared pytest fixtures."""

import pandas as pd
import pytest


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
