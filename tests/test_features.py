"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import ChurnFeatureTransformer


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "tenure": [12, 24, 6, 48, 3],
            "monthly_charges": [50.0, 70.0, 30.0, 90.0, 45.0],
            "total_charges": [600.0, 1680.0, 180.0, 4320.0, 135.0],
            "num_support_tickets": [2, 0, 5, 1, 8],
            "avg_monthly_usage": [100, 200, 50, 300, 80],
            "contract_type": ["monthly", "one_year", "monthly", "two_year", "monthly"],
            "payment_method": ["credit_card", "bank_transfer", "credit_card", "auto_pay", "credit_card"],
        }
    )


@pytest.fixture
def transformer():
    return ChurnFeatureTransformer(
        numerical_cols=["tenure", "monthly_charges", "total_charges", "num_support_tickets", "avg_monthly_usage"],
        categorical_cols=["contract_type", "payment_method"],
    )


def test_fit_transform_shape(transformer, sample_data):
    result = transformer.fit_transform(sample_data)
    assert len(result) == len(sample_data)
    assert len(result.columns) > len(sample_data.columns)


def test_interaction_features_created(transformer, sample_data):
    result = transformer.fit_transform(sample_data)
    assert "lifetime_value" in result.columns
    assert "tenure_bucket" in result.columns
    assert "ticket_rate" in result.columns


def test_numerical_scaling(transformer, sample_data):
    result = transformer.fit_transform(sample_data)
    assert abs(result["tenure"].mean()) < 1e-10
    assert abs(result["monthly_charges"].std() - 1.0) < 0.3


def test_new_customer_zero_tenure_has_no_nan(transformer, sample_data):
    """A brand new customer (tenure == 0) should not produce NaN in any
    engineered column. tenure_bucket in particular used to fall outside
    pd.cut's leftmost bin edge for tenure == 0 and silently turn into NaN."""
    sample_data.loc[0, "tenure"] = 0
    result = transformer.fit_transform(sample_data)

    engineered_cols = ["lifetime_value", "tenure_bucket", "ticket_rate"]
    for col in engineered_cols:
        assert col in result.columns
        assert not result[col].isna().any(), f"{col} has NaN values"

    assert np.isfinite(result["ticket_rate"]).all()
    assert result.loc[0, "tenure_bucket"] == 0.0


def test_unseen_category_maps_to_sentinel(transformer, sample_data):
    """A category value not present at fit time (e.g. a brand new contract
    type at inference) should map to -1 instead of raising and dropping the
    whole batch. Known categories keep their original encoding."""
    transformer.fit(sample_data)

    new_data = sample_data.copy()
    new_data.loc[0, "contract_type"] = "month_to_month_promo"  # unseen at fit
    result = transformer.transform(new_data)

    assert result.loc[0, "contract_type"] == -1
    # Rows with known categories are unaffected and stay non-negative.
    assert (result.loc[1:, "contract_type"] >= 0).all()


def test_no_nan_across_all_engineered_columns(transformer, sample_data):
    """Guard against any future interaction feature reintroducing NaN,
    across every tenure value in the fixture (0, 3, 6, 12, ...)."""
    sample_data.loc[0, "tenure"] = 0
    result = transformer.fit_transform(sample_data)
    assert not result.isna().any().any(), result.isna().sum()
