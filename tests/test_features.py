"""Unit tests for feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import ChurnFeatureTransformer

NUMERICAL_COLS = ["tenure", "monthly_charges", "total_charges", "num_support_tickets", "avg_monthly_usage"]
CATEGORICAL_COLS = ["contract_type", "payment_method"]
RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "churn_data.csv"


@pytest.fixture
def sample_data():
    """Tenure spans four buckets so a collapsed tenure_bucket is visible here."""
    return pd.DataFrame(
        {
            "tenure": [0, 12, 24, 6, 48, 3, 60],
            "monthly_charges": [50.0, 70.0, 30.0, 90.0, 45.0, 110.0, 25.0],
            "total_charges": [0.0, 840.0, 720.0, 540.0, 2160.0, 330.0, 1500.0],
            "num_support_tickets": [2, 0, 5, 1, 8, 3, 0],
            "avg_monthly_usage": [100, 200, 50, 300, 80, 420, 160],
            "contract_type": ["monthly", "one_year", "monthly", "two_year", "monthly", "monthly", "one_year"],
            "payment_method": [
                "credit_card",
                "bank_transfer",
                "credit_card",
                "auto_pay",
                "credit_card",
                "mailed_check",
                "auto_pay",
            ],
        }
    )


@pytest.fixture
def raw_data():
    """The committed synthetic dataset, 2000 rows with tenure from 0 to 72."""
    return pd.read_csv(RAW_CSV)[NUMERICAL_COLS + CATEGORICAL_COLS]


@pytest.fixture
def transformer():
    return ChurnFeatureTransformer(numerical_cols=NUMERICAL_COLS, categorical_cols=CATEGORICAL_COLS)


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


def test_lifetime_value_is_tenure_times_charges(transformer, raw_data):
    """lifetime_value must be the raw product, in currency units. Computing it
    after scaling gives a product of two z-scores instead."""
    result = transformer.fit_transform(raw_data)
    expected = raw_data["tenure"] * raw_data["monthly_charges"]
    pd.testing.assert_series_equal(result["lifetime_value"], expected, check_names=False)
    assert (result["lifetime_value"] >= 0).all()


def test_ticket_rate_is_tickets_per_month(transformer, raw_data):
    """The denominator is raw tenure + 1, which is always positive. On scaled
    tenure it crosses zero and flips the sign of half the column."""
    result = transformer.fit_transform(raw_data)
    expected = raw_data["num_support_tickets"] / (raw_data["tenure"] + 1)
    pd.testing.assert_series_equal(result["ticket_rate"], expected, check_names=False)
    assert (result["ticket_rate"] >= 0).all()


def test_tenure_bucket_spans_the_dataset(transformer, raw_data):
    """Bucketing scaled tenure with month-sized bin edges puts every row in the
    first bucket or outside the range, so check the buckets actually vary."""
    result = transformer.fit_transform(raw_data)
    assert result["tenure_bucket"].isna().sum() == 0
    assert result["tenure_bucket"].nunique() > 1
    assert set(result["tenure_bucket"].unique()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    # tenure 0 and 12 share the first bucket; 13 starts the second.
    assert result.loc[raw_data["tenure"] == 0, "tenure_bucket"].eq(0.0).all()
    assert result.loc[raw_data["tenure"] == 13, "tenure_bucket"].eq(1.0).all()


def test_no_nan_on_the_full_dataset(transformer, raw_data):
    result = transformer.fit_transform(raw_data)
    assert not result.isna().any().any(), result.isna().sum()


def test_new_customer_zero_tenure_has_no_nan(transformer, sample_data):
    """A brand new customer (tenure == 0) should not produce NaN in any
    engineered column. tenure_bucket in particular used to fall outside
    pd.cut's leftmost bin edge for tenure == 0 and silently turn into NaN."""
    result = transformer.fit_transform(sample_data)

    engineered_cols = ["lifetime_value", "tenure_bucket", "ticket_rate"]
    for col in engineered_cols:
        assert col in result.columns
        assert not result[col].isna().any(), f"{col} has NaN values"

    assert np.isfinite(result["ticket_rate"]).all()
    assert result.loc[0, "tenure_bucket"] == 0.0
    assert result.loc[0, "lifetime_value"] == 0.0
    assert result.loc[0, "ticket_rate"] == 2.0


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
    result = transformer.fit_transform(sample_data)
    assert not result.isna().any().any(), result.isna().sum()
