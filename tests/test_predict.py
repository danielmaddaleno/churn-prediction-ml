"""Tests for how predict.py picks the feature columns it sends to the model.

It used to take every column of the input CSV except customer_id and churn.
An export carrying one extra column then reached xgboost, which failed with
a message about DMatrix dtypes instead of about the file.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_sample_data import generate_churn_data  # noqa: E402

import predict as predict_module  # noqa: E402
import train as train_module  # noqa: E402


def test_feature_columns_come_from_the_config(tiny_config):
    assert predict_module.feature_columns(str(tiny_config)) == [
        "tenure",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
        "avg_monthly_usage",
        "contract_type",
        "payment_method",
    ]


def test_extra_input_column_is_ignored(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=200, seed=1)
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    model_name = "xgboost_churn_extra_col"
    train_module.train(str(tiny_config), str(train_path), model_name)

    # Same rows plus a column the model was never trained on.
    df["signup_channel"] = "web"
    scoring_path = tmp_path / "scoring.csv"
    df.to_csv(scoring_path, index=False)

    output_path = tmp_path / "predictions.csv"
    results = predict_module.predict(
        str(scoring_path), str(output_path), model_uri=f"models:/{model_name}/latest", config_path=str(tiny_config)
    )

    assert len(results) == len(df)
    assert results["churn_probability"].between(0, 1).all()


def test_missing_feature_column_names_the_column(tmp_path, tiny_config):
    df = generate_churn_data(n_rows=20, seed=1).drop(columns=["avg_monthly_usage"])
    scoring_path = tmp_path / "scoring.csv"
    df.to_csv(scoring_path, index=False)

    with pytest.raises(ValueError, match="avg_monthly_usage"):
        predict_module.predict(
            str(scoring_path), str(tmp_path / "out.csv"), model_uri="models:/nope/latest", config_path=str(tiny_config)
        )
