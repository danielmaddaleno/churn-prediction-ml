"""Tests for how predict.py picks the feature columns it sends to the model.

It used to take every column of the input CSV except customer_id and churn.
An export carrying one extra column then reached xgboost, which failed with
a message about DMatrix dtypes instead of about the file.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_scores_input_without_churn_column(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=200, seed=1)
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    model_name = "xgboost_churn_no_label"
    train_module.train(str(tiny_config), str(train_path), model_name)

    # New customers to score: the churn label is unknown, so the CSV omits it.
    scoring = df.drop(columns=["churn"])
    scoring_path = tmp_path / "scoring.csv"
    scoring.to_csv(scoring_path, index=False)

    output_path = tmp_path / "predictions.csv"
    results = predict_module.predict(
        str(scoring_path), str(output_path), model_uri=f"models:/{model_name}/latest", config_path=str(tiny_config)
    )

    assert len(results) == len(scoring)
    assert results["churn_probability"].between(0, 1).all()


def test_missing_feature_column_names_the_column(tmp_path, tiny_config):
    df = generate_churn_data(n_rows=20, seed=1).drop(columns=["avg_monthly_usage"])
    scoring_path = tmp_path / "scoring.csv"
    df.to_csv(scoring_path, index=False)

    with pytest.raises(ValueError, match="avg_monthly_usage"):
        predict_module.predict(
            str(scoring_path), str(tmp_path / "out.csv"), model_uri="models:/nope/latest", config_path=str(tiny_config)
        )


def test_threshold_sets_the_churn_label_from_the_probability(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=300, seed=1)
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    model_name = "xgboost_churn_threshold"
    train_module.train(str(tiny_config), str(train_path), model_name)

    scoring_path = tmp_path / "scoring.csv"
    df.drop(columns=["churn"]).to_csv(scoring_path, index=False)
    model_uri = f"models:/{model_name}/latest"

    low = predict_module.predict(
        str(scoring_path), str(tmp_path / "low.csv"), model_uri=model_uri, config_path=str(tiny_config), threshold=0.2
    )
    high = predict_module.predict(
        str(scoring_path), str(tmp_path / "high.csv"), model_uri=model_uri, config_path=str(tiny_config), threshold=0.8
    )

    # The label is exactly the probability compared against the threshold.
    assert (low["churn_prediction"] == (low["churn_probability"] >= 0.2).astype(int)).all()
    assert (high["churn_prediction"] == (high["churn_probability"] >= 0.8).astype(int)).all()
    # A lower cutoff can only flag more churners, never fewer, on the same rows.
    assert low["churn_prediction"].sum() >= high["churn_prediction"].sum()


def test_threshold_out_of_range_is_rejected(tmp_path, tiny_config):
    scoring_path = tmp_path / "scoring.csv"
    generate_churn_data(n_rows=10, seed=1).to_csv(scoring_path, index=False)

    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        predict_module.predict(
            str(scoring_path),
            str(tmp_path / "out.csv"),
            model_uri="models:/nope/latest",
            config_path=str(tiny_config),
            threshold=1.5,
        )


def test_backtest_metrics_scores_a_labelled_holdout(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=300, seed=1)
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    model_name = "xgboost_churn_backtest"
    train_module.train(str(tiny_config), str(train_path), model_name)

    # Score a CSV that still carries the churn column: that is a backtest.
    scoring_path = tmp_path / "scoring.csv"
    df.to_csv(scoring_path, index=False)

    results = predict_module.predict(
        str(scoring_path),
        str(tmp_path / "out.csv"),
        model_uri=f"models:/{model_name}/latest",
        config_path=str(tiny_config),
    )

    metrics = results.attrs["metrics"]
    assert metrics["n_scored"] == len(df)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "precision" in metrics["classification_report"]


def test_backtest_skipped_when_input_has_no_label(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=200, seed=1)
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    model_name = "xgboost_churn_no_backtest"
    train_module.train(str(tiny_config), str(train_path), model_name)

    scoring_path = tmp_path / "scoring.csv"
    df.drop(columns=["churn"]).to_csv(scoring_path, index=False)

    results = predict_module.predict(
        str(scoring_path),
        str(tmp_path / "out.csv"),
        model_uri=f"models:/{model_name}/latest",
        config_path=str(tiny_config),
    )

    assert "metrics" not in results.attrs


def test_backtest_metrics_rank_a_perfect_split():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    predictions = np.array([0, 0, 1, 1])

    metrics = predict_module.backtest_metrics(y_true, probabilities, predictions)

    assert metrics is not None
    assert metrics["n_scored"] == 4
    assert metrics["roc_auc"] == 1.0


def test_backtest_metrics_drop_unlabelled_rows_and_skip_a_single_class():
    # Two rows carry no label and are left out; the labelled rows are all one
    # class, so there is a report but ROC-AUC cannot rank it.
    y_true = pd.Series([1.0, 1.0, np.nan, np.nan])
    probabilities = np.array([0.7, 0.6, 0.4, 0.3])
    predictions = np.array([1, 1, 0, 0])

    metrics = predict_module.backtest_metrics(y_true, probabilities, predictions)

    assert metrics is not None
    assert metrics["n_scored"] == 2
    assert "roc_auc" not in metrics


def test_backtest_metrics_return_none_without_usable_labels():
    y_true = pd.Series([np.nan, np.nan, np.nan])
    probabilities = np.array([0.5, 0.5, 0.5])
    predictions = np.array([1, 1, 1])

    assert predict_module.backtest_metrics(y_true, probabilities, predictions) is None
