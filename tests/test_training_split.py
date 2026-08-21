"""Regression tests for how train.py splits the data.

Two things used to leak the test set into training: the feature transformer
was fit on the whole frame before the split, and early stopping watched the
test set as its eval_set, so the reported AUC was scored on the same rows
that chose the boosting round.
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pytest
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_sample_data import generate_churn_data  # noqa: E402

import train as train_module  # noqa: E402

NUMERICAL = ["tenure", "monthly_charges", "total_charges", "num_support_tickets", "avg_monthly_usage"]
CATEGORICAL = ["contract_type", "payment_method"]
TEST_SIZE = 0.25
RANDOM_STATE = 42


@pytest.fixture
def early_stopping_config(tmp_path):
    """A config small enough to train in seconds, with early stopping on."""
    config = {
        "model": {
            "name": "xgboost_churn_test",
            "type": "xgboost",
            "params": {
                "n_estimators": 30,
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "auc",
                "early_stopping_rounds": 5,
            },
        },
        "features": {"numerical": NUMERICAL, "categorical": CATEGORICAL, "target": "churn"},
        "training": {
            "test_size": TEST_SIZE,
            "validation_size": 0.2,
            "random_state": RANDOM_STATE,
            "cv_folds": 3,
            "optimize": False,
            "n_trials": 1,
        },
        "mlflow": {
            "experiment_name": "test_churn_split",
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _expected_split(df):
    """Reproduce train.py's split to get the rows it must keep for testing."""
    X = df[NUMERICAL + CATEGORICAL]
    y = df["churn"].astype(int)
    X_train_raw, X_test_raw, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train_raw, X_test_raw


def test_early_stopping_never_sees_the_test_rows(tmp_path, early_stopping_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=300, seed=3)
    data_path = tmp_path / "churn_data.csv"
    df.to_csv(data_path, index=False)

    fit_calls = []
    real_cls = train_module.XGBClassifier

    class SpyXGBClassifier(real_cls):
        def fit(self, X, y, **kwargs):
            fit_calls.append({"X": X, "eval_set": kwargs.get("eval_set")})
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(train_module, "XGBClassifier", SpyXGBClassifier)
    train_module.train(str(early_stopping_config), str(data_path), "xgboost_churn_split_test")

    assert len(fit_calls) == 1
    call = fit_calls[0]
    assert call["eval_set"] is not None, "early stopping was configured but no eval_set was passed"

    X_train_raw, X_test_raw = _expected_split(df)
    held_out = set(X_test_raw.index)

    fit_rows = set(call["X"].index)
    eval_rows = set(call["eval_set"][0][0].index)

    assert not fit_rows & held_out
    assert not eval_rows & held_out
    assert eval_rows.isdisjoint(fit_rows)
    assert fit_rows | eval_rows == set(X_train_raw.index)


def test_transformer_is_fit_on_training_rows_only(tmp_path, early_stopping_config, monkeypatch):
    """The scaler inside the registered pipeline must carry the training rows'
    statistics. Fitting before the split gives it the whole dataset's."""
    monkeypatch.chdir(tmp_path)

    df = generate_churn_data(n_rows=300, seed=3)
    data_path = tmp_path / "churn_data.csv"
    df.to_csv(data_path, index=False)

    model_name = "xgboost_churn_scaler_test"
    train_module.train(str(early_stopping_config), str(data_path), model_name)

    mlflow.set_tracking_uri(yaml.safe_load(early_stopping_config.read_text())["mlflow"]["tracking_uri"])
    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
    scaler = pipeline.named_steps["features"].scaler

    X_train_raw, _ = _expected_split(df)
    assert np.allclose(scaler.mean_, X_train_raw[NUMERICAL].mean().to_numpy())
    assert not np.allclose(scaler.mean_, df[NUMERICAL].mean().to_numpy())
