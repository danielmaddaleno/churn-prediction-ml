"""train.py must counterweight the churn/no-churn imbalance.

The target runs about three non-churners per churner, and XGBoost weights both
classes equally unless told otherwise, so without a scale_pos_weight it leans on
the majority and barely flags the churners the model exists to catch. Training
now sets scale_pos_weight from the training split, and lets the config override
it.
"""

import sys
from pathlib import Path

import pandas as pd
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


def _make_config(tmp_path, extra_params=None):
    params = {
        "n_estimators": 30,
        "max_depth": 3,
        "learning_rate": 0.1,
        "eval_metric": "auc",
    }
    if extra_params:
        params.update(extra_params)
    config = {
        "model": {"name": "xgboost_churn_test", "type": "xgboost", "params": params},
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
            "experiment_name": "test_churn_weight",
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _training_labels(df):
    """The y_train slice train.py fits on, reproduced for the expected weight."""
    X = df[NUMERICAL + CATEGORICAL]
    y = df["churn"].astype(int)
    _, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return y_train


def _capture_scale_pos_weight(monkeypatch):
    """Record the scale_pos_weight each XGBClassifier is built with at fit time."""
    seen = []
    real_cls = train_module.XGBClassifier

    class SpyXGBClassifier(real_cls):
        def fit(self, X, y, **kwargs):
            seen.append(self.get_params().get("scale_pos_weight"))
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(train_module, "XGBClassifier", SpyXGBClassifier)
    return seen


def test_positive_class_weight_is_the_negative_positive_ratio():
    y = pd.Series([0, 0, 0, 1])
    assert train_module.positive_class_weight(y) == pytest.approx(3.0)


@pytest.mark.parametrize("y", [pd.Series([1, 1, 1]), pd.Series([0, 0]), pd.Series([], dtype=int)])
def test_positive_class_weight_is_a_no_op_without_both_classes(y):
    assert train_module.positive_class_weight(y) == 1.0


def test_train_weights_the_minority_class_from_the_training_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path)

    df = generate_churn_data(n_rows=400, seed=11)
    data_path = tmp_path / "churn_data.csv"
    df.to_csv(data_path, index=False)

    seen = _capture_scale_pos_weight(monkeypatch)
    train_module.train(str(config_path), str(data_path), "xgboost_churn_weight_test")

    expected = train_module.positive_class_weight(_training_labels(df))
    assert expected > 1.0, "the sample data is imbalanced, so the weight must exceed 1"
    assert seen == [pytest.approx(expected)]


def test_config_scale_pos_weight_overrides_the_computed_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _make_config(tmp_path, extra_params={"scale_pos_weight": 1.0})

    df = generate_churn_data(n_rows=400, seed=11)
    data_path = tmp_path / "churn_data.csv"
    df.to_csv(data_path, index=False)

    seen = _capture_scale_pos_weight(monkeypatch)
    train_module.train(str(config_path), str(data_path), "xgboost_churn_weight_override_test")

    assert seen == [1.0]
