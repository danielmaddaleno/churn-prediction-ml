"""The Optuna objective must score the preprocessing, not just the model.

The scaler and label encoders used to be fit once on the whole training set
before tuning, and cross_val_score then folded that already-transformed frame.
Every fold's validation rows had therefore passed through the fit, so the
roc_auc Optuna maximised was measured on leaked preprocessing. The objective
now cross-validates a pipeline over the raw columns, which refits the transform
on each fold's own training rows.
"""

import sys
from pathlib import Path

import optuna
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_sample_data import generate_churn_data  # noqa: E402

import train as train_module  # noqa: E402
from feature_engineering import ChurnFeatureTransformer  # noqa: E402

NUMERICAL = ["tenure", "monthly_charges", "total_charges", "num_support_tickets", "avg_monthly_usage"]
CATEGORICAL = ["contract_type", "payment_method"]

# Inside the ranges objective() samples from, so FixedTrial stays quiet.
FIXED_PARAMS = {
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
}


@pytest.fixture
def raw_training_data():
    df = generate_churn_data(n_rows=240, seed=7)
    X = df[NUMERICAL + CATEGORICAL]
    y = df["churn"].astype(int)
    X_train_raw, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X_train_raw, y_train


def test_objective_refits_the_transformer_inside_each_fold(raw_training_data, monkeypatch):
    X_train_raw, y_train = raw_training_data
    cv_folds = 3

    fit_sizes = []
    real_fit = ChurnFeatureTransformer.fit

    def spy_fit(self, X, y=None):
        fit_sizes.append(len(X))
        return real_fit(self, X, y)

    monkeypatch.setattr(ChurnFeatureTransformer, "fit", spy_fit)

    trial = optuna.trial.FixedTrial(FIXED_PARAMS)
    score = train_module.objective(trial, X_train_raw, y_train, NUMERICAL, CATEGORICAL, cv_folds)

    assert 0.0 <= score <= 1.0
    # One transformer fit per fold, each on that fold's training portion only.
    # The leaky version fit the transform once on all the training rows and
    # never refit it during cross-validation.
    assert len(fit_sizes) == cv_folds
    assert all(n < len(X_train_raw) for n in fit_sizes)
