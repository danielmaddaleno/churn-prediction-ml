"""Unit tests for SHAP explanation helpers.

These run against a real fitted Pipeline, the same shape train.py registers
with MLflow, because that is the only artifact the pipeline produces.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_sample_data import generate_churn_data  # noqa: E402

from explain import explain_cohort, explain_global, explain_local  # noqa: E402
from feature_engineering import ChurnFeatureTransformer  # noqa: E402

NUMERICAL = ["tenure", "monthly_charges", "total_charges", "num_support_tickets", "avg_monthly_usage"]
CATEGORICAL = ["contract_type", "payment_method"]
ENGINEERED = ["lifetime_value", "tenure_bucket", "ticket_rate"]


@pytest.fixture(scope="module")
def raw_frame():
    df = generate_churn_data(n_rows=150, seed=11)
    return df[NUMERICAL + CATEGORICAL], df["churn"].astype(int)


@pytest.fixture(scope="module")
def fitted_pipeline(raw_frame):
    """Transformer plus classifier in one Pipeline, as train.py registers it."""
    X, y = raw_frame
    pipeline = Pipeline(
        [
            ("features", ChurnFeatureTransformer(numerical_cols=NUMERICAL, categorical_cols=CATEGORICAL)),
            ("model", XGBClassifier(n_estimators=20, max_depth=3, eval_metric="auc", random_state=42)),
        ]
    )
    return pipeline.fit(X, y)


@pytest.fixture(scope="module")
def feature_frame(raw_frame):
    X, _ = raw_frame
    return X


def test_explain_local_runs_on_the_registered_pipeline(fitted_pipeline, feature_frame):
    """The registered artifact is a Pipeline, and TreeExplainer rejects those,
    so explain_local has to unwrap it before building the explainer."""
    result = explain_local(fitted_pipeline, feature_frame, idx=0)

    assert set(result) == {"index", "base_value", "feature_impacts", "top_drivers"}
    assert isinstance(result["base_value"], float)
    assert set(result["feature_impacts"]) == set(NUMERICAL + CATEGORICAL + ENGINEERED)
    assert len(result["top_drivers"]) == 5
    assert all(np.isfinite(v) for v in result["feature_impacts"].values())

    impacts = list(result["feature_impacts"].values())
    assert impacts == sorted(impacts, key=abs, reverse=True)


def test_explain_local_accepts_negative_index(fitted_pipeline, feature_frame):
    last = explain_local(fitted_pipeline, feature_frame, idx=-1)
    same = explain_local(fitted_pipeline, feature_frame, idx=len(feature_frame) - 1)
    assert last["feature_impacts"] == same["feature_impacts"]


def test_explain_local_works_on_a_bare_estimator(fitted_pipeline, feature_frame):
    """Callers who kept the classifier on its own still get explanations."""
    transformed = fitted_pipeline.named_steps["features"].transform(feature_frame)
    result = explain_local(fitted_pipeline.named_steps["model"], transformed, idx=0)
    assert set(result["feature_impacts"]) == set(NUMERICAL + CATEGORICAL + ENGINEERED)


@pytest.mark.parametrize("bad_idx", [150, 400, -151])
def test_explain_local_out_of_range_idx_raises(fitted_pipeline, feature_frame, bad_idx):
    """An out-of-range row index should fail with a clear IndexError before any
    explainer work happens, not a cryptic iloc bounds error later."""
    with pytest.raises(IndexError, match="out of range"):
        explain_local(fitted_pipeline, feature_frame, idx=bad_idx)


def test_explain_cohort_runs_on_the_registered_pipeline(fitted_pipeline, feature_frame):
    mask = (feature_frame["tenure"] < 12).to_numpy()
    assert mask.sum() > 0

    result = explain_cohort(fitted_pipeline, feature_frame, mask)

    assert list(result.columns) == ["mean_abs_shap"]
    assert set(result.index) == set(NUMERICAL + CATEGORICAL + ENGINEERED)
    assert (result["mean_abs_shap"] >= 0).all()
    assert result["mean_abs_shap"].notna().all()
    assert result["mean_abs_shap"].is_monotonic_decreasing


def test_explain_cohort_empty_mask_raises(fitted_pipeline, feature_frame):
    """A cohort mask that matches no customers should fail with a clear
    ValueError up front, rather than passing an empty frame to SHAP and
    returning an importance table that is silently all NaN."""
    empty_mask = np.zeros(len(feature_frame), dtype=bool)
    with pytest.raises(ValueError, match="selected 0 rows"):
        explain_cohort(fitted_pipeline, feature_frame, empty_mask)


def test_explain_global_writes_a_plot(fitted_pipeline, feature_frame, tmp_path):
    output_path = tmp_path / "shap_summary.png"
    explain_global(fitted_pipeline, feature_frame, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cohort_ranking_matches_a_manual_shap_pass(fitted_pipeline, feature_frame):
    """The unwrapping must feed SHAP the transformed frame, not the raw one."""
    import shap

    mask = (feature_frame["tenure"] > 40).to_numpy()
    transformed = fitted_pipeline.named_steps["features"].transform(feature_frame[mask])
    expected = shap.TreeExplainer(fitted_pipeline.named_steps["model"]).shap_values(transformed)
    expected_means = pd.DataFrame(np.abs(expected), columns=transformed.columns).mean()

    result = explain_cohort(fitted_pipeline, feature_frame, mask)
    pd.testing.assert_series_equal(
        result["mean_abs_shap"].sort_index(),
        expected_means.sort_index(),
        check_names=False,
    )
