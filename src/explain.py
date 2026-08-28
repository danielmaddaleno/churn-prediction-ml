"""SHAP-based model explanations."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _tree_step_and_features(model, X: pd.DataFrame) -> tuple:
    """Reduce a fitted Pipeline to its tree step and the frame that step sees.

    train.py registers the feature transformer and the classifier together, so
    anything loaded from the MLflow registry arrives here as a Pipeline, and
    shap.TreeExplainer only accepts the tree estimator itself. A bare
    estimator is passed through with its features untouched.
    """
    if not isinstance(model, Pipeline):
        return model, X

    estimator = model.steps[-1][1]

    # Call the fitted transformers directly rather than slicing the pipeline:
    # ChurnFeatureTransformer keeps its state in self.scaler / self.label_encoders,
    # and check_is_fitted only recognises attributes named with a trailing
    # underscore, so a sliced pipeline reports itself as unfitted.
    X_features = X
    for _, step in model.steps[:-1]:
        X_features = step.transform(X_features)

    if not isinstance(X_features, pd.DataFrame):
        X_features = pd.DataFrame(X_features, columns=getattr(estimator, "feature_names_in_", None), index=X.index)
    return estimator, X_features


def explain_global(model, X: pd.DataFrame, output_path: str = "shap_summary.png"):
    """Generate global SHAP feature importance plot.

    Args:
        model: A fitted tree model, or the Pipeline train.py registers.
        X: The frame the model scores. Raw columns when model is a Pipeline.
    """
    estimator, X_features = _tree_step_and_features(model, X)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_features)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_features, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Global SHAP summary saved to %s", output_path)


def explain_local(model, X: pd.DataFrame, idx: int) -> dict:
    """Explain a single prediction with SHAP.

    Args:
        model: A fitted tree model, or the Pipeline train.py registers.
        X: The frame the model scores. Raw columns when model is a Pipeline.
        idx: Positional row index to explain (negative indexing allowed,
            same as ``DataFrame.iloc``).

    Raises:
        IndexError: If ``idx`` is outside the rows of ``X``.
    """
    # Validate the row index up front so an out-of-range value fails with a
    # clear message here instead of a bare "positional indexers are
    # out-of-bounds" from iloc after building the (expensive) explainer.
    n_rows = len(X)
    if not -n_rows <= idx < n_rows:
        raise IndexError(f"row index {idx} is out of range for {n_rows} rows")

    estimator, X_features = _tree_step_and_features(model, X.iloc[[idx]])
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_features)

    feature_impacts = dict(zip(X_features.columns, shap_values[0]))
    sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    top_drivers = list(sorted_impacts.items())[:5]
    logger.info("Top churn drivers for row %d: %s", idx, top_drivers)

    return {
        "index": idx,
        "base_value": float(explainer.expected_value),
        "feature_impacts": sorted_impacts,
        "top_drivers": {k: round(v, 4) for k, v in top_drivers},
    }


def explain_cohort(model, X: pd.DataFrame, cohort_mask: np.ndarray) -> pd.DataFrame:
    """Aggregate SHAP explanations for a cohort of customers.

    Raises:
        ValueError: If ``cohort_mask`` selects no rows.
    """
    # A cohort filter that matches nobody (e.g. "tenure > 100" on a dataset
    # capped at 72) otherwise slips through as an empty frame. SHAP then
    # returns a zero-row result whose column mean is all NaN, so the caller
    # gets a plausible-looking importance table full of NaN instead of an
    # error. Fail loudly up front, before building the explainer.
    X_cohort = X[cohort_mask]
    if len(X_cohort) == 0:
        raise ValueError("cohort_mask selected 0 rows; nothing to explain")

    estimator, X_features = _tree_step_and_features(model, X_cohort)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_features)

    mean_abs_shap = pd.DataFrame(np.abs(shap_values), columns=X_features.columns).mean().sort_values(ascending=False)

    return mean_abs_shap.to_frame("mean_abs_shap")
