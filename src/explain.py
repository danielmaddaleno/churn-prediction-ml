"""SHAP-based model explanations."""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from data_loader import load_data
from predict import load_registered_model, select_feature_frame

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/model_config.yaml"
DEFAULT_MODEL_URI = "models:/xgboost_churn/latest"


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


def main(argv: list[str] | None = None) -> dict | None:
    """Run SHAP explanations for a registered model from the command line.

    Loads the pipeline train.py registers, reads the same raw feature columns
    predict.py uses, and writes a global importance plot. With --row it also
    returns one customer's local explanation.
    """
    parser = argparse.ArgumentParser(description="SHAP explanations for a registered churn model")
    parser.add_argument("--input", required=True, help="CSV of customers to explain")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--output", default="shap_summary.png", help="path for the global summary plot")
    parser.add_argument("--row", type=int, default=None, help="iloc position of a customer to explain locally")
    args = parser.parse_args(argv)

    df = load_data(args.input, require_target=False)
    X = select_feature_frame(df, args.config)
    model = load_registered_model(args.model_uri)

    explain_global(model, X, args.output)

    if args.row is None:
        return None

    explanation = explain_local(model, X, args.row)
    logger.info("Top churn drivers for row %d: %s", args.row, explanation["top_drivers"])
    return explanation


if __name__ == "__main__":
    main()
