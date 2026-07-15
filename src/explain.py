"""SHAP-based model explanations."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def explain_global(model, X: pd.DataFrame, output_path: str = "shap_summary.png"):
    """Generate global SHAP feature importance plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Global SHAP summary saved to %s", output_path)


def explain_local(model, X: pd.DataFrame, idx: int) -> dict:
    """Explain a single prediction with SHAP."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[idx]])

    feature_impacts = dict(zip(X.columns, shap_values[0]))
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
    """Aggregate SHAP explanations for a cohort of customers."""
    explainer = shap.TreeExplainer(model)
    X_cohort = X[cohort_mask]
    shap_values = explainer.shap_values(X_cohort)

    mean_abs_shap = pd.DataFrame(np.abs(shap_values), columns=X.columns).mean().sort_values(ascending=False)

    return mean_abs_shap.to_frame("mean_abs_shap")
