"""Inference pipeline for churn predictions."""

import argparse
import logging

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from sklearn.metrics import classification_report, roc_auc_score

from data_loader import load_data

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/model_config.yaml"


def feature_columns(config_path: str) -> list[str]:
    """Feature columns the model was trained on, in the order train.py used."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cfg_feat = config["features"]
    return list(cfg_feat["numerical"]) + list(cfg_feat["categorical"])


def select_feature_frame(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """Pick the model's feature columns by name from the training config.

    Selecting by name keeps an export's extra columns out and names any
    missing column instead of letting it reach xgboost, which reports it as a
    DMatrix dtype error that says nothing about the input file.
    """
    feature_cols = feature_columns(config_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"the input CSV is missing feature columns the model needs: {missing}. "
            f"The expected columns are listed under 'features' in {config_path}."
        )
    return df[feature_cols]


def load_registered_model(model_uri: str):
    """Load a registered MLflow pipeline, with a readable error when the
    registry has nothing to serve yet."""
    try:
        return mlflow.sklearn.load_model(model_uri)
    except MlflowException as exc:
        raise RuntimeError(
            f"Could not load model '{model_uri}' from the MLflow registry. "
            "Make sure a model has been trained and registered first "
            "(run train.py), and that MLFLOW_TRACKING_URI points at the "
            "same store used for training."
        ) from exc


def backtest_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict | None:
    """Score predictions against a known churn label, for a labelled holdout.

    New customers have no label, so scoring them produces no metrics. But a
    holdout exported with its true churn column is a backtest, and the obvious
    question is how the registered model did on it. Report that instead of
    dropping the column.

    Returns None when the churn column has no 0/1 label to score against (all
    null, or values outside {0, 1}). With a single class present the report is
    still returned, but ROC-AUC is left out since it cannot rank one class.
    """
    labelled = y_true.notna().to_numpy()
    y = y_true[labelled]
    if y.empty or not set(y.unique()) <= {0, 1}:
        return None

    y = y.astype(int)
    metrics: dict = {
        "n_scored": int(labelled.sum()),
        "classification_report": classification_report(y, predictions[labelled], zero_division=0),
    }
    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities[labelled]))
    return metrics


def predict(
    input_path: str,
    output_path: str,
    model_uri: str = "models:/xgboost_churn/latest",
    config_path: str = DEFAULT_CONFIG,
    threshold: float = 0.5,
):
    """Generate churn predictions for new data.

    threshold is the churn probability at or above which a customer is
    labelled a churner. It defaults to 0.5 (what XGBClassifier.predict uses),
    but churn work usually wants it lower: missing a customer who leaves costs
    far more than offering retention to one who would have stayed, so a lower
    cutoff that trades precision for recall is the common choice.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    df = load_data(input_path, require_target=False)
    X = select_feature_frame(df, config_path)

    # The registered model is a Pipeline (feature transform + classifier),
    # not a bare classifier, so it expects the same raw columns train.py
    # started from. Applying ChurnFeatureTransformer here too would run the
    # transform twice and feed the model data it was never trained on.
    model = load_registered_model(model_uri)

    probabilities = model.predict_proba(X)[:, 1]
    # Label straight off the probability instead of model.predict, which is
    # pinned to a 0.5 cutoff. At threshold 0.5 the two agree; a lower
    # threshold is what lets this catch more churners.
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "churn_probability": probabilities,
            "churn_prediction": predictions,
            # include_lowest so a probability of exactly 0.0 lands in the
            # "low" bin. Without it pd.cut leaves the leftmost edge open and a
            # 0.0 churn probability (which tree models do produce) falls
            # through as a NaN risk_tier.
            "risk_tier": pd.cut(
                probabilities,
                bins=[0, 0.3, 0.6, 1.0],
                labels=["low", "medium", "high"],
                include_lowest=True,
            ),
        }
    )

    results.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", output_path, len(results))

    # Summary
    logger.info("Risk distribution:\n%s", results["risk_tier"].value_counts().to_string())

    # When the scoring input carries the true label (a labelled holdout rather
    # than new customers), report how the model did on it. The label is read
    # at the chosen threshold, so the report matches the churn_prediction column.
    if "churn" in df.columns:
        metrics = backtest_metrics(df["churn"], probabilities, predictions)
        if metrics is None:
            logger.info("Input has a churn column but no 0/1 labels to backtest against")
        else:
            if "roc_auc" in metrics:
                logger.info("Backtest ROC-AUC on %d labelled rows: %.4f", metrics["n_scored"], metrics["roc_auc"])
            logger.info("Backtest classification report:\n%s", metrics["classification_report"])
            results.attrs["metrics"] = metrics

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model-uri", default="models:/xgboost_churn/latest")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="churn probability at or above which a customer is labelled a churner",
    )
    args = parser.parse_args()
    predict(args.input, args.output, args.model_uri, args.config, args.threshold)
