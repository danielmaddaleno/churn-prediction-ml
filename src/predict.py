"""Inference pipeline for churn predictions."""

import argparse
import logging

import mlflow
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException

from data_loader import load_data

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/model_config.yaml"


def feature_columns(config_path: str) -> list[str]:
    """Feature columns the model was trained on, in the order train.py used."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cfg_feat = config["features"]
    return list(cfg_feat["numerical"]) + list(cfg_feat["categorical"])


def predict(
    input_path: str,
    output_path: str,
    model_uri: str = "models:/xgboost_churn/latest",
    config_path: str = DEFAULT_CONFIG,
):
    """Generate churn predictions for new data."""
    df = load_data(input_path)

    # Pick the feature columns by name from the training config instead of
    # taking whatever the CSV happens to carry. An export with an extra
    # column used to reach the model and fail inside xgboost, with a message
    # about DMatrix dtypes rather than about the input file.
    feature_cols = feature_columns(config_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{input_path} is missing feature columns the model needs: {missing}. "
            f"The expected columns are listed under 'features' in {config_path}."
        )
    X = df[feature_cols]

    # The registered model is a Pipeline (feature transform + classifier),
    # not a bare classifier, so it expects the same raw columns train.py
    # started from. Applying ChurnFeatureTransformer here too would run the
    # transform twice and feed the model data it was never trained on.
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except MlflowException as exc:
        raise RuntimeError(
            f"Could not load model '{model_uri}' from the MLflow registry. "
            "Make sure a model has been trained and registered first "
            "(run train.py), and that MLFLOW_TRACKING_URI points at the "
            "same store used for training."
        ) from exc

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

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
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model-uri", default="models:/xgboost_churn/latest")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    predict(args.input, args.output, args.model_uri, args.config)
