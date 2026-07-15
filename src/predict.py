"""Inference pipeline for churn predictions."""

import argparse
import logging

import mlflow
import pandas as pd

from data_loader import load_data

logger = logging.getLogger(__name__)


def predict(input_path: str, output_path: str, model_uri: str = "models:/xgboost_churn/latest"):
    """Generate churn predictions for new data."""
    df = load_data(input_path)

    # The registered model is a Pipeline (feature transform + classifier),
    # not a bare classifier, so it expects the same raw columns train.py
    # started from. Applying ChurnFeatureTransformer here too would run the
    # transform twice and feed the model data it was never trained on.
    model = mlflow.sklearn.load_model(model_uri)

    feature_cols = [c for c in df.columns if c not in ("customer_id", "churn")]
    X = df[feature_cols]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    results = pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            "churn_probability": probabilities,
            "churn_prediction": predictions,
            "risk_tier": pd.cut(
                probabilities,
                bins=[0, 0.3, 0.6, 1.0],
                labels=["low", "medium", "high"],
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
    args = parser.parse_args()
    predict(args.input, args.output, args.model_uri)
