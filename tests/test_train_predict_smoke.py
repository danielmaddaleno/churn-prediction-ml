"""End-to-end smoke test: train a model on tiny synthetic data, register it
with MLflow, then load it back through predict.py and score the same data.

This is also the regression test for the train/serve skew bug: predict.py
used to feed raw columns to a model trained on ChurnFeatureTransformer
output. If someone reintroduces that bug, this test fails because the
model would receive un-encoded strings (e.g. "monthly") where it expects
numbers and raise, instead of silently producing wrong predictions.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_sample_data import generate_churn_data  # noqa: E402

import predict as predict_module  # noqa: E402
import train as train_module  # noqa: E402


def test_train_then_predict_smoke(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data_path = tmp_path / "churn_data.csv"
    df = generate_churn_data(n_rows=200, seed=1)
    df.to_csv(data_path, index=False)

    model_name = "xgboost_churn_smoke"
    train_module.train(str(tiny_config), str(data_path), model_name)

    output_path = tmp_path / "predictions.csv"
    results = predict_module.predict(
        str(data_path), str(output_path), model_uri=f"models:/{model_name}/latest", config_path=str(tiny_config)
    )

    assert output_path.exists()
    assert len(results) == len(df)
    assert set(results.columns) == {
        "customer_id",
        "churn_probability",
        "churn_prediction",
        "risk_tier",
    }
    assert results["churn_probability"].between(0, 1).all()
    assert results["risk_tier"].isin(["low", "medium", "high"]).all()


def test_shipped_config_fills_the_high_risk_tier(tmp_path, monkeypatch):
    """Early stopping watches the metric named in the config. AUC only ranks, so
    the round it picks can leave every probability under 0.55, which puts the
    whole dataset in the low and medium bins predict.py reports."""
    repo = Path(__file__).resolve().parent.parent
    data_path = repo / "data" / "raw" / "churn_data.csv"

    config = yaml.safe_load((repo / "configs" / "model_config.yaml").read_text())
    config["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config["mlflow"]["experiment_name"] = "test_shipped_config"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    monkeypatch.chdir(tmp_path)
    model_name = "xgboost_churn_shipped"
    train_module.train(str(config_path), str(data_path), model_name)

    results = predict_module.predict(
        str(data_path),
        str(tmp_path / "predictions.csv"),
        model_uri=f"models:/{model_name}/latest",
        config_path=str(config_path),
    )
    assert (results["risk_tier"] == "high").any()
