"""Tests for the explain.py command-line entry point.

explain.py shipped three SHAP helpers but no way to run them; train.py and
predict.py both had a CLI and a Makefile target and this module did not. The
entry point loads the registered pipeline and the same raw feature columns
predict.py scores, so these tests train a tiny model, register it, then drive
the CLI the way `make explain` does.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_sample_data import generate_churn_data  # noqa: E402

import explain as explain_module  # noqa: E402
import train as train_module  # noqa: E402

NUMERICAL = ["tenure", "monthly_charges", "total_charges", "num_support_tickets", "avg_monthly_usage"]
CATEGORICAL = ["contract_type", "payment_method"]
ENGINEERED = ["lifetime_value", "tenure_bucket", "ticket_rate"]


def _train_tiny_model(tmp_path, tiny_config, model_name):
    df = generate_churn_data(n_rows=200, seed=1)
    data_path = tmp_path / "churn_data.csv"
    df.to_csv(data_path, index=False)
    train_module.train(str(tiny_config), str(data_path), model_name)
    return data_path


def test_cli_writes_global_summary_without_a_row(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = _train_tiny_model(tmp_path, tiny_config, "xgboost_churn_cli_global")

    output_path = tmp_path / "summary.png"
    result = explain_module.main(
        [
            "--input",
            str(data_path),
            "--config",
            str(tiny_config),
            "--model-uri",
            "models:/xgboost_churn_cli_global/latest",
            "--output",
            str(output_path),
        ]
    )

    assert result is None
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_cli_returns_a_local_explanation_for_a_row(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = _train_tiny_model(tmp_path, tiny_config, "xgboost_churn_cli_local")

    output_path = tmp_path / "summary.png"
    result = explain_module.main(
        [
            "--input",
            str(data_path),
            "--config",
            str(tiny_config),
            "--model-uri",
            "models:/xgboost_churn_cli_local/latest",
            "--output",
            str(output_path),
            "--row",
            "0",
        ]
    )

    assert output_path.exists()
    assert set(result) == {"index", "base_value", "feature_impacts", "top_drivers"}
    assert set(result["feature_impacts"]) == set(NUMERICAL + CATEGORICAL + ENGINEERED)
    assert len(result["top_drivers"]) == 5


def test_cli_names_a_missing_feature_column(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _train_tiny_model(tmp_path, tiny_config, "xgboost_churn_cli_missing")

    short = generate_churn_data(n_rows=20, seed=2).drop(columns=["avg_monthly_usage"])
    short_path = tmp_path / "short.csv"
    short.to_csv(short_path, index=False)

    with pytest.raises(ValueError, match="avg_monthly_usage"):
        explain_module.main(
            [
                "--input",
                str(short_path),
                "--config",
                str(tiny_config),
                "--model-uri",
                "models:/xgboost_churn_cli_missing/latest",
                "--output",
                str(tmp_path / "summary.png"),
            ]
        )
