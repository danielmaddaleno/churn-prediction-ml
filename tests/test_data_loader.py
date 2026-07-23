"""Unit tests for data loading and schema validation."""

import pandas as pd
import pytest

from data_loader import load_data, validate_schema


def test_load_data_reads_valid_csv(tmp_path, sample_data):
    csv_path = tmp_path / "churn.csv"
    sample_data.to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert len(result) == len(sample_data)
    assert list(result.columns) == list(sample_data.columns)


def test_load_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


def test_load_data_missing_required_column_raises(tmp_path, sample_data):
    csv_path = tmp_path / "churn.csv"
    sample_data.drop(columns=["churn"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(csv_path)


def test_validate_schema_passes_on_clean_data(sample_data):
    assert validate_schema(sample_data) is True


def test_validate_schema_flags_negative_tenure(sample_data):
    sample_data.loc[0, "tenure"] = -5
    assert validate_schema(sample_data) is False


def test_validate_schema_flags_negative_monthly_charges(sample_data):
    sample_data.loc[0, "monthly_charges"] = -10.0
    assert validate_schema(sample_data) is False


def test_validate_schema_flags_high_null_columns(sample_data):
    df = pd.concat([sample_data] * 4, ignore_index=True)
    df.loc[: len(df) // 2, "monthly_charges"] = None
    assert validate_schema(df) is False


def test_validate_schema_flags_non_binary_churn(sample_data):
    # A stray label in the churn target (here a 2) must be caught here, while
    # it is still a data issue, rather than turning into a bogus multiclass
    # run once train.py casts the column with astype(int).
    sample_data.loc[0, "churn"] = 2
    assert validate_schema(sample_data) is False
