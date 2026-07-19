"""Data loading and validation module."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load and validate the churn dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)

    required_cols = {"customer_id", "tenure", "monthly_charges", "churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Check data quality: nulls, types, value ranges."""
    issues = []

    null_pct = df.isnull().mean()
    high_nulls = null_pct[null_pct > 0.3]
    if not high_nulls.empty:
        issues.append(f"High null columns: {high_nulls.to_dict()}")

    if df["tenure"].min() < 0:
        issues.append("Negative tenure values found")

    if df["monthly_charges"].min() < 0:
        issues.append("Negative monthly_charges found")

    # The churn target must be binary. A stray label (a 2, or a string
    # "Yes"/"No" that never got encoded) slips past training: train.py does
    # y.astype(int), which crashes on strings and silently turns a 3-value
    # column into a bogus multiclass problem for a binary classifier. Catch
    # it here while it is still a data issue, not a modelling mystery.
    invalid_churn = set(df["churn"].dropna().unique()) - {0, 1}
    if invalid_churn:
        issues.append(f"Non-binary churn target values: {sorted(map(str, invalid_churn))}")

    if issues:
        for issue in issues:
            logger.warning("Data issue: %s", issue)
        return False

    logger.info("Schema validation passed")
    return True
