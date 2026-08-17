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
        ValueError: If required columns are missing or the file has no rows.
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

    # A header-only CSV otherwise loads fine here and only blows up much
    # later, when train.py calls train_test_split with stratify on an empty
    # frame and raises a confusing sklearn error. Fail loudly at load time.
    if df.empty:
        raise ValueError(f"No data rows found in {filepath}; the file has column headers but no records")

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Check data quality: nulls, types, value ranges."""
    issues = []

    null_pct = df.isnull().mean()
    high_nulls = null_pct[null_pct > 0.3]
    if not high_nulls.empty:
        issues.append(f"High null columns: {high_nulls.to_dict()}")

    # tenure and monthly_charges must be numeric for the range checks below.
    # A raw CSV export with a stray "N/A" or "unknown" in one of these columns
    # reads back as an object column, and the .min() < 0 comparison then raises
    # a cryptic TypeError (comparing str to int) that crashes validation.
    # Report it as a data issue and skip the range check on that column.
    non_numeric = [c for c in ("tenure", "monthly_charges") if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"Non-numeric values in numeric columns: {non_numeric}")

    if "tenure" not in non_numeric and df["tenure"].min() < 0:
        issues.append("Negative tenure values found")

    if "monthly_charges" not in non_numeric and df["monthly_charges"].min() < 0:
        issues.append("Negative monthly_charges found")

    # The churn target must be binary. A stray label (a 2, or a string
    # "Yes"/"No" that never got encoded) slips past training: train.py does
    # y.astype(int), which crashes on strings and silently turns a 3-value
    # column into a bogus multiclass problem for a binary classifier. Catch
    # it here while it is still a data issue, not a modelling mystery.
    invalid_churn = set(df["churn"].dropna().unique()) - {0, 1}
    if invalid_churn:
        issues.append(f"Non-binary churn target values: {sorted(map(str, invalid_churn))}")

    # customer_id is the primary key: each customer should appear once. A
    # duplicated id double counts a customer and, worse, lets the same
    # customer end up in both the train and test split, leaking the label
    # and inflating the reported AUC. Flag it here before training splits.
    dup_ids = int(df["customer_id"].duplicated().sum())
    if dup_ids:
        issues.append(f"Duplicate customer_id values: {dup_ids}")

    if issues:
        for issue in issues:
            logger.warning("Data issue: %s", issue)
        return False

    logger.info("Schema validation passed")
    return True
