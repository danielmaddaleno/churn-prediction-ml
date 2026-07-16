"""Feature engineering transformations."""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for churn prediction features."""

    def __init__(self, numerical_cols: list[str], categorical_cols: list[str]):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler.fit(X[self.numerical_cols])
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Scale numerical features
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])

        # Encode categorical features. At inference time a customer can carry a
        # category value that was never seen during fit (a new contract type,
        # a new payment method). LabelEncoder.transform raises on those and
        # drops the whole batch, so map any unseen value to -1 and warn instead.
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            values = df[col].astype(str)
            mapping = {label: idx for idx, label in enumerate(le.classes_)}
            unseen = sorted(set(values) - set(mapping))
            if unseen:
                logger.warning("Unseen categories in %s mapped to -1: %s", col, unseen)
            df[col] = values.map(mapping).fillna(-1).astype(int)

        # Interaction features
        df = self._add_interaction_features(df)

        logger.info("Transformed %d features -> %d features", len(X.columns), len(df.columns))
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features that capture business logic."""
        if "tenure" in df.columns and "monthly_charges" in df.columns:
            df["lifetime_value"] = df["tenure"] * df["monthly_charges"]

        if "tenure" in df.columns:
            # bins start at -1 (not 0) so a brand new customer with tenure == 0
            # still lands in the first bucket instead of falling outside the
            # range and becoming NaN (pd.cut bins are left-open by default).
            df["tenure_bucket"] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 48, 72, np.inf],
                labels=[0, 1, 2, 3, 4],
            ).astype(float)

        if "num_support_tickets" in df.columns and "tenure" in df.columns:
            df["ticket_rate"] = df["num_support_tickets"] / (df["tenure"] + 1)

        return df
