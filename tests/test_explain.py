"""Unit tests for SHAP explanation helpers."""

import numpy as np
import pandas as pd
import pytest

from explain import explain_cohort, explain_local


@pytest.fixture
def feature_frame():
    return pd.DataFrame(
        {
            "tenure": [1, 2, 3],
            "monthly_charges": [50.0, 70.0, 30.0],
        }
    )


@pytest.mark.parametrize("bad_idx", [3, 10, -4])
def test_explain_local_out_of_range_idx_raises(feature_frame, bad_idx):
    """An out-of-range row index should fail with a clear IndexError before
    any explainer work happens, not a cryptic iloc bounds error later. The
    check runs before the model is touched, so a stand-in model of None is
    fine here."""
    with pytest.raises(IndexError, match="out of range"):
        explain_local(model=None, X=feature_frame, idx=bad_idx)


@pytest.mark.parametrize("good_idx", [0, 2, -1, -3])
def test_explain_local_accepts_valid_positional_index(feature_frame, good_idx):
    """Valid positions, including negative ones the way iloc allows, must pass
    the bounds check. They fail later only because the stand-in model of None
    is not a real explainer, never with an IndexError from the guard."""
    with pytest.raises(Exception) as excinfo:
        explain_local(model=None, X=feature_frame, idx=good_idx)
    assert not isinstance(excinfo.value, IndexError)


def test_explain_cohort_empty_mask_raises(feature_frame):
    """A cohort mask that matches no customers should fail with a clear
    ValueError up front, rather than passing an empty frame to SHAP and
    returning an importance table that is silently all NaN. The check runs
    before the explainer is built, so a stand-in model of None is fine."""
    empty_mask = np.zeros(len(feature_frame), dtype=bool)
    with pytest.raises(ValueError, match="selected 0 rows"):
        explain_cohort(model=None, X=feature_frame, cohort_mask=empty_mask)


def test_explain_cohort_nonempty_mask_passes_guard(feature_frame):
    """A mask that selects at least one row must get past the empty-cohort
    guard. It fails later only because the stand-in model of None is not a
    real explainer, never with the ValueError from the guard."""
    mask = np.array([True, False, True])
    with pytest.raises(Exception) as excinfo:
        explain_cohort(model=None, X=feature_frame, cohort_mask=mask)
    assert not (isinstance(excinfo.value, ValueError) and "selected 0 rows" in str(excinfo.value))
