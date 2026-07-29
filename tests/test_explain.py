"""Unit tests for SHAP explanation helpers."""

import pandas as pd
import pytest

from explain import explain_local


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
