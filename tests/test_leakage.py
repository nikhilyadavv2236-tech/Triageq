"""
tests/test_leakage.py — Test that no forbidden columns appear in the feature matrix.
"""

import pytest
import pandas as pd
import numpy as np
from triageq.data.features import FORBIDDEN, assert_no_leakage, ALLOWED


def test_no_forbidden_in_empty_df():
    """An empty DataFrame with no columns should pass the leakage check."""
    X = pd.DataFrame()
    assert_no_leakage(X)


def test_no_forbidden_in_allowed_df():
    """A DataFrame with only allowed columns should pass."""
    X = pd.DataFrame({"age": [30, 40], "pulse": [80, 90]})
    assert_no_leakage(X)


def test_forbidden_column_raises():
    """A DataFrame with a forbidden column should raise ValueError."""
    X = pd.DataFrame({"age": [30], "disp": [1]})   # disp is forbidden
    with pytest.raises(ValueError, match="Leakage detected"):
        assert_no_leakage(X)


def test_forbidden_resi_o_raises():
    """resi_o (the reference label) must be forbidden."""
    X = pd.DataFrame({"age": [30], "resi_o": [2]})
    with pytest.raises(ValueError, match="Leakage detected"):
        assert_no_leakage(X)


def test_forbidden_waittime_raises():
    """waittime (post-triage outcome) must be forbidden."""
    X = pd.DataFrame({"age": [30], "waittime": [15.0]})
    with pytest.raises(ValueError, match="Leakage detected"):
        assert_no_leakage(X)


def test_forbidden_pattern_raises():
    """Columns ending in _icu or _disp should be caught by pattern check."""
    X = pd.DataFrame({"age": [30], "admit_icu": [1]})
    with pytest.raises(ValueError, match="Potential leakage"):
        assert_no_leakage(X)


def test_immedr_is_forbidden():
    """IMMEDR (nurse triage score) must be in the FORBIDDEN list for the main model."""
    assert "immedr" in FORBIDDEN
