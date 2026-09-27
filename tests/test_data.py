"""
tests/test_data.py — Data-pipeline tests on the processed NHAMCS files.

Skipped automatically when data/processed/nhamcs_with_labels.parquet is absent.
  test_labels : rESI-O defined for ≥ 95% of kept visits; every level present
  test_split  : train/val/test years disjoint; tuning folds never share a hospital
  test_leakage_real : the real feature matrix passes the leakage guard and
                      contains no label-source column
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LABELS = Path("data/processed/nhamcs_with_labels.parquet")
pytestmark = pytest.mark.skipif(not LABELS.exists(), reason="NHAMCS data not processed")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(LABELS)


def test_labels(df):
    kept = df["resi_o_excl"].eq("")
    frac = df.loc[kept, "resi_o"].notna().mean()
    assert frac >= 0.95, f"rESI-O defined for only {frac:.1%} of kept visits"
    counts = df["resi_o"].value_counts()
    assert set(counts.index) == {1.0, 2.0, 3.0, 4.0, 5.0}
    assert df.loc[~kept, "resi_o"].isna().all(), "excluded visits must have no label"


def test_split(df):
    from triageq.ml.train import SPLIT_YEARS, tuning_folds
    tr, va, te = (set(SPLIT_YEARS[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    train = df[df["survey_year"].isin(tr) & df["resi_o"].notna()]
    for fit_idx, val_idx in tuning_folds(train, n_splits=5):
        assert not set(train["hosp_id"].iloc[fit_idx]) & set(train["hosp_id"].iloc[val_idx])


def test_leakage_real(df):
    from triageq.data.features import build_features, top_rfv_groups
    from triageq.data.reference import LABEL_SOURCE_COLUMNS
    X = build_features(df.head(2000), top_rfv_groups(df.head(2000), 20))
    assert not set(X.columns) & set(LABEL_SOURCE_COLUMNS)
    assert "immedr" not in X.columns
    assert np.isfinite(X.select_dtypes("number").fillna(0).to_numpy()).all()
