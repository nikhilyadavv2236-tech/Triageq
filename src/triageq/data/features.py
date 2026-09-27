"""
triageq.data.features — Triage-time feature matrix construction (Milestone 9).

Enforces the triage-time-only constraint: only variables that would be
known at the moment of triage are allowed. Any diagnostic/outcome/
resource variable is FORBIDDEN to prevent target leakage.

Leakage guard
-------------
assert_no_leakage(X) raises ValueError if any forbidden column, or any column
read by the reference-label builders, is present.

Allowed features (triage-time only)
-------------------------------------
- age, sex; arrival by ambulance; transfer from another facility
- arrival hour and weekday (cyclically encoded), month
- initial vitals: temperature, pulse, respiratory rate, systolic/diastolic BP, SpO2
  (NHAMCS has no GCS or on-oxygen item), with missing-value flags
- pain scale
- reason for visit: RFV1 at the 3-digit level (top groups + "other"),
  plus RFV2/RFV3 presence
- seen in last 72 h, nursing-home residence, injury flag, follow-up episode

Forbidden: diagnoses, tests, imaging, procedures, medications, disposition,
waiting time, length of visit, every rESI-O / NHAMCS-ESI source column, and
IMMEDR (nurse triage — only allowed in the optional "ML + nurse" variant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triageq.data.reference import LABEL_SOURCE_COLUMNS

# ── Column lists ──────────────────────────────────────────────────────────────

NUMERIC = ["age", "tempf", "pulse", "respr", "bpsys", "bpdias", "popct", "painscale"]
BINARY_RAW = {"sex": 2, "arrems": 1, "ambtransfer": 1, "seen72": 1, "episode": 2}

ALLOWED_BASE = (
    NUMERIC
    + [f"{c}_missing" for c in NUMERIC]
    + ["male", "ambulance", "transfer", "seen_72h", "followup_episode",
       "nursinghome", "injury",
       "arrival_hour_sin", "arrival_hour_cos", "arrival_dow_sin", "arrival_dow_cos",
       "month", "rfv2_present", "rfv3_present"]
)
RFV_PREFIX = "rfv1g_"
ALLOWED = ALLOWED_BASE          # plus rfv1g_* one-hot columns

FORBIDDEN = sorted(set([
    # Diagnoses
    "diag1", "diag2", "diag3", "diag4", "diag5",
    # Outcomes / disposition / timing
    "disp", "los", "waittime", "lov",
    # Reference label columns
    "resi_o", "nhamcs_esi", "n_resources", "resi_o_excl",
    # Nurse triage (excluded from main model; allowed in optional variant)
    "immedr",
    # Legacy names kept for the guard
    "labtest", "imaging", "meds", "ivmed", "proc", "ekgmon", "intub", "defib", "ivpress",
    "admiticu",
] + LABEL_SOURCE_COLUMNS))


def assert_no_leakage(X: pd.DataFrame, allow_nurse: bool = False) -> None:
    """Raise ValueError if any forbidden column is present in X."""
    forbidden = [c for c in FORBIDDEN if not (allow_nurse and c == "immedr")]
    bad = [c for c in forbidden if c in X.columns]
    if bad:
        raise ValueError(f"Leakage detected! Forbidden columns in feature matrix: {bad}")

    forbidden_patterns = ["_diag", "_lab", "_imaging", "_icu", "_disp"]
    pattern_hits = [c for c in X.columns if any(c.endswith(p) for p in forbidden_patterns)]
    if pattern_hits:
        raise ValueError(f"Potential leakage: columns matching forbidden patterns: {pattern_hits}")


def top_rfv_groups(train: pd.DataFrame, n: int = 100) -> list[int]:
    """Most frequent 3-digit RFV1 groups in the TRAINING years only."""
    return [int(v) for v in train["rfv1_3d"].value_counts().head(n).index]


def build_features(df: pd.DataFrame, rfv_groups: list[int],
                   include_nurse: bool = False) -> pd.DataFrame:
    """
    Construct the triage-time feature matrix from canonical NHAMCS columns.

    Numeric vitals keep NaN (LightGBM handles it natively) plus explicit
    *_missing flags; the logistic-regression pipeline imputes medians itself.
    """
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = df[c]
        X[f"{c}_missing"] = df[c].isna().astype(np.int8)
    # Implausible vitals → NaN (data-entry errors)
    X.loc[(X.tempf < 85) | (X.tempf > 110), "tempf"] = np.nan
    X.loc[X.popct > 100, "popct"] = np.nan
    X.loc[X.respr > 80, "respr"] = np.nan

    X["male"]             = df["sex"].eq(2).astype(np.int8)
    X["ambulance"]        = df["arrems"].eq(1).astype(np.int8)
    X["transfer"]         = df["ambtransfer"].eq(1).astype(np.int8)
    X["seen_72h"]         = df["seen72"].eq(1).astype(np.int8)
    X["followup_episode"] = df["episode"].eq(2).astype(np.int8)
    X["nursinghome"]      = df["residnce"].eq(2).astype(np.int8)
    X["injury"]           = df["injury"].eq(1).astype(np.int8)

    hour = np.floor(df["arrtime"] / 100.0) + (df["arrtime"] % 100) / 60.0
    X["arrival_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["arrival_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["arrival_dow_sin"]  = np.sin(2 * np.pi * (df["vdayr"] - 1) / 7)
    X["arrival_dow_cos"]  = np.cos(2 * np.pi * (df["vdayr"] - 1) / 7)
    X["month"]            = df["vmonth"]

    X["rfv2_present"] = df["rfv2"].notna().astype(np.int8) & df["rfv2"].gt(0).astype(np.int8)
    X["rfv3_present"] = df["rfv3"].notna().astype(np.int8) & df["rfv3"].gt(0).astype(np.int8)
    g = df["rfv1_3d"]
    for code in rfv_groups:
        X[f"{RFV_PREFIX}{code}"] = g.eq(code).astype(np.int8)
    X[f"{RFV_PREFIX}other"] = (~g.isin(rfv_groups)).astype(np.int8)

    if include_nurse:
        X["immedr"] = df["immedr"].where(df["immedr"].between(1, 5))

    assert_no_leakage(X, allow_nurse=include_nurse)
    return X
