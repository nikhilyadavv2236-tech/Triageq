"""
triageq.data.nhamcs_load — NHAMCS-ED public-use file ingestion (Milestone 9).

Reads the CDC Stata files per year with pyreadstat, applies the variable-name
mapping in configs/nhamcs_variables.yaml, recodes missing codes to NaN, and
saves parquet.

Source: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/NHAMCS/stata/
        ED{year}-stata.zip, unzipped into data/raw/nhamcs/.

Missing codes (NCHS convention): negative values (-9 blank, -8 unknown,
-7 not applicable, -6 refused) -> NaN. Variable-specific sentinels:
PULSE/BPDIAS 998 (Doppler/palpated) -> NaN. TEMPF is stored x10.

Usage:
    python -m triageq.data.nhamcs_load

Output: data/processed/nhamcs_{year}.parquet per year + nhamcs_pooled.parquet
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import pyreadstat
    _PYREADSTAT_AVAILABLE = True
except ImportError:
    _PYREADSTAT_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw/nhamcs")
PROCESSED_DIR = Path("data/processed")
MAPPING_PATH  = Path("configs/nhamcs_variables.yaml")
YEARS         = [2016, 2017, 2018, 2019, 2021, 2022]

# Variables whose values are real codes/counts: only negatives are missing
SENTINEL_998 = ["pulse", "bpdias"]


def load_variable_mapping(year: int, path: Path = MAPPING_PATH) -> dict[str, str]:
    """Return {raw_name: canonical_name} for one year."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = dict(cfg["default"])
    mapping.update((cfg.get("years") or {}).get(year, {}) or {})
    return {raw: canon for canon, raw in mapping.items() if raw is not None}


def _find_file(year: int, raw_dir: Path) -> Path | None:
    for p in raw_dir.glob("*.dta"):
        if str(year) in p.name:
            return p
    return None


def load_year(year: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    """
    Load one year of NHAMCS-ED data with canonical column names.

    Returns None (with a warning) if the year's file is not present.
    """
    if not _PYREADSTAT_AVAILABLE:
        raise ImportError("pyreadstat is required: pip install pyreadstat")

    file_path = _find_file(year, raw_dir)
    if file_path is None:
        warnings.warn(
            f"NHAMCS {year} file not found in {raw_dir}. Download ED{year}-stata.zip from "
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/NHAMCS/stata/"
        )
        return None

    mapping = load_variable_mapping(year)
    _, meta = pyreadstat.read_dta(str(file_path), metadataonly=True)
    present = [raw for raw in mapping if raw in meta.column_names]
    missing = sorted(set(mapping) - set(present))
    if missing:
        warnings.warn(f"NHAMCS {year}: mapped variables not in file: {missing}")
    extra = [c for c in ("RFV13D",) if c in meta.column_names]

    df, _ = pyreadstat.read_dta(str(file_path), usecols=present + extra)
    df = df.rename(columns=mapping)
    df = df.rename(columns={"RFV13D": "rfv1_3d"})

    if "arrtime" in df and not pd.api.types.is_numeric_dtype(df["arrtime"]):
        df["arrtime"] = pd.to_numeric(df["arrtime"], errors="coerce")   # stored as "HHMM" text
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float64")
            df.loc[df[col] < 0, col] = np.nan
    for col in SENTINEL_998:
        if col in df:
            df.loc[df[col] >= 998, col] = np.nan
    if "tempf" in df:
        df["tempf"] = df["tempf"] / 10.0
    if "rfv1_3d" not in df:
        df["rfv1_3d"] = np.floor(df["rfv1"] / 10.0)
    if "lov" not in df:
        df["lov"] = np.nan

    df["survey_year"] = year
    df["hosp_id"] = year * 1000 + df["hospcode"]       # hospital codes restart each year
    return df


def load_all_years(
    raw_dir: Path = RAW_DIR,
    years: list[int] = YEARS,
) -> pd.DataFrame:
    """Load and pool all available NHAMCS years."""
    frames = [df for y in years if (df := load_year(y, raw_dir)) is not None]
    if not frames:
        raise FileNotFoundError(
            f"No NHAMCS files found in {raw_dir}. "
            "Download data files from CDC AHCD before running the data pipeline."
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """CLI entry point: load all years, save parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in YEARS:
        df = load_year(year)
        if df is not None:
            df.to_parquet(PROCESSED_DIR / f"nhamcs_{year}.parquet", index=False)
            print(f"Saved {len(df):,} rows → {PROCESSED_DIR}/nhamcs_{year}.parquet")
            frames.append(df)
    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        pooled.to_parquet(PROCESSED_DIR / "nhamcs_pooled.parquet", index=False)
        print(f"Pooled {len(pooled):,} rows → {PROCESSED_DIR}/nhamcs_pooled.parquet")


if __name__ == "__main__":
    main()
