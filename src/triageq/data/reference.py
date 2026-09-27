"""
triageq.data.reference — Reference acuity label construction (Milestone 9).

rESI-O (primary, outcome/resource-based; frozen before any model training)
---------------------------------------------------------------------------
Exclusions (label = NaN, reason recorded in `resi_o_excl`):
  DOA (arrived dead); LWBS, LBTC, left AMA (care incomplete, resources unobserved).
Rules, applied in order (first match wins):
  L1  died in ED (DIEDED) or a life-saving intervention: CPR, endotracheal
      intubation (ENDOINT), BiPAP/CPAP (BPAP)
  L2  admitted to critical care (ADMIT=1), operating room (ADMIT=3) or cardiac
      catheterisation lab (ADMIT=5), or died in hospital after admission (HDSTAT=2)
  L3  admitted to this hospital (ADMITHOS), observation then hospitalised (OBSHOS),
      or transferred to another / psychiatric hospital (TRANOTH, TRANPSYC);
      or ≥ 2 ESI resources
  L4  exactly 1 ESI resource
  L5  no ESI resources
  NaN diagnostic AND procedure items both blank (resources unknown) and not L1–L3
      by disposition.

ESI resources (ESI v4 handbook categories, each counted once):
  labs      any blood/urine lab: CBC, BMP, CMP, BUNCREAT, ELECTROL, GLUCOSE, LFT,
            CARDENZ, BNP, ABG, BAC, DDIMER, LACTATE, PTTINR, OTHERBLD, BLOODCX,
            URINECX, URINE, TOXSCREN
  ecg       EKG
  xray      XRAY
  adv_img   CT, MRI, ultrasound or other imaging
  iv_fluids IVFLUIDS
  neb       NEBUTHER (nebulised medication)
  consult   consulting physician seen (CONSULT)
  procedure simple (bladder catheter, suturing, I&D) = 1; complex (LP, central line) = 2
Not counted: point-of-care tests, oral meds, splints, skin adhesive (ESI rules).
Limitation: NHAMCS does not record medication route, so IV/IM medications
(an ESI resource) cannot be counted — rESI-O undercounts resources for patients
whose only resource was a parenteral drug.

NHAMCS-ESI (secondary, sensitivity label)
------------------------------------------
An ESI-style stepwise algorithm on NHAMCS fields in the spirit of the NHAMCS-derived
ESI (West J Emerg Med 2020;21(5)); it is NOT an exact replication of that paper:
  Step 1 (ESI 1): life-saving intervention (CPR, ENDOINT, BPAP) or died in ED
  Step 2 (ESI 2): high risk — severe pain (≥ 7) with ≥ 2 resources, SpO2 < 90%,
                  or admission to critical care / OR / cath lab
  Step 3: resources 0 → ESI 5, 1 → ESI 4, ≥ 2 → step 4
  Step 4: danger-zone vitals (age-banded HR, RR > 20, SpO2 < 92%) → ESI 2, else ESI 3
It uses triage vitals, so it overlaps with model features and is secondary.

Usage:
    python -m triageq.data.reference
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LAB_VARS = ["cbc", "bmp", "cmp", "buncreat", "electrol", "glucose", "lft", "cardenz",
            "bnp", "abg", "bac", "ddimer", "lactate", "pttinr", "otherbld", "bloodcx",
            "urinecx", "urine", "toxscren"]
ADV_IMG_VARS = ["catscan", "mri", "ultrasnd", "othimage"]
SIMPLE_PROC = ["bladcath", "suture", "incdrain"]
COMPLEX_PROC = ["lumbar", "centline"]
LSI_VARS = ["cpr", "endoint", "bpap"]

# Every column read by either label builder (the leakage guard forbids these as features)
LABEL_SOURCE_COLUMNS = sorted(set(
    LAB_VARS + ADV_IMG_VARS + SIMPLE_PROC + COMPLEX_PROC + LSI_VARS +
    ["ekg", "xray", "ivfluids", "nebuther", "consult", "totdiag", "totproc",
     "dieded", "doa", "lwbs", "lbtc", "leftama", "admit", "hdstat", "admithos",
     "obshos", "obsdis", "tranoth", "tranpsyc", "numgiv", "waittime", "lov"]
))


def _flag(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].eq(1).any(axis=1)


def count_resources(df: pd.DataFrame) -> pd.Series:
    """ESI resource count per visit (see module docstring for the mapping)."""
    r = (_flag(df, LAB_VARS).astype(int)
         + _flag(df, ["ekg"]).astype(int)
         + _flag(df, ["xray"]).astype(int)
         + _flag(df, ADV_IMG_VARS).astype(int)
         + _flag(df, ["ivfluids"]).astype(int)
         + _flag(df, ["nebuther"]).astype(int)
         + _flag(df, ["consult"]).astype(int))
    proc = np.where(_flag(df, COMPLEX_PROC), 2, np.where(_flag(df, SIMPLE_PROC), 1, 0))
    return (r + proc).astype("int32")


def exclusion_reason(df: pd.DataFrame) -> pd.Series:
    """Visit-level exclusion reason ('' = kept)."""
    reason = pd.Series("", index=df.index, dtype="object")
    for col, name in [("leftama", "left_ama"), ("lbtc", "lbtc"), ("lwbs", "lwbs"), ("doa", "doa")]:
        if col in df:
            reason[df[col].eq(1)] = name        # later assignments win: DOA > LWBS > LBTC > AMA
    return reason


def build_resi_o(df: pd.DataFrame) -> pd.Series:
    """
    Outcome/resource-based reference acuity rESI-O (1..5, NaN if excluded/unknown).
    """
    level = pd.Series(np.nan, index=df.index, dtype="float64")
    admit = df["admit"] if "admit" in df else pd.Series(np.nan, index=df.index)

    l1 = _flag(df, ["dieded"] + LSI_VARS)
    l2 = admit.isin([1, 3, 5]) | (df["hdstat"].eq(2) if "hdstat" in df else False)
    l3_disp = _flag(df, ["admithos", "obshos", "tranoth", "tranpsyc"])

    res = count_resources(df)
    blank = (df["totdiag"].isna() if "totdiag" in df else True) & \
            (df["totproc"].isna() if "totproc" in df else True)

    level[res.eq(0) & ~blank] = 5
    level[res.eq(1)] = 4
    level[res.ge(2) | l3_disp] = 3
    level[l2] = 2
    level[l1] = 1
    level[exclusion_reason(df).ne("")] = np.nan
    return level


def _danger_vitals(df: pd.DataFrame) -> pd.Series:
    """ESI step-4 danger-zone vitals (ESI v4 age bands for heart rate)."""
    age, hr = df["age"], df["pulse"]
    hr_hi = np.select([age < 1 / 12 * 3, age < 3, age < 8], [180, 160, 140], default=100)
    return (hr > hr_hi) | (df["respr"] > 20) | (df["popct"] < 92)


def build_nhamcs_esi(df: pd.DataFrame) -> pd.Series:
    """ESI-style stepwise sensitivity label (see module docstring)."""
    level = pd.Series(np.nan, index=df.index, dtype="float64")
    res = count_resources(df)
    blank = df["totdiag"].isna() & df["totproc"].isna()
    danger = _danger_vitals(df).fillna(False)
    high_risk = ((df["painscale"] >= 7) & res.ge(2)) | (df["popct"] < 90) | \
        df["admit"].isin([1, 3, 5])

    level[res.eq(0) & ~blank] = 5
    level[res.eq(1)] = 4
    level[res.ge(2)] = np.where(danger[res.ge(2)], 2, 3)
    level[high_risk.fillna(False)] = 2
    level[_flag(df, ["dieded"] + LSI_VARS)] = 1
    level[exclusion_reason(df).ne("")] = np.nan
    return level


def main() -> None:
    """Build both labels, save, and write the T5 data-flow table."""
    processed = Path("data/processed")
    src = processed / "nhamcs_pooled.parquet"
    if not src.exists():
        print("Pooled NHAMCS data not found. Run: python -m triageq.data.nhamcs_load first.")
        return

    df = pd.read_parquet(src)
    df["resi_o_excl"] = exclusion_reason(df)
    df["n_resources"] = count_resources(df)
    df["resi_o"] = build_resi_o(df)
    df["nhamcs_esi"] = build_nhamcs_esi(df)
    df.to_parquet(processed / "nhamcs_with_labels.parquet", index=False)

    # T5: data flow per year
    rows = []
    for y, g in df.groupby("survey_year"):
        kept = g["resi_o_excl"].eq("")
        r = {"year": int(y), "visits": len(g)}
        for reason in ["doa", "lwbs", "lbtc", "left_ama"]:
            r[f"excl_{reason}"] = int(g["resi_o_excl"].eq(reason).sum())
        r["kept"] = int(kept.sum())
        r["label_missing"] = int((kept & g["resi_o"].isna()).sum())
        r["labelled"] = int(g["resi_o"].notna().sum())
        r["pct_labelled_of_kept"] = 100 * r["labelled"] / max(r["kept"], 1)
        for k in range(1, 6):
            r[f"L{k}"] = int(g["resi_o"].eq(k).sum())
        rows.append(r)
    t5 = pd.DataFrame(rows)
    tot = t5.drop(columns=["year", "pct_labelled_of_kept"]).sum()
    tot["year"], tot["pct_labelled_of_kept"] = "all", 100 * tot["labelled"] / tot["kept"]
    t5 = pd.concat([t5, tot.to_frame().T], ignore_index=True)
    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    t5.to_csv(tables / "T5_data_flow.csv", index=False, float_format="%.2f")
    print(t5.to_string(index=False))

    lab = df[df["resi_o"].notna()]
    w = lab.groupby("resi_o")["patwt"].sum()
    pi = (w / w.sum()).round(4)
    print("\nrESI-O weighted distribution (π):", pi.to_dict())
    print("rESI-O unweighted counts:", lab["resi_o"].value_counts().sort_index().to_dict())
    (processed / "pi_resi_o.json").write_text(
        pd.Series(pi.values, index=[f"L{int(k)}" for k in pi.index]).to_json(indent=2))
    print(f"Saved → {processed}/nhamcs_with_labels.parquet, pi_resi_o.json; T5 → {tables}/")


if __name__ == "__main__":
    main()
