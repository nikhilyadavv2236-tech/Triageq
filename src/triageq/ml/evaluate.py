"""
triageq.ml.evaluate — Classifier evaluation metrics (Milestone 10).

On the held-out test years (2021–2022), for every operating point:
  - exact accuracy, within-one accuracy, macro-F1, QWK
  - undertriage / overtriage (overall) and undertriage of true levels 1–2
  - high-acuity (levels 1–2 vs 3–5) sensitivity, specificity, AUROC, AUPRC
  - bootstrap 95% CIs (1,000 visit-level resamples)
  - 5×5 confusion matrix (counts) and fitted ordered-probit (σ, b)
Calibration of P(level ≤ 2) (prior-corrected LightGBM): ECE + reliability curve.

Operating points:
  nurse          IMMEDR (1–5) vs rESI-O, test visits with a valid IMMEDR
  ml             LightGBM expected-urgency score with validation-tuned cut-points (δ = 0)
  ml_argmax      LightGBM argmax
  ml_ordinal     ordinal LightGBM with tuned cut-points
  lr             logistic regression argmax
  ml_nurse_subset  `ml` restricted to the nurse subset (paired comparison)

Usage:
    python -m triageq.ml.evaluate
Outputs: results/tables/T3_classifier_performance.csv, results/e4/operating_points.json,
         results/e4/confusion_*.csv, results/e4/calibration.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 5
PRED_PATH = Path("data/processed/ml_predictions.parquet")
OUT_DIR   = Path("results/e4")
TAB_DIR   = Path("results/tables")


def _cm(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount((y - 1) * K + (p - 1), minlength=K * K).reshape(K, K).astype(float)


def _metrics_from_cm(C: np.ndarray) -> dict[str, float]:
    n = C.sum()
    i, j = np.indices(C.shape)
    rows, cols = C.sum(1), C.sum(0)
    tp = np.diag(C)
    f1 = np.where(rows + cols > 0, 2 * tp / np.maximum(rows + cols, 1e-12), 0.0)
    W = (i - j) ** 2 / (K - 1) ** 2
    E = np.outer(rows, cols) / n
    hi_t, hi_p = i < 2, j < 2
    return {
        "exact_acc":   tp.sum() / n,
        "within1_acc": C[np.abs(i - j) <= 1].sum() / n,
        "macro_f1":    f1.mean(),
        "qwk":         1 - (W * C).sum() / (W * E).sum(),
        "undertriage": C[j > i].sum() / n,
        "overtriage":  C[j < i].sum() / n,
        "under_12":    C[(i < 2) & (j > i)].sum() / max(C[:2].sum(), 1),
        "ha_sens":     C[hi_t & hi_p].sum() / max(C[:2].sum(), 1),
        "ha_spec":     C[~hi_t & ~hi_p].sum() / max(C[2:].sum(), 1),
    }


def point_metrics(y, p, score_hi) -> dict[str, float]:
    """All metrics; `score_hi` is larger for predicted higher acuity (for AUROC/AUPRC)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = _metrics_from_cm(_cm(y, p))
    m["youden_j"] = m["ha_sens"] + m["ha_spec"] - 1
    hi = (y <= 2).astype(int)
    m["ha_auroc"] = float(roc_auc_score(hi, score_hi))
    m["ha_auprc"] = float(average_precision_score(hi, score_hi))
    return m


def bootstrap(y, p, score_hi, n_boot: int = 1000, seed: int = 20260925) -> dict[str, tuple]:
    rng = np.random.default_rng(seed)
    est = point_metrics(y, p, score_hi)
    draws = {k: [] for k in est}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, v in point_metrics(y[idx], p[idx], score_hi[idx]).items():
            draws[k].append(v)
    return {k: (est[k], float(np.percentile(draws[k], 2.5)), float(np.percentile(draws[k], 97.5)))
            for k in est}


def ece(prob: np.ndarray, outcome: np.ndarray, bins: int = 10):
    """Expected calibration error with equal-mass bins; returns (ece, curve DataFrame)."""
    q = np.quantile(prob, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q, prob, side="right") - 1, 0, bins - 1)
    df = pd.DataFrame({"b": b, "p": prob, "o": outcome}).groupby("b").agg(
        mean_pred=("p", "mean"), frac_pos=("o", "mean"), n=("o", "size"))
    e = float((df.n * (df.mean_pred - df.frac_pos).abs()).sum() / df.n.sum())
    return e, df.reset_index(drop=True)


def evaluate_all(n_boot: int = 1000) -> dict:
    """Evaluate every operating point on the test years and save all outputs."""
    from triageq.errors import fit_params, summary_stats

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(PRED_PATH)
    te = d[d["split"] == "test"].reset_index(drop=True)
    y = te["resi_o"].to_numpy().astype(int)
    p_hi = (te["p_lgbm_1"] + te["p_lgbm_2"]).to_numpy()
    p_hi_lr = (te["p_lr_1"] + te["p_lr_2"]).to_numpy()
    nurse_ok = te["immedr"].between(1, 5).to_numpy()

    points = {
        "nurse":           (y[nurse_ok], te.loc[nurse_ok, "immedr"].to_numpy().astype(int),
                            -te.loc[nurse_ok, "immedr"].to_numpy()),
        "ml":              (y, te["pred_lgbm_cut"].to_numpy(), p_hi),
        "ml_nurse_subset": (y[nurse_ok], te.loc[nurse_ok, "pred_lgbm_cut"].to_numpy(), p_hi[nurse_ok]),
        "ml_argmax":       (y, te["pred_lgbm"].to_numpy(), p_hi),
        "ml_ordinal":      (y, te["pred_olgbm"].to_numpy(), -te["score_olgbm"].to_numpy()),
        "lr":              (y, te["pred_lr"].to_numpy(), p_hi_lr),
    }

    pi_path = Path("data/processed/pi_resi_o.json")
    pi = np.array(list(json.loads(pi_path.read_text()).values()))
    rows, ops = [], {}
    for name, (yt, yp, sc) in points.items():
        print(f"  {name}: n = {len(yt):,}")
        res = bootstrap(yt, yp.astype(int), sc, n_boot=n_boot)
        C = _cm(yt, yp.astype(int))
        pd.DataFrame(C.astype(int), index=[f"true_{k}" for k in range(1, 6)],
                     columns=[f"pred_{k}" for k in range(1, 6)]).to_csv(OUT_DIR / f"confusion_{name}.csv")
        sigma, b = fit_params(C)
        M = C / C.sum(1, keepdims=True)
        st_pi = summary_stats(M, pi)          # quality at the population mix π (simulation input)
        ops[name] = {"matrix": M.tolist(), "counts": C.astype(int).tolist(), "n": int(len(yt)),
                     "sigma_fit": sigma, "b_fit": b, "youden_j_at_pi": st_pi["youden_j"],
                     "exact_acc_at_pi": st_pi["exact_acc"], "ha_sens_at_pi": st_pi["ha_sens"]}
        row = {"operating_point": name, "n": len(yt), "sigma_fit": sigma, "b_fit": b}
        for k, (e, lo, hi) in res.items():
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = e, lo, hi
        rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(TAB_DIR / "T3_classifier_performance.csv", index=False, float_format="%.4f")

    # Nurse-label acuity mix (for the E3 "π from nurse labels" variant), survey-weighted
    allv = d[d["immedr"].between(1, 5)]
    wn = allv.groupby("immedr")["patwt"].sum()
    pi_nurse = (wn / wn.sum()).round(4)
    Path("data/processed/pi_nurse.json").write_text(
        pd.Series(pi_nurse.values, index=[f"L{int(k)}" for k in pi_nurse.index]).to_json(indent=2))

    # Calibration of P(level ≤ 2): raw (prior-corrected) and isotonic-recalibrated on
    # the 2019 validation year (monotone, so no operating point changes)
    from sklearn.isotonic import IsotonicRegression
    va = d[d["split"] == "val"]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        va["p_lgbm_1"] + va["p_lgbm_2"], (va["resi_o"] <= 2).astype(int))
    e, curve = ece(p_hi, (y <= 2).astype(int))
    e_iso, curve_iso = ece(iso.predict(p_hi), (y <= 2).astype(int))
    curve["kind"], curve_iso["kind"] = "raw", "isotonic"
    pd.concat([curve, curve_iso]).to_parquet(OUT_DIR / "calibration.parquet", index=False)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve((y <= 2).astype(int), p_hi)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet(OUT_DIR / "roc_ml.parquet", index=False)
    fn, tn, _ = roc_curve((y[nurse_ok] <= 2).astype(int), -te.loc[nurse_ok, "immedr"].to_numpy())
    pd.DataFrame({"fpr": fn, "tpr": tn}).to_parquet(OUT_DIR / "roc_nurse.parquet", index=False)

    out = {"pi_resi_o": pi.tolist(), "pi_nurse": pi_nurse.tolist(), "ece_p_high": e,
           "ece_p_high_isotonic": e_iso, "operating_points": ops}
    (OUT_DIR / "operating_points.json").write_text(json.dumps(out, indent=2))

    show = ["operating_point", "n", "exact_acc", "within1_acc", "qwk", "undertriage", "under_12",
            "ha_sens", "ha_spec", "youden_j", "ha_auroc", "sigma_fit", "b_fit"]
    print(t3[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nECE of P(level ≤ 2): prior-corrected {e:.4f}; isotonic (fit on 2019) {e_iso:.4f}")
    print("Nurse-label π (weighted):", pi_nurse.to_dict())
    return out


if __name__ == "__main__":
    evaluate_all()
