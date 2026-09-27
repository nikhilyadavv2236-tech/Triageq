"""
triageq.metrics — Per-level queue outcome metrics.

Computes all primary and secondary endpoints from a vector of waiting times
and the corresponding true acuity levels.

Primary endpoint
----------------
TC_12  : P(wait ≤ τ_k) for true levels 1–2, pooled (target compliance)

Secondary endpoints
-------------------
  p90, p95   : Percentiles of wait for each true level
  mean_wait  : E[wait] per true level
  starvation : P(wait > 3·τ_k) per true level
  harm_45    : Mean and p95 wait for true levels 4–5
  weighted   : Σ w_k · E[wait_k]  with w = (16,8,4,2,1)
  overall    : Mean wait, all patients (conservation check)
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
from typing import Any


# ── Default targets (minutes) per acuity level ────────────────────────────────

DEFAULT_TAU = [5.0, 15.0, 30.0, 60.0, 120.0]   # τ_1 ... τ_5
DEFAULT_W   = [16.0, 8.0, 4.0, 2.0, 1.0]        # cost weights


def summarise(
    waits: NDArray[np.float64],
    true_levels: NDArray[np.int_],
    tau: list[float] | None = None,
    weights: list[float] | None = None,
    warmup_frac: float = 0.10,
    K: int = 5,
) -> dict[str, Any]:
    """
    Compute all queue outcome metrics from a single replication.

    Parameters
    ----------
    waits       : (N,) waiting times; NaN for warm-up patients or LWBS
    true_levels : (N,) true acuity levels (1..K)
    tau         : (K,) time-to-provider targets in minutes
    weights     : (K,) cost weights for weighted mean wait
    warmup_frac : fraction of patients already discarded (used only for logging)
    K           : number of acuity levels

    Returns
    -------
    dict with all metrics (see module docstring for list)
    """
    if tau is None:
        tau = DEFAULT_TAU[:K]
    if weights is None:
        weights = DEFAULT_W[:K]

    tau     = np.asarray(tau,     dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Mask: valid (post-warm-up, non-LWBS) patients
    valid = ~np.isnan(waits)

    per_level: dict[str, Any] = {}
    tc_ha_numerator   = 0.0
    tc_ha_denominator = 0.0
    mean_wait_all     = float(np.nanmean(waits)) if valid.any() else np.nan
    weighted_cost     = 0.0

    for k in range(1, K + 1):
        mask = valid & (true_levels == k)
        w_k  = waits[mask]
        n_k  = len(w_k)

        if n_k == 0:
            per_level[k] = {
                "n": 0, "mean": np.nan, "p50": np.nan,
                "p90": np.nan, "p95": np.nan,
                "tc":  np.nan, "starvation": np.nan,
            }
            continue

        tau_k   = float(tau[k - 1])
        cost_wt = float(weights[k - 1])

        w_arr   = waits[mask]
        mean_k  = float(np.mean(w_arr))
        p50_k   = float(np.percentile(w_arr, 50))
        p90_k   = float(np.percentile(w_arr, 90))
        p95_k   = float(np.percentile(w_arr, 95))
        tc_k    = float(np.mean(w_arr <= tau_k))
        starv_k = float(np.mean(w_arr > 3.0 * tau_k))

        per_level[k] = {
            "n":          n_k,
            "mean":       mean_k,
            "p50":        p50_k,
            "p90":        p90_k,
            "p95":        p95_k,
            "tc":         tc_k,
            "starvation": starv_k,
        }

        weighted_cost += cost_wt * mean_k

        # High-acuity pool: levels 1–2
        if k <= 2:
            tc_ha_numerator   += float((waits[valid & (true_levels == k)] <= tau[k - 1]).sum())
            tc_ha_denominator += float((valid & (true_levels == k)).sum())

    # Pooled high-acuity TC
    tc_12 = tc_ha_numerator / tc_ha_denominator if tc_ha_denominator > 0 else np.nan

    # Low-acuity harm (levels 4–5)
    mask_lo = valid & (true_levels >= 4)
    harm_45: dict[str, float] = {}
    if mask_lo.any():
        w_lo    = waits[mask_lo]
        tl_lo   = true_levels[mask_lo]          # true levels for these patients
        tau_lo  = tau[tl_lo - 1]                # vectorised tau lookup
        harm_45 = {
            "mean": float(np.mean(w_lo)),
            "p95":  float(np.percentile(w_lo, 95)),
            "tc":   float(np.mean(w_lo <= tau_lo)),
        }
    else:
        harm_45 = {"mean": np.nan, "p95": np.nan, "tc": np.nan}

    return {
        "tc_12":        tc_12,
        "per_level":    per_level,
        "harm_45":      harm_45,
        "weighted_cost": weighted_cost,
        "overall_mean": mean_wait_all,
        "n_valid":      int(valid.sum()),
    }


SWEEP_METRICS = {
    # name: higher_is_better
    "tc12":    True,    # primary: P(wait ≤ τ_k), true levels 1–2 pooled
    "mean12":  False,
    "p90_12":  False,
    "p95_12":  False,
    "starv12": False,   # P(wait > 3τ_k), levels 1–2
    "mean1":   False,
    "mean45":  False,
    "p95_45":  False,
    "tc45":    True,
    "wcost":   False,   # Σ w_k E[wait_k]
    "overall": False,
}


def sweep_metrics(
    waits: NDArray[np.float64],
    true_levels: NDArray[np.int_],
    tau: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """
    Scalar metrics for one replication of a 5-level sweep cell (see SWEEP_METRICS).

    Cheaper than summarise(): percentiles only for the pooled 1–2 and 4–5 groups.
    """
    tau = np.asarray(tau, dtype=np.float64)
    if weights is None:
        weights = np.asarray(DEFAULT_W[:len(tau)], dtype=np.float64)
    valid = ~np.isnan(waits)
    w, t = waits[valid], true_levels[valid]
    tau_i = tau[t - 1]

    hi = t <= 2
    lo = t >= 4
    w_hi, w_lo = w[hi], w[lo]
    nan = float("nan")

    level_means = np.array([w[t == k].mean() if (t == k).any() else nan
                            for k in range(1, len(tau) + 1)])
    return {
        "tc12":    float(np.mean(w_hi <= tau_i[hi])) if hi.any() else nan,
        "mean12":  float(w_hi.mean()) if hi.any() else nan,
        "p90_12":  float(np.percentile(w_hi, 90)) if hi.any() else nan,
        "p95_12":  float(np.percentile(w_hi, 95)) if hi.any() else nan,
        "starv12": float(np.mean(w_hi > 3.0 * tau_i[hi])) if hi.any() else nan,
        "mean1":   float(level_means[0]),
        "mean45":  float(w_lo.mean()) if lo.any() else nan,
        "p95_45":  float(np.percentile(w_lo, 95)) if lo.any() else nan,
        "tc45":    float(np.mean(w_lo <= tau_i[lo])) if lo.any() else nan,
        "wcost":   float(np.nansum(np.asarray(weights) * level_means)),
        "overall": float(w.mean()),
    }


def summarise_replications(
    rep_results: list[dict[str, Any]],
    K: int = 5,
) -> dict[str, Any]:
    """
    Aggregate per-replication summaries into means and 95% CIs.

    Uses t-distribution across replication means (Law & Kelton method).

    Parameters
    ----------
    rep_results : list of dicts, each from summarise()
    K           : number of acuity levels

    Returns
    -------
    dict with same keys as summarise(), values replaced by (mean, lo, hi) tuples
    """
    import scipy.stats as st

    R = len(rep_results)
    if R == 0:
        return {}

    def _ci(vals: list[float]) -> tuple[float, float, float]:
        a = np.array(vals, dtype=np.float64)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return (np.nan, np.nan, np.nan)
        m = float(np.mean(a))
        if len(a) == 1:
            return (m, np.nan, np.nan)
        se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
        t  = float(st.t.ppf(0.975, df=len(a) - 1))
        return (m, m - t * se, m + t * se)

    out: dict[str, Any] = {}

    # Top-level scalars
    for key in ("tc_12", "weighted_cost", "overall_mean"):
        vals = [r[key] for r in rep_results]
        out[key] = _ci(vals)

    # Per-level
    out["per_level"] = {}
    for k in range(1, K + 1):
        out["per_level"][k] = {}
        for metric in ("mean", "p90", "p95", "tc", "starvation"):
            vals = [r["per_level"][k][metric] for r in rep_results if r["per_level"][k]["n"] > 0]
            out["per_level"][k][metric] = _ci(vals)

    # Harm
    out["harm_45"] = {}
    for metric in ("mean", "p95", "tc"):
        vals = [r["harm_45"][metric] for r in rep_results]
        out["harm_45"][metric] = _ci(vals)

    return out
