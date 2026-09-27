"""
triageq.breakeven — Break-even accuracy estimation.

For a given (ρ, b, c, policy, metric), finds the noise level σ* at which
triage performance equals FIFO performance, then converts σ* to interpretable
classifier quality metrics.

Algorithm
---------
1. For each σ on the grid, compute per-replication Δ(σ) = metric(triage) − metric(FIFO).
   Orientation: Δ < 0 means triage is better (lower wait or higher TC difference).
2. Find the grid interval where the mean Δ changes sign; linearly interpolate → σ*.
3. Bootstrap the R replications (1,000 resamples) to get a 95% CI on σ*.
4. Convert σ* to exact confusion matrix stats via errors.summary_stats.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
from typing import Any

from triageq.errors import confusion_from_params, summary_stats


# ── Break-even search ─────────────────────────────────────────────────────────

StatusType = str  # "crossing" | "always_better" | "never_better"


def find_breakeven(
    sigmas: NDArray[np.float64],
    deltas_by_rep: NDArray[np.float64],   # (n_sigma, n_reps)
    metric_higher_is_better: bool = True,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Find the break-even noise level σ* from replication-level metric differences.

    Parameters
    ----------
    sigmas               : (M,) grid of σ values (ascending)
    deltas_by_rep        : (M, R) per-replication metric differences:
                           metric(triage) - metric(FIFO)
                           Positive = triage is better (if metric_higher_is_better).
    metric_higher_is_better : True for TC (higher = better), False for mean wait.
    n_boot               : number of bootstrap resamples for σ* CI
    rng                  : numpy Generator (optional)

    Returns
    -------
    dict with:
        sigma_star : float (or nan if no crossing)
        ci_lo      : float, 95% CI lower bound
        ci_hi      : float, 95% CI upper bound
        status     : "crossing" | "always_better" | "never_better"
        mean_delta : (M,) mean Δ across replications
    """
    if rng is None:
        rng = np.random.default_rng(20260925)

    sigmas         = np.asarray(sigmas,        dtype=np.float64)
    deltas_by_rep  = np.asarray(deltas_by_rep, dtype=np.float64)  # (M, R)
    M, R = deltas_by_rep.shape

    mean_delta = np.nanmean(deltas_by_rep, axis=1)   # (M,)

    # At σ=0, triage is perfect (or close to it) → Δ should be positive (better)
    # As σ increases, performance degrades → Δ crosses zero at σ*

    def _find_crossing(delta_vec: NDArray) -> float | None:
        """Find the σ where delta_vec changes from positive to negative."""
        for i in range(len(delta_vec) - 1):
            d0, d1 = delta_vec[i], delta_vec[i + 1]
            if not (math.isfinite(d0) and math.isfinite(d1)):
                continue
            if metric_higher_is_better:
                # triage better when Δ > 0; break-even at Δ = 0
                if d0 >= 0 and d1 < 0:
                    # Linear interpolation
                    t = d0 / (d0 - d1)
                    return float(sigmas[i] + t * (sigmas[i + 1] - sigmas[i]))
            else:
                # triage better when Δ < 0 (lower wait)
                if d0 <= 0 and d1 > 0:
                    t = -d0 / (d1 - d0)
                    return float(sigmas[i] + t * (sigmas[i + 1] - sigmas[i]))
        return None

    sigma_star = _find_crossing(mean_delta)

    if sigma_star is None:
        # No better→worse crossing: classify by where triage is better on the grid
        finite = mean_delta[np.isfinite(mean_delta)]
        better = finite > 0 if metric_higher_is_better else finite < 0
        if better.all():
            status = "always_better"
        elif not better.any():
            status = "never_better"
        else:
            status = "mixed"     # e.g. worse at σ=0 under strong bias, better later
        return {
            "sigma_star": np.nan,
            "ci_lo":      np.nan,
            "ci_hi":      np.nan,
            "status":     status,
            "mean_delta": mean_delta,
        }

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    boot_stars = []
    rep_indices = np.arange(R)
    for _ in range(n_boot):
        boot_reps  = rng.choice(rep_indices, size=R, replace=True)
        boot_delta = deltas_by_rep[:, boot_reps].mean(axis=1)
        s = _find_crossing(boot_delta)
        if s is not None:
            boot_stars.append(s)

    boot_stars_arr = np.array(boot_stars, dtype=np.float64)
    if len(boot_stars_arr) >= 10:
        ci_lo = float(np.percentile(boot_stars_arr, 2.5))
        ci_hi = float(np.percentile(boot_stars_arr, 97.5))
    else:
        ci_lo = ci_hi = np.nan

    return {
        "sigma_star": sigma_star,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "status":     "crossing",
        "mean_delta": mean_delta,
    }


# ── Convert σ* to interpretable metrics ───────────────────────────────────────

def to_interpretable(
    sigma_star: float,
    b: float,
    pi: NDArray[np.float64],
    K: int = 5,
    ci_lo: float | None = None,
    ci_hi: float | None = None,
) -> dict[str, Any]:
    """
    Convert a break-even σ* to classifier quality metrics.

    Parameters
    ----------
    sigma_star : break-even noise level
    b          : bias used in the sweep
    pi         : (K,) true acuity prior
    K          : number of levels
    ci_lo, ci_hi : optional CI bounds on σ* (also converted)

    Returns
    -------
    dict with exact_acc, within1_acc, undertriage, ha_sens, youden_j, qwk,
    and optionally ci_lo/hi versions of each.
    """
    if not math.isfinite(sigma_star):
        return {"sigma_star": sigma_star, "status": "no_crossing"}

    M = confusion_from_params(sigma_star, b, K)
    stats = summary_stats(M, pi, K)
    result = {"sigma_star": sigma_star, "bias": b, **stats}

    if ci_lo is not None and math.isfinite(ci_lo):
        M_lo = confusion_from_params(ci_lo, b, K)
        s_lo = summary_stats(M_lo, pi, K)
        result["ci_lo_stats"] = s_lo
        result["ci_lo"] = ci_lo

    if ci_hi is not None and math.isfinite(ci_hi):
        M_hi = confusion_from_params(ci_hi, b, K)
        s_hi = summary_stats(M_hi, pi, K)
        result["ci_hi_stats"] = s_hi
        result["ci_hi"] = ci_hi

    return result


# ── Paired delta table helper ─────────────────────────────────────────────────

def delta_table(
    rep_metric_triage: list[float],
    rep_metric_fifo: list[float],
) -> NDArray[np.float64]:
    """
    Compute per-replication metric difference (triage − FIFO).

    Parameters
    ----------
    rep_metric_triage : list of length R with per-replication metric values
    rep_metric_fifo   : list of length R with per-replication FIFO values

    Returns
    -------
    (R,) array of differences
    """
    return np.asarray(rep_metric_triage, dtype=np.float64) - \
           np.asarray(rep_metric_fifo,   dtype=np.float64)


# ── Synthetic test helper ─────────────────────────────────────────────────────

def synthetic_breakeven_test(
    true_crossing: float = 1.0,
    n_sigma: int = 21,
    R: int = 20,
    noise_std: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generate synthetic Δ(σ) data with a known crossing at `true_crossing`.

    Used in test_breakeven_synthetic to verify that find_breakeven recovers
    the true crossing within tolerance.

    The synthetic delta is: Δ(σ) = (true_crossing - σ) + noise
    So mean Δ > 0 for σ < true_crossing and < 0 for σ > true_crossing.
    """
    rng    = np.random.default_rng(seed)
    sigmas = np.linspace(0.0, 2.0, n_sigma)
    noise  = rng.normal(0.0, noise_std, size=(n_sigma, R))
    deltas = (true_crossing - sigmas[:, None]) + noise
    return find_breakeven(
        sigmas,
        deltas,
        metric_higher_is_better=True,
        rng=rng,
    )
