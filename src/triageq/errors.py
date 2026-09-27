"""
triageq.errors — Ordinal misclassification model for acuity levels.

The latent model is:
    y* = k_true + b + σ·ε,   ε ~ N(0,1)
    k_hat = clip(round(y*), 1, K)

Cut-points are at half-integers (0.5, 1.5, ..., K-0.5), so
    P(k_hat = j | k_true = k) = Φ((j+0.5 - k - b) / σ) - Φ((j-0.5 - k - b) / σ)

σ = 0  →  identity permutation (shifted by round(b)).
b > 0  →  undertriage (predicted less urgent than truth).
b < 0  →  overtriage (predicted more urgent than truth).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Any


# ── Core confusion matrix ──────────────────────────────────────────────────────

def confusion_from_params(
    sigma: float,
    b: float,
    K: int = 5,
) -> NDArray[np.float64]:
    """
    Compute the K×K confusion matrix analytically from (sigma, b).

    Entry M[k-1, j-1] = P(predicted = j | true = k), for k, j in 1..K.
    Rows correspond to true levels; columns to predicted levels.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (≥ 0). sigma=0 gives a permutation matrix.
    b     : float
        Systematic bias. b>0 = undertriage, b<0 = overtriage.
    K     : int
        Number of acuity levels (default 5, matching ESI/CTAS).

    Returns
    -------
    M : ndarray, shape (K, K), float64
        Row-stochastic confusion matrix.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    M = np.zeros((K, K), dtype=np.float64)

    for k in range(1, K + 1):          # true level
        mu = float(k) + b              # mean of latent y* (σ normalised out)

        if sigma == 0:
            # Deterministic: k_hat = clip(round(mu), 1, K)
            # Use Python's built-in round() to match predict_levels()
            k_hat = int(np.clip(round(mu), 1, K))
            M[k - 1, k_hat - 1] = 1.0
        else:
            for j in range(1, K + 1):  # predicted level
                # P(pred = j | true = k) = P(j-0.5 < y* <= j+0.5)
                #   where y* ~ N(mu, sigma²)
                # Clipping: anything ≤ 0.5 rounds to 1; anything > K-0.5 rounds to K
                if j == 1:
                    # Left boundary: (-inf, 1.5)
                    hi = (1.5 - mu) / sigma
                    M[k - 1, j - 1] = float(norm.cdf(hi))
                elif j == K:
                    # Right boundary: [K-0.5, +inf)
                    lo = (float(K) - 0.5 - mu) / sigma
                    M[k - 1, j - 1] = 1.0 - float(norm.cdf(lo))
                else:
                    lo = (float(j) - 0.5 - mu) / sigma
                    hi = (float(j) + 0.5 - mu) / sigma
                    M[k - 1, j - 1] = float(norm.cdf(hi)) - float(norm.cdf(lo))

    # Numerical safety: rows must sum to 1
    row_sums = M.sum(axis=1, keepdims=True)
    M = M / np.where(row_sums > 0, row_sums, 1.0)

    return M


# ── Patient-level prediction ───────────────────────────────────────────────────

def predict_levels(
    true: NDArray[np.int_],
    eps: NDArray[np.float64],
    sigma: float,
    b: float,
    K: int = 5,
) -> NDArray[np.int_]:
    """
    Predict acuity levels using the latent ordered-probit model.

    Uses pre-drawn standard-normal noise ε for Common Random Numbers (CRN).

    Parameters
    ----------
    true  : int array, values in [1, K]
    eps   : float array, same shape, pre-drawn from N(0,1)
    sigma : noise std
    b     : bias
    K     : number of levels

    Returns
    -------
    pred : int array, values in [1, K]
    """
    latent = true.astype(np.float64) + b + sigma * eps
    pred = np.clip(np.round(latent), 1, K).astype(np.int_)
    return pred


def predict_from_matrix(
    true: NDArray[np.int_],
    u: NDArray[np.float64],
    M: NDArray[np.float64],
) -> NDArray[np.int_]:
    """
    Sample predicted levels from an empirical confusion matrix using CRN.

    Uses pre-drawn uniforms u ∈ [0,1) for inverse-CDF sampling.

    Parameters
    ----------
    true : int array, values in [1, K]
    u    : float array, same shape, pre-drawn from U(0,1)
    M    : ndarray (K, K) — row-stochastic confusion matrix

    Returns
    -------
    pred : int array, values in [1, K]
    """
    M = np.asarray(M, dtype=np.float64)
    K = M.shape[0]
    cumM = np.cumsum(M, axis=1)       # (K, K), each row: CDF over predicted levels
    # Same as searchsorted(row, u, side='left'): count of CDF entries < u
    j = (cumM[np.asarray(true) - 1] < np.asarray(u)[:, None]).sum(axis=1)
    return np.clip(j + 1, 1, K).astype(np.int_)


# ── Summary statistics ─────────────────────────────────────────────────────────

def summary_stats(
    M: NDArray[np.float64],
    pi: NDArray[np.float64],
    K: int = 5,
) -> dict[str, Any]:
    """
    Compute interpretable classifier quality metrics from a confusion matrix.

    All metrics are π-weighted (population-level, not per-class-averaged).

    Parameters
    ----------
    M  : (K, K) row-stochastic confusion matrix
    pi : (K,) true acuity prior, sums to 1
    K  : number of levels

    Returns
    -------
    dict with keys:
        exact_acc    : P(pred == true)
        within1_acc  : P(|pred - true| <= 1)
        undertriage  : P(pred > true)   — predicted less urgent
        overtriage   : P(pred < true)   — predicted more urgent
        ha_sens      : sensitivity for high-acuity (levels 1-2 vs 3-5)
        ha_spec      : specificity for high-acuity
        youden_j     : ha_sens + ha_spec - 1
        qwk          : quadratic-weighted kappa
    """
    M = np.asarray(M, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()

    # Joint probability matrix: J[k,j] = P(true=k+1, pred=j+1)
    J = M * pi[:, None]

    exact_acc  = float(np.trace(J))
    within1    = float(sum(J[i, j] for i in range(K) for j in range(K) if abs(i - j) <= 1))
    under      = float(sum(J[i, j] for i in range(K) for j in range(K) if j > i))
    over       = float(sum(J[i, j] for i in range(K) for j in range(K) if j < i))

    # High-acuity = levels 1-2 (indices 0, 1)
    ha_idx = [0, 1]
    lo_idx = [2, 3, 4]
    p_ha_true = float(pi[ha_idx].sum())

    # P(pred is HA | true is HA)
    ha_sens = float(J[np.ix_(ha_idx, ha_idx)].sum()) / p_ha_true if p_ha_true > 0 else 0.0
    # P(pred is LO | true is LO)
    p_lo_true = float(pi[lo_idx].sum())
    ha_spec = float(J[np.ix_(lo_idx, lo_idx)].sum()) / p_lo_true if p_lo_true > 0 else 0.0

    youden_j = ha_sens + ha_spec - 1.0

    # Severe undertriage: a true level-1/2 patient predicted into levels 4–5, i.e.
    # queued behind the bulk of low-acuity arrivals (the errors that drive tail waits)
    severe_under_12 = float(J[np.ix_(ha_idx, [3, 4])].sum()) / p_ha_true if p_ha_true > 0 else 0.0

    # Quadratic-weighted kappa
    w = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)], dtype=np.float64)
    w = w / ((K - 1) ** 2)
    p_marginal_row = J.sum(axis=1)        # P(true = k)
    p_marginal_col = J.sum(axis=0)        # P(pred = j)
    E = np.outer(p_marginal_row, p_marginal_col)  # expected under independence
    po = float(np.sum(w * J))
    pe = float(np.sum(w * E))
    qwk = 1.0 - po / pe if pe > 0 else 1.0

    return {
        "exact_acc":   exact_acc,
        "within1_acc": within1,
        "undertriage": under,
        "overtriage":  over,
        "ha_sens":     ha_sens,
        "ha_spec":     ha_spec,
        "youden_j":    youden_j,
        "qwk":         qwk,
        "severe_under_12": severe_under_12,
    }


# ── Fit (σ, b) from empirical counts ──────────────────────────────────────────

def fit_params(
    M_counts: NDArray[np.float64],
    K: int = 5,
    init: tuple[float, float] = (0.8, 0.0),
) -> tuple[float, float]:
    """
    Fit (sigma, b) of the ordered-probit model to an empirical count matrix.

    Maximises the expected log-likelihood (sum of count * log P(j|k)).

    Parameters
    ----------
    M_counts : (K, K) matrix of counts (not probabilities)
    K        : number of levels
    init     : (sigma, b) starting point

    Returns
    -------
    (sigma_hat, b_hat) : MLE estimates
    """
    M_counts = np.asarray(M_counts, dtype=np.float64)

    def neg_log_lik(params: NDArray) -> float:
        sigma, b = params
        if sigma <= 0:
            return 1e9
        M_model = confusion_from_params(sigma, b, K)
        # Clip for numerical safety
        M_model = np.clip(M_model, 1e-12, 1.0)
        ll = float(np.sum(M_counts * np.log(M_model)))
        return -ll

    res = minimize(
        neg_log_lik,
        x0=list(init),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000},
    )
    sigma_hat, b_hat = res.x
    return float(max(sigma_hat, 0.0)), float(b_hat)


# ── Two-class break-even check (Section 5.1) ──────────────────────────────────

def two_class_breakeven_condition(
    sensitivity: float,
    specificity: float,
) -> tuple[bool, float]:
    """
    Analytic break-even condition for mean wait, two classes, non-preemptive.

    From Section 5.1: triage beats FIFO on E[W_H] iff sensitivity + specificity > 1.
    This is load-independent.

    Returns
    -------
    (beats_fifo, youden_j) : bool and J = sens + spec - 1
    """
    j = sensitivity + specificity - 1.0
    return (j > 0, j)
