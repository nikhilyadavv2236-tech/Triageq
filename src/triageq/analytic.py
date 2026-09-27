"""
triageq.analytic — Closed-form queueing formulas.

Implements:
  - M/M/1 FIFO (Little / Pollaczek-Khinchine degenerate case)
  - M/G/1 FIFO  (Pollaczek-Khinchine)
  - M/G/1 non-preemptive priority (Cobham's formula)
  - M/M/c non-preemptive priority with a common service rate
  - Kleinrock's accumulating priority (APQ) for two classes
  - Mean waits for any confusion matrix (mixture over predicted-class Cobham waits)
  - Two-class break-even check from Section 5.1

All times are in the same unit as the service-time inputs (minutes if you pass
service rates in 1/min).

References
----------
Cobham, A. (1954). Priority assignment in waiting line problems.
    Operations Research 2(1), 70-76.
Kleinrock, L. (1964). A delay-dependent queue discipline.
    Naval Research Logistics Quarterly 11(3-4), 329-341.
Kleinrock, L. (1965). A conservation law for a wide class of queueing
    disciplines. Naval Research Logistics Quarterly 12(2), 181-192.
Gross, D., Harris, C. (1998). Fundamentals of Queueing Theory. Wiley.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


# ── Erlang-C (M/M/c probability that a new arrival must wait) ─────────────────

def erlang_c(c: int, a: float) -> float:
    """
    Erlang-C formula: P(wait > 0) for M/M/c with offered load a = λ/μ < c.

    Uses log-domain computation to avoid overflow.

    Parameters
    ----------
    c : number of servers
    a : offered load (λ/μ), must be < c for stability

    Returns
    -------
    C : float in [0, 1]
    """
    if a >= c:
        return 1.0
    # log( a^c / c! * c / (c - a) ) + log( sum_{n=0}^{c-1} a^n / n! )
    log_numerator = c * math.log(a) - math.lgamma(c + 1) + math.log(c) - math.log(c - a)
    # log of denominator sum using log-sum-exp
    log_terms = [n * math.log(a) - math.lgamma(n + 1) for n in range(c)]
    max_log = max(log_terms)
    log_denom_sum = max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))

    log_C_inv_minus1 = log_denom_sum - log_numerator  # log( (1/C) - 1 ) roughly
    # C = 1 / (1 + exp(log_denom_sum - log_numerator))  but more carefully:
    # C = numerator / (numerator + denominator_sum)
    log_num = log_numerator
    log_den = log_denom_sum

    if log_num > log_den:
        log_total = log_num + math.log1p(math.exp(log_den - log_num))
    else:
        log_total = log_den + math.log1p(math.exp(log_num - log_den))

    C = math.exp(log_num - log_total)
    return float(np.clip(C, 0.0, 1.0))


# ── M/M/1 FIFO ────────────────────────────────────────────────────────────────

def mm1_fifo_wq(lam: float, mu: float) -> float:
    """
    Mean waiting time (excluding service) for M/M/1 FIFO.

    W_q = ρ / (μ(1 − ρ))  with ρ = λ/μ.
    """
    rho = lam / mu
    if rho >= 1:
        return math.inf
    return rho / (mu * (1.0 - rho))


# ── M/G/1 FIFO (Pollaczek-Khinchine) ─────────────────────────────────────────

def mg1_pk_wq(lam: float, ES: float, ES2: float) -> float:
    """
    Mean waiting time for M/G/1 FIFO via Pollaczek-Khinchine mean-value formula.

    W_q = λ·E[S²] / (2(1 − ρ))  with ρ = λ·E[S].

    Parameters
    ----------
    lam : arrival rate
    ES  : E[service time]
    ES2 : E[service time²]
    """
    rho = lam * ES
    if rho >= 1:
        return math.inf
    return lam * ES2 / (2.0 * (1.0 - rho))


# ── M/G/1 non-preemptive priority (Cobham) ────────────────────────────────────

def cobham_np(
    lams: NDArray[np.float64],
    ES: NDArray[np.float64],
    ES2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Mean waiting times for K-class M/G/1 non-preemptive priority.

    Classes are numbered 1..K in decreasing priority (class 1 = highest).
    Within each class, service is FIFO.

    Cobham's formula:
        W_k = W_0 / [ (1 - σ_{k-1})(1 - σ_k) ]

    where:
        W_0   = Σ_j λ_j E[S_j²] / 2   (mean residual work)
        σ_k   = Σ_{j=1}^{k} λ_j E[S_j]  (cumulative load of top-k classes)
        σ_0   = 0

    Parameters
    ----------
    lams : (K,) arrival rates per class
    ES   : (K,) mean service times per class
    ES2  : (K,) mean squared service times per class (E[S²] = Var[S] + E[S]²)

    Returns
    -------
    W : (K,) mean waiting times per class (excluding service)
    """
    lams = np.asarray(lams, dtype=np.float64)
    ES   = np.asarray(ES,   dtype=np.float64)
    ES2  = np.asarray(ES2,  dtype=np.float64)
    K = len(lams)

    W0 = 0.5 * float(np.sum(lams * ES2))          # mean residual work
    sigma = np.cumsum(lams * ES)                   # σ_1, σ_2, ..., σ_K
    sigma_prev = np.concatenate([[0.0], sigma[:-1]])  # σ_0=0, σ_1, ..., σ_{K-1}

    rho = float(sigma[-1])
    if rho >= 1:
        return np.full(K, math.inf)

    W = np.empty(K, dtype=np.float64)
    for k in range(K):
        s0 = sigma_prev[k]
        s1 = sigma[k]
        denom = (1.0 - s0) * (1.0 - s1)
        if denom <= 0:
            W[k] = math.inf
        else:
            W[k] = W0 / denom

    return W


# ── M/M/c non-preemptive priority (common service rate) ───────────────────────

def mmc_np_common(
    lams: NDArray[np.float64],
    mu: float,
    c: int,
) -> NDArray[np.float64]:
    """
    Mean waiting times for K-class M/M/c non-preemptive priority.

    With a common service rate μ, the formula is:

        W_k = C(c, a) / (c·μ) * 1 / [(1 - σ_{k-1}/c)(1 - σ_k/c)]

    where a = λ_total / μ is the offered load and C(c, a) is Erlang-C.

    (This is the M/M/c analogue of Cobham: the Erlang-C factor replaces W_0
    in the same role, and load fractions replace ρ_k in the M/G/1 case.)

    Parameters
    ----------
    lams : (K,) arrival rates per class
    mu   : common service rate (scalar)
    c    : number of servers

    Returns
    -------
    W : (K,) mean waiting times per class (excluding service)
    """
    lams  = np.asarray(lams, dtype=np.float64)
    K     = len(lams)
    lam   = float(lams.sum())
    a     = lam / mu                  # offered load
    rho   = a / c                     # utilisation

    if rho >= 1:
        return np.full(K, math.inf)

    C     = erlang_c(c, a)
    base  = C / (c * mu)                 # W_0 analogue; K=1 gives M/M/c FIFO Wq

    # Cumulative loads (fractions of c)
    rho_k  = lams / (c * mu)           # each class's load fraction
    sigma  = np.cumsum(rho_k)          # σ_k / c  in [0, 1)
    sigma_prev = np.concatenate([[0.0], sigma[:-1]])

    W = np.empty(K, dtype=np.float64)
    for k in range(K):
        s0 = sigma_prev[k]
        s1 = sigma[k]
        denom = (1.0 - s0) * (1.0 - s1)
        if denom <= 0:
            W[k] = math.inf
        else:
            W[k] = base / denom

    return W


# ── APQ two-class (Kleinrock 1964) ────────────────────────────────────────────

def apq_two_class_kleinrock(
    lam1: float, lam2: float,
    ES1: float,  ES2: float,
    ES2_1: float, ES2_2: float,
    w1: float, w2: float,
) -> tuple[float, float]:
    """
    Mean waiting times for two-class M/G/1 accumulating priority queue.

    Kleinrock (1964) delay-dependent discipline:
        Priority accrues at rate w_k per unit of waiting time.
        A free server chooses the customer with the highest accrued priority.

    With r = w2/w1 ≤ 1 and W_F = W_0/(1−ρ) the FIFO wait:
        W_2 = W_F / (1 − ρ_1(1 − r))
        W_1 = W_F − ρ_2 W_2 (1 − r)
    r = 1 gives FIFO; r → 0 gives Cobham's non-preemptive priority.
    Thin wrapper around apq_kleinrock().

    Parameters
    ----------
    lam1, lam2     : arrival rates for class 1 (high) and class 2 (low)
    ES1, ES2       : mean service times
    ES2_1, ES2_2   : E[S²] for each class
    w1, w2         : priority accrual rates (w1 > w2 for class 1 being higher priority)

    Returns
    -------
    (W1, W2) : mean waiting times (excluding service)

    Notes
    -----
    For the validation (V7), we use r = w2/w1. When r → 0 the result should
    approach the Cobham non-preemptive result. When r = 1, it gives FIFO.
    """
    W = apq_kleinrock(
        np.array([lam1, lam2]), np.array([ES1, ES2]),
        np.array([ES2_1, ES2_2]), np.array([w1, w2]),
    )
    return (float(W[0]), float(W[1]))


def apq_kleinrock(
    lams: NDArray[np.float64],
    ES: NDArray[np.float64],
    ES2: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Mean waits for the K-class M/G/1 accumulating (delay-dependent) priority queue.

    Kleinrock (1964); see also Kleinrock, Queueing Systems Vol. 2, §3.7.
    Priority of a waiting class-k customer at time t is w_k·(t − arrival).
    Sorting classes so that w_(1) ≤ … ≤ w_(K) (lowest priority first):

        W_p = [ W0/(1−ρ) − Σ_{i<p} ρ_i W_i (1 − w_i/w_p) ]
              / [ 1 − Σ_{i>p} ρ_i (1 − w_p/w_i) ]

    which is solved recursively from the lowest-priority class upwards.
    Equal weights give FIFO; w_i/w_j → 0 recovers Cobham's formula.

    Parameters
    ----------
    lams, ES, ES2 : (K,) arrival rates, E[S], E[S²] per class
    weights       : (K,) accrual rates per class (any order)

    Returns
    -------
    W : (K,) mean waits, in the input class order
    """
    lams = np.asarray(lams, dtype=np.float64)
    ES   = np.asarray(ES, dtype=np.float64)
    ES2  = np.asarray(ES2, dtype=np.float64)
    w    = np.asarray(weights, dtype=np.float64)
    K    = len(lams)

    rho_k = lams * ES
    rho   = float(rho_k.sum())
    if rho >= 1:
        return np.full(K, math.inf)
    W_fifo = 0.5 * float(np.sum(lams * ES2)) / (1.0 - rho)

    idx = np.argsort(w, kind="stable")          # lowest weight first
    W_sorted = np.empty(K, dtype=np.float64)
    for p in range(K):
        wp  = w[idx[p]]
        num = W_fifo
        for i in range(p):
            wi = w[idx[i]]
            num -= rho_k[idx[i]] * W_sorted[i] * (1.0 - wi / wp)
        den = 1.0
        for i in range(p + 1, K):
            wi = w[idx[i]]
            den -= rho_k[idx[i]] * (1.0 - wp / wi)
        W_sorted[p] = num / den if den > 0 else math.inf

    W = np.empty(K, dtype=np.float64)
    W[idx] = W_sorted
    return W


# ── Mean waits under misclassification (any confusion matrix) ─────────────────

def misclassified_mean_waits(
    M: NDArray[np.float64],
    pi: NDArray[np.float64],
    lam: float,
    ES: NDArray[np.float64],
    ES2: NDArray[np.float64],
    c: int = 1,
    mu_common: float | None = None,
) -> NDArray[np.float64]:
    """
    Mean waiting times by TRUE acuity level under a noisy classifier.

    The predicted level k_hat is drawn from confusion matrix M given true level k.
    Under non-preemptive priority by predicted level, we need:
      - predicted-class arrival rates:  lam_j = Σ_k lam_k * M[k-1, j-1]
      - predicted-class service params: computed from the mixture

    For the c=1 case we use Cobham's formula.
    For the c>1 case with a common rate we use mmc_np_common.

    W_true_k = Σ_j M[k-1, j-1] * W_pred[j]   (law of total expectation)

    Parameters
    ----------
    M          : (K,K) row-stochastic confusion matrix
    pi         : (K,) true acuity prior
    lam        : total arrival rate
    ES         : (K,) mean service times by TRUE class
    ES2        : (K,) E[S²] by TRUE class
    c          : number of servers
    mu_common  : if given, use M/M/c formula with this common rate;
                 otherwise use M/G/1 Cobham formula

    Returns
    -------
    W_true : (K,) mean waiting times by true acuity level
    """
    M   = np.asarray(M,   dtype=np.float64)
    pi  = np.asarray(pi,  dtype=np.float64)
    ES  = np.asarray(ES,  dtype=np.float64)
    ES2 = np.asarray(ES2, dtype=np.float64)
    K   = M.shape[0]

    pi = pi / pi.sum()
    lams_true = lam * pi                    # per-true-class arrival rates

    # ── Predicted-class arrival rates ──────────────────────────────────────
    # lam_pred[j] = Σ_k lam_k * M[k, j]
    lams_pred = lams_true @ M               # (K,)

    if mu_common is not None:
        # M/M/c with common rate: service params don't matter for W
        W_pred = mmc_np_common(lams_pred, mu_common, c)
    else:
        # M/G/1 Cobham: need service params per predicted class
        # E[S | pred = j] = Σ_k P(true=k | pred=j) * ES[k]
        # By Bayes: P(true=k | pred=j) = pi_k * M[k,j] / lam_pred[j]
        ES_pred  = np.zeros(K, dtype=np.float64)
        ES2_pred = np.zeros(K, dtype=np.float64)
        for j in range(K):
            if lams_pred[j] > 0:
                weights     = lams_true * M[:, j] / lams_pred[j]
                ES_pred[j]  = float(weights @ ES)
                ES2_pred[j] = float(weights @ ES2)
            else:
                ES_pred[j]  = ES.mean()
                ES2_pred[j] = ES2.mean()

        W_pred = cobham_np(lams_pred, ES_pred, ES2_pred)

    # ── Mix back to true-class waits ───────────────────────────────────────
    # W_true[k] = Σ_j M[k, j] * W_pred[j]
    W_true = M @ W_pred                    # (K,)

    return W_true


# ── Two-class break-even check ────────────────────────────────────────────────

def two_class_breakeven_check(
    s: float,
    p: float,
    rho: float,
) -> dict[str, float | bool]:
    """
    Check the analytic break-even condition for mean wait (Section 5.1).

    Returns the Youden J value and whether triage beats FIFO.
    The condition is load-independent: depends only on s + p > 1.

    Parameters
    ----------
    s   : sensitivity (P(pred = H | true = H))
    p   : specificity (P(pred = L | true = L))
    rho : total utilization (unused in the analytic check, included for logging)

    Returns
    -------
    dict with:
        youden_j     : s + p - 1
        beats_fifo   : bool (youden_j > 0)
        rho_used     : the rho passed in (for reference)
    """
    j = s + p - 1.0
    return {
        "youden_j":   j,
        "beats_fifo": bool(j > 0),
        "rho_used":   rho,
    }


# ── FIFO mean wait for M/M/c ─────────────────────────────────────────────────

def mmc_fifo_wq(lam: float, mu: float, c: int) -> float:
    """Mean waiting time for M/M/c FIFO (excluding service)."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return math.inf
    C = erlang_c(c, a)
    return C / (c * mu * (1.0 - rho))
