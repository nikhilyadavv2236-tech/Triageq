"""
triageq.streams — Common Random Numbers (CRN) stream generation.

All policies and all (σ, b) parameter combinations share the same pre-drawn
random arrays within a replication. This is essential for:
  1. Smooth break-even curves (predicted level varies continuously with σ).
  2. Tight paired confidence intervals (common noise cancels across policies).

Stream contents per replication:
  - arrival_times  : (N,) cumulative inter-arrival times (Poisson process)
  - service_times  : (N,) service duration per patient (from true level's distribution)
  - true_levels    : (N,) true acuity levels drawn from π (1..K)
  - eps            : (N,) standard normal noise for the ordered-probit model
  - u              : (N,) uniform [0,1) for inverse-CDF sampling from empirical matrices
  - patience       : (N,) exponential patience times (for abandonment, optional)

Usage
-----
    streams = make_streams(n=100_000, rho=0.85, c=5, pi=pi, service_cfg=cfg, seed=seed)
    pred = predict_levels(streams["true_levels"], streams["eps"], sigma=0.8, b=0.0)
    waits = simulate(streams, pred, policy="np_pq", c=5, ...)
"""

from __future__ import annotations

import numpy as np
from numpy.random import SeedSequence, Generator
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Any


# ── Service time configuration ─────────────────────────────────────────────────

@dataclass
class ServiceConfig:
    """
    Configuration for service time distributions.

    Attributes
    ----------
    family : "exponential" | "lognormal"
        Distribution family.
    mean_min : float
        Overall mean service time in minutes (target after rescaling).
    level_multipliers : list[float] | None
        Per-level mean multipliers (length K). If None, all levels share the
        same mean (S0: common exponential baseline).
        Default for S1: [1.5, 1.3, 1.0, 0.8, 0.6] (rescaled to hit mean_min).
    cv : float
        Coefficient of variation (only used for family="lognormal"). Default 1.5.
    pi : list[float] | None
        Acuity mix used to rescale level multipliers so the π-weighted overall
        mean equals mean_min. If None, the unweighted mean is used.
    """
    family: str = "exponential"
    mean_min: float = 20.0
    level_multipliers: list[float] | None = None
    cv: float = 1.5
    pi: list[float] | None = None

    def per_level_means(self, K: int = 5) -> NDArray[np.float64]:
        """Compute per-level mean service times, rescaled to hit mean_min overall."""
        if self.level_multipliers is None:
            return np.full(K, self.mean_min, dtype=np.float64)
        m = np.asarray(self.level_multipliers, dtype=np.float64)
        if self.pi is None:
            base = m.mean()
        else:
            p = np.asarray(self.pi, dtype=np.float64)
            base = float(np.dot(p / p.sum(), m))
        return m * (self.mean_min / base)

    def ES_and_ES2(self, K: int = 5) -> tuple[NDArray, NDArray]:
        """Return E[S] and E[S²] per level."""
        ES = self.per_level_means(K)
        if self.family == "exponential":
            # Var[S] = E[S]² for exponential → E[S²] = 2·E[S]²
            ES2 = 2.0 * ES ** 2
        elif self.family == "lognormal":
            # CV = σ/μ → Var = (CV·μ)² → E[S²] = Var + μ²
            ES2 = ES ** 2 * (1.0 + self.cv ** 2)
        else:
            raise ValueError(f"Unknown service family: {self.family}")
        return ES, ES2


# ── Stream generation ─────────────────────────────────────────────────────────

def make_streams(
    n: int,
    rho: float,
    c: int,
    pi: NDArray[np.float64],
    service_cfg: ServiceConfig,
    seed: int | SeedSequence,
    include_patience: bool = False,
    patience_mean: float = 120.0,
) -> dict[str, Any]:
    """
    Pre-draw all random variates for one simulation replication.

    All arrays have length n (number of patients). The same streams are reused
    for every policy and every (σ, b) value within the replication.

    Parameters
    ----------
    n              : number of patients to simulate
    rho            : target utilisation ρ = λ·E[S]/c
    c              : number of servers
    pi             : (K,) true acuity prior (will be normalised)
    service_cfg    : ServiceConfig instance
    seed           : integer seed or numpy SeedSequence
    include_patience : if True, also draw patience times (for abandonment)
    patience_mean  : mean patience in minutes (exponential)

    Returns
    -------
    dict with keys:
        arrival_times  : (n,) float, cumulative arrival times [minutes]
        service_times  : (n,) float, service durations [minutes] by true level
        true_levels    : (n,) int,   values in [1, K]
        eps            : (n,) float, N(0,1) noise for ordered-probit
        u              : (n,) float, U(0,1) for empirical matrix sampling
        patience       : (n,) float, patience times (if include_patience)
        lam            : float, arrival rate used
        ES_per_level   : (K,) mean service times per level
        ES2_per_level  : (K,) E[S²] per level
        rho_actual     : float, ρ = λ · E[S_overall] / c
        K              : int, number of acuity levels
    """
    K = len(pi)
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()

    ES, ES2 = service_cfg.ES_and_ES2(K)
    ES_overall = float(np.dot(pi, ES))          # weighted mean service time

    lam = rho * c / ES_overall                   # arrival rate from ρ

    # Seed the RNG
    if isinstance(seed, SeedSequence):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(seed)

    # ── Arrivals ──────────────────────────────────────────────────────────
    inter_arrivals = rng.exponential(scale=1.0 / lam, size=n)
    arrival_times  = np.cumsum(inter_arrivals)

    # ── True levels ───────────────────────────────────────────────────────
    true_levels = rng.choice(np.arange(1, K + 1), size=n, p=pi)

    # ── Service times ─────────────────────────────────────────────────────
    service_times = np.empty(n, dtype=np.float64)
    for k in range(1, K + 1):
        mask = true_levels == k
        count = int(mask.sum())
        if count == 0:
            continue
        if service_cfg.family == "exponential":
            service_times[mask] = rng.exponential(scale=ES[k - 1], size=count)
        elif service_cfg.family == "lognormal":
            mu_ln  = np.log(ES[k - 1]) - 0.5 * np.log(1 + service_cfg.cv ** 2)
            sig_ln = np.sqrt(np.log(1 + service_cfg.cv ** 2))
            service_times[mask] = rng.lognormal(mean=mu_ln, sigma=sig_ln, size=count)
        else:
            raise ValueError(f"Unknown service family: {service_cfg.family}")

    # ── Noise for ordered-probit ───────────────────────────────────────────
    eps = rng.standard_normal(size=n)

    # ── Uniforms for empirical matrix sampling ─────────────────────────────
    u = rng.uniform(size=n)

    out: dict[str, Any] = {
        "arrival_times":  arrival_times,
        "service_times":  service_times,
        "true_levels":    true_levels,
        "eps":            eps,
        "u":              u,
        "lam":            lam,
        "ES_per_level":   ES,
        "ES2_per_level":  ES2,
        "rho_actual":     lam * ES_overall / c,
        "K":              K,
    }

    if include_patience:
        out["patience"] = rng.exponential(scale=patience_mean, size=n)

    return out


# ── SeedSequence factory ──────────────────────────────────────────────────────

def replication_seeds(
    n_reps: int,
    master_seed: int = 20260925,
) -> list[SeedSequence]:
    """
    Spawn n_reps independent SeedSequences from a master seed.

    Usage
    -----
        seeds = replication_seeds(20)
        streams = [make_streams(..., seed=s) for s in seeds]
    """
    ss = SeedSequence(master_seed)
    return list(ss.spawn(n_reps))
