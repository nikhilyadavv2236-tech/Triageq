"""
tests/test_validation.py — Simulator validation tests V1–V7.

Marked @pytest.mark.slow — run with:
    pytest tests/ -v -m slow

Pass criteria (from Section 5.2):
    - Relative error of simulated vs analytic mean wait < 2% (3% for APQ)
    - 95% CI covers the analytic truth
    - All tests run at rho ∈ {0.5, 0.7, 0.9} with 30 replications each
"""

import math
import numpy as np
import pytest
import scipy.stats as st

from triageq.streams import make_streams, ServiceConfig, replication_seeds
from triageq.errors import predict_levels, confusion_from_params
from triageq.analytic import (
    mm1_fifo_wq,
    mg1_pk_wq,
    cobham_np,
    mmc_np_common,
    mmc_fifo_wq,
    two_class_breakeven_check,
    apq_two_class_kleinrock,
)
from triageq.sim_simpy import run_simpy
from triageq.metrics import summarise


# ── Helpers ───────────────────────────────────────────────────────────────────

K  = 5
PI = np.array([0.01, 0.12, 0.42, 0.35, 0.10])
PI = PI / PI.sum()

N_REPS     = 30
N_BASE     = 100_000
N_HIGH     = 250_000
WARMUP     = 0.10
ATOL_REL   = 0.02    # 2% relative error
ATOL_APQ   = 0.03    # 3% for APQ (Kleinrock)


def _simulate_reps(n_reps, n_patients, rho, c, pi, service_cfg, policy, pred_fn=None, seed_master=20260925):
    """Run n_reps replications, return list of per-level mean waits."""
    seeds   = replication_seeds(n_reps, master_seed=seed_master)
    results = []

    for seed in seeds:
        streams = make_streams(n_patients, rho, c, pi, service_cfg, seed=seed)

        if pred_fn is None:
            # Oracle: use true levels (for validation)
            pred = streams["true_levels"]
        else:
            pred = pred_fn(streams)

        waits, evts = run_simpy(streams, pred, policy=policy, c=c, warmup_frac=WARMUP)
        valid = ~np.isnan(waits)
        true_lv = streams["true_levels"]

        level_means = {}
        for k in range(1, K + 1):
            mask = valid & (true_lv == k)
            level_means[k] = float(np.mean(waits[mask])) if mask.any() else math.nan

        results.append(level_means)

    return results


def _mean_and_ci(values: list[float]) -> tuple[float, float, float]:
    """Compute mean and 95% t-CI from a list of replication values."""
    a  = np.array(values, dtype=np.float64)
    m  = float(np.mean(a))
    se = float(np.std(a, ddof=1) / math.sqrt(len(a)))
    t  = float(st.t.ppf(0.975, df=len(a) - 1))
    return m, m - t * se, m + t * se


def _overall_mean_wait(reps: list[dict[int, float]], pi: np.ndarray) -> list[float]:
    """Weighted mean wait across all levels for each replication."""
    out = []
    for d in reps:
        vals = [d[k] for k in range(1, K + 1) if not math.isnan(d[k])]
        out.append(float(np.mean(vals)) if vals else math.nan)
    return out


# ── V1: M/M/1 FIFO ───────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v1_mm1_fifo(rho):
    """V1: M/M/1 FIFO — simulated vs analytic W_q = ρ/(μ(1−ρ))."""
    mu  = 1.0 / 20.0   # service rate (1 per 20 min)
    lam = rho * mu
    cfg = ServiceConfig(family="exponential", mean_min=20.0)
    pi1 = np.array([1.0])   # single class
    n   = N_HIGH if rho >= 0.9 else N_BASE

    seeds = replication_seeds(N_REPS, master_seed=1001)
    waits_per_rep = []
    for seed in seeds:
        streams = make_streams(n, rho, 1, np.array([1.0]), cfg, seed=seed)
        pred    = streams["true_levels"]
        waits, _ = run_simpy(streams, pred, policy="fifo", c=1, warmup_frac=WARMUP)
        valid    = ~np.isnan(waits)
        waits_per_rep.append(float(np.mean(waits[valid])))

    sim_mean, lo, hi = _mean_and_ci(waits_per_rep)
    analytic         = mm1_fifo_wq(lam, mu)

    rel_err = abs(sim_mean - analytic) / analytic
    assert rel_err < ATOL_REL, \
        f"V1 rho={rho}: rel_err={rel_err:.4f}, sim={sim_mean:.3f}, analytic={analytic:.3f}"
    assert lo <= analytic <= hi, \
        f"V1 rho={rho}: analytic={analytic:.3f} not in 95% CI [{lo:.3f}, {hi:.3f}]"


# ── V2: M/G/1 FIFO, lognormal CV=1.5 ────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v2_mg1_fifo_lognormal(rho):
    """V2: M/G/1 FIFO, lognormal service (CV=1.5) — Pollaczek-Khinchine."""
    mean_s = 20.0
    cv     = 1.5
    ES     = mean_s
    ES2    = ES ** 2 * (1 + cv ** 2)      # lognormal: E[S²] = μ²(1+CV²)
    mu     = 1.0 / ES
    lam    = rho * mu
    cfg    = ServiceConfig(family="lognormal", mean_min=mean_s, cv=cv)
    n      = N_HIGH if rho >= 0.9 else N_BASE

    seeds = replication_seeds(N_REPS, master_seed=1002)
    waits_per_rep = []
    for seed in seeds:
        streams = make_streams(n, rho, 1, np.array([1.0]), cfg, seed=seed)
        pred    = streams["true_levels"]
        waits, _ = run_simpy(streams, pred, policy="fifo", c=1, warmup_frac=WARMUP)
        valid    = ~np.isnan(waits)
        waits_per_rep.append(float(np.mean(waits[valid])))

    sim_mean, lo, hi = _mean_and_ci(waits_per_rep)
    analytic         = mg1_pk_wq(lam, ES, ES2)

    rel_err = abs(sim_mean - analytic) / analytic
    assert rel_err < ATOL_REL, \
        f"V2 rho={rho}: rel_err={rel_err:.4f}, sim={sim_mean:.3f}, analytic={analytic:.3f}"
    assert lo <= analytic <= hi, \
        f"V2 rho={rho}: analytic={analytic:.3f} not in CI [{lo:.3f}, {hi:.3f}]"


# ── V3: M/G/1 NP-PQ, 5 levels, level-dependent service ──────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v3_mg1_nppq_5levels(rho):
    """V3: M/G/1 NP-PQ, 5 levels, S1 service — Cobham mean waits."""
    multipliers = [1.5, 1.3, 1.0, 0.8, 0.6]
    cfg = ServiceConfig(family="exponential", mean_min=20.0, level_multipliers=multipliers)
    ES, ES2 = cfg.ES_and_ES2(K)

    ES_overall = float(np.dot(PI, ES))
    lam = rho / ES_overall   # c=1
    lams_true = lam * PI

    analytic_waits = cobham_np(lams_true, ES, ES2)
    n = N_HIGH if rho >= 0.9 else N_BASE

    seeds = replication_seeds(N_REPS, master_seed=1003)
    level_waits_reps = {k: [] for k in range(1, K + 1)}

    for seed in seeds:
        streams = make_streams(n, rho, 1, PI, cfg, seed=seed)
        pred = streams["true_levels"]   # Oracle (true levels = predicted)
        waits, _ = run_simpy(streams, pred, policy="np_pq", c=1, warmup_frac=WARMUP)
        valid    = ~np.isnan(waits)
        true_lv  = streams["true_levels"]
        for k in range(1, K + 1):
            mask = valid & (true_lv == k)
            if mask.any():
                level_waits_reps[k].append(float(np.mean(waits[mask])))

    for k in range(1, K + 1):
        rvals = level_waits_reps[k]
        if not rvals:
            continue
        sim_mean, lo, hi = _mean_and_ci(rvals)
        analytic = analytic_waits[k - 1]

        if not math.isfinite(analytic):
            continue
        rel_err = abs(sim_mean - analytic) / analytic
        assert rel_err < ATOL_REL, \
            f"V3 rho={rho} level={k}: rel_err={rel_err:.4f}, sim={sim_mean:.3f}, analytic={analytic:.3f}"
        assert lo <= analytic <= hi, \
            f"V3 rho={rho} level={k}: analytic not in CI"


# ── V4: M/M/c NP-PQ, c=5, common rate ───────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v4_mmc_nppq_common(rho):
    """V4: M/M/c NP-PQ, c=5, common rate — M/M/c analytic formula."""
    c   = 5
    mu  = 1.0 / 20.0
    lam = rho * c * mu
    cfg = ServiceConfig(family="exponential", mean_min=20.0)   # common rate
    lams_true = lam * PI
    analytic_waits = mmc_np_common(lams_true, mu, c)
    n = N_HIGH if rho >= 0.9 else N_BASE

    seeds = replication_seeds(N_REPS, master_seed=1004)
    level_waits_reps = {k: [] for k in range(1, K + 1)}

    for seed in seeds:
        streams = make_streams(n, rho, c, PI, cfg, seed=seed)
        pred = streams["true_levels"]   # Oracle
        waits, _ = run_simpy(streams, pred, policy="np_pq", c=c, warmup_frac=WARMUP)
        valid    = ~np.isnan(waits)
        true_lv  = streams["true_levels"]
        for k in range(1, K + 1):
            mask = valid & (true_lv == k)
            if mask.any():
                level_waits_reps[k].append(float(np.mean(waits[mask])))

    for k in range(1, K + 1):
        rvals = level_waits_reps[k]
        if not rvals:
            continue
        sim_mean, lo, hi = _mean_and_ci(rvals)
        analytic = analytic_waits[k - 1]
        if not math.isfinite(analytic):
            continue
        rel_err = abs(sim_mean - analytic) / analytic
        assert rel_err < ATOL_REL, \
            f"V4 rho={rho} level={k}: rel_err={rel_err:.4f}, sim={sim_mean:.3f}, analytic={analytic:.3f}"
        assert lo <= analytic <= hi, \
            f"V4 rho={rho} level={k}: analytic not in CI"


# ── V5: Conservation law ──────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v5_conservation_law(rho):
    """V5: Σ ρ_k·W_k is equal across FIFO, NP-PQ, APQ (Kleinrock 1965)."""
    c   = 1
    cfg = ServiceConfig(family="exponential", mean_min=20.0)
    ES, ES2 = cfg.ES_and_ES2(K)
    ES_overall = float(np.dot(PI, ES))
    lam = rho / ES_overall
    lams_true = lam * PI
    rho_k = lams_true * ES
    n = N_HIGH if rho >= 0.9 else N_BASE

    seeds = replication_seeds(N_REPS, master_seed=1005)

    wcons_fifo = []
    wcons_nppq = []
    wcons_apq  = []

    for seed in seeds:
        streams = make_streams(n, rho, c, PI, cfg, seed=seed)
        true_lv = streams["true_levels"]

        for policy, store in [("fifo", wcons_fifo), ("np_pq", wcons_nppq), ("apq", wcons_apq)]:
            waits, _ = run_simpy(streams, true_lv, policy=policy, c=c, warmup_frac=WARMUP)
            valid    = ~np.isnan(waits)
            level_means = []
            for k in range(1, K + 1):
                mask = valid & (true_lv == k)
                level_means.append(float(np.mean(waits[mask])) if mask.any() else 0.0)
            # Σ ρ_k · W_k
            wc = float(np.dot(rho_k, level_means))
            store.append(wc)

    m_fifo, lo_fifo, hi_fifo = _mean_and_ci(wcons_fifo)
    m_nppq, lo_nppq, hi_nppq = _mean_and_ci(wcons_nppq)
    m_apq,  lo_apq,  hi_apq  = _mean_and_ci(wcons_apq)

    # The paired difference should be zero: each CI should contain the grand mean
    grand = (m_fifo + m_nppq + m_apq) / 3.0
    assert lo_fifo <= grand <= hi_fifo, f"V5 FIFO CI [{lo_fifo:.4f},{hi_fifo:.4f}] doesn't contain grand {grand:.4f}"
    assert lo_nppq <= grand <= hi_nppq, f"V5 NP-PQ CI [{lo_nppq:.4f},{hi_nppq:.4f}] doesn't contain grand {grand:.4f}"
    assert lo_apq  <= grand <= hi_apq,  f"V5 APQ CI  [{lo_apq:.4f},{hi_apq:.4f}] doesn't contain grand {grand:.4f}"


# ── V6: Two-class break-even crosses at s+p=1 ────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v6_two_class_breakeven(rho):
    """V6: Mean-wait break-even is s+p=1, load-independent."""
    # Verify over a grid of (s,p) that crossing is at J=0 for all rho
    c   = 1
    cfg = ServiceConfig(family="exponential", mean_min=20.0)
    pi2 = np.array([0.20, 0.80])   # two classes: H=20%, L=80%
    K2  = 2

    # Test three operating points: below, at, above J=0
    for (s, p) in [(0.9, 0.8), (0.5, 0.5), (0.6, 0.3)]:
        # Build a 2×2 confusion matrix
        M2 = np.array([[s, 1-s], [1-p, p]])

        seeds = replication_seeds(10, master_seed=2000 + int(100*rho))
        delta_means = []
        for seed in seeds:
            streams = make_streams(N_BASE // 2, rho, c, pi2, cfg, seed=seed)
            # FIFO
            fifo_pred = np.ones(len(streams["true_levels"]), dtype=np.int32)
            w_fifo, _ = run_simpy(streams, fifo_pred, policy="fifo", c=c, warmup_frac=WARMUP)
            # Triage: sample from M2
            from triageq.errors import predict_from_matrix
            pred = predict_from_matrix(streams["true_levels"], streams["u"], M2)
            w_pq, _ = run_simpy(streams, pred, policy="np_pq", c=c, warmup_frac=WARMUP)

            # High-acuity mean wait (true level 1)
            valid_f = ~np.isnan(w_fifo)
            valid_p = ~np.isnan(w_pq)
            true_lv = streams["true_levels"]
            m_fifo = float(np.mean(w_fifo[valid_f & (true_lv == 1)]))
            m_pq   = float(np.mean(w_pq[valid_p   & (true_lv == 1)]))
            delta_means.append(m_pq - m_fifo)

        mean_delta = float(np.mean(delta_means))
        j = s + p - 1.0

        if j > 0.05:
            assert mean_delta < 0, f"V6 s={s},p={p},rho={rho}: expected triage better (Δ<0), got Δ={mean_delta:.4f}"
        elif j < -0.05:
            assert mean_delta > 0, f"V6 s={s},p={p},rho={rho}: expected triage worse (Δ>0), got Δ={mean_delta:.4f}"


# ── V7: APQ two-class Kleinrock ───────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_v7_apq_two_class(rho):
    """V7: APQ two-class — mean waits match Kleinrock (1964) within 3%."""
    c     = 1
    mean_s = 20.0
    cv    = 0.0    # use exponential (CV=1 → ES2 = 2·ES²) for clean analytic check
    cfg   = ServiceConfig(family="exponential", mean_min=mean_s)
    pi2   = np.array([0.20, 0.80])
    K2    = 2
    ES, ES2 = cfg.ES_and_ES2(K2)

    lam = rho / float(np.dot(pi2, ES))
    lam1, lam2 = lam * pi2
    ES1, ES2_1 = ES[0], ES2[0]
    ES_2, ES2_2 = ES[1], ES2[1]
    w1, w2 = 8.0, 1.0

    analytic_W1, analytic_W2 = apq_two_class_kleinrock(
        lam1, lam2, ES1, ES_2, ES2_1, ES2_2, w1, w2
    )
    n = N_HIGH if rho >= 0.9 else N_BASE

    seeds = replication_seeds(N_REPS, master_seed=1007)
    w1_reps, w2_reps = [], []

    for seed in seeds:
        streams = make_streams(n, rho, c, pi2, cfg, seed=seed)
        pred    = streams["true_levels"]   # APQ on true levels (oracle)
        waits, _ = run_simpy(
            streams, pred, policy="apq", c=c,
            apq_weights=[w1, w2], warmup_frac=WARMUP
        )
        valid   = ~np.isnan(waits)
        true_lv = streams["true_levels"]
        m1 = valid & (true_lv == 1)
        m2 = valid & (true_lv == 2)
        if m1.any(): w1_reps.append(float(np.mean(waits[m1])))
        if m2.any(): w2_reps.append(float(np.mean(waits[m2])))

    sim1, lo1, hi1 = _mean_and_ci(w1_reps)
    sim2, lo2, hi2 = _mean_and_ci(w2_reps)

    if math.isfinite(analytic_W1):
        rel1 = abs(sim1 - analytic_W1) / analytic_W1
        assert rel1 < ATOL_APQ, f"V7 rho={rho} level=1: rel_err={rel1:.4f}"
        assert lo1 <= analytic_W1 <= hi1, f"V7 rho={rho} level=1: analytic not in CI"

    if math.isfinite(analytic_W2):
        rel2 = abs(sim2 - analytic_W2) / analytic_W2
        assert rel2 < ATOL_APQ, f"V7 rho={rho} level=2: rel_err={rel2:.4f}"
        assert lo2 <= analytic_W2 <= hi2, f"V7 rho={rho} level=2: analytic not in CI"
