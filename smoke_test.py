"""
smoke_test.py — Quick end-to-end smoke test (< 30 seconds).

Runs a tiny simulation to verify the full pipeline is wired correctly:
  1. Stream generation
  2. Error model → predicted levels
  3. SimPy and Numba simulation
  4. Metrics computation
  5. Break-even search
  6. Figure generation (F1, T1 only — no data required)

Usage:
    python smoke_test.py
"""

import sys
import time
import numpy as np

print("triageq smoke test", flush=True)
t0 = time.time()

# ── 1. Import all modules ────────────────────────────────────────────────────
print("  [1] Importing modules...", end=" ")
from triageq.streams import ServiceConfig, make_streams, replication_seeds
from triageq.errors import (
    confusion_from_params, predict_levels, predict_from_matrix,
    summary_stats, fit_params, two_class_breakeven_condition,
)
from triageq.analytic import (
    mm1_fifo_wq, mg1_pk_wq, cobham_np, erlang_c,
    mmc_np_common, mmc_fifo_wq, two_class_breakeven_check,
    misclassified_mean_waits,
)
from triageq.sim_simpy import run_simpy
from triageq.sim_fast import run_fast
from triageq.metrics import summarise, summarise_replications
from triageq.breakeven import find_breakeven, to_interpretable, delta_table
print("OK")

# ── 2. Analytic checks ───────────────────────────────────────────────────────
print("  [2] Analytic checks...", end=" ")
wq = mm1_fifo_wq(lam=0.04, mu=0.05)
assert abs(wq - 800.0) < 1, f"M/M/1 Wq: {wq}"

lams = np.array([0.004, 0.005, 0.01, 0.015, 0.006])
ES   = np.array([20.0] * 5)
ES2  = 2 * ES ** 2
W = cobham_np(lams, ES, ES2)
assert W[0] < W[4], "Level 1 (highest priority) must wait less than level 5"
print("OK")

# ── 3. Error model ───────────────────────────────────────────────────────────
print("  [3] Error model...", end=" ")
K  = 5
PI = np.array([0.01, 0.12, 0.42, 0.35, 0.10])
M  = confusion_from_params(sigma=0.8, b=0.0, K=K)
assert M.shape == (5, 5)
assert np.allclose(M.sum(axis=1), 1.0)

stats = summary_stats(M, PI)
assert 0 < stats["exact_acc"] < 1
assert 0 <= stats["ha_sens"] <= 1
print("OK")

# ── 4. Streams ───────────────────────────────────────────────────────────────
print("  [4] Stream generation...", end=" ")
cfg    = ServiceConfig(family="exponential", mean_min=20.0)
seeds  = replication_seeds(3, master_seed=42)
st     = make_streams(n=5000, rho=0.7, c=1, pi=PI, service_cfg=cfg, seed=seeds[0])
assert st["arrival_times"].shape == (5000,)
assert 1 <= st["true_levels"].min() <= st["true_levels"].max() <= 5
print("OK")

# ── 5. SimPy simulation ──────────────────────────────────────────────────────
print("  [5] SimPy FIFO and NP-PQ...", end=" ")
pred = predict_levels(st["true_levels"], st["eps"], sigma=0.8, b=0.0, K=K)
w_fifo, _ = run_simpy(st, pred, policy="fifo",  c=1, warmup_frac=0.10)
w_pq,   _ = run_simpy(st, pred, policy="np_pq", c=1, warmup_frac=0.10)
assert np.nanmean(w_pq[st["true_levels"] == 1]) < np.nanmean(w_fifo[st["true_levels"] == 1]), \
    "NP-PQ should reduce level-1 waits"
print("OK")

# ── 6. Numba simulation ──────────────────────────────────────────────────────
print("  [6] Numba kernel (first call compiles)...", end=" ")
w_fast = run_fast(st, pred, policy="np_pq", c=1, warmup_frac=0.10)
valid  = ~np.isnan(w_pq) & ~np.isnan(w_fast)
assert np.allclose(w_pq[valid], w_fast[valid], atol=1e-6), "SimPy and Numba must agree"
print("OK")

# ── 7. Metrics ───────────────────────────────────────────────────────────────
print("  [7] Metrics...", end=" ")
tau = [5.0, 15.0, 30.0, 60.0, 120.0]
s   = summarise(w_pq, st["true_levels"], tau=tau, K=K)
assert 0 <= s["tc_12"] <= 1
assert s["overall_mean"] > 0
print("OK")

# ── 8. Break-even ────────────────────────────────────────────────────────────
print("  [8] Break-even search...", end=" ")
from triageq.breakeven import synthetic_breakeven_test
res = synthetic_breakeven_test(true_crossing=0.9, R=10, noise_std=0.02, seed=7)
assert res["status"] == "crossing"
assert abs(res["sigma_star"] - 0.9) < 0.3, f"sigma_star={res['sigma_star']:.3f}"
print("OK")

# ── 9. Figures (no data required for F1/T1) ──────────────────────────────────
print("  [9] F1 system diagram and T1 table...", end=" ")
from pathlib import Path
out = Path("results/smoke_figures")
out.mkdir(parents=True, exist_ok=True)
from triageq.figures import make_f1, make_t1
make_f1(out)
make_t1(out)
print("OK")

elapsed = time.time() - t0
print(f"\n✅ All smoke tests passed in {elapsed:.1f}s")
print(f"   F1 and T1 saved to results/smoke_figures/")
