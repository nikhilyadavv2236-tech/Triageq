"""
triageq.run — CLI entry point for running experiments from YAML configs.

Usage:
    python -m triageq.run --config configs/e0.yaml
    python -m triageq.run --config configs/e2.yaml --jobs 8

The script:
  1. Loads the YAML config
  2. Dispatches to the appropriate experiment runner
  3. Saves results as parquet to output_dir/

Common random numbers: every worker draws ONE set of streams per
(ρ, c, replication) and runs every policy and every (σ, b) on it. Seeds depend
on the replication index only (never on σ, b or policy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_pi(cfg: dict) -> np.ndarray:
    """Resolve the pi parameter — either from config or placeholder until NHAMCS ready."""
    return _pi_from(cfg.get("pi", [0.01, 0.12, 0.42, 0.35, 0.10]))


PI_FILES = {"from_data": Path("data/processed/pi_resi_o.json"),
            "nurse":     Path("data/processed/pi_nurse.json")}


def _pi_from(pi_raw) -> np.ndarray:
    """A list, 'from_data' (weighted rESI-O mix) or 'nurse' (weighted IMMEDR mix)."""
    if isinstance(pi_raw, str):
        path = PI_FILES[pi_raw]
        if path.exists():
            pi = np.array(list(json.loads(path.read_text()).values()), dtype=np.float64)
            print(f"[run] pi = {pi_raw} ({path}): {np.round(pi, 4).tolist()}")
        else:
            print(f"[run] NOTE: pi='{pi_raw}' requested but {path} not found (run the data pipeline).")
            print("[run]       Using placeholder pi=[0.01, 0.12, 0.42, 0.35, 0.10]")
            pi = np.array([0.01, 0.12, 0.42, 0.35, 0.10])
    else:
        pi = np.array(pi_raw, dtype=np.float64)
    return pi / pi.sum()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _n_patients(cfg: dict, rho: float) -> int:
    n = cfg["n_patients"]
    return n["high_load"] if rho >= n["high_load_threshold"] else n["default"]


def _svc_dict(svc: dict, pi: np.ndarray | None = None) -> dict:
    d = {
        "family":            svc["family"],
        "mean_min":          float(svc["mean_min"]),
        "level_multipliers": svc.get("level_multipliers"),
    }
    if "cv" in svc:
        d["cv"] = float(svc["cv"])
    if d["level_multipliers"] is not None and pi is not None:
        d["pi"] = [float(x) for x in pi]
    return d


def _ci(vals) -> tuple[float, float, float]:
    """Mean and 95% t-interval across replications."""
    import scipy.stats as st
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (math.nan, math.nan, math.nan)
    m = float(a.mean())
    if len(a) < 2:
        return (m, math.nan, math.nan)
    h = float(st.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))
    return (m, m - h, m + h)


def _sim(engine: str, streams, pred, policy: str, c: int, warmup: float, apq_w=None):
    if engine == "simpy":
        from triageq.sim_simpy import run_simpy
        return run_simpy(streams, pred, policy=policy, c=c,
                         apq_weights=apq_w, warmup_frac=warmup)[0]
    from triageq.sim_fast import run_fast
    return run_fast(streams, pred, policy=policy, c=c,
                    apq_weights=apq_w, warmup_frac=warmup)


def _warm_numba() -> None:
    """Compile the kernel once in the parent so workers load it from cache."""
    from triageq.streams import ServiceConfig, make_streams
    from triageq.sim_fast import run_fast
    s = make_streams(200, 0.5, 1, np.array([0.5, 0.5]), ServiceConfig(), seed=0)
    run_fast(s, s["true_levels"], "apq", 1)


def _parallel(fn, tasks, n_jobs: int, desc: str):
    from joblib import Parallel, delayed
    from tqdm import tqdm
    return Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(fn)(*t) for t in tqdm(tasks, desc=desc)
    )


# ── E0: validation (V1–V8) ────────────────────────────────────────────────────

def _e0_worker(case: dict, rho: float, seed, engine: str, warmup: float) -> dict:
    """One replication of one validation case. Returns per-run level means."""
    from triageq.streams import ServiceConfig, make_streams
    from triageq.errors import predict_from_matrix

    pi = np.asarray(case["pi"], dtype=np.float64)
    K  = len(pi)
    streams = make_streams(case["n"], rho, case["c"], pi,
                           ServiceConfig(**case["svc"]), seed=seed)
    true = streams["true_levels"]
    out = {}
    for label, policy, M in case["runs"]:
        pred = true if M is None else predict_from_matrix(true, streams["u"], np.asarray(M))
        w = _sim(engine, streams, pred, policy, case["c"], warmup, case.get("apq_w"))
        valid = ~np.isnan(w)
        out[label] = [float(np.mean(w[valid & (true == k)])) if (valid & (true == k)).any()
                      else math.nan for k in range(1, K + 1)]
    return out


def _e0_cases(cfg: dict) -> list[dict]:
    """Build V1–V7 case definitions with their analytic references."""
    from triageq.streams import ServiceConfig
    from triageq.analytic import (
        mm1_fifo_wq, mg1_pk_wq, cobham_np, mmc_np_common,
        misclassified_mean_waits, apq_kleinrock,
    )

    pi5 = [0.01, 0.12, 0.42, 0.35, 0.10]
    pi2 = [0.20, 0.80]
    exp20 = {"family": "exponential", "mean_min": 20.0, "level_multipliers": None}
    s1    = {"family": "exponential", "mean_min": 20.0,
             "level_multipliers": [1.5, 1.3, 1.0, 0.8, 0.6]}
    logn  = {"family": "lognormal", "mean_min": 20.0, "level_multipliers": None, "cv": 1.5}
    apq5  = [float(x) for x in cfg.get("apq_weights", [16, 8, 4, 2, 1])]

    cases = []
    for rho in cfg["rho"]:
        n = _n_patients(cfg, rho)

        def lam_of(pi, svc, c):
            ES, _ = ServiceConfig(**svc).ES_and_ES2(len(pi))
            return rho * c / float(np.dot(np.asarray(pi) / sum(pi), ES))

        def es(svc, K):
            return ServiceConfig(**svc).ES_and_ES2(K)

        # V1 M/M/1 FIFO
        lam = lam_of([1.0], exp20, 1)
        cases.append(dict(test="V1", name="M/M/1 FIFO", rho=rho, n=n, c=1, pi=[1.0], svc=exp20,
                          runs=[("fifo", "fifo", None)],
                          ref={"fifo": [mm1_fifo_wq(lam, 1 / 20.0)]}))
        # V2 M/G/1 FIFO lognormal CV 1.5
        ES, ES2 = es(logn, 1)
        lam = lam_of([1.0], logn, 1)
        cases.append(dict(test="V2", name="M/G/1 FIFO lognormal", rho=rho, n=n, c=1, pi=[1.0],
                          svc=logn, runs=[("fifo", "fifo", None)],
                          ref={"fifo": [mg1_pk_wq(lam, ES[0], ES2[0])]}))
        # V3 M/G/1 NP-PQ, 5 levels, S1
        ES, ES2 = es(s1, 5)
        lam = lam_of(pi5, s1, 1)
        cases.append(dict(test="V3", name="M/G/1 NP-PQ S1", rho=rho, n=n, c=1, pi=pi5, svc=s1,
                          runs=[("np_pq", "np_pq", None)],
                          ref={"np_pq": list(cobham_np(lam * np.array(pi5), ES, ES2))}))
        # V4 M/M/5 NP-PQ common rate
        lam = lam_of(pi5, exp20, 5)
        cases.append(dict(test="V4", name="M/M/5 NP-PQ", rho=rho, n=n, c=5, pi=pi5, svc=exp20,
                          runs=[("np_pq", "np_pq", None)],
                          ref={"np_pq": list(mmc_np_common(lam * np.array(pi5), 1 / 20.0, 5))}))
        # V5 conservation law (M/G/1): Σ ρ_k W_k = ρ W0 / (1 − ρ)
        ES, ES2 = es(exp20, 5)
        lam = lam_of(pi5, exp20, 1)
        W0 = 0.5 * lam * float(np.dot(pi5, ES2))
        cases.append(dict(test="V5", name="Conservation", rho=rho, n=n, c=1, pi=pi5, svc=exp20,
                          apq_w=apq5,
                          runs=[("fifo", "fifo", None), ("np_pq", "np_pq", None),
                                ("apq", "apq", None)],
                          ref={"conservation": rho * W0 / (1 - rho)}))
        # V6 two-class misclassification (true-H mean wait)
        ES, ES2 = es(exp20, 2)
        lam = lam_of(pi2, exp20, 1)
        runs, ref = [("fifo", "fifo", None)], {"fifo": [mg1_pk_wq(lam, ES[0], ES2[0])] * 2}
        for s, p in [(0.9, 0.8), (0.7, 0.6), (0.5, 0.5), (0.4, 0.3)]:
            M2 = [[s, 1 - s], [1 - p, p]]
            lab = f"s={s:.1f},p={p:.1f}"
            runs.append((lab, "np_pq", M2))
            ref[lab] = list(misclassified_mean_waits(np.array(M2), np.array(pi2), lam, ES, ES2, c=1))
        cases.append(dict(test="V6", name="2-class misclassified NP-PQ", rho=rho, n=n, c=1,
                          pi=pi2, svc=exp20, runs=runs, ref=ref))
        # V7 APQ two-class, w = (8, 1)
        lam = lam_of(pi2, exp20, 1)
        cases.append(dict(test="V7", name="M/G/1 APQ 2-class", rho=rho, n=n, c=1, pi=pi2,
                          svc=exp20, apq_w=[8.0, 1.0], runs=[("apq", "apq", None)],
                          ref={"apq": list(apq_kleinrock(lam * np.array(pi2), ES, ES2, [8.0, 1.0]))}))
        # V7b APQ five-class, w = (16, 8, 4, 2, 1)
        ES, ES2 = es(exp20, 5)
        lam = lam_of(pi5, exp20, 1)
        cases.append(dict(test="V7b", name="M/G/1 APQ 5-class", rho=rho, n=n, c=1, pi=pi5,
                          svc=exp20, apq_w=apq5, runs=[("apq", "apq", None)],
                          ref={"apq": list(apq_kleinrock(lam * np.array(pi5), ES, ES2, apq5))}))
    return cases


def _e0_v8(out_dir: Path) -> "pd.DataFrame":
    """V8: empirical confusion matrices (10^6 draws per true level) vs closed form."""
    import pandas as pd
    from triageq.errors import confusion_from_params, predict_levels, predict_from_matrix

    rng, K, n = np.random.default_rng(20260925), 5, 1_000_000
    rows = []
    for sigma, b in [(0.3, 0.0), (0.5, 0.0), (1.0, 0.25), (1.5, -0.25), (2.0, 0.5)]:
        M = confusion_from_params(sigma, b, K)
        M_lv = np.zeros((K, K))
        M_mx = np.zeros((K, K))
        for k in range(1, K + 1):
            true = np.full(n, k)
            p1 = predict_levels(true, rng.standard_normal(n), sigma, b, K)
            p2 = predict_from_matrix(true, rng.uniform(size=n), M)
            M_lv[k - 1] = np.bincount(p1, minlength=K + 1)[1:] / n
            M_mx[k - 1] = np.bincount(p2, minlength=K + 1)[1:] / n
        for sampler, Me in [("ordered-probit", M_lv), ("inverse-CDF", M_mx)]:
            d = float(np.abs(Me - M).max())
            rows.append({"test": "V8", "sigma": sigma, "b": b, "sampler": sampler,
                         "max_abs_diff": d, "pass": d < 0.002})
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "v8_results.parquet", index=False)
    return df


def _e0_welch_worker(rho, c, pi, svc_dict, seed, n, stride, window):
    from triageq.streams import ServiceConfig, make_streams
    from triageq.sim_fast import run_fast
    s = make_streams(n, rho, c, pi, ServiceConfig(**svc_dict), seed=seed)
    out = {}
    for pol in ("fifo", "np_pq"):
        w = run_fast(s, s["true_levels"], pol, c, warmup_frac=0.0)
        out[pol] = w
    out["low"] = np.where(s["true_levels"] == 5, out["np_pq"], np.nan)
    return out


def _e0_welch(cfg: dict, out_dir: Path, n_jobs: int) -> None:
    """F8 data: Welch moving averages of waits by patient index at ρ = 0.95, c = 5."""
    import pandas as pd
    from triageq.streams import replication_seeds

    rho, c, n, R, window = 0.95, 5, 250_000, 20, 2_500
    pi = np.array([0.01, 0.12, 0.42, 0.35, 0.10])
    svc = {"family": "exponential", "mean_min": 20.0, "level_multipliers": None}
    seeds = replication_seeds(R, master_seed=cfg["master_seed"])
    res = _parallel(_e0_welch_worker,
                    [(rho, c, pi, svc, s, n, 50, window) for s in seeds], n_jobs, "Welch")
    df = {"index": np.arange(n)}
    for key in ("fifo", "np_pq", "low"):
        avg = np.nanmean(np.vstack([r[key] for r in res]), axis=0)
        # Centred moving average (Welch 1983), NaN-aware for the level-5 series
        s = pd.Series(avg).rolling(window, center=True, min_periods=window // 5).mean()
        df[key] = s.values
    pd.DataFrame(df).iloc[::50].to_parquet(out_dir / "welch_rho095.parquet", index=False)


def run_e0(cfg: dict, n_jobs: int = -1, engine: str | None = None) -> None:
    """Run experiment E0: validation (V1–V8), plus Welch warm-up data (F8)."""
    import pandas as pd
    from triageq.streams import replication_seeds

    engine  = engine or cfg.get("engine", "simpy")
    n_reps  = cfg["replications"]
    warmup  = cfg["warmup_frac"]
    out_dir = Path(cfg.get("output_dir", "results/e0/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if engine == "fast":
        _warm_numba()

    cases = _e0_cases(cfg)
    seeds = replication_seeds(n_reps, master_seed=cfg["master_seed"])
    print(f"\n{'='*60}\nE0: Validation — {len(cases)} cases × {n_reps} reps, engine={engine}\n{'='*60}")

    tasks = [(case, case["rho"], seed, engine, warmup) for case in cases for seed in seeds]
    flat = _parallel(_e0_worker, tasks, n_jobs, "E0 reps")

    records = []
    for ci, case in enumerate(cases):
        reps = flat[ci * n_reps:(ci + 1) * n_reps]
        K = len(case["pi"])
        base = {"test": case["test"], "case": case["name"], "rho": case["rho"]}

        if case["test"] == "V5":
            ES = np.full(K, 20.0)
            lam = case["rho"] / 20.0
            rho_k = lam * np.asarray(case["pi"]) * ES
            cons = {lab: [float(np.dot(rho_k, r[lab])) for r in reps]
                    for lab in ("fifo", "np_pq", "apq")}
            truth = case["ref"]["conservation"]
            for lab, vals in cons.items():
                m, lo, hi = _ci(vals)
                d_m, d_lo, d_hi = _ci(np.array(vals) - np.array(cons["fifo"]))
                rel = abs(m - truth) / truth
                ok = rel < 0.02 and lo <= truth <= hi
                if lab != "fifo":
                    ok = ok and d_lo <= 0 <= d_hi
                records.append({**base, "run": lab, "level": "Σρ_kW_k", "analytic": truth,
                                "simulated": m, "ci_lo": lo, "ci_hi": hi, "rel_err": rel,
                                "ci_covers": lo <= truth <= hi,
                                "paired_diff_vs_fifo": d_m if lab != "fifo" else math.nan,
                                "paired_diff_ci_lo": d_lo if lab != "fifo" else math.nan,
                                "paired_diff_ci_hi": d_hi if lab != "fifo" else math.nan,
                                "pass": ok})
            continue

        tol = 0.03 if case["test"] in ("V7", "V7b") else 0.02
        for lab, ref in case["ref"].items():
            levels = [1] if case["test"] == "V6" else range(1, len(ref) + 1)
            for k in levels:
                truth = float(ref[k - 1])
                vals = [r[lab][k - 1] for r in reps]
                m, lo, hi = _ci(vals)
                rel = abs(m - truth) / truth if truth > 0 else math.nan
                rec = {**base, "run": lab, "level": str(k) if K > 1 else "all",
                       "analytic": truth, "simulated": m, "ci_lo": lo, "ci_hi": hi,
                       "rel_err": rel, "ci_covers": bool(lo <= truth <= hi),
                       "pass": bool(rel < tol and lo <= truth <= hi)}
                if case["test"] == "V6" and lab != "fifo":
                    s, p = (float(x.split("=")[1]) for x in lab.split(","))
                    d = np.array(vals) - np.array([r["fifo"][0] for r in reps])
                    d_m, d_lo, d_hi = _ci(d)
                    J = s + p - 1
                    sign_ok = (d_hi < 0) if J > 1e-9 else (d_lo > 0) if J < -1e-9 else (d_lo <= 0 <= d_hi)
                    rec.update({"youden_j": J, "paired_diff_vs_fifo": d_m,
                                "paired_diff_ci_lo": d_lo, "paired_diff_ci_hi": d_hi,
                                "sign_ok": bool(sign_ok),
                                "pass": bool(rec["pass"] and sign_ok)})
                records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(out_dir / "validation_results.parquet", index=False)

    v8 = _e0_v8(out_dir)
    print("\nV8 (confusion sampling):")
    print(v8.to_string(index=False))

    print("\nWelch warm-up data (ρ = 0.95, c = 5) ...")
    _e0_welch(cfg, out_dir, n_jobs)

    cols = ["test", "rho", "run", "level", "analytic", "simulated", "rel_err", "ci_covers", "pass"]
    print("\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[E0] V1–V7: {int(df['pass'].sum())}/{len(df)} checks pass; "
          f"V8: {int(v8['pass'].sum())}/{len(v8)}")
    print(f"[E0] Results saved to {out_dir}/")


# ── E1: two-class sweep ───────────────────────────────────────────────────────

def _e1_worker(rho, rep, seed, n, pi2, svc_dict, tau, grid, warmup):
    from triageq.streams import ServiceConfig, make_streams
    from triageq.errors import predict_from_matrix
    from triageq.sim_fast import run_fast

    s = make_streams(n, rho, 1, np.asarray(pi2), ServiceConfig(**svc_dict), seed=seed)
    true = s["true_levels"]

    def metrics(w):
        v = ~np.isnan(w)
        h, l = w[v & (true == 1)], w[v & (true == 2)]
        return {"mean_H": float(h.mean()), "tc_H": float(np.mean(h <= tau[0])),
                "p90_H": float(np.percentile(h, 90)), "p95_H": float(np.percentile(h, 95)),
                "starv_H": float(np.mean(h > 3 * tau[0])), "mean_L": float(l.mean()),
                "p95_L": float(np.percentile(l, 95))}

    rows = [{"rho": rho, "rep": rep, "policy": "fifo", "sensitivity": math.nan,
             "specificity": math.nan, **metrics(run_fast(s, true, "fifo", 1, warmup_frac=warmup))}]
    for sens, spec in grid:
        M2 = np.array([[sens, 1 - sens], [1 - spec, spec]])
        pred = predict_from_matrix(true, s["u"], M2).astype(np.int32)
        rows.append({"rho": rho, "rep": rep, "policy": "np_pq", "sensitivity": sens,
                     "specificity": spec,
                     **metrics(run_fast(s, pred, "np_pq", 1, warmup_frac=warmup))})
    return rows


E1_METRICS = {"mean_H": False, "tc_H": True, "p90_H": False, "p95_H": False,
              "starv_H": False, "mean_L": False, "p95_L": False}


def run_e1(cfg: dict, n_jobs: int = -1, engine: str | None = None) -> None:
    """Run experiment E1: two-class break-even sweep (Figure F2)."""
    import itertools
    import pandas as pd
    from triageq.streams import ServiceConfig, replication_seeds
    from triageq.analytic import misclassified_mean_waits, mg1_pk_wq
    from triageq.breakeven import find_breakeven

    pi2     = resolve_pi(cfg)
    assert len(pi2) == 2, "E1 is a two-class experiment: pi must have two entries"
    rhos    = cfg["rho"]
    n_reps  = cfg["replications"]
    warmup  = cfg["warmup_frac"]
    tau     = np.array(cfg["tau_min"], dtype=np.float64)
    out_dir = Path(cfg.get("output_dir", "results/e1/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    svc     = _svc_dict(cfg["service"])
    grid    = [(round(s, 4), round(p, 4)) for s, p in
               itertools.product(cfg["sensitivity"], cfg["specificity"])]
    seeds   = replication_seeds(n_reps, master_seed=cfg["master_seed"])

    print(f"\n{'='*60}\nE1: Two-class sweep — {len(grid)} (s,p) × {len(rhos)} ρ × {n_reps} reps\n{'='*60}")
    _warm_numba()
    tasks = [(rho, r, seeds[r], _n_patients(cfg, rho), pi2, svc, tau, grid, warmup)
             for rho in rhos for r in range(n_reps)]
    reps = pd.DataFrame([row for rows in _parallel(_e1_worker, tasks, n_jobs, "E1") for row in rows])
    reps.to_parquet(out_dir / "e1_reps.parquet", index=False)

    # Paired deltas vs FIFO per replication
    fifo = reps[reps.policy == "fifo"].set_index(["rho", "rep"])
    tri  = reps[reps.policy == "np_pq"].copy()
    for m in E1_METRICS:
        tri[f"d_{m}"] = tri[m].values - fifo.loc[list(zip(tri.rho, tri.rep)), m].values

    ES, ES2 = ServiceConfig(**svc).ES_and_ES2(2)
    agg = []
    for (rho, s, p), g in tri.groupby(["rho", "sensitivity", "specificity"]):
        lam = rho / float(np.dot(pi2, ES))
        W_tri = misclassified_mean_waits(np.array([[s, 1 - s], [1 - p, p]]), pi2, lam, ES, ES2, c=1)
        rec = {"rho": rho, "sensitivity": s, "specificity": p, "youden_j": round(s + p - 1, 6),
               "analytic_mean_H": float(W_tri[0]),
               "analytic_delta_mean_H": float(W_tri[0]) - mg1_pk_wq(lam, float(np.dot(pi2, ES)),
                                                                     float(np.dot(pi2, ES2)))}
        for m in E1_METRICS:
            rec[m] = float(g[m].mean())
            rec[f"delta_{m}"], rec[f"delta_{m}_lo"], rec[f"delta_{m}_hi"] = _ci(g[f"d_{m}"])
        agg.append(rec)
    agg = pd.DataFrame(agg)
    agg["delta_mean_wait_H"] = agg["delta_mean_H"]      # names used by F2
    agg["delta_tc_H"] = agg["delta_tc_H"]
    agg.to_parquet(out_dir / "e1_results.parquet", index=False)

    # Break-even along slices; x = 1 − s increases as quality falls (like σ)
    be = []
    slices = {"diagonal s=p": None, **{f"p={p:.2f}": p for p in (0.8, 0.9, 1.0)}}
    for rho in rhos:
        for sl_name, p_fix in slices.items():
            g = tri[tri.rho == rho]
            g = g[np.isclose(g.sensitivity, g.specificity)] if p_fix is None \
                else g[np.isclose(g.specificity, p_fix)]
            g = g.sort_values("sensitivity", ascending=False)
            s_vals = np.sort(g.sensitivity.unique())[::-1]
            x = 1.0 - s_vals
            for m, hib in E1_METRICS.items():
                if m.endswith("_L"):
                    continue
                D = np.vstack([g[np.isclose(g.sensitivity, sv)].sort_values("rep")[f"d_{m}"].values
                               for sv in s_vals])
                res = find_breakeven(x, D, metric_higher_is_better=hib, n_boot=1000)
                s_star = 1 - res["sigma_star"]
                p_star = s_star if p_fix is None else p_fix
                be.append({"rho": rho, "slice": sl_name, "metric": m, "status": res["status"],
                           "s_star": s_star, "s_lo": 1 - res["ci_hi"], "s_hi": 1 - res["ci_lo"],
                           "J_star": s_star + p_star - 1 if math.isfinite(s_star) else math.nan})
    be = pd.DataFrame(be)
    be.to_parquet(out_dir / "e1_breakeven.parquet", index=False)
    print("\nBreak-even Youden J* by ρ (NP-PQ vs FIFO, true-H patients):")
    piv = be[be.slice == "diagonal s=p"].pivot(index="rho", columns="metric", values="J_star")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\n[E1] Results saved to {out_dir}/")


# ── E2 / E3: five-level sweep ─────────────────────────────────────────────────

def _e2_worker(rho, c, rep, seed, n, pi, svc_dict, tau, sigmas, biases, policies,
               apq_w, warmup, variant=None):
    """All policies × (σ, b) on ONE set of streams for (ρ, c, replication)."""
    from triageq.streams import ServiceConfig, make_streams
    from triageq.errors import predict_levels
    from triageq.sim_fast import run_fast
    from triageq.metrics import sweep_metrics

    K = len(pi)
    s = make_streams(n, rho, c, np.asarray(pi), ServiceConfig(**svc_dict), seed=seed)
    true = s["true_levels"]
    base = {"rho": rho, "c": c, "rep": rep}
    if variant is not None:
        base["variant"] = variant

    def run(pred, pol):
        return sweep_metrics(run_fast(s, pred, pol, c, apq_weights=apq_w, warmup_frac=warmup),
                             true, tau)

    rows = [{**base, "policy": "fifo", "sigma": math.nan, "b": math.nan,
             **run(true, "fifo")},
            {**base, "policy": "oracle", "sigma": 0.0, "b": 0.0, **run(true, "np_pq")},
            {**base, "policy": "oracle_apq", "sigma": 0.0, "b": 0.0, **run(true, "apq")}]
    for b in biases:
        for sigma in sigmas:
            pred = predict_levels(true, s["eps"], float(sigma), float(b), K).astype(np.int32)
            for pol in policies:
                rows.append({**base, "policy": pol, "sigma": round(float(sigma), 4),
                             "b": float(b), **run(pred, pol)})
    return rows


def _sigma_grid(sc: dict) -> np.ndarray:
    """Uniform σ grid from start/stop/step, plus optional `extra` values beyond it."""
    sigmas = np.arange(sc["start"], sc["stop"] + sc["step"] / 2, sc["step"])
    sigmas = np.concatenate([sigmas, np.asarray(sc.get("extra", []), dtype=np.float64)])
    return np.unique(np.round(sigmas, 4))


def _paired_deltas(reps, keys):
    """Attach per-replication deltas vs FIFO for every sweep metric."""
    from triageq.metrics import SWEEP_METRICS
    fifo = reps[reps.policy == "fifo"].set_index(keys + ["rep"])
    tri  = reps[reps.policy != "fifo"].copy()
    idx  = list(zip(*[tri[k] for k in keys + ["rep"]]))
    for m in SWEEP_METRICS:
        tri[f"d_{m}"] = tri[m].values - fifo.loc[idx, m].values
    return tri


def _summarise_cells(tri, keys):
    from triageq.metrics import SWEEP_METRICS
    out = []
    for key, g in tri.groupby(keys + ["policy", "b", "sigma"], dropna=False):
        rec = dict(zip(keys + ["policy", "b", "sigma"], key))
        for m in SWEEP_METRICS:
            rec[m] = float(g[m].mean())
            rec[f"d_{m}"], rec[f"d_{m}_lo"], rec[f"d_{m}_hi"] = _ci(g[f"d_{m}"])
        rec["n_reps"] = len(g)
        out.append(rec)
    import pandas as pd
    return pd.DataFrame(out)


def _breakeven_table(tri, keys, pi, policies, metrics, n_boot=1000):
    """σ* with bootstrap CI for every (keys, policy, b, metric)."""
    import pandas as pd
    from triageq.metrics import SWEEP_METRICS
    from triageq.breakeven import find_breakeven
    from triageq.errors import confusion_from_params, summary_stats

    K = len(pi)
    out = []
    sub = tri[tri.policy.isin(policies)]
    for key, g in sub.groupby(keys + ["policy", "b"]):
        sigmas = np.sort(g.sigma.unique())
        piv = {m: g.pivot(index="sigma", columns="rep", values=f"d_{m}").loc[sigmas].values
               for m in metrics}
        for m in metrics:
            res = find_breakeven(sigmas, piv[m], metric_higher_is_better=SWEEP_METRICS[m],
                                 n_boot=n_boot)
            rec = {**dict(zip(keys + ["policy", "b"], key)), "metric": m,
                   "status": res["status"], "sigma_star": res["sigma_star"],
                   "sigma_lo": res["ci_lo"], "sigma_hi": res["ci_hi"]}
            b = float(key[-1])
            for tag, sg in (("", res["sigma_star"]), ("_at_sigma_lo", res["ci_lo"]),
                            ("_at_sigma_hi", res["ci_hi"])):
                if math.isfinite(sg):
                    st = summary_stats(confusion_from_params(sg, b, K), pi, K)
                    for k in (("exact_acc", "within1_acc", "undertriage", "overtriage",
                               "ha_sens", "ha_spec", "youden_j", "qwk", "severe_under_12") if tag == "" else
                              ("exact_acc", "undertriage", "ha_sens", "youden_j", "severe_under_12")):
                        rec[f"{k}{tag}"] = st[k]
            out.append(rec)
    return pd.DataFrame(out)


def run_e2(cfg: dict, n_jobs: int = -1, engine: str | None = None) -> None:
    """Run experiment E2: five-level core break-even sweep."""
    import pandas as pd
    from triageq.streams import replication_seeds
    from triageq.metrics import SWEEP_METRICS

    pi      = resolve_pi(cfg)
    rhos    = cfg["rho"]
    n_reps  = cfg["replications"]
    warmup  = cfg["warmup_frac"]
    out_dir = Path(cfg.get("output_dir", "results/e2/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sc      = cfg["sigma"]
    sigmas  = _sigma_grid(sc)
    biases  = cfg["bias"]
    servers = cfg["servers"]
    policies = [p for p in cfg["policies"] if p not in ("fifo", "oracle")]
    apq_w   = [float(x) for x in cfg["apq_weights"]]
    tau     = np.array(cfg["tau_min"], dtype=np.float64)
    svc     = _svc_dict(cfg["service"], pi)
    seeds   = replication_seeds(n_reps, master_seed=cfg["master_seed"])

    n_cells = len(sigmas) * len(biases) * len(rhos) * len(servers) * len(policies)
    print(f"\n{'='*60}\nE2: Five-level sweep — {n_cells:,} cells × {n_reps} reps")
    print(f"    {len(sigmas)} σ × {len(biases)} b × {len(rhos)} ρ × {len(servers)} c × {policies}\n{'='*60}")

    if cfg.get("post_only"):
        reps = pd.read_parquet(out_dir / "e2_reps.parquet")
        print("[E2] --post-only: rebuilding tables from saved e2_reps.parquet")
    else:
        _warm_numba()
        tasks = [(rho, c, r, seeds[r], _n_patients(cfg, rho), pi, svc, tau, sigmas, biases,
                  policies, apq_w, warmup) for rho in rhos for c in servers for r in range(n_reps)]
        reps = pd.DataFrame([row for rows in _parallel(_e2_worker, tasks, n_jobs, "E2 (ρ,c,rep)")
                             for row in rows])
        reps.to_parquet(out_dir / "e2_reps.parquet", index=False)

    keys = ["rho", "c"]
    tri = _paired_deltas(reps, keys)
    cells = _summarise_cells(tri, keys)
    fifo_mean = reps[reps.policy == "fifo"].groupby(keys)[list(SWEEP_METRICS)].mean().reset_index()
    fifo_mean["policy"] = "fifo"
    pd.concat([fifo_mean, cells], ignore_index=True).to_parquet(out_dir / "e2_raw.parquet", index=False)
    print(f"\n[E2] Per-replication rows → {out_dir}/e2_reps.parquet; cell means → e2_raw.parquet")

    be = _breakeven_table(tri, keys, pi, policies, list(SWEEP_METRICS))
    be.to_parquet(out_dir / "e2_breakeven.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps({"pi": list(map(float, pi)),
                                                   "tau": list(map(float, tau)),
                                                   "apq_weights": apq_w}, indent=2))
    tab = Path("results/tables")
    tab.mkdir(parents=True, exist_ok=True)
    be.to_csv(tab / "e2_breakeven_all.csv", index=False, float_format="%.4f")

    print(f"[E2] Break-even table → {out_dir}/e2_breakeven.parquet  (+ results/tables/e2_breakeven_all.csv)")
    print("\nBreak-even status counts by metric:")
    print(be.groupby(["metric", "status"]).size().unstack(fill_value=0).to_string())
    show = be[(be.metric == "p95_12") & (be.b == 0.0)]
    print("\np95 wait (true levels 1–2) break-even, b = 0:")
    print(show[["policy", "c", "rho", "status", "sigma_star", "sigma_lo", "sigma_hi",
                "youden_j", "ha_sens", "exact_acc"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


def run_e3(cfg: dict, n_jobs: int = -1, engine: str | None = None) -> None:
    """Run experiment E3: robustness across service variants and APQ weights."""
    import pandas as pd
    from triageq.streams import replication_seeds
    from triageq.metrics import SWEEP_METRICS

    pi      = resolve_pi(cfg)
    rhos    = cfg["rho"]
    n_reps  = cfg["replications"]
    warmup  = cfg["warmup_frac"]
    out_dir = Path(cfg.get("output_dir", "results/e3/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sc      = cfg["sigma"]
    sigmas  = _sigma_grid(sc)
    biases  = cfg["bias"]
    servers = cfg["servers"]
    policies = [p for p in cfg["policies"] if p not in ("fifo", "oracle")]
    tau     = np.array(cfg["tau_min"], dtype=np.float64)
    seeds   = replication_seeds(n_reps, master_seed=cfg["master_seed"])

    w_default = [float(x) for x in cfg["apq_weights"]]
    base_svc = cfg["service_variants"][0]
    variants = []                                   # (name, service dict, APQ weights, π)
    for v in cfg["service_variants"]:
        variants.append((v["name"], _svc_dict(v, pi), w_default, pi))
    for v in cfg.get("apq_weight_variants", []):
        variants.append((v["name"], _svc_dict(base_svc, pi), [float(x) for x in v["weights"]], pi))
    for v in cfg.get("pi_variants", []):
        pv = _pi_from(v["pi"])
        variants.append((v["name"], _svc_dict(base_svc, pv), w_default, pv))

    print(f"\nE3: Robustness — {len(variants)} variants × {len(rhos)} ρ × {len(servers)} c × {n_reps} reps")
    _warm_numba()
    tasks = [(rho, c, r, seeds[r], _n_patients(cfg, rho), vpi, svc, tau, sigmas, biases,
              policies, w, warmup, name)
             for name, svc, w, vpi in variants for rho in rhos for c in servers for r in range(n_reps)]
    reps = pd.DataFrame([row for rows in _parallel(_e2_worker, tasks, n_jobs, "E3")
                         for row in rows])
    reps.to_parquet(out_dir / "e3_reps.parquet", index=False)

    keys = ["variant", "rho", "c"]
    tri = _paired_deltas(reps, keys)
    _summarise_cells(tri, keys).to_parquet(out_dir / "e3_raw.parquet", index=False)
    # Interpret σ* with each variant's own acuity mix
    be = pd.concat([_breakeven_table(tri[tri.variant == name], keys, vpi, policies,
                                     ["tc12", "p95_12", "mean12"])
                    for name, _, _, vpi in variants], ignore_index=True)
    be.to_parquet(out_dir / "e3_breakeven.parquet", index=False)
    show = be[be.metric == "p95_12"]
    print("p95 wait (true levels 1–2) break-even by variant:")
    print(show[["variant", "policy", "c", "rho", "status", "sigma_star", "youden_j"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[E3] Results saved → {out_dir}/")


# ── E4: real operating points (plug-in confusion matrices) ────────────────────

def _e4_worker(rho, c, rep, seed, n, pi, svc_dict, tau, mats, policies, apq_w, warmup):
    """FIFO, oracle and every plug-in confusion matrix × policy on ONE set of streams."""
    from triageq.streams import ServiceConfig, make_streams
    from triageq.errors import predict_from_matrix
    from triageq.sim_fast import run_fast
    from triageq.metrics import sweep_metrics

    s = make_streams(n, rho, c, np.asarray(pi), ServiceConfig(**svc_dict), seed=seed)
    true = s["true_levels"]
    base = {"rho": rho, "c": c, "rep": rep}

    def run(pred, pol):
        return sweep_metrics(run_fast(s, pred, pol, c, apq_weights=apq_w, warmup_frac=warmup),
                             true, tau)

    rows = [{**base, "point": "fifo", "policy": "fifo", **run(true, "fifo")}]
    for pol in policies:
        rows.append({**base, "point": "oracle", "policy": pol, **run(true.astype(np.int32), pol)})
    for name, M in mats.items():
        pred = predict_from_matrix(true, s["u"], np.asarray(M)).astype(np.int32)
        for pol in policies:
            rows.append({**base, "point": name, "policy": pol, **run(pred, pol)})
    return rows


def run_e4(cfg: dict, n_jobs: int = -1, engine: str | None = None) -> None:
    """Run experiment E4: nurse, ML and ML δ-curve operating points in the queue."""
    import pandas as pd
    from triageq.streams import replication_seeds
    from triageq.metrics import SWEEP_METRICS

    pi      = resolve_pi(cfg)
    rhos    = cfg["rho"]
    n_reps  = cfg["replications"]
    warmup  = cfg["warmup_frac"]
    servers = cfg["servers"]
    policies = [p for p in cfg["policies"] if p != "fifo"]
    apq_w   = [float(x) for x in cfg["apq_weights"]]
    tau     = np.array(cfg["tau_min"], dtype=np.float64)
    svc     = _svc_dict(cfg["service"], pi)
    out_dir = Path(cfg.get("output_dir", "results/e4/"))
    seeds   = replication_seeds(n_reps, master_seed=cfg["master_seed"])

    ops = json.loads((out_dir / "operating_points.json").read_text())["operating_points"]
    curve = json.loads((out_dir / "delta_curve.json").read_text())
    mats = {name: ops[name]["matrix"] for name in cfg["operating_points"]}
    mats.update({f"delta_{c['delta']:+.2f}": c["matrix"] for c in curve})

    print(f"\n{'='*60}\nE4: {len(mats)} operating points × {policies} × {len(rhos)} ρ × c={servers} × {n_reps} reps\n{'='*60}")
    if cfg.get("post_only"):
        reps = pd.read_parquet(out_dir / "e4_reps.parquet")
        print("[E4] --post-only: rebuilding tables from saved e4_reps.parquet")
    else:
        _warm_numba()
        tasks = [(rho, c, r, seeds[r], _n_patients(cfg, rho), pi, svc, tau, mats, policies, apq_w, warmup)
                 for rho in rhos for c in servers for r in range(n_reps)]
        reps = pd.DataFrame([row for rows in _parallel(_e4_worker, tasks, n_jobs, "E4") for row in rows])
        reps.to_parquet(out_dir / "e4_reps.parquet", index=False)

    # Paired deltas vs FIFO
    fifo = reps[reps.point == "fifo"].set_index(["rho", "c", "rep"])
    tri = reps[reps.point != "fifo"].copy()
    idx = list(zip(tri.rho, tri.c, tri.rep))
    for m in SWEEP_METRICS:
        tri[f"d_{m}"] = tri[m].values - fifo.loc[idx, m].values
    summ = []
    for key, g in tri.groupby(["rho", "c", "point", "policy"]):
        rec = dict(zip(["rho", "c", "point", "policy"], key))
        for m in SWEEP_METRICS:
            rec[m] = float(g[m].mean())
            rec[f"d_{m}"], rec[f"d_{m}_lo"], rec[f"d_{m}_hi"] = _ci(g[f"d_{m}"])
        summ.append(rec)
    fifo_mean = reps[reps.point == "fifo"].groupby(["rho", "c"])[list(SWEEP_METRICS)].mean().reset_index()
    fifo_mean["point"], fifo_mean["policy"] = "fifo", "fifo"
    summ = pd.concat([fifo_mean, pd.DataFrame(summ)], ignore_index=True)
    summ.to_parquet(out_dir / "e4_summary.parquet", index=False)

    # Break-even gap vs the E2 surface (p95 of true levels 1–2), in two currencies:
    #   gap_j  = J(op) − J*               (HA Youden J; misleading for real matrices,
    #                                      whose errors are mostly adjacent)
    #   gap_su = SU* − SU(op)             (severe undertriage P(pred ≥ 4 | true ≤ 2);
    #                                      positive = fewer severe errors than the bar)
    from triageq.errors import summary_stats
    be_path = Path("results/e2/e2_breakeven.parquet")
    gap_rows = []
    if be_path.exists():
        be = pd.read_parquet(be_path)
        be = be[be.metric == "p95_12"]
        biases = np.array(sorted(be.b.unique()))
        for name in cfg["operating_points"]:
            op = ops[name]
            su_op = summary_stats(np.array(op["matrix"]), pi)["severe_under_12"]
            b_near = float(biases[np.argmin(np.abs(biases - op["b_fit"]))])
            for (rho, c, pol), g in be[be.b == b_near].groupby(["rho", "c", "policy"]):
                if c not in servers:
                    continue
                r = g.iloc[0]
                sim = summ[(summ.rho == rho) & (summ.c == c) & (summ.point == name) & (summ.policy == pol)]
                gap_rows.append({
                    "point": name, "rho": rho, "c": c, "policy": pol, "b_fit": op["b_fit"],
                    "b_surface": b_near, "sigma_fit": op["sigma_fit"],
                    "youden_j_at_pi": op["youden_j_at_pi"], "breakeven_status": r.status,
                    "breakeven_j": r.youden_j if r.status == "crossing" else np.nan,
                    "gap_j": op["youden_j_at_pi"] - r.youden_j if r.status == "crossing" else np.inf,
                    "severe_under_12_op": su_op,
                    "breakeven_severe_under_12": r.severe_under_12 if r.status == "crossing" else np.nan,
                    "gap_su": r.severe_under_12 - su_op if r.status == "crossing" else np.inf,
                    "sim_d_p95_12": float(sim.d_p95_12.iloc[0]) if len(sim) else np.nan,
                    "sim_d_p95_12_lo": float(sim.d_p95_12_lo.iloc[0]) if len(sim) else np.nan,
                    "sim_d_p95_12_hi": float(sim.d_p95_12_hi.iloc[0]) if len(sim) else np.nan,
                })
    gaps = pd.DataFrame(gap_rows)
    gaps.to_parquet(out_dir / "e4_breakeven_gap.parquet", index=False)

    # Queue-aware δ: best high-acuity p95 with low-acuity p95 ≤ cap × FIFO's
    cap = float(cfg.get("harm_cap_ratio", 1.5))
    dsum = summ[summ.point.str.startswith("delta_")].copy()
    dsum["delta"] = dsum.point.str.replace("delta_", "").astype(float)
    sel = []
    for (rho, c, pol), g in dsum.groupby(["rho", "c", "policy"]):
        f = fifo_mean[(fifo_mean.rho == rho) & (fifo_mean.c == c)].iloc[0]
        ok = g[g.p95_45 <= cap * f.p95_45]
        best_unc = g.loc[g.p95_12.idxmin()]
        best_cap = ok.loc[ok.p95_12.idxmin()] if len(ok) else None
        best_tc  = (ok if len(ok) else g).loc[(ok if len(ok) else g).tc12.idxmax()]
        sel.append({"rho": rho, "c": c, "policy": pol, "harm_cap_ratio": cap,
                    "delta_min_p95_12": best_unc.delta, "p95_12_at_that": best_unc.p95_12,
                    "p95_45_at_that": best_unc.p95_45,
                    "cap_feasible": best_cap is not None,
                    "delta_capped": best_cap.delta if best_cap is not None else np.nan,
                    "p95_12_capped": best_cap.p95_12 if best_cap is not None else np.nan,
                    "p95_45_capped": best_cap.p95_45 if best_cap is not None else np.nan,
                    "delta_max_tc12_capped": best_tc.delta,
                    "fifo_p95_12": f.p95_12, "fifo_p95_45": f.p95_45})
    sel = pd.DataFrame(sel)
    sel.to_parquet(out_dir / "e4_queue_aware_delta.parquet", index=False)
    tab = Path("results/tables")
    tab.mkdir(parents=True, exist_ok=True)
    sel.to_csv(tab / "T7_queue_aware_delta.csv", index=False, float_format="%.3f")

    show = summ[summ.point.isin(["nurse", "ml", "oracle"]) & (summ.policy == "np_pq")]
    print("\nNP-PQ vs FIFO, Δ p95 wait of true levels 1–2 (min) [95% CI]:")
    print(show.assign(v=show.d_p95_12.map("{:+.1f}".format) + " [" + show.d_p95_12_lo.map("{:+.1f}".format)
                      + ", " + show.d_p95_12_hi.map("{:+.1f}".format) + "]")
          .pivot_table(index="point", columns="rho", values="v", aggfunc="first").to_string())
    if len(gaps):
        for col, lab in (("gap_j", "HA Youden J of operating point − break-even J*"),
                         ("gap_su", "break-even severe undertriage − operating point's")):
            print(f"\nBreak-even gap ({lab}), NP-PQ:")
            print(gaps[gaps.policy == "np_pq"].pivot_table(index="point", columns="rho", values=col)
                  .to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\nQueue-aware δ (NP-PQ):")
    print(sel[sel.policy == "np_pq"][["rho", "delta_min_p95_12", "cap_feasible", "delta_capped",
                                      "p95_12_capped", "fifo_p95_12"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n[E4] Results saved → {out_dir}/")


def run_figures(cfg: dict, n_jobs: int = -1, engine: str | None = None) -> None:
    """Generate all figures and tables from saved experiment results."""
    from triageq.figures import generate_all
    results_dir = cfg.get("results_dir", "results/")
    output_dir  = cfg.get("output_dir",  "results/figures/")
    generate_all(results_dir, output_dir)


RUNNERS = {
    "e0": run_e0,
    "e1": run_e1,
    "e2": run_e2,
    "e3": run_e3,
    "e4": run_e4,
    "figures": run_figures,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="triageq experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all CPUs)")
    parser.add_argument("--engine", choices=["simpy", "fast"], default=None,
                        help="Override the config's simulation engine (E0 only)")
    parser.add_argument("--post-only", action="store_true",
                        help="E2/E4: rebuild summary and break-even tables from saved *_reps.parquet")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg["post_only"] = args.post_only
    exp = cfg.get("experiment", "").lower()

    t0 = time.time()
    if exp in RUNNERS:
        RUNNERS[exp](cfg, n_jobs=args.jobs, engine=args.engine)
    else:
        print(f"[run] Experiment '{exp}' not recognised. Available: {list(RUNNERS)}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n[run] Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
