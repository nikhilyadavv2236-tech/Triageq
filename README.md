# triageq — When Is Imperfect Triage Better Than None?

A Python package implementing the simulation, analytic models, and ML pipeline for the DASA'26 paper:

> **"When Is Imperfect Triage Better Than None? Load-Dependent Break-Even Accuracy for ML Triage in Emergency Priority Queues"**

## Structure

```
triageq/
  configs/            # YAML experiment configs
  data/raw/           # NHAMCS files (git-ignored)
  data/processed/     # cleaned parquet per year + pooled
  src/triageq/
    errors.py         # ordinal error model, confusion matrices, (σ,b) fitting
    analytic.py       # Cobham, Erlang-C, P-K, Kleinrock APQ, misclassified mean waits
    streams.py        # CRN: pre-drawn arrivals, services, true levels, eps, u
    sim_simpy.py      # reference simulator (FIFO, NP-PQ, APQ, P-PQ, abandonment)
    sim_fast.py       # Numba kernel (FIFO, NP-PQ, APQ; non-preemptive, c servers)
    metrics.py        # per-level waits, TC, percentiles, harm, weighted cost
    breakeven.py      # sign-change search, interpolation, bootstrap CI, conversion
    run.py            # CLI: python -m triageq.run --config configs/e2.yaml
    data/
      nhamcs_load.py  # per-year read + variable-name mapping
      reference.py    # rESI-O and NHAMCS-ESI label construction
      features.py     # triage-time feature matrix, leakage guard
    ml/
      train.py        # LR, LightGBM, ordinal LightGBM; temporal split; tuning
      evaluate.py     # classifier metrics + bootstrap CIs, confusion matrices
      thresholds.py   # delta-shifted cut-points → operating curve
    figures.py        # F1–F11, T1–T6
  tests/
  results/            # parquet per experiment, figures/, tables/
  Makefile
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e src/

# 3. Run fast tests (< 60 s)
make test

# 4. Run all validation experiments (E0: V1–V8)
make validate

# 5. Run two-class sweep (E1)
make e1

# 6. Run full core sweep (E2, ~3-8 CPU-hours)
make sweep
```

## Milestones

| # | Description | Status |
|---|---|---|
| 1 | Setup: repo, env, Makefile, fast tests | ✅ |
| 2 | Error model (errors.py) | ✅ |
| 3 | Analytic module (analytic.py) | ✅ (APQ and M/M/c formulas corrected) |
| 4 | SimPy simulator (sim_simpy.py) | ✅ |
| 5 | Numba kernel (sim_fast.py) | ✅ O(N·K) rewrite: 250k patients in ~15 ms; K1 passes |
| 6 | Full validation (E0, V1–V8) | ✅ SimPy engine, 30 reps: all 81 rel. errors < tol; 2 chance CI misses; V8 10/10 |
| 7 | Two-class E1 | ✅ mean-wait J* ≈ 0 at every ρ; p95 J* rises 0.06 → 0.35 |
| 8 | Core sweep E2 | ✅ σ grid extended to 20 (tail crossings lie beyond σ = 2) |
| 9 | Data pipeline (NHAMCS) | ✅ 108,180 visits (2016–19, 2021–22); rESI-O for 99.2% of kept visits; T5 |
| 10 | Classifier (LR, LightGBM) | ✅ LR / LightGBM / ordinal LightGBM; T3, F6, F10; nurse & ML (σ, b) |
| 11 | E3/E4/E5 + figures | ✅ E3 (incl. nurse-π variant), E4 (nurse, ML, δ-curve); E5 and P-PQ not run |
| 12 | Full reproducibility | ◐ every step scripted; `make` not available on this machine |

## Running on this machine

Windows Application Control blocks venv launchers, so use the uv-managed 3.11
interpreter directly with packages installed to a target directory:

```powershell
$py = uv python find 3.11
uv pip install --python $py --target $env:LOCALAPPDATA\triageq-site -r requirements.txt
$env:PYTHONPATH = "$env:LOCALAPPDATA\triageq-site;$PWD\src"
& $py -m triageq.run --config configs/e0.yaml --jobs 8   # ~40 min (SimPy reference engine)
& $py -m triageq.run --config configs/e1.yaml            # ~5 min
& $py -m triageq.run --config configs/e2.yaml            # ~10 min on 8 workers
& $py -m triageq.run --config configs/e3.yaml

# Data + classifier (Stata files from ftp.cdc.gov/.../NHAMCS/stata/ED{year}-stata.zip,
# unzipped into data/raw/nhamcs/)
& $py -m triageq.data.nhamcs_load        # canonical columns via configs/nhamcs_variables.yaml
& $py -m triageq.data.reference          # rESI-O + NHAMCS-ESI labels, π, T5
& $py -m triageq.ml.train                # ~45 min (80 LightGBM configs)
& $py -m triageq.ml.evaluate             # T3, confusion matrices, (σ, b), calibration
& $py -m triageq.ml.thresholds           # δ-curve
& $py -m triageq.run --config configs/e4.yaml
& $py -m triageq.figures
```

`--post-only` (E2, E4) rebuilds summary/break-even tables from the saved
`*_reps.parquet` without re-simulating.

Note: E2/E3/E4 use `pi: from_data` (survey-weighted rESI-O mix). Results with the
original placeholder π are kept in `results/e2_placeholder_pi/` and `results/e3_placeholder_pi/`.

Measured runtimes (8 cores): E0 40 min, E1 12 min (3 workers), E2 14 min (8 workers), E3 ~5 min.

## Key Design Decisions

- **Common Random Numbers (CRN):** All policies share the same pre-drawn arrival times, service times, true levels, and noise ε. This makes break-even curves smooth and paired CIs tight.
- **Two engines:** SimPy for correctness (reference, extensions), Numba for speed (sweeps). `test_kernel_equivalence` (K1) ensures they give identical per-patient waits.
- **Metric of record:** Target compliance (TC₁₂ = P(wait ≤ τₖ) for true levels 1–2), not mean wait. Mean wait is load-independent at break-even (Proposition in Section 5.1); TC is the clinically meaningful metric.
- **Master seed:** `20260925`. All replications derived via `numpy.random.SeedSequence`.

## References

- Argon & Ziya (2009) MSOM: priority under imperfect type identification
- Sun, Argon & Ziya (2022) POMS: when to triage under errors
- Cobham (1954): non-preemptive priority mean waits
- Kleinrock (1964, 1965): conservation law, delay-dependent priority
- Stanford, Taylor & Ziedins (2014): accumulating priority queue for EDs
