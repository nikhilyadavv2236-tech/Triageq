# Where each number in the draft comes from

| Paper item | Source |
|---|---|
| Table I (validation), "81 checks, ≤1.4%, 80/81 cover" | `results/e0/validation_results.parquet`, `results/tables/T2_validation_compact.csv`, `T2b_v8_confusion_sampling.csv` |
| 5-level analytic cross-check (median 0.55%, p95 1.7%, 2,380 cells) | `results/tables/E2_analytic_crosscheck.csv` |
| Fig. 1, two-class J* (mean ≈0; p95 0.06→0.35; TC +0.004–0.024 at J=−0.4) | `results/e1/e1_breakeven.parquet`, `results/e1/e1_results.parquet` |
| Five-level J*, SU*, bias effect, TC₁₂ at σ=20 | `results/e2/e2_breakeven.parquet`, `results/e2/e2_raw.parquet` (π from data) |
| Robustness range 0.190–0.214 | `results/e3/e3_breakeven.parquet`, `results/tables/T6_e3_robustness.csv` |
| Data flow (108,180 / 3,336 excluded / 99.2% labelled), π | `results/tables/T5_data_flow.csv`, `data/processed/pi_resi_o.json` |
| Table III (classifiers), ECE 0.023 → 0.007 | `results/tables/T3_classifier_performance.csv`, `results/e4/operating_points.json` |
| SU of operating points (0.04 / 0.12 / 0.06) | `summary_stats(matrix, π)` on `results/e4/operating_points.json` |
| Table II Δp95 columns, harm numbers, APQ comparison | `results/e4/e4_summary.parquet` |
| Queue-aware δ (−0.2 / +0.6 / APQ −0.1) | `results/tables/T7_queue_aware_delta.csv` |

Figures: `figures/F2_…` = Fig. 1, `figures/F3_breakeven_p95_12_c5_severe_under_12` = Fig. 2,
`figures/F5_tradeoff` = Fig. 3. `F6_confusion_matrices` is included in the folder but not
used in the text (candidate for the paper if space allows, otherwise supplementary).
Regenerate all with `python -m triageq.figures`, then copy from `results/figures/`.
