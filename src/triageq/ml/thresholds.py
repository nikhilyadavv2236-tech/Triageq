"""
triageq.ml.thresholds — Queue-aware threshold selection (Milestone 10).

The LightGBM expected-urgency score s = Σ_k k·P(level = k) (prior-corrected
probabilities) is cut into five levels with the validation-tuned cut-points
c₁ < … < c₄ shifted by a single offset δ:

    level = 1 + #{ i : c_i − δ ≤ s }

δ > 0 moves every cut-point down, so predictions become LESS urgent
(undertriage, same sign convention as the bias b of the error model);
δ < 0 gives overtriage. δ ∈ [−1, 1], step 0.05.

For each δ the test-year 5×5 confusion matrix is saved; E4 feeds these into
the simulator's plug-in mode and picks, at each ρ, the δ that best serves
high-acuity patients subject to a cap on low-acuity harm.

Usage:
    python -m triageq.ml.thresholds
Output: results/e4/delta_curve.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 5


def compute_operating_curve(score: np.ndarray, y: np.ndarray, base_cuts: np.ndarray,
                            delta_range=(-1.0, 1.0), step: float = 0.05) -> list[dict]:
    """Sweep δ and return one confusion matrix (counts and row-normalised) per δ."""
    from triageq.ml.train import cut_predict
    deltas = np.round(np.arange(delta_range[0], delta_range[1] + step / 2, step), 4)
    out = []
    for d in deltas:
        p = cut_predict(score, np.asarray(base_cuts) - d)
        C = np.bincount((y - 1) * K + (p - 1), minlength=K * K).reshape(K, K).astype(float)
        M = C / np.maximum(C.sum(1, keepdims=True), 1)
        out.append({"delta": float(d), "counts": C.astype(int).tolist(), "matrix": M.tolist()})
    return out


def main() -> None:
    from triageq.errors import summary_stats
    d = pd.read_parquet("data/processed/ml_predictions.parquet")
    te = d[d["split"] == "test"]
    meta = json.loads(Path("models/meta.json").read_text())
    curve = compute_operating_curve(te["score_lgbm"].to_numpy(), te["resi_o"].to_numpy().astype(int),
                                    np.array(meta["lgbm_base_cuts"]))
    pi = np.array(list(json.loads(Path("data/processed/pi_resi_o.json").read_text()).values()))
    for c in curve:
        c.update({k: float(v) for k, v in summary_stats(np.array(c["matrix"]), pi).items()})
    out = Path("results/e4")
    out.mkdir(parents=True, exist_ok=True)
    (out / "delta_curve.json").write_text(json.dumps(curve, indent=2))
    print(pd.DataFrame(curve)[["delta", "exact_acc", "undertriage", "overtriage", "ha_sens",
                               "ha_spec", "youden_j"]].iloc[::4].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
