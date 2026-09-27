"""
triageq.ml.train — Classifier training: LR, LightGBM, ordinal LightGBM (Milestone 10).

Temporal split:
  Train: 2016–2018
  Validation: 2019  (hyperparameter tuning by QWK, cut-point tuning)
  Test: 2021–2022   (touched only by evaluate.py)

Hospital codes restart each year, so a hospital is (year, HOSPCODE) = `hosp_id`.
LR's C is chosen by GroupKFold over hospitals within the training years; the
LightGBM models are tuned on the 2019 validation year (different hospitals by
construction), so neither tuning step leaks sites.

Models:
  LR   : multinomial logistic regression, median impute + standardise, balanced weights
  LGBM : LightGBM multiclass, balanced class weights, random search (≤ 40 configs)
  OLGBM: LightGBM regression on the level with 4 cut-points tuned on validation QWK

Class weights distort predicted probabilities; `prior_correct` divides them back
out (p_k ∝ p̃_k / w_k), which restores probabilities for the true class mix.

Usage:
    python -m triageq.ml.train
Outputs: models/*.joblib, data/processed/ml_predictions.parquet
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_YEARS = {"train": [2016, 2017, 2018], "val": [2019], "test": [2021, 2022]}
LABELS_PATH = Path("data/processed/nhamcs_with_labels.parquet")
MODEL_DIR   = Path("models")
PRED_PATH   = Path("data/processed/ml_predictions.parquet")
SEED        = 20260925
K           = 5


def tuning_folds(train: pd.DataFrame, n_splits: int = 5):
    """GroupKFold splits over hospitals (hosp_id) within the training years."""
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n_splits).split(train, groups=train["hosp_id"]))


def class_weights(y: np.ndarray) -> np.ndarray:
    """Balanced class weights w_k = n / (K · n_k) for labels 0..K-1."""
    counts = np.bincount(y, minlength=K).astype(float)
    return len(y) / (K * counts)


def prior_correct(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Undo class weighting: p_k ∝ p̃_k / w_k."""
    Q = P / w[None, :]
    return Q / Q.sum(axis=1, keepdims=True)


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def cut_predict(score: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Level 1..K from a continuous urgency score and K-1 ascending cut-points."""
    return 1 + np.searchsorted(np.sort(cuts), score, side="right")


def tune_cuts(score: np.ndarray, y: np.ndarray, init=(1.5, 2.5, 3.5, 4.5)) -> np.ndarray:
    """Cut-points maximising QWK on (score, y); coordinate search then Nelder-Mead."""
    from scipy.optimize import minimize
    cuts = np.array(init, dtype=float)
    for _ in range(3):                        # coarse coordinate search
        for i in range(len(cuts)):
            grid = np.linspace(cuts[i] - 1.0, cuts[i] + 1.0, 41)
            vals = [qwk(y, cut_predict(score, np.r_[cuts[:i], g, cuts[i + 1:]])) for g in grid]
            cuts[i] = grid[int(np.argmax(vals))]
        cuts = np.sort(cuts)
    res = minimize(lambda c: -qwk(y, cut_predict(score, c)), cuts, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
    return np.sort(res.x)


def load_split():
    from triageq.data.features import build_features, top_rfv_groups
    df = pd.read_parquet(LABELS_PATH)
    df = df[df["resi_o"].notna()].reset_index(drop=True)
    tr = df["survey_year"].isin(SPLIT_YEARS["train"])
    groups = top_rfv_groups(df[tr], n=100)
    X = build_features(df, groups)
    y = df["resi_o"].astype(int).to_numpy() - 1           # 0..4
    split = np.select([tr, df["survey_year"].isin(SPLIT_YEARS["val"])], ["train", "val"], "test")
    return df, X, y, split, groups


def _fit_lr(X, y, df_tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    def make(C):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    best = (-np.inf, None)
    for C in (0.01, 0.1, 1.0):
        scores = []
        for fit_i, val_i in tuning_folds(df_tr, n_splits=5):
            m = make(C).fit(X.iloc[fit_i], y[fit_i])
            scores.append(qwk(y[val_i], m.predict(X.iloc[val_i])))
        print(f"    LR C={C}: GroupKFold QWK = {np.mean(scores):.4f}")
        best = max(best, (float(np.mean(scores)), C))
    return make(best[1]).fit(X, y), {"C": best[1], "cv_qwk": best[0]}


def _lgbm_params(rng) -> dict:
    return {
        "num_leaves":        int(rng.choice([15, 31, 63, 127])),
        "learning_rate":     float(rng.choice([0.02, 0.05, 0.1])),
        "min_child_samples": int(rng.choice([20, 50, 100, 200])),
        "feature_fraction":  float(rng.choice([0.5, 0.7, 0.9])),
        "bagging_fraction":  float(rng.choice([0.7, 0.9, 1.0])),
        "bagging_freq":      1,
        "lambda_l2":         float(rng.choice([0.0, 1.0, 10.0])),
    }


def _fit_lgbm(X_tr, y_tr, X_va, y_va, n_configs: int, ordinal: bool):
    import lightgbm as lgb
    rng = np.random.default_rng(SEED + (1 if ordinal else 0))
    w = class_weights(y_tr)
    base = ({"objective": "regression"} if ordinal else
            {"objective": "multiclass", "num_class": K})
    base.update({"verbosity": -1, "seed": SEED, "num_threads": 0})
    dtr = lgb.Dataset(X_tr, y_tr.astype(float) if ordinal else y_tr, weight=w[y_tr],
                      params={"feature_pre_filter": False})
    dva = lgb.Dataset(X_va, y_va.astype(float) if ordinal else y_va, weight=w[y_va], reference=dtr)

    best = {"qwk": -np.inf}
    for i in range(n_configs):
        params = {**base, **_lgbm_params(rng)}
        booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        if ordinal:
            s = booster.predict(X_va, num_iteration=booster.best_iteration) + 1
            cuts = tune_cuts(s, y_va + 1)
            score = qwk(y_va, cut_predict(s, cuts) - 1)
        else:
            P = booster.predict(X_va, num_iteration=booster.best_iteration)
            score, cuts = qwk(y_va, P.argmax(1)), None
        if score > best["qwk"]:
            best = {"qwk": score, "params": params, "booster": booster,
                    "iters": booster.best_iteration, "cuts": cuts}
        print(f"    {'OLGBM' if ordinal else 'LGBM'} config {i + 1:2d}/{n_configs}: "
              f"val QWK = {score:.4f} (iters {booster.best_iteration})  best {best['qwk']:.4f}")
    return best


def train_all(n_configs: int = 40) -> None:
    """Train LR, LightGBM, and ordinal LightGBM on temporal train/val/test split."""
    import joblib
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    df, X, y, split, groups = load_split()
    tr, va, te = (split == s for s in ("train", "val", "test"))
    print(f"Rows: train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}  features {X.shape[1]}")
    w = class_weights(y[tr])

    print("  Logistic regression ...")
    lr, lr_info = _fit_lr(X[tr].reset_index(drop=True), y[tr], df[tr].reset_index(drop=True))
    print("  LightGBM multiclass ...")
    lgbm = _fit_lgbm(X[tr], y[tr], X[va], y[va], n_configs, ordinal=False)
    print("  LightGBM ordinal ...")
    olgbm = _fit_lgbm(X[tr], y[tr], X[va], y[va], n_configs, ordinal=True)

    # Predictions on every row (val rows drive threshold tuning, test rows evaluation)
    P_lr   = prior_correct(lr.predict_proba(X), w)
    P_lgbm = prior_correct(lgbm["booster"].predict(X, num_iteration=lgbm["iters"]), w)
    s_ord  = olgbm["booster"].predict(X, num_iteration=olgbm["iters"]) + 1

    # Expected-urgency score for the LightGBM model; base cut-points tuned on validation
    score = P_lgbm @ np.arange(1, K + 1)
    base_cuts = tune_cuts(score[va], y[va] + 1)

    out = pd.DataFrame({
        "split": split, "survey_year": df["survey_year"], "hosp_id": df["hosp_id"],
        "patwt": df["patwt"], "resi_o": y + 1, "immedr": df["immedr"],
        "nhamcs_esi": df["nhamcs_esi"],
        "pred_lr": P_lr.argmax(1) + 1, "pred_lgbm": P_lgbm.argmax(1) + 1,
        "pred_lgbm_cut": cut_predict(score, base_cuts),
        "pred_olgbm": cut_predict(s_ord, olgbm["cuts"]), "score_lgbm": score,
        "score_olgbm": s_ord,
    })
    for k in range(K):
        out[f"p_lgbm_{k + 1}"] = P_lgbm[:, k]
        out[f"p_lr_{k + 1}"] = P_lr[:, k]
    out.to_parquet(PRED_PATH, index=False)

    joblib.dump({"model": lr, "info": lr_info}, MODEL_DIR / "lr.joblib")
    joblib.dump({"booster": lgbm["booster"], "iters": lgbm["iters"], "params": lgbm["params"],
                 "val_qwk": lgbm["qwk"], "class_weights": w, "base_cuts": base_cuts},
                MODEL_DIR / "lgbm.joblib")
    joblib.dump({"booster": olgbm["booster"], "iters": olgbm["iters"], "params": olgbm["params"],
                 "val_qwk": olgbm["qwk"], "cuts": olgbm["cuts"]}, MODEL_DIR / "olgbm.joblib")
    meta = {"rfv_groups": groups, "features": list(X.columns), "split_years": SPLIT_YEARS,
            "lr": lr_info, "lgbm_val_qwk": lgbm["qwk"], "olgbm_val_qwk": olgbm["qwk"],
            "lgbm_params": {k: v for k, v in lgbm["params"].items() if k != "num_threads"},
            "lgbm_base_cuts": base_cuts.tolist(), "olgbm_cuts": np.asarray(olgbm["cuts"]).tolist(),
            "class_weights": w.tolist(), "n_configs": n_configs}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"Saved models → {MODEL_DIR}/, predictions → {PRED_PATH}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    train_all()
