"""
triageq.figures — Figure and table generation for the DASA'26 paper.

Generates (whatever the available results allow):
  F1: System diagram (block diagram)
  F2: Two-class hook figure (Δ mean wait / Δ TC / Δ p95 vs Youden J, one line per ρ)
  F3: Core break-even curves (break-even accuracy vs ρ, NP-PQ vs APQ panels, 95% CI)
  F4: TC₁₂ gain heatmap on (accuracy, ρ) with zero contour
  F5: Trade-off placeholder (needs nurse/ML points, Milestone 10)
  F7: Validation parity plot (sim vs analytic mean waits, V1–V7)
  F8: Welch warm-up plot at ρ = 0.95
  F11: Robustness tornado (E3)

Tables:
  T1: Model parameters
  T2: Validation results (analytic vs simulated ± CI)
  T4: Break-even lookup table (b = 0, c = 5)
  T6: E3 robustness numbers
  E2 analytic cross-check: 5-level mean waits, simulation vs Cobham/M/M/c mixture

All outputs: vector PDF, Okabe-Ito colour-blind palette, IEEE two-column layout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Design constants ──────────────────────────────────────────────────────────

PALETTE = {
    "fifo":   "#E69F00",   # orange
    "np_pq":  "#0072B2",   # blue
    "apq":    "#009E73",   # green
    "oracle": "#D55E00",   # vermillion
    "nurse":  "#CC79A7",   # pink
    "ml":     "#56B4E9",   # sky blue
}
BIAS_COLORS = {-0.5: "#0072B2", -0.25: "#56B4E9", 0.0: "#000000",
               0.25: "#E69F00", 0.5: "#D55E00"}

SINGLE_W = 3.5    # inches  — one IEEE column
DOUBLE_W = 7.16   # inches  — two IEEE columns
DPI      = 300

FONT_SIZE = 8
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans"],
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE - 1,
    "ytick.labelsize":  FONT_SIZE - 1,
    "legend.fontsize":  FONT_SIZE - 1,
    "lines.linewidth":  1.2,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "pdf.fonttype":     42,
})

RHO_VALS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
RHO_CMAP = plt.cm.viridis
DEFAULT_PI = [0.01, 0.12, 0.42, 0.35, 0.10]
POLICY_LABEL = {"np_pq": "NP-PQ", "apq": "APQ", "oracle": "Oracle", "fifo": "FIFO"}
METRIC_LABEL = {"tc12": "TC$_{12}$", "p95_12": "p95 wait (L1–2)", "p90_12": "p90 wait (L1–2)",
                "mean12": "mean wait (L1–2)", "starv12": "P(wait > 3τ) (L1–2)",
                "wcost": "weighted cost"}


def _rho_color(rho: float) -> tuple:
    t = (rho - min(RHO_VALS)) / (max(RHO_VALS) - min(RHO_VALS))
    return RHO_CMAP(0.9 * t)


def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _load_pi(res: Path) -> np.ndarray:
    meta = res / "e2" / "meta.json"
    pi = json.loads(meta.read_text())["pi"] if meta.exists() else DEFAULT_PI
    pi = np.asarray(pi, dtype=np.float64)
    return pi / pi.sum()


# ── F1: System diagram ────────────────────────────────────────────────────────

def make_f1(out_dir: Path) -> None:
    """F1: System block diagram — arrivals → classifier → queue → servers."""
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 1.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, label, color="#0072B2", fontsize=7):
        rect = plt.Rectangle((x, y), w, h, fc=color, ec="white", alpha=0.9, lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")

    def arrow(x1, y, x2):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))

    box(0.1, 1.0, 1.4, 1.0, "Poisson\narrivals λ\ntrue level k~π", color="#555555")
    box(1.7, 1.0, 2.0, 1.0, "Classifier\n(σ, b) or\nempirical M", color=PALETTE["np_pq"])
    box(3.9, 0.4, 2.2, 2.2, "Queue by\npredicted k̂\nFIFO / NP-PQ\nAPQ / Oracle", color=PALETTE["apq"])
    box(6.3, 1.0, 2.0, 1.0, "c providers\nE[S] = 20 min", color=PALETTE["oracle"])
    box(8.5, 1.0, 1.4, 1.0, "Waits by\ntrue k:\nTC, p95", color="#555555")

    arrow(1.5, 1.5, 1.7)
    arrow(3.7, 1.5, 3.9)
    arrow(6.1, 1.5, 6.3)
    arrow(8.3, 1.5, 8.5)

    ax.text(2.7, 0.75, "k̂ = clip(round(k + b + σε))", ha="center", va="top",
            fontsize=6, color="#555555", style="italic")
    _save(fig, out_dir / "F1_system_diagram.pdf")


# ── F2: Two-class hook figure ─────────────────────────────────────────────────

def make_f2(e1_path: Path, out_dir: Path) -> None:
    """F2: Δ (triage − FIFO) for true-H patients vs Youden J along s = p, one line per ρ."""
    if not e1_path.exists():
        print("  [F2] E1 results not found. Skipping.")
        return

    df = pd.read_parquet(e1_path)
    df = df[np.isclose(df["sensitivity"], df["specificity"])]
    panels = [("delta_mean_H", "Δ mean wait, high acuity (min)", "(a) Mean wait: crosses at J = 0"),
              ("delta_tc_H",   "Δ P(wait ≤ τ$_H$)",               "(b) Target compliance"),
              ("delta_p95_H",  "Δ p95 wait, high acuity (min)",   "(c) 95th percentile")]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.3))

    for rho in sorted(df["rho"].unique()):
        sub = df[df["rho"] == rho].sort_values("youden_j")
        col = _rho_color(rho)
        for ax, (col_name, _, _) in zip(axes, panels):
            ax.plot(sub["youden_j"], sub[col_name], color=col, lw=1.1, label=f"ρ = {rho:.2f}")
            ax.fill_between(sub["youden_j"], sub[f"{col_name}_lo"], sub[f"{col_name}_hi"],
                            color=col, alpha=0.15, lw=0)
        axes[0].plot(sub["youden_j"], sub["analytic_delta_mean_H"], color=col, lw=0.6, ls=":")

    for ax, (_, ylab, title) in zip(axes, panels):
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("Youden J  (s = p)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].set_yscale("symlog", linthresh=10)
    axes[2].set_yscale("symlog", linthresh=10)
    axes[1].legend(fontsize=5, loc="lower right", framealpha=0.6)
    _save(fig, out_dir / "F2_two_class_breakeven.pdf")


# ── F3: Core break-even curves ────────────────────────────────────────────────

QUALITY_LABEL = {"youden_j": "high-acuity Youden J", "exact_acc": "exact accuracy",
                 "ha_sens": "high-acuity sensitivity",
                 "severe_under_12": "severe undertriage"}


def make_f3(be_path: Path, out_dir: Path, metric: str = "p95_12", c: int = 5,
            y: str = "youden_j") -> None:
    """F3: break-even classifier quality vs ρ, one line per bias b, NP-PQ vs APQ panels."""
    if not be_path.exists():
        print("  [F3] Break-even table not found. Skipping.")
        return
    df = pd.read_parquet(be_path)
    df = df[(df["metric"] == metric) & (df["c"] == c)]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.6), sharey=True)
    for ax, policy in zip(axes, ["np_pq", "apq"]):
        for b, grp in df[df["policy"] == policy].groupby("b"):
            grp = grp.sort_values("rho")
            col = BIAS_COLORS.get(round(float(b), 2), "grey")
            cr = grp[grp["status"] == "crossing"]
            ax.plot(cr["rho"], cr[y], color=col, marker="o", ms=2.5,
                    label=f"b = {float(b):+.2f}")
            if f"{y}_at_sigma_hi" in cr:
                # larger σ = lower quality, so σ_hi gives the lower band edge
                ax.fill_between(cr["rho"], cr[f"{y}_at_sigma_hi"], cr[f"{y}_at_sigma_lo"],
                                color=col, alpha=0.15, lw=0)
            ab = grp[grp["status"] == "always_better"] if y != "severe_under_12" else grp.iloc[:0]
            ax.scatter(ab["rho"], np.zeros(len(ab)), marker="v", color=col, s=10, clip_on=False)
        ops_path = be_path.parent.parent / "e4" / "operating_points.json"
        if ops_path.exists():
            from triageq.errors import summary_stats
            ops = json.loads(ops_path.read_text())["operating_points"]
            pi = _load_pi(be_path.parent.parent)
            for name, lab in (("nurse", "nurse (IMMEDR)"), ("ml", "ML (LightGBM)")):
                if name in ops:
                    v = summary_stats(np.array(ops[name]["matrix"]), pi)[y]
                    ax.axhline(v, color=PALETTE[name], ls="--", lw=1.0)
                    ax.text(0.475, v, f" {lab}: {v:.2f}", color=PALETTE[name],
                            fontsize=5.5, va="bottom")
        ax.set_xlabel("Utilisation ρ")
        ax.set_title(f"({'ab'[['np_pq', 'apq'].index(policy)]}) {POLICY_LABEL[policy]}, c = {c}")
        ax.set_xlim(0.47, 0.98)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel(f"Break-even {QUALITY_LABEL.get(y, y)}\n({METRIC_LABEL.get(metric, metric)})")
    axes[1].legend(fontsize=5.5, loc="center left" if y == "severe_under_12" else "upper left",
                   framealpha=0.8, ncol=1)
    _save(fig, out_dir / f"F3_breakeven_{metric}_c{c}_{y}.pdf")


# ── F4: TC₁₂ gain heatmap ─────────────────────────────────────────────────────

def make_f4(e2_path: Path, out_dir: Path, pi: np.ndarray, c: int = 5, b: float = 0.0,
            policy: str = "np_pq", metric: str = "p95_12") -> None:
    """F4: heatmap of a metric's gain over FIFO on (HA Youden J, ρ), zero contour drawn."""
    if not e2_path.exists():
        print("  [F4] E2 raw not found. Skipping.")
        return
    from triageq.errors import confusion_from_params, summary_stats
    from triageq.metrics import SWEEP_METRICS

    df  = pd.read_parquet(e2_path)
    sub = df[(df["c"] == c) & (df["b"].round(2) == round(b, 2)) & (df["policy"] == policy)]
    if sub.empty:
        print("  [F4] No data. Skipping.")
        return
    piv = sub.pivot(index="rho", columns="sigma", values=f"d_{metric}").sort_index()
    sig = piv.columns.values
    jj = np.array([summary_stats(confusion_from_params(float(s), b), pi)["youden_j"] for s in sig])
    order = np.argsort(jj)
    jj, grid = jj[order], piv.values[:, order]
    if not SWEEP_METRICS[metric]:
        grid = -grid            # orient so positive = triage better
    rhos = piv.index.values

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.5))
    lim = float(np.nanpercentile(np.abs(grid), 95))
    im = ax.pcolormesh(jj, rhos, grid, cmap="RdBu", vmin=-lim, vmax=lim, shading="nearest")
    try:
        ax.contour(jj, rhos, grid, levels=[0], colors=["black"], linewidths=[0.9])
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, extend="both")
    unit = "" if metric.startswith("tc") else " (min)"
    cbar.set_label(f"FIFO − triage, {METRIC_LABEL.get(metric, metric)}{unit}"
                   if not SWEEP_METRICS[metric] else f"triage − FIFO, {METRIC_LABEL.get(metric, metric)}",
                   fontsize=FONT_SIZE - 2)
    ax.set_xscale("log")
    ax.set_xlabel("High-acuity Youden J (log scale)")
    ax.set_ylabel("Utilisation ρ")
    ax.set_title(f"{POLICY_LABEL[policy]}, c = {c}, b = {b:+.2f}; blue = triage better", fontsize=7)
    _save(fig, out_dir / f"F4_{metric}_heatmap_{policy}_c{c}.pdf")


# ── F5: Trade-off: high-acuity benefit vs low-acuity harm ─────────────────────

def make_f5(e4_path: Path, out_dir: Path, rhos=(0.85, 0.95), c: int = 5) -> None:
    """F5: high-acuity p95 reduction vs low-acuity p95 increase (nurse, ML, δ-curve, oracle)."""
    if not e4_path.exists():
        print("  [F5] E4 summary not found. Skipping.")
        return
    s = pd.read_parquet(e4_path)
    s = s[s.c == c]
    fig, axes = plt.subplots(1, len(rhos), figsize=(DOUBLE_W, 2.6))
    for ax, rho in zip(axes, rhos):
        g = s[np.isclose(s.rho, rho)]
        for pol, mk in (("np_pq", "o"), ("apq", "s")):
            d = g[(g.policy == pol) & g.point.str.startswith("delta_")].copy()
            d["delta"] = d.point.str.replace("delta_", "").astype(float)
            d = d.sort_values("delta")
            ax.plot(-d.d_p95_12, d.d_p95_45, color=PALETTE[pol], lw=0.9, alpha=0.8,
                    label=f"ML δ-curve, {POLICY_LABEL[pol]}")
            for dv in (-0.5, 0.0, 0.5):
                r = d[np.isclose(d.delta, dv)]
                if len(r):
                    ax.annotate(f"δ={dv:+.1f}", (-r.d_p95_12.iloc[0], r.d_p95_45.iloc[0]),
                                fontsize=5, color=PALETTE[pol], xytext=(2, 2),
                                textcoords="offset points")
            for pt in ("nurse", "ml", "oracle"):
                r = g[(g.policy == pol) & (g.point == pt)]
                if len(r):
                    ax.scatter(-r.d_p95_12, r.d_p95_45, marker=mk, s=22, zorder=4,
                               color=PALETTE[pt], edgecolor="black", lw=0.4,
                               label=f"{pt}, {POLICY_LABEL[pol]}")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("High-acuity p95 wait reduction vs FIFO (min)")
        ax.set_title(f"ρ = {rho}, c = {c}")
    axes[0].set_ylabel("Low-acuity (L4–5) p95 wait increase (min)")
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].legend(h, l, fontsize=5, loc="upper left", framealpha=0.7)
    _save(fig, out_dir / "F5_tradeoff.pdf")


# ── F6: Confusion matrices ────────────────────────────────────────────────────

def make_f6(e4_dir: Path, out_dir: Path) -> None:
    """F6: row-normalised confusion matrices, nurse vs rESI-O and ML vs rESI-O."""
    ops_path = e4_dir / "operating_points.json"
    if not ops_path.exists():
        print("  [F6] operating_points.json not found. Skipping.")
        return
    ops = json.loads(ops_path.read_text())["operating_points"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W * 0.7, 2.6))
    for ax, (name, title) in zip(axes, (("nurse", "(a) Nurse IMMEDR"), ("ml", "(b) ML (LightGBM)"))):
        M = np.array(ops[name]["matrix"])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.55 else "black")
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(5), range(1, 6))
        ax.set_xlabel("Predicted level")
        ax.set_title(f"{title}, n = {ops[name]['n']:,}")
    axes[0].set_ylabel("rESI-O reference level")
    fig.colorbar(im, ax=axes, shrink=0.8, label="P(pred | true)")
    _save(fig, out_dir / "F6_confusion_matrices.pdf", tight=False)


# ── F9: Waiting-time ECDFs ────────────────────────────────────────────────────

def make_f9(e4_dir: Path, out_dir: Path, pi: np.ndarray, rho: float = 0.9, c: int = 5) -> None:
    """F9: ECDF of waits of true level-2 patients under FIFO / NP-PQ (nurse, ML) / APQ (ML)."""
    ops_path = e4_dir / "operating_points.json"
    if not ops_path.exists():
        print("  [F9] operating_points.json not found. Skipping.")
        return
    from triageq.streams import ServiceConfig, make_streams, replication_seeds
    from triageq.errors import predict_from_matrix
    from triageq.sim_fast import run_fast

    ops = json.loads(ops_path.read_text())["operating_points"]
    s = make_streams(250_000, rho, c, pi, ServiceConfig(), seed=replication_seeds(1, 20260925)[0])
    true = s["true_levels"]
    runs = [("FIFO", true, "fifo", PALETTE["fifo"], "-"),
            ("NP-PQ, nurse", predict_from_matrix(true, s["u"], np.array(ops["nurse"]["matrix"])),
             "np_pq", PALETTE["nurse"], "-"),
            ("NP-PQ, ML", predict_from_matrix(true, s["u"], np.array(ops["ml"]["matrix"])),
             "np_pq", PALETTE["ml"], "-"),
            ("APQ, ML", predict_from_matrix(true, s["u"], np.array(ops["ml"]["matrix"])),
             "apq", PALETTE["apq"], "--"),
            ("NP-PQ, oracle", true, "np_pq", PALETTE["oracle"], ":")]
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
    for lab, pred, pol, col, ls in runs:
        w = run_fast(s, pred.astype(np.int32), pol, c)
        w = np.sort(w[~np.isnan(w) & (true == 2)])
        ax.plot(np.maximum(w, 0.1), np.arange(1, len(w) + 1) / len(w), color=col, ls=ls, label=lab)
    ax.axvline(15, color="black", lw=0.6, ls="--")
    ax.text(15, 0.05, r" $\tau_2$ = 15 min", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Wait to provider, true level 2 (min, log)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"ρ = {rho}, c = {c}", fontsize=7)
    ax.legend(fontsize=5.5, loc="lower right")
    _save(fig, out_dir / "F9_ecdf_level2.pdf")


# ── F10: Calibration and ROC ──────────────────────────────────────────────────

def make_f10(e4_dir: Path, out_dir: Path) -> None:
    """F10: reliability curve of P(level ≤ 2) and high-acuity ROC (ML vs nurse)."""
    cal, roc_ml, roc_n = (e4_dir / f for f in ("calibration.parquet", "roc_ml.parquet",
                                               "roc_nurse.parquet"))
    if not cal.exists():
        print("  [F10] calibration data not found. Skipping.")
        return
    ops = json.loads((e4_dir / "operating_points.json").read_text())
    cv = pd.read_parquet(cal)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W * 0.7, 2.5))
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.6)
    for kind, col, key in (("raw", PALETTE["ml"], "ece_p_high"),
                           ("isotonic", PALETTE["apq"], "ece_p_high_isotonic")):
        c = cv[cv["kind"] == kind] if "kind" in cv else cv
        if len(c) and key in ops:
            axes[0].plot(c.mean_pred, c.frac_pos, "o-", color=col, ms=3,
                         label=f"{kind} (ECE {ops[key]:.3f})")
    axes[0].legend(fontsize=6, loc="upper left")
    axes[0].set_xlabel("Predicted P(level ≤ 2)")
    axes[0].set_ylabel("Observed fraction (test years)")
    axes[0].set_title("(a) Calibration, equal-mass bins")
    lim = max(cv.mean_pred.max(), cv.frac_pos.max()) * 1.1
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    for path, name, lab in ((roc_ml, "ml", "ML"), (roc_n, "nurse", "Nurse")):
        r = pd.read_parquet(path)
        axes[1].plot(r.fpr, r.tpr, color=PALETTE[name], label=lab)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.6)
    axes[1].set_xlabel("1 − specificity (levels 3–5)")
    axes[1].set_ylabel("Sensitivity (levels 1–2)")
    axes[1].set_title("(b) High-acuity ROC, test years")
    t3 = Path("results/tables/T3_classifier_performance.csv")
    if t3.exists():
        t = pd.read_csv(t3).set_index("operating_point")
        axes[1].legend([f"ML (AUROC {t.loc['ml', 'ha_auroc']:.3f})",
                        f"Nurse (AUROC {t.loc['nurse', 'ha_auroc']:.3f})"], fontsize=6)
    _save(fig, out_dir / "F10_calibration_roc.pdf")


# ── F7: Validation parity plot ────────────────────────────────────────────────

def make_f7(val_path: Path, out_dir: Path) -> None:
    """F7: simulated vs analytic mean waits (V1–V7), log-log with y = x and 95% CIs."""
    if not val_path.exists():
        print("  [F7] Validation results not found. Skipping.")
        return
    df = pd.read_parquet(val_path).dropna(subset=["analytic", "simulated"])
    df = df[df["analytic"] > 0]
    if df.empty:
        print("  [F7] Empty validation results. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.0))
    tests = sorted(df["test"].unique())
    colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    mks = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, test in enumerate(tests):
        g = df[df["test"] == test]
        ax.errorbar(g["analytic"], g["simulated"],
                    yerr=[g["simulated"] - g["ci_lo"], g["ci_hi"] - g["simulated"]],
                    fmt=mks[i % len(mks)], color=colors[i % len(colors)], ms=3.5, lw=0.6,
                    capsize=1.2, label=test, alpha=0.9)
    vals = pd.concat([df["analytic"], df["simulated"]])
    lim = [vals.min() * 0.7, vals.max() * 1.4]
    ax.plot(lim, lim, "k--", lw=0.6, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Analytic mean wait (min)")
    ax.set_ylabel("Simulated mean wait (min)")
    ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    _save(fig, out_dir / "F7_validation_parity.pdf")


# ── F8: Welch warm-up ─────────────────────────────────────────────────────────

def make_f8(welch_path: Path, out_dir: Path) -> None:
    """F8: Welch moving average of waits vs patient index (ρ = 0.95, c = 5)."""
    if not welch_path.exists():
        print("  [F8] Welch data not found. Skipping.")
        return
    df = pd.read_parquet(welch_path)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.3))
    ax.plot(df["index"], df["fifo"], color=PALETTE["fifo"], label="FIFO, all patients")
    ax.plot(df["index"], df["np_pq"], color=PALETTE["np_pq"], label="NP-PQ, all patients")
    ax.plot(df["index"], df["low"], color=PALETTE["oracle"], label="NP-PQ, true level 5")
    ax.axvline(0.10 * df["index"].max(), color="black", lw=0.7, ls="--")
    ax.text(0.10 * df["index"].max(), ax.get_ylim()[1] * 0.95, " 10% warm-up",
            fontsize=6, va="top")
    ax.set_xlabel("Patient index")
    ax.set_ylabel("Moving-average wait (min)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(fontsize=5.5)
    _save(fig, out_dir / "F8_welch_warmup.pdf")


# ── F11 / T6: robustness ──────────────────────────────────────────────────────

def make_f11_t6(e3_path: Path, out_dir: Path, rho: float = 0.9, c: int = 5,
                policy: str = "np_pq", metric: str = "p95_12") -> None:
    """F11: change in break-even HA Youden J across E3 variants; T6: full E3 table."""
    if not e3_path.exists():
        print("  [F11/T6] E3 break-even table not found. Skipping.")
        return
    be = pd.read_parquet(e3_path)
    cols = ["variant", "policy", "c", "rho", "metric", "status", "sigma_star",
            "sigma_lo", "sigma_hi", "youden_j", "exact_acc", "ha_sens", "undertriage"]
    be[[k for k in cols if k in be]].sort_values(["metric", "policy", "c", "rho", "variant"]).to_csv(
        out_dir / "T6_e3_robustness.csv", index=False, float_format="%.4f")
    print("  Saved T6_e3_robustness.csv")

    sub = be[(be.rho.round(2) == rho) & (be.c == c) & (be.policy == policy) & (be.metric == metric)]
    base = sub[sub.variant == "S0_exp_common"]
    if base.empty or base.iloc[0]["status"] != "crossing":
        print("  [F11] Baseline has no crossing at this cell. Skipping plot.")
        return
    b_acc = float(base.iloc[0]["youden_j"])
    rows = sub[sub.variant != "S0_exp_common"].copy()
    rows["delta"] = rows["youden_j"] - b_acc
    rows = rows.sort_values("delta")
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.0))
    ax.barh(rows["variant"], rows["delta"].fillna(0),
            color=[PALETTE["np_pq"] if d >= 0 else PALETTE["fifo"] for d in rows["delta"].fillna(0)])
    for y, (d, st) in enumerate(zip(rows["delta"], rows["status"])):
        if st != "crossing":
            ax.text(0, y, f" {st}", va="center", fontsize=6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel(f"Δ break-even HA Youden J vs S0 (baseline J* = {b_acc:.3f})")
    ax.set_title(f"{POLICY_LABEL[policy]}, {METRIC_LABEL[metric]}, ρ = {rho}, c = {c}", fontsize=7)
    _save(fig, out_dir / "F11_robustness_tornado.pdf")


# ── Tables ────────────────────────────────────────────────────────────────────

def make_t1(out_dir: Path, pi: np.ndarray) -> None:
    """T1: Model parameters table."""
    rows = [
        ("K — acuity levels", "5  (ESI-like)"),
        ("True-level prior π", f"{np.round(pi, 4).tolist()}  (survey-weighted rESI-O, NHAMCS 2016–22)"),
        ("Targets τ (min)", "[5, 15, 30, 60, 120]  for levels 1–5"),
        ("Mean service E[S]", "20 min  (sensitivity: 10, 40)"),
        ("Service family", "S0: Exp common; S1: Exp level-dep; S2: Lognormal CV = 1.5"),
        ("Servers c", "1 (analytic), 5 (main results)"),
        ("Utilisation ρ", "0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95"),
        ("Noise σ", "0.0, 0.1, …, 2.0  (21 grid points)"),
        ("Bias b", "−0.50, −0.25, 0, +0.25, +0.50"),
        ("Policies", "FIFO, NP-PQ, APQ, Oracle"),
        ("APQ weights w", "[16, 8, 4, 2, 1]"),
        ("Replications R", "20  (E1/E2);  30  (E0 validation)"),
        ("Run length", "100 k (ρ ≤ 0.85);  250 k (ρ ≥ 0.90)"),
        ("Warm-up fraction", "10 %"),
        ("Master seed", "20260925"),
    ]
    pd.DataFrame(rows, columns=["Parameter", "Value"]).to_csv(
        out_dir / "T1_model_parameters.csv", index=False)
    print("  Saved T1_model_parameters.csv")


def make_t2(val_path: Path, v8_path: Path, out_dir: Path) -> None:
    """T2: analytic vs simulated mean waits with 95% CI and relative error."""
    if not val_path.exists():
        print("  [T2] Not found. Skipping.")
        return
    df = pd.read_parquet(val_path)
    out = pd.DataFrame({
        "Test": df["test"], "Case": df["case"], "rho": df["rho"], "Run": df["run"],
        "Level": df["level"], "Analytic": df["analytic"], "Simulated": df["simulated"],
        "CI lo": df["ci_lo"], "CI hi": df["ci_hi"], "|Rel err| %": 100 * df["rel_err"],
        "CI covers": df["ci_covers"], "Result": df["pass"].map({True: "PASS", False: "FAIL"}),
    })
    out.to_csv(out_dir / "T2_validation.csv", index=False, float_format="%.3f")

    # Compact paper version: worst relative error per test × ρ
    compact = (df.groupby(["test", "rho"])
                 .agg(n_checks=("pass", "size"), n_pass=("pass", "sum"),
                      max_rel_err_pct=("rel_err", lambda x: 100 * x.max()),
                      all_ci_cover=("ci_covers", "all"))
                 .reset_index())
    compact.to_csv(out_dir / "T2_validation_compact.csv", index=False, float_format="%.2f")
    if v8_path.exists():
        pd.read_parquet(v8_path).to_csv(out_dir / "T2b_v8_confusion_sampling.csv",
                                        index=False, float_format="%.5f")
    print("  Saved T2_validation.csv, T2_validation_compact.csv")


def make_t4(be_path: Path, out_dir: Path) -> None:
    """T4: break-even accuracy by ρ × policy (b = 0, c = 5) for the key metrics."""
    if not be_path.exists():
        print("  [T4] Not found. Skipping.")
        return
    df = pd.read_parquet(be_path)
    sub = df[(df["b"].round(2) == 0.0) & (df["c"] == 5) &
             df["metric"].isin(["tc12", "p95_12", "p90_12", "mean12"])]
    cols = ["metric", "policy", "rho", "status", "sigma_star", "sigma_lo", "sigma_hi",
            "exact_acc", "exact_acc_at_sigma_hi", "exact_acc_at_sigma_lo",
            "ha_sens", "undertriage", "youden_j", "qwk"]
    sub = sub[[k for k in cols if k in sub]].sort_values(["metric", "policy", "rho"])
    sub.to_csv(out_dir / "T4_breakeven_table.csv", index=False, float_format="%.4f")
    print("  Saved T4_breakeven_table.csv")


def make_e2_crosscheck(e2_path: Path, out_dir: Path, pi: np.ndarray) -> None:
    """Simulated vs analytic mean wait of true levels 1–2 under NP-PQ misclassification."""
    if not e2_path.exists():
        return
    from triageq.errors import confusion_from_params
    from triageq.analytic import misclassified_mean_waits

    df = pd.read_parquet(e2_path)
    sub = df[df.policy == "np_pq"]
    rows = []
    ES, ES2 = np.full(5, 20.0), np.full(5, 800.0)
    w12 = pi[:2] / pi[:2].sum()
    for _, r in sub.iterrows():
        M = confusion_from_params(float(r.sigma), float(r.b))
        lam = r.rho * r.c / 20.0
        W = misclassified_mean_waits(M, pi, lam, ES, ES2, c=int(r.c),
                                     mu_common=(1 / 20.0 if r.c > 1 else None))
        a = float(np.dot(w12, W[:2]))
        rows.append({"rho": r.rho, "c": r.c, "b": r.b, "sigma": r.sigma,
                     "analytic_mean12": a, "sim_mean12": r.mean12,
                     "rel_err": abs(r.mean12 - a) / a})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "E2_analytic_crosscheck.csv", index=False, float_format="%.4f")
    print(f"  Saved E2_analytic_crosscheck.csv  (median |rel err| = "
          f"{out.rel_err.median():.2%}, 95th pct = {out.rel_err.quantile(0.95):.2%})")


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_all(results_dir: str = "results/", output_dir: str = "results/figures/") -> None:
    """Generate all available figures and tables from saved parquet results."""
    res = Path(results_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab = res / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    pi = _load_pi(res)

    print(f"\nGenerating figures → {out}/, tables → {tab}/")

    make_f1(out)
    make_t1(tab, pi)
    e4 = res / "e4"
    make_f5(e4 / "e4_summary.parquet", out)
    make_f6(e4, out)
    make_f9(e4, out, pi)
    make_f10(e4, out)
    gap = e4 / "e4_breakeven_gap.parquet"
    if gap.exists():
        g = pd.read_parquet(gap)
        g.to_csv(tab / "T4b_breakeven_gap.csv", index=False, float_format="%.4f")
        print("  Saved T4b_breakeven_gap.csv")

    val_path = res / "e0" / "validation_results.parquet"
    make_f7(val_path, out)
    make_t2(val_path, res / "e0" / "v8_results.parquet", tab)
    make_f8(res / "e0" / "welch_rho095.parquet", out)

    make_f2(res / "e1" / "e1_results.parquet", out)

    e2_raw = res / "e2" / "e2_raw.parquet"
    e2_be  = res / "e2" / "e2_breakeven.parquet"
    for pol in ("np_pq", "apq"):
        for metric in ("p95_12", "tc12"):
            make_f4(e2_raw, out, pi, policy=pol, metric=metric)
    if e2_be.exists():
        for metric in ("p95_12", "p90_12"):
            for c in (1, 5):
                for yq in ("youden_j", "severe_under_12"):
                    make_f3(e2_be, out, metric=metric, c=c, y=yq)
        make_t4(e2_be, tab)
    make_e2_crosscheck(e2_raw, tab, pi)

    make_f11_t6(res / "e3" / "e3_breakeven.parquet", out)

    print(f"\nAll done — check {out}/ and {tab}/")


if __name__ == "__main__":
    generate_all()
