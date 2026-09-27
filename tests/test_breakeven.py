"""
tests/test_breakeven.py — Unit tests for the break-even estimation module.

Tests:
  - find_breakeven recovers a known crossing on synthetic linear data
  - Status codes: always_better, never_better, crossing
  - to_interpretable returns valid metrics
  - delta_table computes correct differences
"""

import numpy as np
import pytest

from triageq.breakeven import (
    find_breakeven,
    to_interpretable,
    delta_table,
    synthetic_breakeven_test,
)


# ── Synthetic break-even recovery ─────────────────────────────────────────────

def test_synthetic_recovery_known_crossing():
    """find_breakeven should recover a known crossing at sigma=1.0 within 0.1."""
    result = synthetic_breakeven_test(true_crossing=1.0, n_sigma=21, R=30, noise_std=0.01)
    assert result["status"] == "crossing", f"Expected 'crossing', got {result['status']}"
    sigma_star = result["sigma_star"]
    assert abs(sigma_star - 1.0) < 0.15, \
        f"Recovered sigma_star={sigma_star:.4f}, expected ~1.0"


def test_synthetic_recovery_always_better():
    """When all deltas are positive (triage always better), status should reflect that."""
    sigmas = np.linspace(0.0, 2.0, 11)
    # All deltas > 0 (triage always better)
    deltas = np.ones((11, 10)) * 0.5
    result = find_breakeven(sigmas, deltas, metric_higher_is_better=True)
    assert result["status"] == "always_better"
    assert np.isnan(result["sigma_star"])


def test_synthetic_recovery_never_better():
    """When all deltas are negative (triage never better), status = never_better."""
    sigmas = np.linspace(0.0, 2.0, 11)
    deltas = -np.ones((11, 10)) * 0.5
    result = find_breakeven(sigmas, deltas, metric_higher_is_better=True)
    assert result["status"] == "never_better"
    assert np.isnan(result["sigma_star"])


def test_ci_is_ordered():
    """Bootstrap CI should have lo ≤ sigma_star ≤ hi."""
    result = synthetic_breakeven_test(true_crossing=0.8, R=20, noise_std=0.03, seed=7)
    if result["status"] == "crossing":
        assert result["ci_lo"] <= result["sigma_star"] <= result["ci_hi"], \
            f"CI not ordered: [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}], star={result['sigma_star']:.4f}"


# ── to_interpretable ──────────────────────────────────────────────────────────

def test_to_interpretable_valid_metrics():
    """to_interpretable should return valid classifier metrics for finite sigma_star."""
    pi = np.array([0.01, 0.12, 0.42, 0.35, 0.10])
    result = to_interpretable(sigma_star=1.0, b=0.0, pi=pi)
    assert 0.0 <= result["exact_acc"] <= 1.0
    assert 0.0 <= result["ha_sens"] <= 1.0
    assert 0.0 <= result["ha_spec"] <= 1.0
    assert -1.0 <= result["youden_j"] <= 1.0


def test_to_interpretable_nan():
    """NaN sigma_star should return a dict with no_crossing status."""
    result = to_interpretable(sigma_star=float("nan"), b=0.0, pi=np.ones(5) / 5)
    assert result["status"] == "no_crossing"


# ── delta_table ───────────────────────────────────────────────────────────────

def test_delta_table_basic():
    """delta_table should compute element-wise differences."""
    triage = [0.8, 0.7, 0.6]
    fifo   = [0.7, 0.7, 0.7]
    d      = delta_table(triage, fifo)
    np.testing.assert_allclose(d, [0.1, 0.0, -0.1])
