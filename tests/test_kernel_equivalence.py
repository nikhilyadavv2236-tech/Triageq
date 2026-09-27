"""
tests/test_kernel_equivalence.py — Test K1: SimPy == Numba on the same streams.

For FIFO, NP-PQ, and APQ, with c ∈ {1, 5}, the two simulators must return
IDENTICAL per-patient waiting times (post-warm-up) when given the same input.

This is the key correctness test: once K1 passes, all sweep results from
sim_fast.py are trusted as equivalent to the reference sim_simpy.py.

Test uses N = 5,000 patients (fast enough to run without @slow marker).
"""

import numpy as np
import pytest

from triageq.streams import make_streams, ServiceConfig, replication_seeds
from triageq.errors import predict_levels
from triageq.sim_simpy import run_simpy
from triageq.sim_fast import run_fast

K   = 5
PI  = np.array([0.01, 0.12, 0.42, 0.35, 0.10])
N   = 5_000
RHO = 0.7


def _get_streams_and_pred(c: int, seed_int: int = 42):
    cfg     = ServiceConfig(family="exponential", mean_min=20.0)
    seeds   = replication_seeds(1, master_seed=seed_int)
    streams = make_streams(N, RHO, c, PI, cfg, seed=seeds[0])
    eps     = streams["eps"]
    true_lv = streams["true_levels"]
    pred    = predict_levels(true_lv, eps, sigma=0.8, b=0.0, K=K)
    return streams, pred


@pytest.mark.parametrize("policy", ["fifo", "np_pq", "apq"])
@pytest.mark.parametrize("c", [1, 5])
def test_k1_kernel_equivalence(policy: str, c: int):
    """K1: SimPy and Numba must give identical per-patient waits (atol=1e-6)."""
    streams, pred = _get_streams_and_pred(c)

    apq_w = [16.0, 8.0, 4.0, 2.0, 1.0]

    waits_simpy, _ = run_simpy(
        streams, pred,
        policy=policy,
        c=c,
        apq_weights=apq_w,
        warmup_frac=0.10,
    )

    waits_fast = run_fast(
        streams, pred,
        policy=policy,
        c=c,
        apq_weights=apq_w,
        warmup_frac=0.10,
    )

    # Both should have the same NaN mask
    nan_simpy = np.isnan(waits_simpy)
    nan_fast  = np.isnan(waits_fast)
    assert np.array_equal(nan_simpy, nan_fast), \
        f"NaN mask differs: policy={policy}, c={c}"

    # Where both are valid, waits must match
    valid = ~nan_simpy
    np.testing.assert_allclose(
        waits_simpy[valid],
        waits_fast[valid],
        atol=1e-6,
        err_msg=f"Waits differ for policy={policy}, c={c}",
    )
