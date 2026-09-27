"""
triageq.sim_fast — High-speed Numba event-loop kernel for queue simulation.

This implements FIFO, NP-PQ, and APQ for a c-server non-preemptive queue.
It is ~100× faster than the SimPy reference implementation, enabling the
large parameter sweeps in E2.

Test K1 requires sim_fast and sim_simpy to give IDENTICAL per-patient waits
for the same input streams (FIFO, NP-PQ, APQ; c ∈ {1, 5}).

Algorithm
---------
For each patient in arrival order:
  1. Find the earliest free server (min over server_free_at).
  2. If all servers are busy at that time, advance clock to next arrival.
  3. Enqueue all patients whose arrival ≤ clock.
  4. Select the next patient to serve according to the policy:
       FIFO   : earliest arrival among queued patients
       NP-PQ  : head of the lowest-numbered non-empty priority queue
       APQ    : patient with highest accrued priority w_k * (clock - arrival)
  5. Record wait = clock − patient_arrival.
  6. Advance the server's free time by service_time.

Data structures
---------------
  - server_free_at: (c,) array of server-free times
  - One FIFO queue per predicted level (FIFO policy uses a single level).
    Patients arrive in index order, so each level's queue is a contiguous
    slice of `order` (patient ids grouped by level, arrival order preserved)
    with a head pointer (next to serve) and a tail pointer (next to enqueue).

Within a level every policy is FIFO, so selection only compares the K queue
heads: NP-PQ takes the lowest non-empty level; APQ takes the head with the
largest w_k * (t - arrival), ties to the earlier arrival. Cost is O(N·K).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Try to import numba; fall back to pure-numpy if unavailable
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    # Provide a no-op decorator so the code still runs (slower)
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator
    _NUMBA_AVAILABLE = False


# ── Policy constants ──────────────────────────────────────────────────────────

POLICY_FIFO  = 0
POLICY_NPPQ  = 1
POLICY_APQ   = 2


# ── Core Numba kernel ─────────────────────────────────────────────────────────

@njit(cache=True)
def _simulate_kernel(
    arrival_times:  NDArray[np.float64],   # (N,)
    service_times:  NDArray[np.float64],   # (N,)
    pred_levels:    NDArray[np.int32],     # (N,) values in [1, K]
    policy:         int,                   # POLICY_FIFO/NPPQ/APQ
    c:              int,
    K:              int,
    apq_weights:    NDArray[np.float64],   # (K,) one per level
) -> NDArray[np.float64]:                  # (N,) waiting times
    """
    Core simulation kernel — runs in Numba nopython mode.

    Returns waiting times for all N patients (no warm-up removal here;
    caller sets waits[0:warmup_n] = NaN).
    """
    N = len(arrival_times)
    waits = np.full(N, np.nan, dtype=np.float64)

    # ── Per-level queues as slices of `order` ─────────────────────────────
    # FIFO ignores levels: everyone goes into queue 0.
    lvl = np.empty(N, dtype=np.int64)
    counts = np.zeros(K, dtype=np.int64)
    for i in range(N):
        q = 0 if policy == POLICY_FIFO else int(pred_levels[i]) - 1
        lvl[i] = q
        counts[q] += 1
    start = np.zeros(K, dtype=np.int64)
    for q in range(1, K):
        start[q] = start[q - 1] + counts[q - 1]
    head = start.copy()          # next patient to serve in level q
    tail = start.copy()          # next free slot in level q
    order = np.empty(N, dtype=np.int64)
    fill = start.copy()
    for i in range(N):
        order[fill[lvl[i]]] = i
        fill[lvl[i]] += 1

    # ── Server state ──────────────────────────────────────────────────────
    server_free_at = np.zeros(c, dtype=np.float64)

    next_arrival_idx = 0
    n_queued = 0
    n_served = 0

    while n_served < N:
        # Earliest free server
        earliest_free = server_free_at[0]
        free_server   = 0
        for s in range(1, c):
            if server_free_at[s] < earliest_free:
                earliest_free = server_free_at[s]
                free_server   = s
        clock = earliest_free

        # Enqueue all arrivals up to clock
        while next_arrival_idx < N and arrival_times[next_arrival_idx] <= clock:
            tail[lvl[next_arrival_idx]] += 1
            n_queued += 1
            next_arrival_idx += 1

        # Idle server and empty queue: jump to the next arrival
        if n_queued == 0:
            clock = arrival_times[next_arrival_idx]
            while next_arrival_idx < N and arrival_times[next_arrival_idx] <= clock:
                tail[lvl[next_arrival_idx]] += 1
                n_queued += 1
                next_arrival_idx += 1

        # ── Select among queue heads ──────────────────────────────────────
        best_q = -1
        if policy == POLICY_APQ:
            best_pri = -math.inf
            best_arr = math.inf
            for q in range(K):
                if head[q] < tail[q]:
                    a   = arrival_times[order[head[q]]]
                    pri = apq_weights[q] * (clock - a)
                    if pri > best_pri or (pri == best_pri and a < best_arr):
                        best_pri = pri
                        best_arr = a
                        best_q   = q
        else:
            # FIFO (single queue) or NP-PQ: lowest non-empty level
            for q in range(K):
                if head[q] < tail[q]:
                    best_q = q
                    break

        chosen = order[head[best_q]]
        head[best_q] += 1

        waits[chosen] = clock - arrival_times[chosen]
        server_free_at[free_server] = clock + service_times[chosen]
        n_queued -= 1
        n_served += 1

    return waits


# ── Public interface ──────────────────────────────────────────────────────────

def run_fast(
    streams: dict[str, Any],
    pred_levels: NDArray[np.int_],
    policy: str,
    c: int,
    apq_weights: list[float] | None = None,
    warmup_frac: float = 0.10,
) -> NDArray[np.float64]:
    """
    Run the Numba simulation kernel.

    Parameters
    ----------
    streams      : output of make_streams()
    pred_levels  : (N,) predicted acuity levels (1..K), or true levels for oracle
    policy       : "fifo" | "np_pq" | "apq" | "oracle"
    c            : number of servers
    apq_weights  : (K,) priority accrual rates; default [16,8,4,2,1]
    warmup_frac  : fraction of patients to discard (set to NaN)

    Returns
    -------
    waits : (N,) float, waiting times; NaN for warm-up patients
    """
    arrival_times = streams["arrival_times"]
    service_times = streams["service_times"]
    true_levels   = streams["true_levels"]
    K             = streams["K"]
    N             = len(arrival_times)

    if apq_weights is None:
        apq_weights = [16.0, 8.0, 4.0, 2.0, 1.0][:K]

    w = np.asarray(apq_weights, dtype=np.float64)

    if policy == "oracle":
        effective_pred = true_levels.astype(np.int32)
        pol_id = POLICY_NPPQ
    elif policy == "fifo":
        effective_pred = np.ones(N, dtype=np.int32)  # ignored
        pol_id = POLICY_FIFO
    elif policy == "np_pq":
        effective_pred = pred_levels.astype(np.int32)
        pol_id = POLICY_NPPQ
    elif policy == "apq":
        effective_pred = pred_levels.astype(np.int32)
        pol_id = POLICY_APQ
    else:
        raise ValueError(f"Unknown policy for sim_fast: {policy}")

    waits = _simulate_kernel(
        arrival_times.astype(np.float64),
        service_times.astype(np.float64),
        effective_pred,
        pol_id,
        c,
        K,
        w,
    )

    # Mark warm-up patients
    warmup_n = int(math.ceil(N * warmup_frac))
    waits[:warmup_n] = np.nan

    return waits
