"""
triageq.sim_simpy — Reference discrete-event simulator using SimPy.

Supports policies: FIFO, NP-PQ, APQ, P-PQ, and abandonment (LWBS).
This is the ground-truth reference; validated against analytic formulas (V1–V7).
The Numba kernel (sim_fast.py) must produce identical per-patient waits (test K1).

Policy codes
------------
    "fifo"   : First-come, first-served (ignore predicted level)
    "np_pq"  : Non-preemptive priority by predicted level (lower int = higher priority)
    "apq"    : Accumulating priority queue (Kleinrock/Stanford-Taylor-Ziedins)
    "p_pq"   : Preemptive-resume priority by predicted level
    "oracle" : NP-PQ on TRUE level (upper bound)

Usage
-----
    from triageq.streams import make_streams, ServiceConfig, replication_seeds
    from triageq.sim_simpy import run_simpy

    streams = make_streams(n=100_000, rho=0.8, c=5, ...)
    waits, events = run_simpy(streams, pred_levels=pred, policy="np_pq", c=5)
"""

from __future__ import annotations

import math
import heapq
from typing import Any

import numpy as np
import simpy
from numpy.typing import NDArray


# ── Internal state for APQ ────────────────────────────────────────────────────

class _APQQueue:
    """
    Accumulating priority queue built on top of SimPy.

    Each waiting patient accrues priority at rate w_k (their predicted level's
    weight) per minute of waiting. The server always takes the patient with the
    highest accrued priority (= w_k * (now - arrival_time)).
    Within a tied priority, earlier arrival wins.
    """

    def __init__(self, env: simpy.Environment, c: int, weights: list[float]):
        self.env     = env
        self.c       = c
        self.weights = weights                  # w_k, index 0 = level 1
        self._queue  : list[tuple] = []        # heap: (-priority, arrival, patient_id)
        self._servers_free = c
        self._server_event = env.event()       # triggered when a server frees up

    def request(self, arrival_t: float, pred_level: int, patient_id: int) -> simpy.Event:
        """Register a patient and return an event that fires when service starts."""
        ev = self.env.event()
        heapq.heappush(self._queue, (arrival_t, pred_level, patient_id, ev))
        self._try_dispatch()
        return ev

    def release(self) -> None:
        """Signal that a server has freed up."""
        self._servers_free += 1
        self._try_dispatch()

    def _try_dispatch(self) -> None:
        while self._servers_free > 0 and self._queue:
            # Find best patient: highest accrued priority w_k * (now - arrival)
            now = self.env.now
            best_idx = 0
            best_pri = -math.inf
            for i, (arr, pred, pid, ev) in enumerate(self._queue):
                w = self.weights[pred - 1]
                pri = w * (now - arr)
                if pri > best_pri or (pri == best_pri and arr < self._queue[best_idx][0]):
                    best_pri = pri
                    best_idx = i
            # Pop best (rebuild heap after arbitrary removal)
            item = self._queue.pop(best_idx)
            heapq.heapify(self._queue)
            self._servers_free -= 1
            _, _, _, ev = item
            ev.succeed()


# ── Main simulation function ───────────────────────────────────────────────────

def run_simpy(
    streams: dict[str, Any],
    pred_levels: NDArray[np.int_],
    policy: str,
    c: int,
    apq_weights: list[float] | None = None,
    warmup_frac: float = 0.10,
    patience: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """
    Run a discrete-event simulation using SimPy.

    Parameters
    ----------
    streams      : output of make_streams()
    pred_levels  : (N,) predicted acuity levels (1..K)
    policy       : "fifo" | "np_pq" | "apq" | "p_pq" | "oracle"
    c            : number of servers
    apq_weights  : (K,) priority accrual rates (APQ only); default [16,8,4,2,1]
    warmup_frac  : fraction of patients to discard as warm-up
    patience     : (N,) patience times for abandonment; None = no abandonment

    Returns
    -------
    waits  : (N,) float, waiting times (NaN for LWBS patients, NaN for warm-up)
    events : dict with summary info (n_served, n_lwbs, warmup_cutoff)
    """
    arrival_times = streams["arrival_times"]
    service_times = streams["service_times"]
    true_levels   = streams["true_levels"]
    N = len(arrival_times)
    K = streams["K"]

    if apq_weights is None:
        apq_weights = [16.0, 8.0, 4.0, 2.0, 1.0][:K]

    # Determine effective priority per patient
    if policy == "oracle":
        priorities = true_levels
    else:
        priorities = pred_levels

    waits       = np.full(N, np.nan, dtype=np.float64)
    lwbs_flags  = np.zeros(N, dtype=bool)
    warmup_n    = int(math.ceil(N * warmup_frac))

    env = simpy.Environment()

    # ── Build resource according to policy ────────────────────────────────
    if policy in ("fifo", "np_pq", "oracle"):
        resource = simpy.PriorityResource(env, capacity=c)
    elif policy == "p_pq":
        resource = simpy.PreemptiveResource(env, capacity=c)
    elif policy == "apq":
        apq_queue = _APQQueue(env, c, apq_weights)
        resource  = None  # APQ uses its own scheduler
    else:
        raise ValueError(f"Unknown policy: {policy}")

    def patient_process(pid: int) -> None:
        arr_t  = arrival_times[pid]
        svc_t  = service_times[pid]
        pred_k = int(priorities[pid])
        pati   = patience[pid] if patience is not None else math.inf

        # Wait until arrival time
        yield env.timeout(max(0.0, arr_t - env.now))
        arrive_time = env.now

        if policy == "apq":
            # APQ: wait for dispatch event
            dispatch_event = apq_queue.request(arrive_time, pred_k, pid)
            pat_timeout    = env.timeout(pati)
            result = yield dispatch_event | pat_timeout
            if pat_timeout in result:  # abandoned
                lwbs_flags[pid] = True
                return
            wait = env.now - arrive_time
            if pid >= warmup_n:
                waits[pid] = wait
            # Serve
            yield env.timeout(svc_t)
            apq_queue.release()

        elif policy == "p_pq":
            # Preemptive: priority resource
            req = resource.request(priority=pred_k, preempt=True)
            pat_timeout = env.timeout(pati)
            result = yield req | pat_timeout
            if pat_timeout in result:
                resource.release(req)
                lwbs_flags[pid] = True
                return
            wait = env.now - arrive_time
            if pid >= warmup_n:
                waits[pid] = wait
            yield env.timeout(svc_t)
            resource.release(req)

        else:
            # FIFO or NP-PQ: PriorityResource
            # priority = 0 for FIFO (same for all), else pred_k (lower = higher priority)
            pri = 0 if policy == "fifo" else pred_k
            req = resource.request(priority=pri)
            pat_timeout = env.timeout(pati)
            result = yield req | pat_timeout
            if pat_timeout in result:
                resource.release(req)
                lwbs_flags[pid] = True
                return
            wait = env.now - arrive_time
            if pid >= warmup_n:
                waits[pid] = wait
            yield env.timeout(svc_t)
            resource.release(req)

    # Launch all patient processes in arrival order
    for pid in range(N):
        env.process(patient_process(pid))

    env.run()

    events = {
        "n_served":      int((~lwbs_flags).sum()),
        "n_lwbs":        int(lwbs_flags.sum()),
        "warmup_cutoff": warmup_n,
        "true_levels":   true_levels,
        "pred_levels":   pred_levels,
    }

    return waits, events
