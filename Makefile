PYTHON   ?= python
PYTEST   := $(PYTHON) -m pytest
CONFIGS  := configs
RESULTS  := results

# ── Setup ──────────────────────────────────────────────────────────────────────
# Python 3.11 (numba 0.59 / numpy 1.26 do not support 3.13+). Install from project root.
install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# ── Testing ────────────────────────────────────────────────────────────────────
# Fast tests only (< 60 s): skips anything marked @pytest.mark.slow
test:
	$(PYTEST) tests/ -v -m "not slow" --timeout=60

# All tests including the slow serial SimPy validation suite (hours).
# `make validate` runs the same checks in parallel and writes T2/F7/F8.
test-all:
	$(PYTEST) tests/ -v --run-slow --timeout=20000

# ── Experiments ────────────────────────────────────────────────────────────────
# E0: validation (V1–V8) — generates results/validation/ and T2, F7, F8
validate:
	$(PYTHON) -m triageq.run --config $(CONFIGS)/e0.yaml

# E1: two-class break-even vs utilization (mean wait and TC)
e1:
	$(PYTHON) -m triageq.run --config $(CONFIGS)/e1.yaml

# E2: five-level core sweep (the main result)
e2:
	$(PYTHON) -m triageq.run --config $(CONFIGS)/e2.yaml

# E3: robustness (service distributions, server count)
e3:
	$(PYTHON) -m triageq.run --config $(CONFIGS)/e3.yaml

# Alias: run e1 + e2 (the main computational work)
sweep: e1 e2

# ── Data pipeline ──────────────────────────────────────────────────────────────
data:
	$(PYTHON) -m triageq.data.nhamcs_load
	$(PYTHON) -m triageq.data.reference
	$(PYTHON) -m triageq.data.features

# ── ML ─────────────────────────────────────────────────────────────────────────
ml:
	$(PYTHON) -m triageq.ml.train
	$(PYTHON) -m triageq.ml.evaluate
	$(PYTHON) -m triageq.ml.thresholds

# ── Figures and tables ─────────────────────────────────────────────────────────
figures:
	$(PYTHON) -m triageq.figures

# ── Full pipeline ──────────────────────────────────────────────────────────────
all: validate e1 e2 e3 data ml figures

.PHONY: install test test-all validate e1 e2 e3 sweep data ml figures all clean

# ── Housekeeping ───────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
