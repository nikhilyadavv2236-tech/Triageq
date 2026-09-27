# tests/conftest.py — pytest configuration and shared fixtures

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow validation tests (V1–V7, ~10-20 minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (validation suite)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
