"""Shared test fixtures and pytest config.

Redirects the BSE cache into a per-test-session temporary directory so we
never pollute the user's ``~/.jax_qc`` during ``pytest``. Also registers
the ``slow`` marker for tests that should opt out of default runs.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect by default; run with '-m slow')",
    )


def pytest_collection_modifyitems(config, items):
    # If the user explicitly selected markers via -m, honor that and do
    # nothing here. Otherwise skip anything tagged @pytest.mark.slow.
    if config.getoption("-m"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with 'pytest -m slow'")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True, scope="session")
def _isolated_cache_dir() -> pathlib.Path:
    with tempfile.TemporaryDirectory(prefix="jax_qc_cache_") as tmp:
        os.environ["JAX_QC_CACHE_DIR"] = tmp
        yield pathlib.Path(tmp)
        os.environ.pop("JAX_QC_CACHE_DIR", None)
