"""Shared test fixtures.

Redirects the BSE cache into a per-test-session temporary directory so we
never pollute the user's ``~/.jax_qc`` during ``pytest``.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_cache_dir() -> pathlib.Path:
    with tempfile.TemporaryDirectory(prefix="jax_qc_cache_") as tmp:
        os.environ["JAX_QC_CACHE_DIR"] = tmp
        yield pathlib.Path(tmp)
        os.environ.pop("JAX_QC_CACHE_DIR", None)
