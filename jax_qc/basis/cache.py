"""Local cache for BSE basis-set fetches.

First call to ``fetch_basis('cc-pVTZ', ...)`` hits the BSE library (which
itself is a local database). We additionally cache the returned dict on
disk as JSON keyed by ``(basis_name, elements)`` to shave repeated
lookups during development.

Cache dir defaults to ``~/.jax_qc/basis_cache/``. Override with the
``JAX_QC_CACHE_DIR`` environment variable (used by tests).

FP: These functions have side effects (filesystem) and are isolated here
so the rest of the pipeline stays pure.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Iterable, Optional


def _default_cache_dir() -> pathlib.Path:
    env = os.environ.get("JAX_QC_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".jax_qc" / "basis_cache"


CACHE_DIR: pathlib.Path = _default_cache_dir()


def cache_key(basis_name: str, elements: Iterable[int]) -> str:
    """Deterministic cache key from (basis name, sorted element list)."""
    sorted_elems = sorted({int(z) for z in elements})
    raw = f"{basis_name.lower().strip()}|{sorted_elems}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(basis_name: str, elements: Iterable[int]) -> pathlib.Path:
    return _default_cache_dir() / f"{cache_key(basis_name, elements)}.json"


def get_cached(basis_name: str, elements: Iterable[int]) -> Optional[dict]:
    """Return the cached BSE dict or ``None`` if not present / unreadable."""
    path = _cache_path(basis_name, elements)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def put_cache(
    basis_name: str, elements: Iterable[int], bse_data: dict
) -> pathlib.Path:
    """Write ``bse_data`` to the cache and return the file path."""
    cache_dir = _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(basis_name, elements)}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(bse_data, f)
    tmp.replace(path)
    return path
