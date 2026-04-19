"""Basis set acquisition and parsing (Pure, Applicative).

Fetches basis set definitions from Basis Set Exchange, parses them into
internal Shell/BasisSet structures, applies normalization, and caches the
parsed data locally.
"""

from jax_qc.basis.bse_fetch import (
    fetch_basis,
    list_available_bases,
    get_basis_info,
)
from jax_qc.basis.parse import bse_dict_to_shells, parse_electron_shell
from jax_qc.basis.normalize import (
    primitive_norm,
    contraction_norm,
    normalize_shell,
)
from jax_qc.basis.cache import cache_key, get_cached, put_cache, CACHE_DIR
from jax_qc.basis.build import build_basis_set

__all__ = [
    "fetch_basis",
    "list_available_bases",
    "get_basis_info",
    "bse_dict_to_shells",
    "parse_electron_shell",
    "primitive_norm",
    "contraction_norm",
    "normalize_shell",
    "cache_key",
    "get_cached",
    "put_cache",
    "CACHE_DIR",
    "build_basis_set",
]
