"""Tests for the basis set layer.

Coverage:
  * BSE fetch returns expected structure for H/O STO-3G.
  * Normalization: contracted s-shell self-overlap = 1 after normalization.
  * Parse builds one Shell per (l, contraction) pair with Bohr coordinates.
  * build_basis_set produces a self-consistent BasisSet.
  * On-disk cache round-trips.
  * 6-31G* on water has the right shape (s, s, sp, d on O plus s on each H).
"""

from __future__ import annotations

import json
import math
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from jax_qc.basis import (
    bse_dict_to_shells,
    build_basis_set,
    cache,
    contraction_norm,
    fetch_basis,
    normalize_shell,
    primitive_norm,
)
from jax_qc.basis.cache import cache_key, get_cached, put_cache
from jax_qc.core.types import make_molecule


# ---- Molecules used across tests ----------------------------------------

def _h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _water():
    # Geometry in Bohr, close to equilibrium.
    return make_molecule(
        elements=("O", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.22143],
                [0.0, 1.43052, -0.88572],
                [0.0, -1.43052, -0.88572],
            ]
        ),
        atomic_numbers=jnp.array([8, 1, 1], dtype=jnp.int32),
    )


# ---- bse_fetch ----------------------------------------------------------

def test_fetch_basis_sto3g_hydrogen():
    data = fetch_basis("sto-3g", [1])
    assert "elements" in data
    assert "1" in data["elements"]
    shells = data["elements"]["1"]["electron_shells"]
    assert len(shells) == 1
    assert shells[0]["angular_momentum"] == [0]
    assert len(shells[0]["exponents"]) == 3


def test_fetch_basis_sto3g_oxygen_has_sp_split():
    # With uncontract_spdf=True, the sp shell of oxygen should be split
    # into an s-shell and a p-shell (3 shells total: 1s core, 2s, 2p).
    data = fetch_basis("sto-3g", [8])
    shells = data["elements"]["8"]["electron_shells"]
    ls = [s["angular_momentum"] for s in shells]
    assert [0] in ls and [1] in ls
    assert sum(1 for s in shells if s["angular_momentum"] == [0]) == 2
    assert sum(1 for s in shells if s["angular_momentum"] == [1]) == 1


def test_fetch_basis_rejects_empty_elements():
    with pytest.raises(ValueError):
        fetch_basis("sto-3g", [])


# ---- normalize ----------------------------------------------------------

def test_primitive_norm_s_formula():
    # For l=0: N = (2 alpha / pi)^(3/4).
    for a in (0.1, 1.0, 10.0):
        expected = (2.0 * a / math.pi) ** 0.75
        assert primitive_norm(a, 0) == pytest.approx(expected)


def test_primitive_norm_p_formula():
    # For l=1: N = (2 alpha / pi)^(3/4) * sqrt(4 alpha / 1).
    for a in (0.5, 1.0, 2.5):
        expected = (2.0 * a / math.pi) ** 0.75 * math.sqrt(4.0 * a / 1.0)
        assert primitive_norm(a, 1) == pytest.approx(expected)


def test_normalize_shell_self_overlap_unity_s():
    # STO-3G hydrogen 1s.
    alphas = np.array([3.42525091, 0.62391373, 0.16885540])
    d = np.array([0.15432897, 0.53532814, 0.44463454])
    c = normalize_shell(alphas, d, 0)
    # After normalize_shell, the overall overlap <chi|chi> must be 1.
    alpha_sum = alphas[:, None] + alphas[None, :]
    prefactor = (math.pi / alpha_sum) ** 1.5
    overlap = float(np.sum(c[:, None] * c[None, :] * prefactor))
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_normalize_shell_self_overlap_unity_p():
    # 2p primitives (example set).
    alphas = np.array([5.0, 1.0, 0.3])
    d = np.array([0.1, 0.4, 0.5])
    c = normalize_shell(alphas, d, 1)
    alpha_sum = alphas[:, None] + alphas[None, :]
    prefactor = (math.pi / alpha_sum) ** 1.5
    angular = 1.0 / (2.0 * alpha_sum) ** 1
    overlap = float(np.sum(c[:, None] * c[None, :] * prefactor * angular))
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_contraction_norm_positive():
    alphas = np.array([1.0, 0.5])
    d = np.array([0.2, 0.8])
    norms = np.array([primitive_norm(float(a), 0) for a in alphas])
    M = contraction_norm(alphas, d * norms, 0)
    assert M > 0.0


# ---- parse --------------------------------------------------------------

def test_bse_dict_to_shells_h2_sto3g():
    mol = _h2()
    bse_data = fetch_basis("sto-3g", [1])
    shells = bse_dict_to_shells(bse_data, mol)
    # H2 in STO-3G: 2 atoms * 1 shell = 2 shells, each s-type, 3 primitives.
    assert len(shells) == 2
    for i, sh in enumerate(shells):
        assert sh.angular_momentum == 0
        assert sh.exponents.shape == (3,)
        assert sh.coefficients.shape == (3,)
        assert sh.atom_index == i
        # Shell centers should match molecule coords.
        assert np.allclose(np.asarray(sh.center), np.asarray(mol.coords[i]))


def test_bse_dict_to_shells_water_sto3g():
    mol = _water()
    # Need both H and O data.
    bse_data = fetch_basis("sto-3g", [1, 8])
    shells = bse_dict_to_shells(bse_data, mol)
    # O: 3 shells (1s, 2s, 2p). H: 1 shell each.
    assert len(shells) == 3 + 1 + 1
    assert sum(1 for s in shells if s.atom_index == 0) == 3
    l_values = sorted(s.angular_momentum for s in shells if s.atom_index == 0)
    assert l_values == [0, 0, 1]


# ---- build_basis_set ----------------------------------------------------

def test_build_basis_set_h2_sto3g():
    mol = _h2()
    basis = build_basis_set(mol, "sto-3g", use_cache=False)
    assert basis.n_basis == 2
    assert len(basis.shells) == 2
    assert basis.name == "sto-3g"
    assert basis.spherical is True
    # shell_to_basis must partition [0, n_basis).
    flat = [idx for group in basis.shell_to_basis for idx in group]
    assert flat == list(range(basis.n_basis))
    assert np.asarray(basis.basis_to_atom).tolist() == [0, 1]


def test_build_basis_set_water_sto3g():
    mol = _water()
    basis = build_basis_set(mol, "sto-3g", use_cache=False)
    # 1s, 2s, 2p(3) on O + 1 on each H = 5 + 1 + 1 = 7.
    assert basis.n_basis == 7
    assert len(basis.shells) == 5
    atom_mapping = np.asarray(basis.basis_to_atom).tolist()
    # First 5 belong to O (atom 0), then one H each.
    assert atom_mapping == [0, 0, 0, 0, 0, 1, 2]


def test_build_basis_set_water_6_31gstar():
    # 6-31G* adds a d polarization on O.
    mol = _water()
    basis = build_basis_set(mol, "6-31G*", use_cache=False)
    # On O: 1s(1) + 2s(1) + 2s'(1) + 2p(3) + 2p'(3) + d(5, spherical) = 14.
    # On each H: 1s(1) + 1s'(1) = 2, so 4 total.
    # Grand total = 18.
    assert basis.n_basis == 18
    o_atoms = [s for s in basis.shells if s.atom_index == 0]
    h_atoms = [s for s in basis.shells if s.atom_index > 0]
    ls = sorted(s.angular_momentum for s in o_atoms)
    assert ls == [0, 0, 0, 1, 1, 2]
    for h in h_atoms:
        assert h.angular_momentum == 0


# ---- cache --------------------------------------------------------------

def test_cache_key_deterministic():
    k1 = cache_key("sto-3g", [1, 8])
    k2 = cache_key("STO-3g", [8, 1])
    assert k1 == k2
    assert cache_key("6-31g*", [1]) != k1


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_QC_CACHE_DIR", str(tmp_path))
    # Reimport cache module fields that resolve the dir dynamically.
    payload = {"elements": {"1": {"electron_shells": []}}, "name": "test"}
    put_cache("dummy", [1], payload)
    recovered = get_cached("dummy", [1])
    assert recovered == payload


def test_build_basis_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_QC_CACHE_DIR", str(tmp_path))
    mol = _h2()
    # First build: cache miss, populates.
    basis = build_basis_set(mol, "sto-3g", use_cache=True)
    cache_files = list(pathlib.Path(tmp_path).glob("*.json"))
    assert len(cache_files) == 1
    # Second build: cache hit; result should match.
    basis2 = build_basis_set(mol, "sto-3g", use_cache=True)
    assert basis.n_basis == basis2.n_basis
    assert [s.angular_momentum for s in basis.shells] == [
        s.angular_momentum for s in basis2.shells
    ]
