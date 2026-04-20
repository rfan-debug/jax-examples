"""Tests for the Step 3 RHF SCF.

Tier 1 benchmark: for every closed-shell test molecule (H2, He, HeH+, H3+)
we compare the total RHF/STO-3G energy to PySCF to <=1 microhartree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.scf.damping import damp
from jax_qc.scf.density import density_rhf
from jax_qc.scf.diis import DIISHistory, diis_extrapolate, diis_history_init
from jax_qc.scf.energy import electronic_energy_rhf
from jax_qc.scf.fock import build_fock_rhf
from jax_qc.scf.guess import core_guess
from jax_qc.scf.orthogonalize import (
    canonical_orthogonalization,
    symmetric_orthogonalization,
)
from jax_qc.scf.rhf import run_rhf

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402


# ----- Tier 1 test molecules ---------------------------------------------

def _h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _he():
    return make_molecule(
        elements=("He",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([2], dtype=jnp.int32),
    )


def _heh_cation():
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


def _h3_plus():
    r = 1.65
    return make_molecule(
        elements=("H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r, 0.0, 0.0],
                [r / 2.0, r * np.sqrt(3) / 2.0, 0.0],
            ]
        ),
        atomic_numbers=jnp.array([1, 1, 1], dtype=jnp.int32),
        charge=+1,
    )


TEST_MOLECULES = {
    "H2": _h2,
    "He": _he,
    "HeH+": _heh_cation,
    "H3+": _h3_plus,
}


def _pyscf_energy(mol, basis_name: str = "sto-3g") -> float:
    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    pmol = gto.M(
        atom=atoms,
        basis=basis_name,
        unit="bohr",
        charge=int(mol.charge),
        spin=int(mol.spin),
    )
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    return float(mf.kernel())


# ----- Unit tests for individual pieces ---------------------------------

def test_symmetric_orthogonalization_gives_identity():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    S = np.asarray(jax_qc.compute_integrals(mol, basis).S)
    X = np.asarray(symmetric_orthogonalization(jnp.asarray(S)))
    np.testing.assert_allclose(X.T @ S @ X, np.eye(S.shape[0]), atol=1e-12)


def test_symmetric_orthogonalization_is_symmetric():
    mol = _h3_plus()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    S = jax_qc.compute_integrals(mol, basis).S
    X = symmetric_orthogonalization(S)
    np.testing.assert_allclose(np.asarray(X), np.asarray(X).T, atol=1e-12)


def test_canonical_orthogonalization_gives_identity():
    mol = _h3_plus()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    S = np.asarray(jax_qc.compute_integrals(mol, basis).S)
    X = np.asarray(canonical_orthogonalization(jnp.asarray(S)))
    np.testing.assert_allclose(X.T @ S @ X, np.eye(X.shape[1]), atol=1e-12)


def test_canonical_orthogonalization_drops_small_eigenvalues():
    # Build a rank-deficient S by hand: two identical rows.
    S = np.array(
        [
            [1.0, 0.9999999, 0.0],
            [0.9999999, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    X = np.asarray(canonical_orthogonalization(jnp.asarray(S), eps=1e-4))
    assert X.shape[1] == 2  # one eigenvalue dropped


def test_density_rhf_trace_equals_n_electrons():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    X = symmetric_orthogonalization(ints.S)
    D = core_guess(ints.H_core, X, n_occ=mol.n_electrons // 2)
    # Tr(D S) = n_electrons.
    assert float(jnp.trace(D @ ints.S)) == pytest.approx(mol.n_electrons, abs=1e-10)


def test_fock_reduces_to_h_core_at_zero_density():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    D = jnp.zeros_like(ints.S)
    F = build_fock_rhf(ints.H_core, D, ints.ERI)
    np.testing.assert_allclose(np.asarray(F), np.asarray(ints.H_core), atol=1e-14)


def test_damp_endpoints():
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(np.asarray(damp(A, B, 0.0)), np.asarray(A))
    np.testing.assert_allclose(np.asarray(damp(A, B, 1.0)), np.asarray(B))


def test_electronic_energy_matches_0_5_trace():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    X = symmetric_orthogonalization(ints.S)
    D = core_guess(ints.H_core, X, n_occ=1)
    F = build_fock_rhf(ints.H_core, D, ints.ERI)
    E = float(electronic_energy_rhf(D, ints.H_core, F))
    E_ref = 0.5 * float(jnp.trace(D @ (ints.H_core + F)))
    assert E == pytest.approx(E_ref, abs=1e-14)


def test_diis_history_init():
    h = diis_history_init(n_basis=4, max_size=6)
    assert h.fock_list.shape == (6, 4, 4)
    assert h.error_list.shape == (6, 4, 4)
    assert h.size == 0
    assert h.max_size == 6


def test_diis_single_entry_returns_input():
    h = diis_history_init(n_basis=2, max_size=6)
    F = jnp.eye(2) * 1.234
    D = jnp.eye(2)
    S = jnp.eye(2)
    X = jnp.eye(2)
    F_ext, h_new, _ = diis_extrapolate(h, F, D, S, X)
    np.testing.assert_allclose(np.asarray(F_ext), np.asarray(F))
    assert h_new.size == 1


# ----- End-to-end RHF vs PySCF ------------------------------------------

@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_rhf_total_energy_matches_pyscf(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", basis="sto-3g", scf_conv=1e-10, max_scf_iter=64)
    result = run_rhf(mol, ints, config)
    assert result.converged, f"{name} RHF did not converge"
    e_ref = _pyscf_energy(mol)
    # Plan Section 9.3 target: <= 1 microhartree.
    assert result.energy == pytest.approx(e_ref, abs=1e-6)


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_rhf_e_total_equals_elec_plus_nuc(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", basis="sto-3g", scf_conv=1e-10)
    result = run_rhf(mol, ints, config)
    assert result.energy == pytest.approx(result.E_elec + result.E_nuc, abs=1e-14)


def test_run_scf_dispatches_to_rhf():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", basis="sto-3g")
    result = jax_qc.run_scf(mol, ints, config)
    assert result.converged


def test_run_scf_rejects_unknown_method():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="ccsd", basis="sto-3g")
    with pytest.raises(NotImplementedError):
        jax_qc.run_scf(mol, ints, config)


def test_rhf_rejects_open_shell():
    # H atom: 1 electron, spin=1 (doublet).
    mol = make_molecule(
        elements=("H",),
        coords=jnp.zeros((1, 3)),
        atomic_numbers=jnp.array([1], dtype=jnp.int32),
        spin=1,
    )
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    # Can't compute integrals/energy because compute_integrals needs no spin info,
    # but run_rhf should reject the spin mismatch.
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf")
    with pytest.raises(ValueError, match="closed-shell"):
        run_rhf(mol, ints, config)


def test_rhf_density_is_symmetric():
    mol = _h3_plus()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", basis="sto-3g", scf_conv=1e-10)
    result = run_rhf(mol, ints, config)
    D = np.asarray(result.state.density)
    np.testing.assert_allclose(D, D.T, atol=1e-12)


def test_rhf_populates_stage_timer():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    timer = jax_qc.StageTimer(sync_device=False)
    config = CalcConfig(method="rhf", basis="sto-3g", scf_conv=1e-10)
    run_rhf(mol, ints, config, timer=timer)
    scf_node = timer.root.children["scf"]
    assert scf_node.fp_abstraction == "monad"
    assert "fock_build" in scf_node.children
    assert scf_node.children["fock_build"].fp_abstraction == "applicative"


def test_rhf_orbital_energies_sorted_ascending():
    mol = _heh_cation()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = run_rhf(mol, ints, config)
    orb = np.asarray(result.state.orbital_energies)
    assert np.all(np.diff(orb) >= 0)


def test_rhf_result_is_pytree():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = run_rhf(mol, ints, config)
    leaves, treedef = jax.tree_util.tree_flatten(result)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.energy == result.energy
