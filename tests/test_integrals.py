"""Tests for the Step 2 s-type molecular integrals.

Strategy: for every tier-1 test molecule (H2, He, HeH+, H3+) we build the
integral set with jax_qc and compare element-wise to PySCF's reference.

PySCF ships a well-tested STO-3G implementation; matching it to ~1e-10 is
more than enough to catch algorithmic errors at this stage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import make_molecule
from jax_qc.integrals.gaussian_product import (
    distance_squared,
    gaussian_product_center,
    gaussian_product_exponent,
)
from jax_qc.integrals.nuclear import nuclear_repulsion_energy
from jax_qc.integrals.overlap import overlap_primitive_ss

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402  (import after skip guard)


# ----- Test molecules ---------------------------------------------------

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
    # HeH+: 2 electrons, charge +1. Bond length ~1.4632 Bohr.
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


def _h3_plus():
    # Equilateral H3+: bond length ~1.65 Bohr. Closed-shell (2 electrons).
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


def _pyscf_mol(mol, basis_name: str = "sto-3g"):
    """Build the matching PySCF Mole object for a jax_qc Molecule."""
    coords = np.asarray(mol.coords)
    atoms = []
    for sym, xyz in zip(mol.elements, coords):
        atoms.append([sym, tuple(xyz.tolist())])
    # PySCF needs an explicit spin for odd-electron systems.
    spin = int(mol.spin)
    return gto.M(
        atom=atoms,
        basis=basis_name,
        unit="bohr",
        charge=int(mol.charge),
        spin=spin,
    )


# ----- Gaussian product helpers ----------------------------------------

def test_distance_squared():
    a = jnp.array([0.0, 0.0, 0.0])
    b = jnp.array([1.0, 2.0, 2.0])
    assert float(distance_squared(a, b)) == pytest.approx(9.0)


def test_gaussian_product_center_midpoint_when_exponents_equal():
    A = jnp.array([0.0, 0.0, 0.0])
    B = jnp.array([1.0, 0.0, 0.0])
    P = gaussian_product_center(1.0, A, 1.0, B)
    np.testing.assert_allclose(np.asarray(P), [0.5, 0.0, 0.0])


def test_gaussian_product_exponent():
    assert float(gaussian_product_exponent(2.0, 3.5)) == pytest.approx(5.5)


def test_overlap_primitive_self_is_normalized_factor():
    # (pi / (2 alpha))^(3/2) is the unnormalized primitive self-overlap.
    alpha = 1.234
    A = jnp.zeros(3)
    got = float(overlap_primitive_ss(alpha, A, alpha, A))
    expected = (np.pi / (2 * alpha)) ** 1.5
    assert got == pytest.approx(expected, rel=1e-14)


# ----- Overlap / kinetic / nuclear vs PySCF -----------------------------

@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_overlap_matches_pyscf(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    S = np.asarray(jax_qc.compute_integrals(mol, basis).S)
    S_ref = _pyscf_mol(mol).intor("int1e_ovlp")
    np.testing.assert_allclose(S, S_ref, atol=1e-10)


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_kinetic_matches_pyscf(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    T = np.asarray(jax_qc.compute_integrals(mol, basis).T)
    T_ref = _pyscf_mol(mol).intor("int1e_kin")
    np.testing.assert_allclose(T, T_ref, atol=1e-10)


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_nuclear_matches_pyscf(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    V = np.asarray(jax_qc.compute_integrals(mol, basis).V)
    V_ref = _pyscf_mol(mol).intor("int1e_nuc")
    np.testing.assert_allclose(V, V_ref, atol=1e-10)


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_h_core_equals_t_plus_v(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    np.testing.assert_allclose(
        np.asarray(ints.H_core), np.asarray(ints.T + ints.V), atol=1e-14
    )


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_eri_matches_pyscf(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    eri = np.asarray(jax_qc.compute_integrals(mol, basis).ERI)
    eri_ref = _pyscf_mol(mol).intor("int2e").reshape(eri.shape)
    np.testing.assert_allclose(eri, eri_ref, atol=1e-9)


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_eri_is_8fold_symmetric(name: str):
    mol = TEST_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    eri = np.asarray(jax_qc.compute_integrals(mol, basis).ERI)
    # Check all 8 permutations agree.
    np.testing.assert_allclose(eri, np.transpose(eri, (1, 0, 2, 3)), atol=1e-14)
    np.testing.assert_allclose(eri, np.transpose(eri, (0, 1, 3, 2)), atol=1e-14)
    np.testing.assert_allclose(eri, np.transpose(eri, (2, 3, 0, 1)), atol=1e-14)


@pytest.mark.parametrize("name", list(TEST_MOLECULES))
def test_nuclear_repulsion_matches_pyscf(name: str):
    mol = TEST_MOLECULES[name]()
    e_nuc = nuclear_repulsion_energy(mol)
    e_ref = _pyscf_mol(mol).energy_nuc()
    assert e_nuc == pytest.approx(e_ref, abs=1e-12)


def test_overlap_is_symmetric_and_unit_diagonal():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    S = np.asarray(jax_qc.compute_integrals(mol, basis).S)
    np.testing.assert_allclose(S, S.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(S), np.ones(basis.n_basis), atol=1e-12)


def test_nuclear_repulsion_single_atom_is_zero():
    mol = _he()
    assert nuclear_repulsion_energy(mol) == 0.0


def test_nuclear_repulsion_rejects_coincident_atoms():
    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    with pytest.raises(ValueError):
        nuclear_repulsion_energy(mol)


def test_compute_integrals_rejects_non_s_shell():
    # 6-31G* puts d-shells on oxygen — must raise until Step 4.
    water = make_molecule(
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
    basis = jax_qc.build_basis_set(water, "6-31G*")
    with pytest.raises(NotImplementedError, match="s-type"):
        jax_qc.compute_integrals(water, basis)


def test_compute_integrals_populates_stage_timer():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    timer = jax_qc.StageTimer(sync_device=False)
    ints = jax_qc.compute_integrals(mol, basis, timer=timer)
    # Integrals parent and all five sub-stages must be recorded.
    integrals = timer.root.children["integrals"]
    assert integrals.fp_abstraction == "applicative"
    for sub in ("overlap", "kinetic", "nuclear", "eri", "nuclear_repulsion"):
        assert sub in integrals.children
    # Sanity: the S diagonal is still 1 even when timed.
    np.testing.assert_allclose(np.diag(np.asarray(ints.S)), np.ones(basis.n_basis))


def test_integral_set_is_pytree():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    leaves, treedef = jax.tree_util.tree_flatten(ints)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt.S), np.asarray(ints.S))
    assert rebuilt.E_nuc == ints.E_nuc
