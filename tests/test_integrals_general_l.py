"""Tests for the Step 4 general angular momentum integrals.

We exercise the McMurchie-Davidson primitives + Cartesian-to-spherical
transformation against PySCF on a small set of molecules that include
p- and d-type basis functions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import make_molecule
from jax_qc.integrals.mcmurchie_davidson import (
    cartesian_components,
    hermite_e,
    hermite_r_aux,
    n_cartesian,
)
from jax_qc.integrals.obara_saika import (
    contracted_overlap_block,
    contracted_kinetic_block,
    contracted_eri_block,
    primitive_overlap_block,
    primitive_kinetic_block,
    primitive_nuclear_block,
    primitive_eri_block,
)
from jax_qc.integrals.spherical import cart_to_spherical
from jax_qc.integrals.screening import compute_shell_pair_bounds

pyscf = pytest.importorskip("pyscf")
from pyscf import gto  # noqa: E402


def _h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _water():
    return make_molecule(
        elements=("O", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ]
        ),
        atomic_numbers=jnp.array([8, 1, 1], dtype=jnp.int32),
    )


def _pyscf_mol(mol, basis_name):
    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    return gto.M(
        atom=atoms,
        basis=basis_name,
        unit="bohr",
        charge=int(mol.charge),
        spin=int(mol.spin),
    )


# ----- McMurchie-Davidson core ---------------------------------------

def test_cartesian_components_count():
    for l in range(4):
        comps = cartesian_components(l)
        assert len(comps) == n_cartesian(l)
        for (lx, ly, lz) in comps:
            assert lx + ly + lz == l


def test_cartesian_components_libint_order_p():
    # libint order for l=1 puts (1,0,0), (0,1,0), (0,0,1).
    assert cartesian_components(1) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def test_cartesian_components_libint_order_d():
    assert cartesian_components(2) == (
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    )


def test_hermite_e_base_case():
    # E[0, 0, 0] = 1 by construction; everything else zero for (i, j) = (0, 0).
    E = hermite_e(0, 0, 0.5, -0.7, 1.3)
    assert E.shape == (1, 1, 1)
    assert E[0, 0, 0] == pytest.approx(1.0)


def test_hermite_e_one_zero():
    # E[1, 0, 0] = X_PA, E[1, 0, 1] = 1/(2p).
    Xpa, Xpb, p = 0.3, -0.2, 1.5
    E = hermite_e(1, 0, Xpa, Xpb, p)
    assert E[1, 0, 0] == pytest.approx(Xpa)
    assert E[1, 0, 1] == pytest.approx(1.0 / (2.0 * p))


def test_hermite_r_aux_base_case():
    # R[0,0,0,n] = (-2p)^n F_n(p |PC|^2). Check n = 0 against direct.
    p = 1.4
    PC = np.array([0.3, -0.1, 0.2])
    R = hermite_r_aux(0, 0, 0, 0, p, PC)
    from jax_qc.integrals.boys import boys_f0

    T = p * float(np.dot(PC, PC))
    assert float(R[0, 0, 0, 0]) == pytest.approx(float(boys_f0(T)))


# ----- Spherical transformations -------------------------------------

def test_cart_to_spherical_l0_l1_identity_for_p():
    np.testing.assert_allclose(cart_to_spherical(1), np.eye(3))


def test_cart_to_spherical_d_shape():
    C = cart_to_spherical(2)
    assert C.shape == (5, 6)


def test_cart_to_spherical_unsupported_l_raises():
    with pytest.raises(NotImplementedError):
        cart_to_spherical(3)


# ----- Primitive sanity (vs analytic s|s) ----------------------------

def test_primitive_overlap_ss_matches_analytic():
    alpha, beta = 1.0, 1.5
    A = np.array([0.1, 0.0, 0.0]); B = np.array([0.0, 0.0, 0.7])
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = float(np.sum((A - B) ** 2))
    expected = (np.pi / p) ** 1.5 * np.exp(-mu * AB2)
    got = primitive_overlap_block(alpha, A, 0, beta, B, 0)[0, 0]
    assert got == pytest.approx(expected, rel=1e-14)


def test_primitive_kinetic_ss_matches_analytic():
    alpha, beta = 1.0, 1.5
    A = np.array([0.1, 0.0, 0.0]); B = np.array([0.0, 0.0, 0.7])
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = float(np.sum((A - B) ** 2))
    S_ss = (np.pi / p) ** 1.5 * np.exp(-mu * AB2)
    expected = mu * (3.0 - 2.0 * mu * AB2) * S_ss
    got = primitive_kinetic_block(alpha, A, 0, beta, B, 0)[0, 0]
    assert got == pytest.approx(expected, rel=1e-14)


def test_primitive_nuclear_ss_matches_analytic():
    from jax_qc.integrals.boys import boys_f0
    alpha, beta = 0.7, 1.2
    A = np.array([0.0, 0.1, 0.0]); B = np.array([0.0, 0.3, 0.7])
    Cnuc = np.array([[0.2, 0.0, 0.5]])
    Z = np.array([3.0])
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    PC2 = float(np.sum((P - Cnuc[0]) ** 2))
    AB2 = float(np.sum((A - B) ** 2))
    expected = -2.0 * np.pi / p * 3.0 * np.exp(-mu * AB2) * float(boys_f0(p * PC2))
    got = primitive_nuclear_block(alpha, A, 0, beta, B, 0, Cnuc, Z)[0, 0]
    assert got == pytest.approx(expected, rel=1e-14)


def test_primitive_eri_ssss_matches_analytic():
    from jax_qc.integrals.boys import boys_f0
    alpha, beta = 1.0, 1.5
    gamma, delta = 0.8, 1.2
    A = np.array([0.1, 0.0, 0.0]); B = np.array([0.0, 0.0, 0.7])
    Cc = np.array([0.2, 0.3, 0.0]); Dd = np.array([0.0, 0.5, 0.4])
    p = alpha + beta; q = gamma + delta
    mu = alpha * beta / p
    mu_q = gamma * delta / q
    rho = p * q / (p + q)
    P = (alpha * A + beta * B) / p
    Q = (gamma * Cc + delta * Dd) / q
    AB2 = float(np.sum((A - B) ** 2))
    CD2 = float(np.sum((Cc - Dd) ** 2))
    PQ2 = float(np.sum((P - Q) ** 2))
    expected = (
        2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
        * np.exp(-mu * AB2 - mu_q * CD2)
        * float(boys_f0(rho * PQ2))
    )
    got = primitive_eri_block(
        alpha, A, 0, beta, B, 0, gamma, Cc, 0, delta, Dd, 0
    )[0, 0, 0, 0]
    assert got == pytest.approx(expected, rel=1e-14)


# ----- End-to-end vs PySCF -------------------------------------------

@pytest.mark.parametrize(
    "mol_name,basis_name",
    [
        ("H2", "sto-3g"),
        ("H2", "6-31g"),
        ("H2O", "sto-3g"),
        ("H2O", "6-31g"),
    ],
)
def test_overlap_kinetic_match_pyscf(mol_name, basis_name):
    mol = _h2() if mol_name == "H2" else _water()
    basis = jax_qc.build_basis_set(mol, basis_name)
    ints = jax_qc.compute_integrals(mol, basis)
    pmol = _pyscf_mol(mol, basis_name)
    S_ref = pmol.intor("int1e_ovlp")
    T_ref = pmol.intor("int1e_kin")
    np.testing.assert_allclose(np.asarray(ints.S), S_ref, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ints.T), T_ref, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_name,basis_name",
    [
        ("H2O", "6-31g"),
    ],
)
def test_nuclear_and_eri_match_pyscf(mol_name, basis_name):
    """Heavier element-wise check vs PySCF (deselected by default)."""
    mol = _water() if mol_name == "H2O" else _h2()
    basis = jax_qc.build_basis_set(mol, basis_name)
    ints = jax_qc.compute_integrals(mol, basis)
    pmol = _pyscf_mol(mol, basis_name)
    V_ref = pmol.intor("int1e_nuc")
    ERI_ref = pmol.intor("int2e").reshape(*ints.ERI.shape)
    np.testing.assert_allclose(np.asarray(ints.V), V_ref, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ints.ERI), ERI_ref, atol=1e-5)


# ----- Schwarz screening ---------------------------------------------

def test_shell_pair_bounds_h2():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "6-31g")
    Q = compute_shell_pair_bounds(basis)
    assert Q.shape == (len(basis.shells), len(basis.shells))
    assert np.all(Q >= 0.0)
    np.testing.assert_allclose(Q, Q.T, atol=1e-14)


def test_schwarz_bound_holds():
    """For every shell quartet, |(ij|kl)| <= Q[i,j] * Q[k,l]."""
    # H2 / sto-3g is small (2 shells), so the n^4 inner loop is tiny.
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    Q = compute_shell_pair_bounds(basis)
    n_shells = len(basis.shells)
    for i in range(n_shells):
        for j in range(n_shells):
            for k in range(n_shells):
                for l in range(n_shells):
                    block = contracted_eri_block(
                        basis.shells[i], basis.shells[j],
                        basis.shells[k], basis.shells[l],
                    )
                    assert float(np.max(np.abs(block))) <= Q[i, j] * Q[k, l] + 1e-10
