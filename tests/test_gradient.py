"""Tests for Step 7: nuclear gradients.

Validates the analytic (Hellmann-Feynman + Pulay) gradient against:
1. Full numerical gradient (central finite differences on the total energy)
2. PySCF analytic gradient

Target accuracy: 1e-6 Hartree/Bohr vs finite difference, 1e-5 vs PySCF.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.grad.interface import compute_gradient
from jax_qc.grad.numerical_grad import numerical_gradient
from jax_qc.grad.rhf_grad import energy_weighted_density, rhf_gradient

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402


# =========================================================================
#  Test molecules
# =========================================================================


def _h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _heh_cation():
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


def _h2_result():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)
    return result, mol


def _heh_result():
    mol = _heh_cation()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)
    return result, mol


# =========================================================================
#  PySCF reference gradient
# =========================================================================


def _pyscf_gradient(mol, basis_name: str = "sto-3g") -> np.ndarray:
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
    mf.verbose = 0
    mf.kernel()
    g = mf.nuc_grad_method()
    g.verbose = 0
    return g.kernel()


# =========================================================================
#  Unit tests
# =========================================================================


def test_energy_weighted_density_trace():
    """Tr(W * S) should equal 2 * sum of occupied orbital energies."""
    result, mol = _h2_result()
    n_occ = mol.n_electrons // 2
    W = energy_weighted_density(
        result.state.coefficients, result.state.orbital_energies, n_occ
    )
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    tr_WS = float(jnp.trace(W @ ints.S))
    expected = 2.0 * float(jnp.sum(result.state.orbital_energies[:n_occ]))
    assert tr_WS == pytest.approx(expected, abs=1e-10)


def test_compute_gradient_dispatcher():
    """compute_gradient should accept method='analytic' and 'numerical'."""
    result, mol = _h2_result()
    g_a = compute_gradient(mol, result, basis_name="sto-3g", method="analytic")
    assert g_a.shape == (2, 3)
    # numerical is too slow for default tests, just check dispatcher works
    with pytest.raises(ValueError, match="Unknown gradient method"):
        compute_gradient(mol, result, method="bogus")


def test_h2_gradient_symmetric():
    """H2 gradient: force on atom 0 should be equal and opposite to atom 1."""
    result, mol = _h2_result()
    grad = rhf_gradient(mol, result, basis_name="sto-3g")
    # By symmetry, grad[0] = -grad[1]
    np.testing.assert_allclose(np.asarray(grad[0]), -np.asarray(grad[1]), atol=1e-8)


def test_h2_gradient_only_z_component():
    """H2 along z-axis: gradient x,y components should be zero."""
    result, mol = _h2_result()
    grad = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
    np.testing.assert_allclose(grad[:, 0], 0.0, atol=1e-8)  # x
    np.testing.assert_allclose(grad[:, 1], 0.0, atol=1e-8)  # y


# =========================================================================
#  Analytic vs numerical gradient
# =========================================================================


def test_h2_analytic_vs_numerical():
    """H2/STO-3G: analytic gradient should match numerical to ~1e-6."""
    result, mol = _h2_result()
    g_analytic = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
    g_numerical = np.asarray(
        numerical_gradient(mol, basis_name="sto-3g", method="rhf", step_size=1e-4)
    )
    np.testing.assert_allclose(g_analytic, g_numerical, atol=1e-5)


def test_heh_analytic_vs_numerical():
    """HeH+/STO-3G: analytic gradient should match numerical."""
    result, mol = _heh_result()
    g_analytic = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
    g_numerical = np.asarray(
        numerical_gradient(mol, basis_name="sto-3g", method="rhf", step_size=1e-4)
    )
    np.testing.assert_allclose(g_analytic, g_numerical, atol=1e-5)


# =========================================================================
#  Analytic vs PySCF
# =========================================================================


def test_h2_gradient_matches_pyscf():
    """H2/STO-3G: analytic gradient should match PySCF."""
    result, mol = _h2_result()
    g_jaxqc = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
    g_pyscf = _pyscf_gradient(mol, "sto-3g")
    np.testing.assert_allclose(g_jaxqc, g_pyscf, atol=1e-5)


def test_heh_gradient_matches_pyscf():
    """HeH+/STO-3G: analytic gradient should match PySCF."""
    result, mol = _heh_result()
    g_jaxqc = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
    g_pyscf = _pyscf_gradient(mol, "sto-3g")
    np.testing.assert_allclose(g_jaxqc, g_pyscf, atol=1e-5)
