"""Tests for Step 8: geometry optimization.

Validates the BFGS optimizer on H2 and HeH+ by checking:
1. Convergence to the equilibrium bond length
2. Gradient near zero at the optimized geometry
3. Energy lower than the starting geometry
4. Comparison with known equilibrium geometries
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.geomopt.interface import optimize_geometry
from jax_qc.geomopt.optimizer import OptResult

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402


# =========================================================================
#  Test molecules — starting from non-equilibrium geometries
# =========================================================================


def _h2_stretched():
    """H2 at R=2.0 Bohr (stretched from equilibrium ~1.39)."""
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _h2_compressed():
    """H2 at R=1.0 Bohr (compressed from equilibrium)."""
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _heh_cation():
    """HeH+ at R=2.0 Bohr (non-equilibrium)."""
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


# =========================================================================
#  PySCF reference equilibrium
# =========================================================================


def _pyscf_equilibrium_energy(atoms_str, basis_name, charge=0, spin=0):
    """Run PySCF geometry optimization and return the equilibrium energy."""
    pmol = gto.M(
        atom=atoms_str, basis=basis_name, unit="bohr", charge=charge, spin=spin
    )
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    return float(mf.e_tot)


# =========================================================================
#  Unit tests
# =========================================================================


def test_opt_result_fields():
    """OptResult should have the expected fields."""
    result = OptResult(
        converged=True,
        molecule=_h2_stretched(),
        energy=-1.0,
        gradient=jnp.zeros((2, 3)),
        n_steps=5,
        trajectory=[],
    )
    assert result.converged is True
    assert result.n_steps == 5


# =========================================================================
#  H2 optimization tests
# =========================================================================


def test_h2_optimization_converges():
    """H2 stretched: BFGS should converge."""
    mol = _h2_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"H2 optimization did not converge in {opt.n_steps} steps"


def test_h2_optimization_energy_decreases():
    """Optimized energy should be lower than the starting energy."""
    mol = _h2_stretched()
    # Starting energy
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result_start = jax_qc.run_rhf(mol, ints, config)

    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.energy < result_start.energy


def test_h2_optimization_gradient_near_zero():
    """At equilibrium, the gradient should be near zero."""
    mol = _h2_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    g_max = float(jnp.max(jnp.abs(opt.gradient)))
    assert g_max < 1e-4, f"max |grad| = {g_max:.2e} > 1e-4"


def test_h2_equilibrium_bond_length():
    """H2/STO-3G equilibrium bond length should be ~1.346 Bohr."""
    mol = _h2_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    # PySCF H2/STO-3G equilibrium: ~1.346 Bohr
    assert r_eq == pytest.approx(1.346, abs=0.05), (
        f"H2 equilibrium R = {r_eq:.4f} Bohr, expected ~1.346"
    )


def test_h2_compressed_also_converges():
    """H2 compressed: BFGS should also find the minimum."""
    mol = _h2_compressed()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged


def test_h2_both_directions_same_minimum():
    """Starting from stretched and compressed should give the same energy."""
    mol_s = _h2_stretched()
    mol_c = _h2_compressed()
    opt_s = optimize_geometry(
        mol_s, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    opt_c = optimize_geometry(
        mol_c, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    assert opt_s.energy == pytest.approx(opt_c.energy, abs=1e-6)


def test_h2_trajectory_is_populated():
    """The trajectory should contain coordinates and energies for each step."""
    mol = _h2_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert len(opt.trajectory) >= 2  # at least start + 1 step
    # First entry should be near the starting energy
    _, e0 = opt.trajectory[0]
    assert e0 != opt.energy  # should have changed


def test_h2_energy_matches_pyscf_at_equilibrium():
    """Energy at optimized geometry should match PySCF at the same geometry."""
    mol = _h2_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    # Run PySCF at our optimized geometry
    coords = np.asarray(opt.molecule.coords)
    atoms = [["H", tuple(coords[0].tolist())], ["H", tuple(coords[1].tolist())]]
    pmol = gto.M(atom=atoms, basis="sto-3g", unit="bohr")
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    e_pyscf = float(mf.kernel())
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-6)


# =========================================================================
#  HeH+ optimization test
# =========================================================================


def test_heh_optimization_converges():
    """HeH+/STO-3G: BFGS should converge."""
    mol = _heh_cation()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"HeH+ optimization did not converge in {opt.n_steps} steps"
    # HeH+/STO-3G equilibrium: ~1.76 Bohr (PySCF PES scan minimum)
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    assert r_eq == pytest.approx(1.76, abs=0.05)
