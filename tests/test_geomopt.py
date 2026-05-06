"""Tests for Step 8: geometry optimization.

Comprehensive benchmark suite validating the BFGS optimizer:

Tier A — Fast (s-type only, no p-shell integrals):
  H2:   homonuclear diatomic, 2 electrons
  HeH+: heteronuclear diatomic, 2 electrons, charged
  H3+:  3-atom polyatomic, equilateral triangle, charged

Tier B — Slow (p-shell integrals, ~60s each eval):
  HF:   heteronuclear diatomic with p-shells, 10 electrons
  H2O:  3-atom polyatomic with p-shells, bond + angle optimization

Each benchmark checks:
  1. Convergence to a stationary point (max |grad| < tol)
  2. Equilibrium geometry vs PySCF reference (bond length within tolerance)
  3. Energy at the optimized geometry matches PySCF single-point
  4. Energy monotonically decreasing in the trajectory

Robustness tests:
  - Compressed and stretched starts converge to the same minimum
  - Significantly distorted multi-atom geometries still converge
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
#  Tier A molecules — s-type only (fast integrals)
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


def _h3_plus_distorted():
    """H3+ starting from a distorted triangle (not equilateral).

    Equilibrium is equilateral with R~1.85 Bohr.  We start from an
    isoceles triangle with sides 2.2, 2.2, 1.6 to test multi-atom
    optimization with a non-trivial distortion.
    """
    return make_molecule(
        elements=("H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
                [1.1, 0.8, 0.0],
            ]
        ),
        atomic_numbers=jnp.array([1, 1, 1], dtype=jnp.int32),
        charge=+1,
    )


def _h3_plus_equilateral_stretched():
    """H3+ equilateral but stretched (R=2.2 vs equilibrium ~1.85)."""
    r = 2.2
    s3 = np.sqrt(3.0)
    return make_molecule(
        elements=("H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r, 0.0, 0.0],
                [r / 2.0, r * s3 / 2.0, 0.0],
            ]
        ),
        atomic_numbers=jnp.array([1, 1, 1], dtype=jnp.int32),
        charge=+1,
    )


# =========================================================================
#  Tier B molecules — p-shells (slow integrals)
# =========================================================================


def _hf_stretched():
    """HF at R=2.2 Bohr (stretched from equilibrium ~1.80)."""
    return make_molecule(
        elements=("F", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]]),
        atomic_numbers=jnp.array([9, 1], dtype=jnp.int32),
    )


def _h2o_distorted():
    """H2O starting from a distorted geometry.

    Equilibrium: R_OH ~ 1.85 Bohr, angle ~ 100-107 deg (STO-3G).
    Start from R_OH = 2.1 Bohr, angle = 90 deg (significantly distorted).
    """
    r = 2.1
    theta = np.deg2rad(90.0)
    return make_molecule(
        elements=("O", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, r * np.sin(theta / 2.0), -r * np.cos(theta / 2.0)],
                [0.0, -r * np.sin(theta / 2.0), -r * np.cos(theta / 2.0)],
            ]
        ),
        atomic_numbers=jnp.array([8, 1, 1], dtype=jnp.int32),
    )


# =========================================================================
#  PySCF reference helper
# =========================================================================


def _pyscf_energy_at_coords(mol, basis_name):
    """Run PySCF single-point at the given molecule's coordinates."""
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
    return float(mf.kernel())


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
#  Tier A: H2 — comprehensive diatomic tests
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
    _, e0 = opt.trajectory[0]
    assert e0 != opt.energy  # should have changed


def test_h2_energy_matches_pyscf_at_equilibrium():
    """Energy at optimized geometry should match PySCF at the same geometry."""
    mol = _h2_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-6)


# =========================================================================
#  Tier A: HeH+ — charged heteronuclear diatomic
# =========================================================================


def test_heh_optimization_converges():
    """HeH+/STO-3G: BFGS should converge to R~1.76 Bohr."""
    mol = _heh_cation()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"HeH+ did not converge in {opt.n_steps} steps"
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    assert r_eq == pytest.approx(1.76, abs=0.05)


def test_heh_energy_matches_pyscf():
    """HeH+: energy at optimized geometry matches PySCF single-point."""
    mol = _heh_cation()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-6)


# =========================================================================
#  Tier A: H3+ — 3-atom polyatomic (s-type, fast)
# =========================================================================


@pytest.mark.slow
def test_h3_plus_distorted_converges():
    """H3+ from distorted triangle: should converge."""
    mol = _h3_plus_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"H3+ distorted did not converge in {opt.n_steps} steps"


@pytest.mark.slow
def test_h3_plus_equilateral_result():
    """H3+ should optimize to an equilateral triangle with R~1.85 Bohr."""
    mol = _h3_plus_equilateral_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    assert opt.converged
    coords = np.asarray(opt.molecule.coords)
    # Compute all three bond lengths
    r01 = float(np.linalg.norm(coords[1] - coords[0]))
    r02 = float(np.linalg.norm(coords[2] - coords[0]))
    r12 = float(np.linalg.norm(coords[2] - coords[1]))
    # All should be approximately equal (equilateral)
    r_mean = (r01 + r02 + r12) / 3.0
    assert r01 == pytest.approx(r_mean, abs=0.05), f"r01={r01:.4f} != mean={r_mean:.4f}"
    assert r02 == pytest.approx(r_mean, abs=0.05), f"r02={r02:.4f} != mean={r_mean:.4f}"
    assert r12 == pytest.approx(r_mean, abs=0.05), f"r12={r12:.4f} != mean={r_mean:.4f}"
    # Equilibrium side length ~1.85 Bohr
    assert r_mean == pytest.approx(1.85, abs=0.1)


@pytest.mark.slow
def test_h3_plus_energy_matches_pyscf():
    """H3+: energy at optimized geometry matches PySCF single-point."""
    mol = _h3_plus_equilateral_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-5, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-6)


@pytest.mark.slow
def test_h3_plus_both_starts_same_energy():
    """H3+ from distorted and equilateral start should reach same energy."""
    opt_d = optimize_geometry(
        _h3_plus_distorted(),
        basis_name="sto-3g",
        max_steps=50,
        grad_tol=1e-4,
        verbose=0,
    )
    opt_e = optimize_geometry(
        _h3_plus_equilateral_stretched(),
        basis_name="sto-3g",
        max_steps=50,
        grad_tol=1e-4,
        verbose=0,
    )
    assert opt_d.converged and opt_e.converged
    assert opt_d.energy == pytest.approx(opt_e.energy, abs=1e-5)


# =========================================================================
#  Tier B: HF — diatomic with p-shells (slow)
# =========================================================================


@pytest.mark.slow
def test_hf_optimization_converges():
    """HF/STO-3G: should converge to R~1.80 Bohr."""
    mol = _hf_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"HF did not converge in {opt.n_steps} steps"
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    # PySCF HF/STO-3G equilibrium: ~1.80 Bohr
    assert r_eq == pytest.approx(1.80, abs=0.1), f"HF R_eq={r_eq:.4f}, expected ~1.80"


@pytest.mark.slow
def test_hf_energy_matches_pyscf():
    """HF: energy at optimized geometry matches PySCF single-point."""
    mol = _hf_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-6)


# =========================================================================
#  Tier B: H2O — 3-atom with p-shells, bond + angle optimization (slow)
# =========================================================================


@pytest.mark.slow
def test_h2o_optimization_converges():
    """H2O/STO-3G from distorted geometry: should converge."""
    mol = _h2o_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"H2O did not converge in {opt.n_steps} steps"


@pytest.mark.slow
def test_h2o_equilibrium_geometry():
    """H2O/STO-3G: optimized R_OH ~ 1.87 Bohr, angle ~ 100-107 deg."""
    mol = _h2o_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    assert opt.converged
    coords = np.asarray(opt.molecule.coords)
    # O is at index 0, H at 1 and 2
    r_oh1 = float(np.linalg.norm(coords[1] - coords[0]))
    r_oh2 = float(np.linalg.norm(coords[2] - coords[0]))
    # Both OH bond lengths should be equal by symmetry
    assert r_oh1 == pytest.approx(r_oh2, abs=0.05)
    # Equilibrium R_OH ~ 1.85-1.87 Bohr for STO-3G
    r_mean = (r_oh1 + r_oh2) / 2.0
    assert r_mean == pytest.approx(1.87, abs=0.1), (
        f"H2O R_OH = {r_mean:.4f}, expected ~1.87"
    )
    # HOH angle
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    # STO-3G HOH angle: ~100-107 deg
    assert 95.0 < angle_deg < 115.0, (
        f"HOH angle = {angle_deg:.1f} deg, expected ~100-107"
    )


@pytest.mark.slow
def test_h2o_energy_matches_pyscf():
    """H2O: energy at optimized geometry matches PySCF single-point."""
    mol = _h2o_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-5)


@pytest.mark.slow
def test_h2o_energy_decreases_monotonically():
    """H2O trajectory energy should be non-increasing (with line search)."""
    mol = _h2o_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    energies = [e for _, e in opt.trajectory]
    for i in range(1, len(energies)):
        # Allow tiny numerical noise (1e-8 Ha)
        assert energies[i] <= energies[i - 1] + 1e-8, (
            f"Energy increased at step {i}: {energies[i - 1]:.10f} -> {energies[i]:.10f}"
        )


# =========================================================================
#  Tier C: Larger basis sets — 6-31G (tests basis-set dependence)
# =========================================================================


def _h2_for_631g():
    """H2 at R=1.6 Bohr for 6-31G optimization (eq ~1.38 Bohr)."""
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _heh_for_631g():
    """HeH+ at R=1.8 Bohr for 6-31G optimization (eq ~1.45 Bohr)."""
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


@pytest.mark.slow
def test_h2_631g_optimization():
    """H2/6-31G: larger basis gives different equilibrium (~1.38 Bohr)."""
    mol = _h2_for_631g()
    opt = optimize_geometry(
        mol, basis_name="6-31g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"H2/6-31G did not converge in {opt.n_steps} steps"
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    # 6-31G equilibrium is slightly different from STO-3G
    assert r_eq == pytest.approx(1.38, abs=0.05), f"H2/6-31G R_eq={r_eq:.4f}"
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "6-31g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-5)


@pytest.mark.slow
def test_heh_631g_optimization():
    """HeH+/6-31G: equilibrium ~1.45 Bohr, matches PySCF."""
    mol = _heh_for_631g()
    opt = optimize_geometry(
        mol, basis_name="6-31g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    assert r_eq == pytest.approx(1.45, abs=0.1), f"HeH+/6-31G R_eq={r_eq:.4f}"
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "6-31g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-5)


# =========================================================================
#  Tier D: 2nd-row atoms — LiH (lithium hydride)
# =========================================================================


def _lih_stretched():
    """LiH at R=3.5 Bohr (stretched from eq ~2.9 for STO-3G)."""
    return make_molecule(
        elements=("Li", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.5]]),
        atomic_numbers=jnp.array([3, 1], dtype=jnp.int32),
    )


def _lih_compressed():
    """LiH at R=2.4 Bohr (compressed from equilibrium)."""
    return make_molecule(
        elements=("Li", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]]),
        atomic_numbers=jnp.array([3, 1], dtype=jnp.int32),
    )


@pytest.mark.slow
def test_lih_sto3g_converges():
    """LiH/STO-3G from stretched geometry: should converge to R~2.9 Bohr."""
    mol = _lih_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"LiH/STO-3G did not converge in {opt.n_steps} steps"
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    # PySCF LiH/STO-3G equilibrium: ~2.9 Bohr
    assert r_eq == pytest.approx(2.9, abs=0.1), f"LiH R_eq={r_eq:.4f}, expected ~2.9"


@pytest.mark.slow
def test_lih_sto3g_energy_matches_pyscf():
    """LiH/STO-3G: energy at optimized geometry matches PySCF."""
    mol = _lih_stretched()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-5)


@pytest.mark.slow
def test_lih_bidirectional_same_minimum():
    """LiH: stretched and compressed starts converge to same energy."""
    opt_s = optimize_geometry(
        _lih_stretched(),
        basis_name="sto-3g",
        max_steps=30,
        grad_tol=1e-4,
        verbose=0,
    )
    opt_c = optimize_geometry(
        _lih_compressed(),
        basis_name="sto-3g",
        max_steps=30,
        grad_tol=1e-4,
        verbose=0,
    )
    assert opt_s.converged and opt_c.converged
    assert opt_s.energy == pytest.approx(opt_c.energy, abs=1e-5)


@pytest.mark.slow
def test_lih_631g_optimization():
    """LiH/6-31G: larger basis shifts equilibrium to R~3.1 Bohr."""
    mol = make_molecule(
        elements=("Li", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.5]]),
        atomic_numbers=jnp.array([3, 1], dtype=jnp.int32),
    )
    opt = optimize_geometry(
        mol, basis_name="6-31g", max_steps=30, grad_tol=1e-4, verbose=0
    )
    assert opt.converged
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    # PySCF LiH/6-31G equilibrium: ~3.1 Bohr
    assert r_eq == pytest.approx(3.1, abs=0.15), (
        f"LiH/6-31G R_eq={r_eq:.4f}, expected ~3.1"
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "6-31g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-5)


# =========================================================================
#  Tier D: NH3 — 4-atom pyramidal with p-shells
# =========================================================================


def _nh3_distorted():
    """NH3 starting from a distorted pyramidal geometry.

    Equilibrium: R_NH ~ 2.0 Bohr, HNH angle ~ 104-108 deg (STO-3G).
    Start from R_NH = 2.3 Bohr with slightly compressed angles.
    """
    r = 2.3
    # Approximate C3v geometry with 3 equivalent N-H bonds
    theta = np.deg2rad(100.0 / 2.0)  # half the HNH angle (distorted)
    s = r * np.sin(theta)
    c = r * np.cos(theta)
    return make_molecule(
        elements=("N", "H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [s, 0.0, -c],
                [-s * 0.5, s * np.sqrt(3.0) / 2.0, -c],
                [-s * 0.5, -s * np.sqrt(3.0) / 2.0, -c],
            ]
        ),
        atomic_numbers=jnp.array([7, 1, 1, 1], dtype=jnp.int32),
    )


@pytest.mark.slow
def test_nh3_optimization_converges():
    """NH3/STO-3G from distorted geometry: should converge."""
    mol = _nh3_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    assert opt.converged, f"NH3 did not converge in {opt.n_steps} steps"


@pytest.mark.slow
def test_nh3_equilibrium_geometry():
    """NH3/STO-3G: R_NH ~ 2.0 Bohr, all NH bonds equivalent."""
    mol = _nh3_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    assert opt.converged
    coords = np.asarray(opt.molecule.coords)
    # N is at index 0, three H at 1,2,3
    r_nh = [float(np.linalg.norm(coords[i] - coords[0])) for i in range(1, 4)]
    r_mean = sum(r_nh) / 3.0
    # All NH bonds should be equivalent
    for i, r in enumerate(r_nh):
        assert r == pytest.approx(r_mean, abs=0.05), (
            f"NH bond {i + 1}: {r:.4f} != mean {r_mean:.4f}"
        )
    # Equilibrium R_NH ~ 2.0 Bohr for STO-3G
    assert r_mean == pytest.approx(2.0, abs=0.15), (
        f"NH3 R_NH = {r_mean:.4f}, expected ~2.0"
    )


@pytest.mark.slow
def test_nh3_energy_matches_pyscf():
    """NH3: energy at optimized geometry matches PySCF single-point."""
    mol = _nh3_distorted()
    opt = optimize_geometry(
        mol, basis_name="sto-3g", max_steps=50, grad_tol=1e-4, verbose=0
    )
    e_pyscf = _pyscf_energy_at_coords(opt.molecule, "sto-3g")
    assert opt.energy == pytest.approx(e_pyscf, abs=1e-5)


# =========================================================================
#  Cross-basis consistency: same molecule, different basis sets
# =========================================================================


@pytest.mark.slow
def test_lih_basis_set_dependence():
    """LiH equilibrium R should increase with larger basis (STO-3G < 6-31G).

    This validates that the optimizer correctly captures the basis-set
    dependence of the potential energy surface.
    """
    opt_sto = optimize_geometry(
        _lih_stretched(),
        basis_name="sto-3g",
        max_steps=30,
        grad_tol=1e-4,
        verbose=0,
    )
    opt_631 = optimize_geometry(
        _lih_stretched(),
        basis_name="6-31g",
        max_steps=30,
        grad_tol=1e-4,
        verbose=0,
    )
    assert opt_sto.converged and opt_631.converged
    r_sto = float(
        np.linalg.norm(
            np.asarray(opt_sto.molecule.coords[1])
            - np.asarray(opt_sto.molecule.coords[0])
        )
    )
    r_631 = float(
        np.linalg.norm(
            np.asarray(opt_631.molecule.coords[1])
            - np.asarray(opt_631.molecule.coords[0])
        )
    )
    # 6-31G equilibrium is longer than STO-3G for LiH
    assert r_631 > r_sto, f"Expected R(6-31G)={r_631:.4f} > R(STO-3G)={r_sto:.4f}"
    # 6-31G energy should be lower (better basis)
    assert opt_631.energy < opt_sto.energy, (
        f"Expected E(6-31G)={opt_631.energy:.8f} < E(STO-3G)={opt_sto.energy:.8f}"
    )
