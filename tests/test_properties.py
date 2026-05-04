"""Tests for Step 6: properties and convenience API.

Validates Mulliken charges, dipole moments, orbital analysis, and the
high-level convenience functions (energy, run, run_xyz, build_molecule)
against PySCF reference values.
"""

from __future__ import annotations

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.io.input_parser import build_molecule, energy, run, run_xyz
from jax_qc.properties.dipole import (
    AU_TO_DEBYE,
    dipole_moment,
    dipole_moment_debye,
    nuclear_dipole,
)
from jax_qc.properties.mulliken import (
    mulliken_analysis,
    mulliken_charges,
    mulliken_populations,
)
from jax_qc.properties.orbital_analysis import OrbitalInfo, orbital_analysis

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


def _h2_result():
    mol = _h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)
    return result, mol, basis


def _he():
    return make_molecule(
        elements=("He",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([2], dtype=jnp.int32),
    )


# =========================================================================
#  Mulliken tests
# =========================================================================


def test_mulliken_charges_sum_to_molecular_charge():
    """Total Mulliken charge must equal the molecular charge."""
    result, mol, basis = _h2_result()
    charges = mulliken_charges(result, mol, basis)
    assert float(jnp.sum(charges)) == pytest.approx(mol.charge, abs=1e-10)


def test_mulliken_populations_sum_to_n_electrons():
    """Sum of Mulliken populations = total electron count."""
    result, mol, basis = _h2_result()
    pops, _ = mulliken_analysis(result, mol, basis)
    assert float(jnp.sum(pops)) == pytest.approx(mol.n_electrons, abs=1e-10)


def test_mulliken_h2_symmetric():
    """H2 has equal charges on both atoms by symmetry."""
    result, mol, basis = _h2_result()
    charges = mulliken_charges(result, mol, basis)
    np.testing.assert_allclose(float(charges[0]), float(charges[1]), atol=1e-10)


def test_mulliken_charges_match_pyscf():
    """Compare Mulliken charges to PySCF on H2/STO-3G."""
    result, mol, basis = _h2_result()
    charges = np.asarray(mulliken_charges(result, mol, basis))

    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    pmol = gto.M(atom=atoms, basis="sto-3g", unit="bohr")
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    _, ref_charges = mf.mulliken_pop(verbose=0)

    np.testing.assert_allclose(charges, ref_charges, atol=1e-4)


# =========================================================================
#  Orbital analysis tests
# =========================================================================


def test_orbital_analysis_h2():
    """H2 has 1 occupied MO (HOMO) and 1 virtual (LUMO)."""
    result, mol, basis = _h2_result()
    info = orbital_analysis(result, mol)
    assert info.homo_index == 0
    assert info.lumo_index == 1
    assert info.homo_lumo_gap is not None
    assert info.homo_lumo_gap > 0.0
    assert float(info.occupations[0]) == 2.0
    assert float(info.occupations[1]) == 0.0


def test_orbital_analysis_gap_matches_eigenvalues():
    """HOMO-LUMO gap = eps_LUMO - eps_HOMO."""
    result, mol, basis = _h2_result()
    info = orbital_analysis(result, mol)
    expected_gap = float(info.orbital_energies[1]) - float(info.orbital_energies[0])
    assert info.homo_lumo_gap == pytest.approx(expected_gap, abs=1e-14)


def test_orbital_energies_sorted():
    """Orbital energies must be in non-decreasing order."""
    result, mol, basis = _h2_result()
    info = orbital_analysis(result, mol)
    orb_e = np.asarray(info.orbital_energies)
    assert np.all(np.diff(orb_e) >= -1e-12)


def test_orbital_analysis_matches_pyscf():
    """HOMO/LUMO energies should match PySCF."""
    result, mol, basis = _h2_result()
    info = orbital_analysis(result, mol)

    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    pmol = gto.M(atom=atoms, basis="sto-3g", unit="bohr")
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()

    np.testing.assert_allclose(
        np.asarray(info.orbital_energies),
        mf.mo_energy,
        atol=1e-6,
    )


# =========================================================================
#  Dipole moment tests
# =========================================================================


def test_nuclear_dipole_h2_symmetric():
    """H2 centered at the origin: nuclear dipole has z-component only."""
    mol = _h2()
    mu_nuc = nuclear_dipole(mol)
    # H2 at (0,0,0) and (0,0,1.4): nuclear dipole = 1*0 + 1*1.4 = 1.4 on z
    assert float(mu_nuc[0]) == pytest.approx(0.0, abs=1e-14)
    assert float(mu_nuc[1]) == pytest.approx(0.0, abs=1e-14)
    assert float(mu_nuc[2]) == pytest.approx(1.4, abs=1e-14)


def test_dipole_h2_near_zero():
    """H2 is homonuclear — total dipole should be near zero."""
    result, mol, basis = _h2_result()
    mu = dipole_moment(result, mol, basis)
    # The x,y components are exactly zero by symmetry.
    assert float(jnp.abs(mu[0])) < 1e-10
    assert float(jnp.abs(mu[1])) < 1e-10
    # The z component should also be near zero for H2 at equilibrium.
    assert float(jnp.abs(mu[2])) < 0.1  # loose check; exact cancellation


def test_dipole_he_zero():
    """He atom: dipole must be exactly zero."""
    mol = _he()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)
    mu = dipole_moment(result, mol, basis)
    np.testing.assert_allclose(np.asarray(mu), 0.0, atol=1e-10)


def test_dipole_debye_positive():
    """Dipole magnitude in Debye must be non-negative."""
    result, mol, basis = _h2_result()
    d = dipole_moment_debye(result, mol, basis)
    assert d >= 0.0


# =========================================================================
#  Convenience API tests
# =========================================================================


def test_build_molecule_angstrom():
    """build_molecule converts Angstrom to Bohr by default."""
    from jax_qc.core.constants import ANGSTROM_TO_BOHR

    mol = build_molecule(
        atoms=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    )
    # 0.74 Angstrom * conversion factor
    expected_z = 0.74 * ANGSTROM_TO_BOHR
    assert float(mol.coords[1, 2]) == pytest.approx(expected_z, abs=1e-8)
    assert mol.n_electrons == 2


def test_build_molecule_bohr():
    """build_molecule with unit='bohr' skips conversion."""
    mol = build_molecule(
        atoms=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        unit="bohr",
    )
    assert float(mol.coords[1, 2]) == pytest.approx(1.4, abs=1e-14)


def test_energy_convenience():
    """jax_qc.energy() should produce a converged result."""
    mol = _h2()
    result = energy(mol, method="rhf", basis="sto-3g")
    assert result.converged
    # Compare with the low-level pipeline
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result_ref = jax_qc.run_rhf(mol, ints, config)
    assert result.energy == pytest.approx(result_ref.energy, abs=1e-10)


def test_run_dict():
    """jax_qc.run() from a dict should work."""
    result = run(
        {
            "molecule": {
                "atoms": ["H", "H"],
                "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
                "unit": "bohr",
            },
            "method": "rhf",
            "basis": "sto-3g",
        }
    )
    assert result.converged


def test_run_xyz_file():
    """jax_qc.run_xyz() should read an XYZ file and produce a result."""
    mol = _h2()
    with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
        f.write("2\nH2 test\n")
        f.write("H  0.0  0.0  0.0\n")
        f.write("H  0.0  0.0  0.74\n")  # Angstrom
        f.flush()
        path = f.name

    try:
        result = run_xyz(path, method="rhf", basis="sto-3g")
        assert result.converged
    finally:
        os.unlink(path)


def test_build_molecule_with_charge_and_spin():
    """build_molecule propagates charge and spin correctly."""
    mol = build_molecule(
        atoms=["H"],
        coords=[[0.0, 0.0, 0.0]],
        unit="bohr",
        spin=1,
    )
    assert mol.n_electrons == 1
    assert mol.n_alpha == 1
    assert mol.n_beta == 0
    assert mol.spin == 1
