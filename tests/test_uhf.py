"""Tests for Step 5 UHF SCF.

Tier 3 benchmark: for every open-shell test molecule we compare the
total UHF energy to PySCF to <= 1 microhartree. Tier 4 charged species
and Tier 5 stress tests are also included.

Unit tests cover the individual Applicative sub-computations:
build_fock_uhf, density_uhf, electronic_energy_uhf, and core_guess_uhf.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.scf.density import density_uhf
from jax_qc.scf.energy import electronic_energy_uhf
from jax_qc.scf.fock import build_fock_uhf
from jax_qc.scf.guess import core_guess_uhf
from jax_qc.scf.orthogonalize import symmetric_orthogonalization
from jax_qc.scf.uhf import run_uhf

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402


# =========================================================================
#  Tier 3 open-shell molecules
# =========================================================================


def _h_atom():
    """H atom — 1 electron, doublet."""
    return make_molecule(
        elements=("H",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([1], dtype=jnp.int32),
        spin=1,
    )


def _li_atom():
    """Li atom — 3 electrons, doublet (1s^2 2s^1)."""
    return make_molecule(
        elements=("Li",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([3], dtype=jnp.int32),
        spin=1,
    )


def _b_atom():
    """B atom — 5 electrons, doublet (2p^1)."""
    return make_molecule(
        elements=("B",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([5], dtype=jnp.int32),
        spin=1,
    )


def _o_atom():
    """O atom — 8 electrons, triplet (2S=2)."""
    return make_molecule(
        elements=("O",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([8], dtype=jnp.int32),
        spin=2,
    )


def _ch3():
    """Methyl radical — 9 electrons, doublet."""
    # Planar CH3: C at origin, H atoms at 120 deg in xy-plane
    r = 2.039  # Bohr
    return make_molecule(
        elements=("C", "H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r, 0.0, 0.0],
                [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0.0],
                [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0.0],
            ]
        ),
        atomic_numbers=jnp.array([6, 1, 1, 1], dtype=jnp.int32),
        spin=1,
    )


def _oh():
    """OH radical — 9 electrons, doublet."""
    return make_molecule(
        elements=("O", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8324]]),
        atomic_numbers=jnp.array([8, 1], dtype=jnp.int32),
        spin=1,
    )


def _no():
    """NO radical — 15 electrons, doublet."""
    return make_molecule(
        elements=("N", "O"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.175]]),
        atomic_numbers=jnp.array([7, 8], dtype=jnp.int32),
        spin=1,
    )


def _o2():
    """O2 — 16 electrons, triplet (2S=2)."""
    return make_molecule(
        elements=("O", "O"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.282]]),
        atomic_numbers=jnp.array([8, 8], dtype=jnp.int32),
        spin=2,
    )


# =========================================================================
#  Tier 4 charged species (closed-shell cations/anions via UHF)
# =========================================================================


def _li_cation():
    """Li+ — 2 electrons, singlet."""
    return make_molecule(
        elements=("Li",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([3], dtype=jnp.int32),
        charge=+1,
        spin=0,
    )


def _oh_anion():
    """OH- — 10 electrons, singlet."""
    return make_molecule(
        elements=("O", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8324]]),
        atomic_numbers=jnp.array([8, 1], dtype=jnp.int32),
        charge=-1,
        spin=0,
    )


def _f_anion():
    """F- — 10 electrons, singlet."""
    return make_molecule(
        elements=("F",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([9], dtype=jnp.int32),
        charge=-1,
        spin=0,
    )


# =========================================================================
#  Tier 5 stress test
# =========================================================================


def _stretched_h2():
    """H2 at R=5 Bohr — near dissociation, UHF breaks symmetry."""
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
        spin=0,
    )


# =========================================================================
#  Molecule registries
# =========================================================================

TIER3_MOLECULES = {
    "H_atom": _h_atom,
    "Li_atom": _li_atom,
    "B_atom": _b_atom,
    "O_atom": _o_atom,
    "CH3": _ch3,
    "OH": _oh,
    "O2": _o2,
}

# NO is excluded from default Tier 3 because its UHF convergence with
# STO-3G requires SOSCF or level shifting — not yet implemented.
TIER3_HARD = {
    "NO": _no,
}

TIER4_MOLECULES = {
    "Li+": _li_cation,
    "OH-": _oh_anion,
    "F-": _f_anion,
}


# =========================================================================
#  PySCF reference helper
# =========================================================================


def _pyscf_uhf_energy(mol, basis_name: str = "sto-3g") -> float:
    """Run PySCF UHF with stability analysis to find the global minimum.

    PySCF's default initial guess can converge to a metastable UHF
    solution (e.g. for O2 triplet). Running one round of stability
    analysis ensures both PySCF and jax_qc are compared at the true
    UHF minimum.
    """
    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    pmol = gto.M(
        atom=atoms,
        basis=basis_name,
        unit="bohr",
        charge=int(mol.charge),
        spin=int(mol.spin),
    )
    mf = scf.UHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    # One round of internal stability analysis to find the true minimum.
    try:
        mo_stable = mf.stability()[0]
        dm = mf.make_rdm1(mo_stable, mf.mo_occ)
        mf2 = scf.UHF(pmol)
        mf2.conv_tol = 1e-12
        mf2.verbose = 0
        mf2.kernel(dm0=dm)
        return float(mf2.e_tot)
    except Exception:
        return float(mf.e_tot)


# =========================================================================
#  Unit tests for UHF sub-computations
# =========================================================================


def test_build_fock_uhf_reduces_to_rhf_for_closed_shell():
    """When D_alpha == D_beta == D/2, UHF Fock should equal RHF Fock."""
    from jax_qc.scf.fock import build_fock_rhf

    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    X = symmetric_orthogonalization(ints.S)

    # RHF density: D total
    from jax_qc.scf.guess import core_guess

    D_total = core_guess(ints.H_core, X, n_occ=1)
    F_rhf = build_fock_rhf(ints.H_core, D_total, ints.ERI)

    # UHF: split D_total into alpha/beta (each = D_total / 2)
    D_half = D_total / 2.0
    F_a, F_b = build_fock_uhf(ints.H_core, D_half, D_half, ints.ERI)

    np.testing.assert_allclose(np.asarray(F_a), np.asarray(F_rhf), atol=1e-12)
    np.testing.assert_allclose(np.asarray(F_b), np.asarray(F_rhf), atol=1e-12)


def test_build_fock_uhf_at_zero_density():
    """At zero density, F_alpha = F_beta = H_core."""
    mol = make_molecule(
        elements=("H",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([1], dtype=jnp.int32),
        spin=1,
    )
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    n = ints.S.shape[0]
    D_zero = jnp.zeros((n, n))
    F_a, F_b = build_fock_uhf(ints.H_core, D_zero, D_zero, ints.ERI)
    np.testing.assert_allclose(np.asarray(F_a), np.asarray(ints.H_core), atol=1e-14)
    np.testing.assert_allclose(np.asarray(F_b), np.asarray(ints.H_core), atol=1e-14)


def test_density_uhf_trace():
    """Tr(D_alpha @ S) = n_alpha, Tr(D_beta @ S) = n_beta."""
    mol = _o_atom()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    X = symmetric_orthogonalization(ints.S)
    D_a, D_b, _ = core_guess_uhf(ints.H_core, X, mol.n_alpha, mol.n_beta, ints.ERI)
    tr_a = float(jnp.trace(D_a @ ints.S))
    tr_b = float(jnp.trace(D_b @ ints.S))
    assert tr_a == pytest.approx(mol.n_alpha, abs=1e-10)
    assert tr_b == pytest.approx(mol.n_beta, abs=1e-10)


def test_density_uhf_is_symmetric():
    """Both alpha and beta density matrices must be symmetric."""
    mol = _oh()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    X = symmetric_orthogonalization(ints.S)
    D_a, D_b, _ = core_guess_uhf(ints.H_core, X, mol.n_alpha, mol.n_beta, ints.ERI)
    np.testing.assert_allclose(np.asarray(D_a), np.asarray(D_a).T, atol=1e-12)
    np.testing.assert_allclose(np.asarray(D_b), np.asarray(D_b).T, atol=1e-12)


def test_electronic_energy_uhf_equals_trace_formula():
    """E_elec = 0.5 * Tr[D_a (H + F_a)] + 0.5 * Tr[D_b (H + F_b)]."""
    mol = _h_atom()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    X = symmetric_orthogonalization(ints.S)
    D_a, D_b, _ = core_guess_uhf(ints.H_core, X, mol.n_alpha, mol.n_beta, ints.ERI)
    F_a, F_b = build_fock_uhf(ints.H_core, D_a, D_b, ints.ERI)
    E = float(electronic_energy_uhf(D_a, D_b, ints.H_core, F_a, F_b))
    E_ref = 0.5 * float(jnp.trace(D_a @ (ints.H_core + F_a))) + 0.5 * float(
        jnp.trace(D_b @ (ints.H_core + F_b))
    )
    assert E == pytest.approx(E_ref, abs=1e-14)


def test_run_scf_dispatches_to_uhf():
    """run_scf with method='uhf' should use the UHF driver."""
    mol = _h_atom()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="uhf", basis="sto-3g", scf_conv=1e-10)
    result = jax_qc.run_scf(mol, ints, config)
    assert result.converged


def test_uhf_e_total_equals_elec_plus_nuc():
    """E_total = E_elec + E_nuc."""
    mol = _li_atom()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="uhf", scf_conv=1e-10)
    result = run_uhf(mol, ints, config)
    assert result.energy == pytest.approx(result.E_elec + result.E_nuc, abs=1e-14)


def test_uhf_density_is_symmetric():
    """Converged total density must be symmetric."""
    mol = _oh()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="uhf", scf_conv=1e-10)
    result = run_uhf(mol, ints, config)
    D = np.asarray(result.state.density)
    np.testing.assert_allclose(D, D.T, atol=1e-12)


def test_uhf_populates_stage_timer():
    """Timer should record the UHF SCF stages."""
    mol = _h_atom()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    timer = jax_qc.StageTimer(sync_device=False)
    config = CalcConfig(method="uhf", scf_conv=1e-10)
    run_uhf(mol, ints, config, timer=timer)
    scf_node = timer.root.children["scf"]
    assert scf_node.fp_abstraction == "monad"
    assert "fock_build" in scf_node.children
    assert scf_node.children["fock_build"].fp_abstraction == "applicative"


def test_uhf_on_closed_shell_matches_rhf():
    """UHF on H2 (singlet) should give the same energy as RHF."""
    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(basis="sto-3g", scf_conv=1e-10)

    result_rhf = jax_qc.run_rhf(mol, ints, config)
    result_uhf = run_uhf(mol, ints, config)

    assert result_uhf.converged
    assert result_uhf.energy == pytest.approx(result_rhf.energy, abs=1e-8)


# =========================================================================
#  End-to-end Tier 3 UHF vs PySCF
# =========================================================================


@pytest.mark.parametrize(
    "name",
    [
        "H_atom",
        "Li_atom",
        "B_atom",
        "O_atom",
        pytest.param("CH3", marks=pytest.mark.slow),
        pytest.param("OH", marks=pytest.mark.slow),
        pytest.param("O2", marks=pytest.mark.slow),
    ],
)
def test_uhf_tier3_energy_matches_pyscf(name: str):
    """Tier 3 open-shell molecules: UHF/STO-3G vs PySCF <= 1 uHa."""
    mol = TIER3_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(
        method="uhf",
        basis="sto-3g",
        scf_conv=1e-10,
        max_scf_iter=128,
        diis_space=8,
    )
    result = run_uhf(mol, ints, config)
    assert result.converged, f"{name} UHF did not converge"
    e_ref = _pyscf_uhf_energy(mol, "sto-3g")
    assert result.energy == pytest.approx(e_ref, abs=1e-6), (
        f"{name}/sto-3g: jax_qc={result.energy:.10f}, pyscf={e_ref:.10f}, "
        f"diff={abs(result.energy - e_ref):.2e}"
    )


# =========================================================================
#  Tier 3 with 6-31G (marked slow — larger basis)
# =========================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    ["Li_atom", "O_atom", "OH", "O2"],
)
def test_uhf_tier3_6_31g_energy_matches_pyscf(name: str):
    """Tier 3 open-shell with 6-31G basis vs PySCF."""
    mol = TIER3_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "6-31g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(
        method="uhf",
        basis="6-31g",
        scf_conv=1e-10,
        max_scf_iter=128,
        diis_space=8,
    )
    result = run_uhf(mol, ints, config)
    assert result.converged, f"{name}/6-31g UHF did not converge"
    e_ref = _pyscf_uhf_energy(mol, "6-31g")
    assert result.energy == pytest.approx(e_ref, abs=1e-6)


# =========================================================================
#  Tier 4 charged species
# =========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("name", list(TIER4_MOLECULES))
def test_uhf_tier4_energy_matches_pyscf(name: str):
    """Tier 4 charged species: UHF/STO-3G vs PySCF <= 1 uHa."""
    mol = TIER4_MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(
        method="uhf",
        basis="sto-3g",
        scf_conv=1e-10,
        max_scf_iter=128,
        diis_space=8,
    )
    result = run_uhf(mol, ints, config)
    assert result.converged, f"{name} UHF did not converge"
    e_ref = _pyscf_uhf_energy(mol, "sto-3g")
    assert result.energy == pytest.approx(e_ref, abs=1e-6), (
        f"{name}/sto-3g: jax_qc={result.energy:.10f}, pyscf={e_ref:.10f}"
    )


# =========================================================================
#  Tier 5 stress test — stretched H2
# =========================================================================


@pytest.mark.slow
def test_uhf_stretched_h2():
    """Stretched H2 at R=5 Bohr: UHF should break symmetry and be lower
    than RHF (or equal if RHF already dissociates at this geometry)."""
    mol = _stretched_h2()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(
        method="uhf",
        basis="sto-3g",
        scf_conv=1e-10,
        max_scf_iter=128,
    )
    result = run_uhf(mol, ints, config)
    assert result.converged
    e_ref = _pyscf_uhf_energy(mol, "sto-3g")
    assert result.energy == pytest.approx(e_ref, abs=1e-6)
