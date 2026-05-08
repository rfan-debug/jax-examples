"""Progressive benchmark suite toward COF simulation.

Validates jax_qc on increasingly complex systems, from small molecules
to aromatic building blocks and COF linkage motifs.  All energies are
compared against PySCF (RHF/STO-3G, conv_tol=1e-12).

Level 2: small polyatomics (plan gaps: CO, N2, HCN, H2CO, C2H4, N2H2)
Level 3: COF-relevant functional groups (imine, amide, boronic acid)
Level 4: aromatic building blocks (benzene, pyridine, triazine, boroxine)
Level 5: COF linkage models (methyl-capped imine, boronate, amide)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.properties.mulliken import mulliken_charges
from jax_qc.properties.orbital_analysis import orbital_analysis

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402

# Angstrom -> Bohr
A2B = 1.8897259886


# =========================================================================
#  Helpers
# =========================================================================


def _mol_from_xyz_angstrom(elements, coords_ang, charge=0, spin=0):
    """Build a Molecule from Angstrom coordinates."""
    coords_bohr = np.array(coords_ang, dtype=np.float64) * A2B
    Z = np.array(
        [jax_qc.core.constants.symbol_to_z(e) for e in elements], dtype=np.int32
    )
    return make_molecule(
        elements=tuple(elements),
        coords=jnp.asarray(coords_bohr),
        atomic_numbers=jnp.asarray(Z),
        charge=charge,
        spin=spin,
    )


def _pyscf_energy(mol, basis_name="sto-3g"):
    """PySCF single-point energy at the molecule's geometry."""
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


def _jaxqc_energy(mol, basis_name="sto-3g"):
    """Run jax_qc RHF and return the total energy."""
    basis = jax_qc.build_basis_set(mol, basis_name)
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)
    return result


# =========================================================================
#  Level 2: Small polyatomics — plan gaps
# =========================================================================


def _co():
    return _mol_from_xyz_angstrom(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])


def _n2():
    return _mol_from_xyz_angstrom(["N", "N"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.098]])


def _hcn():
    return _mol_from_xyz_angstrom(
        ["H", "C", "N"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.066], [0.0, 0.0, 2.212]],
    )


def _h2co():
    return _mol_from_xyz_angstrom(
        ["C", "O", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.203],
            [0.0, 0.934, -0.587],
            [0.0, -0.934, -0.587],
        ],
    )


def _c2h4():
    return _mol_from_xyz_angstrom(
        ["C", "C", "H", "H", "H", "H"],
        [
            [0.0, 0.0, 0.667],
            [0.0, 0.0, -0.667],
            [0.0, 0.923, 1.237],
            [0.0, -0.923, 1.237],
            [0.0, 0.923, -1.237],
            [0.0, -0.923, -1.237],
        ],
    )


def _n2h2():
    """Diazene (trans-HN=NH)."""
    return _mol_from_xyz_angstrom(
        ["N", "N", "H", "H"],
        [
            [0.0, 0.0, 0.626],
            [0.0, 0.0, -0.626],
            [0.493, 0.0, 1.258],
            [-0.493, 0.0, -1.258],
        ],
    )


LEVEL2 = {
    "CO": (_co, -111.2246),
    "N2": (_n2, -107.4960),
    "HCN": (_hcn, -91.6750),
    "H2CO": (_h2co, -112.3539),
    "C2H4": (_c2h4, -77.0727),
    "N2H2": (_n2h2, -108.2669),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL2))
def test_level2_energy(name):
    """Level 2: small polyatomic energy matches PySCF to 1 mHa."""
    make_fn, e_approx = LEVEL2[name]
    mol = make_fn()
    result = _jaxqc_energy(mol)
    e_ref = _pyscf_energy(mol)
    assert result.converged, f"{name} SCF did not converge"
    assert result.energy == pytest.approx(e_ref, abs=1e-5), (
        f"{name}: jax_qc={result.energy:.8f}, pyscf={e_ref:.8f}"
    )


# =========================================================================
#  Level 3: COF-relevant functional groups
# =========================================================================


def _methanimine():
    """CH2=NH — simplest imine bond."""
    return _mol_from_xyz_angstrom(
        ["C", "N", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.267, 0.0, 0.0],
            [-0.540, 0.940, 0.0],
            [-0.540, -0.940, 0.0],
            [1.807, 0.940, 0.0],
        ],
    )


def _formamide():
    """HCONH2 — amide bond model."""
    return _mol_from_xyz_angstrom(
        ["C", "O", "N", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.219, 0.0, 0.0],
            [-0.600, 1.155, 0.0],
            [-0.100, -0.970, 0.0],
            [-0.165, 2.032, 0.0],
            [-1.561, 1.120, 0.0],
        ],
    )


def _boric_acid():
    """B(OH)3 — boronic acid, boron COF precursor."""
    return _mol_from_xyz_angstrom(
        ["B", "O", "O", "O", "H", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.361, 0.0, 0.0],
            [-0.681, 1.179, 0.0],
            [-0.681, -1.179, 0.0],
            [1.843, 0.760, 0.0],
            [-1.524, 1.199, 0.0],
            [-0.319, -1.959, 0.0],
        ],
    )


def _methylamine():
    """CH3NH2 — primary amine, COF monomer end group."""
    return _mol_from_xyz_angstrom(
        ["C", "N", "H", "H", "H", "H", "H"],
        [
            [0.054, 0.678, 0.0],
            [0.054, -0.778, 0.0],
            [-0.467, 1.081, 0.890],
            [-0.467, 1.081, -0.890],
            [1.085, 1.044, 0.0],
            [0.551, -1.126, 0.816],
            [0.551, -1.126, -0.816],
        ],
    )


LEVEL3 = {
    "methanimine": (_methanimine, -92.8153),
    "formamide": (_formamide, -166.5931),
    "boric_acid": (_boric_acid, -247.7221),
    "methylamine": (_methylamine, -94.0297),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL3))
def test_level3_energy(name):
    """Level 3: COF functional group energy matches PySCF to 1 mHa."""
    make_fn, e_approx = LEVEL3[name]
    mol = make_fn()
    result = _jaxqc_energy(mol)
    e_ref = _pyscf_energy(mol)
    assert result.converged, f"{name} SCF did not converge"
    assert result.energy == pytest.approx(e_ref, abs=1e-5), (
        f"{name}: jax_qc={result.energy:.8f}, pyscf={e_ref:.8f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL3))
def test_level3_mulliken_charges_sum(name):
    """Mulliken charges must sum to the molecular charge."""
    make_fn, _ = LEVEL3[name]
    mol = make_fn()
    result = _jaxqc_energy(mol)
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    charges = mulliken_charges(result, mol, basis)
    assert float(jnp.sum(charges)) == pytest.approx(mol.charge, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL3))
def test_level3_homo_lumo_gap_positive(name):
    """HOMO-LUMO gap should be positive for closed-shell molecules."""
    make_fn, _ = LEVEL3[name]
    mol = make_fn()
    result = _jaxqc_energy(mol)
    info = orbital_analysis(result, mol)
    assert info.homo_lumo_gap is not None
    assert info.homo_lumo_gap > 0.0


# =========================================================================
#  Level 4: Aromatic building blocks
# =========================================================================


def _benzene():
    """C6H6 — universal aromatic linker core."""
    return _mol_from_xyz_angstrom(
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        [
            [1.397, 0.0, 0.0],
            [0.699, 1.210, 0.0],
            [-0.699, 1.210, 0.0],
            [-1.397, 0.0, 0.0],
            [-0.699, -1.210, 0.0],
            [0.699, -1.210, 0.0],
            [2.481, 0.0, 0.0],
            [1.240, 2.148, 0.0],
            [-1.240, 2.148, 0.0],
            [-2.481, 0.0, 0.0],
            [-1.240, -2.148, 0.0],
            [1.240, -2.148, 0.0],
        ],
    )


def _triazine():
    """C3N3H3 — Covalent Triazine Framework node."""
    return _mol_from_xyz_angstrom(
        ["N", "C", "N", "C", "N", "C", "H", "H", "H"],
        [
            [1.338, 0.0, 0.0],
            [0.669, 1.159, 0.0],
            [-0.669, 1.159, 0.0],
            [-1.338, 0.0, 0.0],
            [-0.669, -1.159, 0.0],
            [0.669, -1.159, 0.0],
            [1.248, 2.162, 0.0],
            [-2.496, 0.0, 0.0],
            [1.248, -2.162, 0.0],
        ],
    )


def _pyridine():
    """C5H5N — N-heterocycle in CTF nodes."""
    return _mol_from_xyz_angstrom(
        ["N", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H"],
        [
            [0.0, 1.340, 0.0],
            [1.145, 0.682, 0.0],
            [1.145, -0.682, 0.0],
            [0.0, -1.375, 0.0],
            [-1.145, -0.682, 0.0],
            [-1.145, 0.682, 0.0],
            [2.060, 1.255, 0.0],
            [2.060, -1.255, 0.0],
            [0.0, -2.455, 0.0],
            [-2.060, -1.255, 0.0],
            [-2.060, 1.255, 0.0],
        ],
    )


def _boroxine():
    """B3O3H3 — boroxine ring, linkage in COF-1."""
    return _mol_from_xyz_angstrom(
        ["B", "O", "B", "O", "B", "O", "H", "H", "H"],
        [
            [1.376, 0.0, 0.0],
            [0.688, 1.192, 0.0],
            [-0.688, 1.192, 0.0],
            [-1.376, 0.0, 0.0],
            [-0.688, -1.192, 0.0],
            [0.688, -1.192, 0.0],
            [2.480, 0.0, 0.0],
            [-1.240, 2.148, 0.0],
            [-1.240, -2.148, 0.0],
        ],
    )


def _phenol():
    """C6H5OH — boronate ester fragment (diol side)."""
    return _mol_from_xyz_angstrom(
        ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "O", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.396, 0.0],
            [1.209, 2.093, 0.0],
            [2.418, 1.396, 0.0],
            [2.418, 0.0, 0.0],
            [1.209, -0.698, 0.0],
            [-0.929, -0.540, 0.0],
            [-0.929, 1.936, 0.0],
            [1.209, 3.172, 0.0],
            [3.347, 1.936, 0.0],
            [3.347, -0.540, 0.0],
            [1.209, -2.068, 0.0],
            [1.209, -2.558, 0.810],
        ],
    )


def _aniline():
    """C6H5NH2 — amine-functionalized benzene, imine COF monomer."""
    return _mol_from_xyz_angstrom(
        [
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "N",
            "H",
            "H",
        ],
        [
            [1.397, 0.0, 0.0],
            [0.699, 1.210, 0.0],
            [-0.699, 1.210, 0.0],
            [-1.397, 0.0, 0.0],
            [-0.699, -1.210, 0.0],
            [0.699, -1.210, 0.0],
            [2.481, 0.0, 0.0],
            [1.240, 2.148, 0.0],
            [-1.240, 2.148, 0.0],
            [-2.481, 0.0, 0.0],
            [-1.240, -2.148, 0.0],
            [1.397, -2.420, 0.0],
            [2.121, -3.078, 0.0],
            [0.673, -3.078, 0.0],
        ],
    )


LEVEL4 = {
    "benzene": (_benzene, -227.8906),
    "triazine": (_triazine, -275.1120),
    "pyridine": (_pyridine, -243.6241),
    "boroxine": (_boroxine, -296.7479),
    "phenol": (_phenol, -301.7123),
    "aniline": (_aniline, -282.1471),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL4))
def test_level4_energy(name):
    """Level 4: aromatic building block energy matches PySCF to 1 mHa."""
    make_fn, e_approx = LEVEL4[name]
    mol = make_fn()
    result = _jaxqc_energy(mol)
    e_ref = _pyscf_energy(mol)
    assert result.converged, f"{name} SCF did not converge"
    assert result.energy == pytest.approx(e_ref, abs=1e-4), (
        f"{name}: jax_qc={result.energy:.8f}, pyscf={e_ref:.8f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL4))
def test_level4_homo_lumo_gap(name):
    """Level 4: HOMO-LUMO gap should be positive and reasonable."""
    make_fn, _ = LEVEL4[name]
    mol = make_fn()
    result = _jaxqc_energy(mol)
    info = orbital_analysis(result, mol)
    assert info.homo_lumo_gap is not None
    assert info.homo_lumo_gap > 0.0
    # Aromatic molecules in STO-3G: gap typically 0.2-0.8 Ha
    assert info.homo_lumo_gap < 2.0, (
        f"{name} gap={info.homo_lumo_gap:.4f} Ha unreasonably large"
    )


# =========================================================================
#  Level 5: COF linkage models (methyl-capped)
# =========================================================================


def _imine_model():
    """CH3-CH=N-H — simplest imine linkage model (N-methylmethanimine).

    Models the -CH=N- linkage in imine-linked COFs (LZU-1, COF-300).
    """
    return _mol_from_xyz_angstrom(
        ["C", "C", "N", "H", "H", "H", "H", "H"],
        [
            [-1.270, 0.0, 0.0],
            [0.0, 0.518, 0.0],
            [1.112, -0.144, 0.0],
            [-1.340, -0.617, 0.890],
            [-1.340, -0.617, -0.890],
            [-2.073, 0.733, 0.0],
            [0.010, 1.600, 0.0],
            [1.952, 0.432, 0.0],
        ],
    )


def _nma():
    """N-methylacetamide (CH3-CO-NH-CH3) — amide linkage model.

    Models the -C(=O)-NH- linkage in amide-linked COFs.
    """
    return _mol_from_xyz_angstrom(
        ["C", "C", "O", "N", "C", "H", "H", "H", "H", "H", "H", "H"],
        [
            [-1.532, 0.0, 0.0],  # CH3 (acetyl)
            [0.0, 0.0, 0.0],  # C=O
            [0.598, 1.076, 0.0],  # O
            [0.659, -1.137, 0.0],  # NH
            [2.078, -1.230, 0.0],  # CH3 (methyl)
            [-1.889, 0.515, 0.890],
            [-1.889, 0.515, -0.890],
            [-1.889, -1.031, 0.0],
            [0.180, -2.009, 0.0],  # NH
            [2.453, -0.715, 0.890],
            [2.453, -0.715, -0.890],
            [2.453, -2.250, 0.0],
        ],
    )


LEVEL5 = {
    "imine_model": _imine_model,
    "nma_amide_model": _nma,
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL5))
def test_level5_energy(name):
    """Level 5: COF linkage model energy matches PySCF to 1 mHa."""
    mol = LEVEL5[name]()
    result = _jaxqc_energy(mol)
    e_ref = _pyscf_energy(mol)
    assert result.converged, f"{name} SCF did not converge"
    assert result.energy == pytest.approx(e_ref, abs=1e-4), (
        f"{name}: jax_qc={result.energy:.8f}, pyscf={e_ref:.8f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", list(LEVEL5))
def test_level5_mulliken_charge_on_linkage(name):
    """Level 5: Mulliken charge on the linkage heteroatom should be negative.

    The N in imine (C=N) and amide (CO-NH) linkages should carry partial
    negative charge due to its higher electronegativity.
    """
    mol = LEVEL5[name]()
    result = _jaxqc_energy(mol)
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    charges = np.asarray(mulliken_charges(result, mol, basis))
    # Find the N atom
    n_indices = [i for i, e in enumerate(mol.elements) if e == "N"]
    assert len(n_indices) > 0
    for idx in n_indices:
        assert charges[idx] < 0.0, (
            f"{name}: N(atom {idx}) charge = {charges[idx]:.4f}, expected negative"
        )
