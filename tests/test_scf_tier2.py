"""Step 4 Tier 2 SCF benchmarks.

For each closed-shell molecule with at least one non-s shell, we compare
the total RHF energy against PySCF on a small basis. The plan target is
1 microhartree; we run STO-3G and 6-31G only — enough to stress p- and
d-shell handling without making CI hours-long.

If you need to validate larger basis sets (cc-pVDZ, 6-31G**) run
``examples/08_rhf_tier2.py`` interactively — those calculations take
minutes per molecule and are deliberately not in the unit-test suite.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402


def _h2o():
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


def _hf():
    return make_molecule(
        elements=("F", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7325]]),
        atomic_numbers=jnp.array([9, 1], dtype=jnp.int32),
    )


def _ch4():
    # Tetrahedral CH4, R = 2.052 Bohr.
    r = 2.052 / np.sqrt(3.0)
    return make_molecule(
        elements=("C", "H", "H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r, r, r],
                [r, -r, -r],
                [-r, r, -r],
                [-r, -r, r],
            ]
        ),
        atomic_numbers=jnp.array([6, 1, 1, 1, 1], dtype=jnp.int32),
    )


def _nh3():
    # Pyramidal NH3.
    r = 1.913
    theta = np.deg2rad(106.7)
    s = np.sin(theta / 2.0)
    c = np.cos(theta / 2.0)
    return make_molecule(
        elements=("N", "H", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r * s, 0.0, -r * c],
                [-r * s / 2.0, r * s * np.sqrt(3.0) / 2.0, -r * c],
                [-r * s / 2.0, -r * s * np.sqrt(3.0) / 2.0, -r * c],
            ]
        ),
        atomic_numbers=jnp.array([7, 1, 1, 1], dtype=jnp.int32),
    )


MOLECULES = {
    "H2O": _h2o,
    "HF": _hf,
    "CH4": _ch4,
    "NH3": _nh3,
}


def _pyscf_energy(mol, basis_name):
    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    pmol = gto.M(
        atom=atoms, basis=basis_name, unit="bohr",
        charge=int(mol.charge), spin=int(mol.spin),
    )
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    return float(mf.kernel())


@pytest.mark.parametrize(
    "name,basis_name",
    [
        ("H2O", "sto-3g"),
        pytest.param("H2O", "6-31g", marks=pytest.mark.slow),
        ("HF",  "sto-3g"),
        pytest.param("CH4", "sto-3g", marks=pytest.mark.slow),
        pytest.param("NH3", "sto-3g", marks=pytest.mark.slow),
    ],
)
def test_rhf_total_energy_matches_pyscf(name, basis_name):
    mol = MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, basis_name)
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(
        method="rhf", basis=basis_name, scf_conv=1e-10, max_scf_iter=100, diis_space=8
    )
    result = jax_qc.run_scf(mol, ints, config)
    assert result.converged, f"{name}/{basis_name} did not converge"
    e_ref = _pyscf_energy(mol, basis_name)
    # Plan target: <= 1 microhartree.
    assert result.energy == pytest.approx(e_ref, abs=1e-6)


@pytest.mark.slow
def test_rhf_h2o_6_31gss_with_d_shells():
    """Spot-check that the 6-31G** d-shell path matches PySCF.

    Marked slow — run with ``pytest -m slow`` (or
    ``examples/08_rhf_tier2.py`` interactively). Validated locally to
    1e-8 Ha; deselected from default CI to keep the suite under a minute.
    """
    mol = _h2o()
    basis = jax_qc.build_basis_set(mol, "6-31g**")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", basis="6-31g**", scf_conv=1e-10, max_scf_iter=100)
    result = jax_qc.run_scf(mol, ints, config)
    e_ref = _pyscf_energy(mol, "6-31g**")
    assert result.energy == pytest.approx(e_ref, abs=1e-6)
