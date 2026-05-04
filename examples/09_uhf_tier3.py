"""Step 5: UHF on Tier 3 open-shell molecules.

For each (molecule, basis) we run jax_qc's UHF and compare the total
energy against PySCF (when installed). This validates the UHF Fock
build with per-spin densities, the UHF SCF loop, and DIIS on open-shell
systems.

Tier 3 molecules include H atom, Li, B, O, CH3, OH, NO, O2.
Tier 4 includes charged species (Li+, OH-, F-).
Tier 5 includes the stretched H2 stress test.
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from examples._colors import banner, compare, header, label, ok, value, warn

try:
    from pyscf import gto, scf

    _HAS_PYSCF = True
except ImportError:  # pragma: no cover
    _HAS_PYSCF = False


# ---- Tier 3 open-shell molecules ----------------------------------------


def _h_atom():
    return make_molecule(
        elements=("H",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([1], dtype=jnp.int32),
        spin=1,
    )


def _li_atom():
    return make_molecule(
        elements=("Li",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([3], dtype=jnp.int32),
        spin=1,
    )


def _b_atom():
    return make_molecule(
        elements=("B",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([5], dtype=jnp.int32),
        spin=1,
    )


def _o_atom():
    return make_molecule(
        elements=("O",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([8], dtype=jnp.int32),
        spin=2,
    )


def _ch3():
    r = 2.039
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
    return make_molecule(
        elements=("O", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8324]]),
        atomic_numbers=jnp.array([8, 1], dtype=jnp.int32),
        spin=1,
    )


def _no():
    return make_molecule(
        elements=("N", "O"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.175]]),
        atomic_numbers=jnp.array([7, 8], dtype=jnp.int32),
        spin=1,
    )


def _o2():
    return make_molecule(
        elements=("O", "O"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.282]]),
        atomic_numbers=jnp.array([8, 8], dtype=jnp.int32),
        spin=2,
    )


# ---- Tier 4 charged species ---------------------------------------------


def _li_cation():
    return make_molecule(
        elements=("Li",),
        coords=jnp.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=jnp.array([3], dtype=jnp.int32),
        charge=+1,
        spin=0,
    )


def _oh_anion():
    return make_molecule(
        elements=("O", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8324]]),
        atomic_numbers=jnp.array([8, 1], dtype=jnp.int32),
        charge=-1,
        spin=0,
    )


# ---- Tier 5 stress test -------------------------------------------------


def _stretched_h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
        spin=0,
    )


MOLECULES = {
    "H_atom": _h_atom,
    "Li_atom": _li_atom,
    "B_atom": _b_atom,
    "O_atom": _o_atom,
    "CH3": _ch3,
    "OH": _oh,
    "NO": _no,
    "O2": _o2,
    "Li+": _li_cation,
    "OH-": _oh_anion,
    "stretched_H2": _stretched_h2,
}


def _pyscf_reference(mol, basis_name) -> float | None:
    if not _HAS_PYSCF:
        return None
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
    return float(mf.kernel())


def _run(name: str, basis_name: str) -> None:
    print()
    print(banner(f"{name} / {basis_name} — UHF SCF"))
    mol = MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, basis_name)
    timer = jax_qc.StageTimer(sync_device=False)

    with timer.stage("integrals", "applicative"):
        ints = jax_qc.compute_integrals(mol, basis)

    config = CalcConfig(
        method="uhf",
        basis=basis_name,
        scf_conv=1e-10,
        max_scf_iter=128,
        diis_space=8,
    )
    result = jax_qc.run_scf(mol, ints, config, timer=timer)

    conv_tag = ok("converged") if result.converged else warn("NOT converged")
    print(f"  {label('n_basis')}     = {value(str(basis.n_basis))}")
    print(f"  {label('n_shells')}    = {value(str(len(basis.shells)))}")
    print(f"  {label('n_alpha')}     = {value(str(mol.n_alpha))}")
    print(f"  {label('n_beta')}      = {value(str(mol.n_beta))}")
    print(f"  {label('spin (2S)')}   = {value(str(mol.spin))}")
    print(f"  {label('iterations')}  = {value(str(result.n_iterations))}  [{conv_tag}]")
    print(f"  {label('E_total')}     = {value(f'{result.energy:.10f}')} Ha")

    ref = _pyscf_reference(mol, basis_name)
    if ref is None:
        print(f"  {label('no PySCF reference available')}")
    else:
        print()
        print(header("Reference comparison"))
        print(
            "  "
            + compare(
                result.energy,
                ref,
                label_got="E_total (jax_qc)",
                label_ref="E_total (pyscf)",
                tol=1e-6,
            )
        )


def main() -> None:
    pairs = [
        # Tier 3 open-shell
        ("H_atom", "sto-3g"),
        ("Li_atom", "sto-3g"),
        ("B_atom", "sto-3g"),
        ("O_atom", "sto-3g"),
        ("CH3", "sto-3g"),
        ("OH", "sto-3g"),
        ("NO", "sto-3g"),
        ("O2", "sto-3g"),
        # Tier 4 charged species
        ("Li+", "sto-3g"),
        ("OH-", "sto-3g"),
        # Tier 5 stress test
        ("stretched_H2", "sto-3g"),
    ]
    for name, basis_name in pairs:
        _run(name, basis_name)


if __name__ == "__main__":
    main()
