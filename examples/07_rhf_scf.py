"""Run RHF SCF on the Tier 1 closed-shell molecules.

Step 3 plugs the Monadic SCF loop on top of the Step 2 Applicative
integral layer. We run H2, He, HeH+, and H3+ at STO-3G, compare the
total energy against PySCF (when installed) or against hard-coded
literature values otherwise, and print the full stage-timer report.
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


# PySCF / STO-3G reference RHF total energies (Hartree).
LITERATURE_REFS = {
    "H2":   -1.1167143251,
    "He":   -2.8077839575,
    "HeH+": -2.8418364993,
    "H3+":  -1.2375476999,
}


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
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


def _h3_plus():
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


MOLECULES = {
    "H2": _h2,
    "He": _he,
    "HeH+": _heh_cation,
    "H3+": _h3_plus,
}


def _pyscf_reference(mol) -> float | None:
    if not _HAS_PYSCF:
        return None
    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    pmol = gto.M(
        atom=atoms,
        basis="sto-3g",
        unit="bohr",
        charge=int(mol.charge),
        spin=int(mol.spin),
    )
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    return float(mf.kernel())


def _run(name: str) -> None:
    print()
    print(banner(f"{name} / STO-3G — RHF SCF"))
    mol = MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    timer = jax_qc.StageTimer(sync_device=False)

    with timer.stage("integrals", "applicative"):
        ints = jax_qc.compute_integrals(mol, basis)

    config = CalcConfig(
        method="rhf",
        basis="sto-3g",
        scf_conv=1e-10,
        max_scf_iter=64,
        diis_space=8,
    )
    result = jax_qc.run_scf(mol, ints, config, timer=timer)

    conv_tag = ok("converged") if result.converged else warn("NOT converged")
    print(f"  {label('n_basis')}      = {value(str(basis.n_basis))}")
    print(f"  {label('n_electrons')}  = {value(str(mol.n_electrons))}")
    print(f"  {label('iterations')}   = {value(str(result.n_iterations))}  [{conv_tag}]")
    print(f"  {label('E_elec')}       = {value(f'{result.E_elec:.10f}')} Ha")
    print(f"  {label('E_nuc')}        = {value(f'{result.E_nuc:.10f}')} Ha")
    print(f"  {label('E_total')}      = {value(f'{result.energy:.10f}')} Ha")

    ref = _pyscf_reference(mol)
    source = "pyscf" if ref is not None else "literature"
    if ref is None:
        ref = LITERATURE_REFS[name]

    print()
    print(header("Reference comparison"))
    print(f"  {label('source')}: {value(source)}")
    # Plan target: <= 1 microhartree
    print("  " + compare(
        result.energy, ref,
        label_got="E_total (jax_qc)",
        label_ref=f"E_total ({source})",
        tol=1e-6,
    ))

    # Orbital energies (HOMO-LUMO for closed-shell systems)
    orb = np.asarray(result.state.orbital_energies)
    n_occ = mol.n_electrons // 2
    if n_occ > 0 and n_occ < len(orb):
        homo = float(orb[n_occ - 1])
        lumo = float(orb[n_occ])
        gap = lumo - homo
        print()
        print(header("Frontier orbitals"))
        print(f"  {label('HOMO')} = {value(f'{homo:+.6f}')} Ha")
        print(f"  {label('LUMO')} = {value(f'{lumo:+.6f}')} Ha")
        print(f"  {label('gap ')} = {value(f'{gap:+.6f}')} Ha")

    print()
    print(header("Profiling report"))
    print(timer.report(min_percent=0.5))


def main() -> None:
    for name in MOLECULES:
        _run(name)


if __name__ == "__main__":
    main()
