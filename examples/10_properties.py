"""Step 6: Post-SCF properties and convenience API demo.

Demonstrates:
  1. Mulliken population analysis (charges per atom)
  2. Orbital analysis (HOMO/LUMO energies, gap)
  3. Dipole moment
  4. High-level convenience API (energy(), run(), build_molecule())
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from examples._colors import banner, header, label, ok, value, warn


def _section(title: str) -> None:
    print()
    print(banner(title))


def demo_properties() -> None:
    """Run RHF on H2O and display all post-SCF properties."""
    _section("H2O / STO-3G — Post-SCF Properties")

    mol = make_molecule(
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
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", basis="sto-3g", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)

    conv_tag = ok("converged") if result.converged else warn("NOT converged")
    print(f"  {label('E_total')}  = {value(f'{result.energy:.10f}')} Ha  [{conv_tag}]")
    print(f"  {label('n_basis')}  = {value(str(basis.n_basis))}")
    print()

    # --- Mulliken charges ---
    print(header("Mulliken Population Analysis"))
    pops, charges = jax_qc.mulliken_analysis(result, mol, basis)
    for i, (sym, pop, q) in enumerate(zip(mol.elements, pops, charges)):
        print(
            f"  {label(sym):>4s}  population = {value(f'{float(pop):.4f}')}"
            f"   charge = {value(f'{float(q):+.4f}')}"
        )
    print(f"  {label('Total charge')} = {value(f'{float(jnp.sum(charges)):.6f}')}")
    print()

    # --- Orbital analysis ---
    print(header("Orbital Analysis"))
    info = jax_qc.orbital_analysis(result, mol)
    print(f"  {label('HOMO index')}  = {value(str(info.homo_index))}")
    print(f"  {label('HOMO energy')} = {value(f'{info.homo_energy:.6f}')} Ha")
    if info.lumo_energy is not None:
        print(f"  {label('LUMO energy')} = {value(f'{info.lumo_energy:.6f}')} Ha")
        print(f"  {label('HOMO-LUMO gap')} = {value(f'{info.homo_lumo_gap:.6f}')} Ha")
    print()
    print("  Orbital energies (Ha):")
    for i, (e, occ) in enumerate(zip(info.orbital_energies, info.occupations)):
        tag = "occ" if float(occ) > 0 else "vir"
        marker = " <-- HOMO" if i == info.homo_index else ""
        marker = " <-- LUMO" if i == info.lumo_index else marker
        print(f"    {i:3d}  {float(e):12.6f}  ({tag}){marker}")
    print()

    # --- Dipole moment ---
    print(header("Dipole Moment"))
    mu = jax_qc.dipole_moment(result, mol, basis)
    mu_d = jax_qc.dipole_moment_debye(result, mol, basis)
    print(f"  {label('mu_x')} = {value(f'{float(mu[0]):+.6f}')} a.u.")
    print(f"  {label('mu_y')} = {value(f'{float(mu[1]):+.6f}')} a.u.")
    print(f"  {label('mu_z')} = {value(f'{float(mu[2]):+.6f}')} a.u.")
    print(f"  {label('|mu|')} = {value(f'{mu_d:.4f}')} Debye")


def demo_convenience_api() -> None:
    """Demonstrate the high-level convenience API."""
    _section("Convenience API — jax_qc.energy()")

    # Style 1: build_molecule + energy()
    mol = jax_qc.build_molecule(
        atoms=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        unit="bohr",
    )
    result = jax_qc.energy(mol, method="rhf", basis="sto-3g")
    print(f"  H2 (build_molecule + energy): E = {value(f'{result.energy:.10f}')} Ha")

    # Style 2: run() from dict
    _section("Convenience API — jax_qc.run(dict)")
    result2 = jax_qc.run(
        {
            "molecule": {
                "atoms": ["He"],
                "coords": [[0.0, 0.0, 0.0]],
                "unit": "bohr",
            },
            "method": "rhf",
            "basis": "sto-3g",
        }
    )
    print(f"  He (dict API): E = {value(f'{result2.energy:.10f}')} Ha")


def main() -> None:
    demo_properties()
    demo_convenience_api()


if __name__ == "__main__":
    main()
