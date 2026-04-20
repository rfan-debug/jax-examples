"""Water across several basis sets.

Same molecule, different bases — shows that swapping basis names is all
it takes to move up in basis-set quality. Prints the number of basis
functions, shells, and angular momenta on oxygen for each choice.
"""

from __future__ import annotations

import jax.numpy as jnp

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule
from examples._colors import banner, header, label, value


def _water() -> "jax_qc.Molecule":
    # Near-equilibrium geometry in Bohr.
    return make_molecule(
        elements=("O", "H", "H"),
        coords=jnp.array(
            [
                [0.0, 0.0, 0.22143],
                [0.0, 1.43052, -0.88572],
                [0.0, -1.43052, -0.88572],
            ]
        ),
        atomic_numbers=jnp.array([8, 1, 1], dtype=jnp.int32),
    )


def _summary(mol, basis_name: str) -> None:
    basis = jax_qc.build_basis_set(mol, basis_name)
    o_ls = sorted(s.angular_momentum for s in basis.shells if s.atom_index == 0)
    h_ls = sorted(s.angular_momentum for s in basis.shells if s.atom_index > 0)
    print(
        f"  {label(f'{basis_name:<12s}')} "
        f"{label('n_basis')}={value(f'{basis.n_basis:>3d}')}  "
        f"{label('shells')}={value(f'{len(basis.shells):>2d}')}  "
        f"{label('O:')}{o_ls}  "
        f"{label('H:')}{h_ls[:len(h_ls) // 2]} (each)"
    )


def main() -> None:
    water = _water()
    print(banner("Water — Basis-Set Sweep"))
    print(
        f"{label('elements')}  : {water.elements}, "
        f"{label('n_electrons')}={value(str(water.n_electrons))}"
    )
    print()
    print(header("Basis summaries:"))
    for name in ("sto-3g", "3-21g", "6-31g", "6-31G*", "cc-pVDZ", "def2-SVP"):
        _summary(water, name)


if __name__ == "__main__":
    main()
