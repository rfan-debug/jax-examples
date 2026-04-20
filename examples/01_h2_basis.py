"""H2 molecule: build STO-3G basis, print normalized shells.

The simplest possible test case: two hydrogens at the equilibrium bond
length (1.4 Bohr). Shows how a Molecule and BasisSet are constructed and
what a normalized contracted s-shell looks like.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule
from examples._colors import banner, header, label, ok, value


def main() -> None:
    h2 = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    print(banner("H2 / STO-3G — Molecule + Basis Construction"))
    print(f"{label('elements')}    : {h2.elements}")
    print(f"{label('coords/Bohr')} : {np.asarray(h2.coords).tolist()}")
    print(
        f"{label('electrons')}   : {value(str(h2.n_electrons))}  "
        f"({label('alpha')}={value(str(h2.n_alpha))}, "
        f"{label('beta')}={value(str(h2.n_beta))})"
    )

    basis = jax_qc.build_basis_set(h2, "sto-3g")
    print()
    print(header("Basis: sto-3g"))
    print(f"  {label('n_basis')}        = {value(str(basis.n_basis))}")
    print(f"  {label('shells')}         = {value(str(len(basis.shells)))}")
    print(f"  {label('shell_to_basis')} = {basis.shell_to_basis}")
    print(f"  {label('basis_to_atom')}  = {np.asarray(basis.basis_to_atom).tolist()}")

    for i, shell in enumerate(basis.shells):
        print()
        print(header(f"  Shell {i}  (atom {shell.atom_index}, l={shell.angular_momentum})", color="bright_green"))
        print(f"    {label('exponents')}    = {value(str(np.asarray(shell.exponents)))}")
        print(f"    {label('coefficients')} = {value(str(np.asarray(shell.coefficients)))} {ok('(normalized)')}")


if __name__ == "__main__":
    main()
