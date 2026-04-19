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


def main() -> None:
    h2 = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    print(f"Molecule: {h2.elements}")
    print(f"  coords (Bohr) = {np.asarray(h2.coords).tolist()}")
    print(f"  n_electrons   = {h2.n_electrons} (alpha={h2.n_alpha}, beta={h2.n_beta})")

    basis = jax_qc.build_basis_set(h2, "sto-3g")
    print(f"\nBasis: {basis.name}")
    print(f"  n_basis           = {basis.n_basis}")
    print(f"  shells            = {len(basis.shells)}")
    print(f"  shell_to_basis    = {basis.shell_to_basis}")
    print(f"  basis_to_atom     = {np.asarray(basis.basis_to_atom).tolist()}")

    for i, shell in enumerate(basis.shells):
        print(f"\n  Shell {i} on atom {shell.atom_index} (l={shell.angular_momentum}):")
        print(f"    exponents     = {np.asarray(shell.exponents)}")
        print(f"    coefficients  = {np.asarray(shell.coefficients)}  (normalized)")


if __name__ == "__main__":
    main()
