"""HeH+ cation: heteronuclear two-atom system with a non-zero charge.

Demonstrates:
  * charge handling in make_molecule (electron count drops by +charge);
  * mixing elements (He + H) in one basis set request.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule


def main() -> None:
    heh = make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),  # Bohr
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )
    print(f"Molecule: HeH+  ({heh.elements})")
    print(f"  Z total       = {int(jnp.sum(heh.atomic_numbers))}")
    print(f"  charge        = {heh.charge}")
    print(f"  n_electrons   = {heh.n_electrons} (closed-shell 2-electron system)")

    basis = jax_qc.build_basis_set(heh, "sto-3g")
    print(f"\nBasis: sto-3g  (n_basis={basis.n_basis}, {len(basis.shells)} shells)")
    for i, shell in enumerate(basis.shells):
        center = np.asarray(shell.center)
        print(
            f"  Shell {i}: atom {shell.atom_index} ({heh.elements[shell.atom_index]}), "
            f"l={shell.angular_momentum}, center={center.tolist()}"
        )

    # Try a bigger basis to show that BSE names just work.
    basis_ccpvdz = jax_qc.build_basis_set(heh, "cc-pVDZ")
    print(f"\nBasis: cc-pVDZ (n_basis={basis_ccpvdz.n_basis}, "
          f"{len(basis_ccpvdz.shells)} shells)")


if __name__ == "__main__":
    main()
