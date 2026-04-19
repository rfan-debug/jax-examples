"""Build S, T, V, ERI, and E_nuc for H2 / STO-3G and H3+ / STO-3G.

This is what Step 2 adds on top of Step 1: analytical s-type molecular
integrals. The script wraps the build in a StageTimer so you can see
where time is spent.
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule


def _h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
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


def _print(name, mol):
    print(f"\n=== {name} / STO-3G ===")
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    timer = jax_qc.StageTimer(sync_device=False)
    ints = jax_qc.compute_integrals(mol, basis, timer=timer)
    np.set_printoptions(precision=6, suppress=True)
    print(f"n_basis = {basis.n_basis}")
    print(f"S =\n{np.asarray(ints.S)}")
    print(f"T =\n{np.asarray(ints.T)}")
    print(f"V =\n{np.asarray(ints.V)}")
    print(f"H_core =\n{np.asarray(ints.H_core)}")
    print(f"ERI shape: {ints.ERI.shape}  (J-like diagonal = {float(ints.ERI[0,0,0,0]):.6f})")
    print(f"E_nuc = {ints.E_nuc:.10f}")
    print()
    print(timer.report())


def main() -> None:
    _print("H2", _h2())
    _print("H3+", _h3_plus())


if __name__ == "__main__":
    main()
