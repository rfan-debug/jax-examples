"""StageTimer demo.

Builds basis sets for a handful of molecules under a nested StageTimer and
prints the resulting report. Illustrates how driver code will use
`optional_stage` / `timer.stage(...)` once real integrals and SCF arrive.
"""

from __future__ import annotations

import jax.numpy as jnp

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule
from examples._colors import banner, header, label, value


def _h2():
    return make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )


def _heh_cation():
    return make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )


def _water():
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


def main() -> None:
    print(banner("StageTimer — Basis Construction Across Molecules"))
    timer = jax_qc.StageTimer(sync_device=False)

    with timer.stage("build_molecules", "pure"):
        molecules = {
            "H2": _h2(),
            "HeH+": _heh_cation(),
            "H2O": _water(),
        }

    with timer.stage("build_bases", "pure"):
        for name, mol in molecules.items():
            with timer.stage(f"basis_{name}", "pure"):
                for basis_name in ("sto-3g", "6-31G", "6-31G*"):
                    with timer.stage(basis_name, "applicative"):
                        basis = jax_qc.build_basis_set(mol, basis_name)
                        print(
                            f"  {label(f'{name:<5s}')} / "
                            f"{label(f'{basis_name:<8s}')} -> "
                            f"{label('n_basis')}={value(str(basis.n_basis))}"
                        )

    print()
    print(header("Profiling report:"))
    print(timer.report())


if __name__ == "__main__":
    main()
