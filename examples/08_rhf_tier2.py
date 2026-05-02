"""Step 4: RHF on Tier 2 closed-shell molecules with general angular momentum.

For each (molecule, basis) we run jax_qc's RHF and compare the total
energy against PySCF (when installed). The MMD primitives now support
arbitrary l up to d-shells (cc-pVDZ, 6-31G** all OK).

Note: this script runs more shells / quartets than the Tier 1 example so
expect 1-2 minutes for 6-31G** / cc-pVDZ on H2O.
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


# Reference RHF total energies from PySCF (Hartree).
REFS = {
    ("H2O", "sto-3g"):    -73.2144728363,
    ("H2O", "6-31g"):     -74.4943924164,
    ("H2O", "6-31g**"):   -74.5428909110,
    ("HF",  "sto-3g"):    -98.5708712083,
    ("HF",  "6-31g**"):   -100.0091831090,
    ("CH4", "sto-3g"):    -39.7268589862,
    ("NH3", "sto-3g"):    -55.4540090032,
}


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


MOLECULES = {"H2O": _h2o, "HF": _hf, "CH4": _ch4, "NH3": _nh3}


def _pyscf_reference(mol, basis_name) -> float | None:
    if not _HAS_PYSCF:
        return None
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


def _run(name: str, basis_name: str) -> None:
    print()
    print(banner(f"{name} / {basis_name} — RHF SCF"))
    mol = MOLECULES[name]()
    basis = jax_qc.build_basis_set(mol, basis_name)
    timer = jax_qc.StageTimer(sync_device=False)

    with timer.stage("integrals", "applicative"):
        ints = jax_qc.compute_integrals(mol, basis)

    config = CalcConfig(
        method="rhf", basis=basis_name, scf_conv=1e-10,
        max_scf_iter=128, diis_space=8,
    )
    result = jax_qc.run_scf(mol, ints, config, timer=timer)

    conv_tag = ok("converged") if result.converged else warn("NOT converged")
    print(f"  {label('n_basis')}     = {value(str(basis.n_basis))}")
    print(f"  {label('n_shells')}    = {value(str(len(basis.shells)))}")
    print(f"  {label('iterations')}  = {value(str(result.n_iterations))}  [{conv_tag}]")
    print(f"  {label('E_total')}     = {value(f'{result.energy:.10f}')} Ha")

    ref = _pyscf_reference(mol, basis_name)
    src = "pyscf" if ref is not None else "literature"
    if ref is None:
        ref = REFS.get((name, basis_name))
    if ref is None:
        print(f"  {label('no reference available for')} {name}/{basis_name}")
    else:
        print()
        print(header("Reference comparison"))
        print(f"  {label('source')}: {value(src)}")
        print("  " + compare(
            result.energy, ref,
            label_got="E_total (jax_qc)",
            label_ref=f"E_total ({src})",
            tol=1e-6,
        ))


def main() -> None:
    pairs = [
        ("H2O", "sto-3g"),
        ("HF",  "sto-3g"),
        ("CH4", "sto-3g"),
        ("NH3", "sto-3g"),
        ("H2O", "6-31g"),
        ("H2O", "6-31g**"),
    ]
    for (name, basis_name) in pairs:
        _run(name, basis_name)


if __name__ == "__main__":
    main()
