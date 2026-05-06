"""Step 8: Geometry optimization with BFGS.

Demonstrates:
  1. Optimizing H2 bond length from a stretched geometry
  2. Optimizing HeH+ bond length
  3. Displaying the optimization trajectory
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule
from examples._colors import banner, header, label, ok, value, warn


def demo_h2_optimization() -> None:
    """Optimize H2 from a stretched geometry."""
    print()
    print(banner("H2 / STO-3G — Geometry Optimization (BFGS)"))

    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    r_start = float(jnp.linalg.norm(mol.coords[1] - mol.coords[0]))
    print(f"  {label('Starting R')} = {value(f'{r_start:.4f}')} Bohr")
    print()

    opt = jax_qc.optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=1
    )
    print()

    conv_tag = ok("converged") if opt.converged else warn("NOT converged")
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))
    g_max = float(jnp.max(jnp.abs(opt.gradient)))

    print(header("Result"))
    print(f"  {label('Status')}      = {conv_tag}")
    print(f"  {label('Steps')}       = {value(str(opt.n_steps))}")
    print(f"  {label('E_final')}     = {value(f'{opt.energy:.10f}')} Ha")
    print(f"  {label('R_eq')}        = {value(f'{r_eq:.6f}')} Bohr")
    print(f"  {label('max |grad|')}  = {value(f'{g_max:.2e}')} Ha/Bohr")
    print()

    # Trajectory
    print(header("Optimization Trajectory"))
    print(f"  {'Step':>4s}  {'R (Bohr)':>10s}  {'E (Ha)':>16s}  {'dE':>12s}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 16}  {'─' * 12}")
    for i, (c, e) in enumerate(opt.trajectory):
        r = float(np.linalg.norm(c[1] - c[0]))
        de = e - opt.trajectory[0][1] if i > 0 else 0.0
        print(f"  {i:4d}  {r:10.6f}  {e:16.10f}  {de:+12.8f}")


def demo_heh_optimization() -> None:
    """Optimize HeH+ from a non-equilibrium geometry."""
    print()
    print(banner("HeH+ / STO-3G — Geometry Optimization (BFGS)"))

    mol = make_molecule(
        elements=("He", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        atomic_numbers=jnp.array([2, 1], dtype=jnp.int32),
        charge=+1,
    )
    r_start = float(jnp.linalg.norm(mol.coords[1] - mol.coords[0]))
    print(f"  {label('Starting R')} = {value(f'{r_start:.4f}')} Bohr")
    print()

    opt = jax_qc.optimize_geometry(
        mol, basis_name="sto-3g", max_steps=30, grad_tol=1e-5, verbose=1
    )
    print()

    conv_tag = ok("converged") if opt.converged else warn("NOT converged")
    coords = np.asarray(opt.molecule.coords)
    r_eq = float(np.linalg.norm(coords[1] - coords[0]))

    print(header("Result"))
    print(f"  {label('Status')} = {conv_tag}")
    print(f"  {label('R_eq')}   = {value(f'{r_eq:.6f}')} Bohr")
    print(f"  {label('E_final')}= {value(f'{opt.energy:.10f}')} Ha")


def main() -> None:
    demo_h2_optimization()
    demo_heh_optimization()


if __name__ == "__main__":
    main()
