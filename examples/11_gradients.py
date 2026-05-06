"""Step 7: Nuclear gradients and PES scan.

Demonstrates:
  1. Analytic gradient on H2/STO-3G
  2. Comparison with numerical (finite-difference) gradient
  3. PES scan along the H-H bond stretch
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import CalcConfig, make_molecule
from jax_qc.grad.rhf_grad import rhf_gradient
from jax_qc.grad.numerical_grad import numerical_gradient
from examples._colors import banner, compare, header, label, ok, value, warn


def demo_gradient() -> None:
    """Compute and display the H2 nuclear gradient."""
    print()
    print(banner("H2 / STO-3G — Nuclear Gradient"))

    mol = make_molecule(
        elements=("H", "H"),
        coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
    )
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    ints = jax_qc.compute_integrals(mol, basis)
    config = CalcConfig(method="rhf", scf_conv=1e-10)
    result = jax_qc.run_rhf(mol, ints, config)

    print(f"  {label('E_total')} = {value(f'{result.energy:.10f}')} Ha")
    print()

    # Analytic gradient
    print(header("Analytic Gradient (Hellmann-Feynman + Pulay)"))
    g_analytic = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
    for i, sym in enumerate(mol.elements):
        gx, gy, gz = g_analytic[i]
        print(
            f"  {label(sym)}  dE/dx = {value(f'{gx:+.8f}')}"
            f"  dE/dy = {value(f'{gy:+.8f}')}"
            f"  dE/dz = {value(f'{gz:+.8f}')}"
        )
    print()

    # Numerical gradient
    print(header("Numerical Gradient (central finite difference)"))
    g_numerical = np.asarray(numerical_gradient(mol, basis_name="sto-3g", method="rhf"))
    for i, sym in enumerate(mol.elements):
        gx, gy, gz = g_numerical[i]
        print(
            f"  {label(sym)}  dE/dx = {value(f'{gx:+.8f}')}"
            f"  dE/dy = {value(f'{gy:+.8f}')}"
            f"  dE/dz = {value(f'{gz:+.8f}')}"
        )
    print()

    # Comparison
    max_diff = float(np.max(np.abs(g_analytic - g_numerical)))
    tag = ok("PASS") if max_diff < 1e-5 else warn("FAIL")
    print(
        f"  {label('max |analytic - numerical|')} = {value(f'{max_diff:.2e}')}  [{tag}]"
    )


def demo_pes_scan() -> None:
    """Scan the H-H bond length and plot energy + gradient."""
    print()
    print(banner("H2 PES Scan — Energy and Force along bond stretch"))

    distances = np.linspace(0.8, 5.0, 15)
    energies = []
    forces_z = []

    for r in distances:
        mol = make_molecule(
            elements=("H", "H"),
            coords=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(r)]]),
            atomic_numbers=jnp.array([1, 1], dtype=jnp.int32),
        )
        basis = jax_qc.build_basis_set(mol, "sto-3g")
        ints = jax_qc.compute_integrals(mol, basis)
        config = CalcConfig(method="rhf", scf_conv=1e-10)
        result = jax_qc.run_rhf(mol, ints, config)
        energies.append(result.energy)

        grad = np.asarray(rhf_gradient(mol, result, basis_name="sto-3g"))
        forces_z.append(-grad[1, 2])  # force on atom 1 along z

    print(f"  {'R (Bohr)':>10s}  {'E (Ha)':>14s}  {'F_z (Ha/Bohr)':>14s}")
    print(f"  {'─' * 10}  {'─' * 14}  {'─' * 14}")
    for r, e, f in zip(distances, energies, forces_z):
        print(f"  {r:10.4f}  {e:14.8f}  {f:+14.8f}")

    # Find equilibrium (minimum energy)
    i_min = int(np.argmin(energies))
    print()
    print(
        f"  {label('Approx equilibrium')} R = {value(f'{distances[i_min]:.2f}')} Bohr"
        f"  E = {value(f'{energies[i_min]:.8f}')} Ha"
    )


def main() -> None:
    demo_gradient()
    demo_pes_scan()


if __name__ == "__main__":
    main()
