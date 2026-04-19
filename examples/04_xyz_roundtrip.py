"""XYZ input/output: parse, inspect, write back.

Reads a small water XYZ string, prints the derived Molecule, writes it
to disk in Bohr units, reads it back, and confirms the round-trip.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.io.xyz import parse_xyz_string


WATER_XYZ = """\
3
water at equilibrium, angstrom
O   0.000000   0.000000   0.117176
H   0.000000   0.757087  -0.468706
H   0.000000  -0.757087  -0.468706
"""


def main() -> None:
    mol = parse_xyz_string(WATER_XYZ)  # default unit: Angstrom on disk
    print("Parsed water:")
    print(f"  elements    = {mol.elements}")
    print(f"  n_electrons = {mol.n_electrons}")
    print(f"  coords (Bohr) =\n{np.asarray(mol.coords)}")

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "water.xyz"
        # Write in Bohr so we can verify the internal-coord round-trip.
        jax_qc.write_xyz(path, mol, unit="bohr", comment="water (bohr)")
        print(f"\nWrote {path}:")
        print(path.read_text())

        mol2 = jax_qc.read_xyz(path, unit="bohr")
        delta = float(np.max(np.abs(np.asarray(mol.coords) - np.asarray(mol2.coords))))
        print(f"Max coordinate difference after round-trip: {delta:.2e} Bohr")


if __name__ == "__main__":
    main()
