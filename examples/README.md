# jax_qc examples

Runnable examples that exercise the Step 1 foundation layer: molecule
construction, Basis Set Exchange integration, XYZ IO, and profiling.

SCF energies, integrals, and gradients land in later steps; these scripts
currently print basis-set and timing information only.

## Running

From the repo root:

```bash
pip install -e ".[test]"
python examples/01_h2_basis.py
python examples/02_heh_cation.py
python examples/03_water_basis_sets.py
python examples/04_xyz_roundtrip.py
python examples/05_profiling.py
```

Every script is self-contained and uses only functions exported from
`jax_qc` itself — no external QC packages required.
