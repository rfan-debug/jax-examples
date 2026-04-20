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
python examples/06_integrals_h2.py   # computes S/T/V/ERI/E_nuc
```

Every script is self-contained and uses only functions exported from
`jax_qc` itself. Example `06_integrals_h2.py` will use PySCF as a
reference when it is installed (`pip install -e ".[full]"`); otherwise
it falls back to hard-coded literature values so the comparison still
runs on a minimal install.

## Colored output

The examples print with ANSI colors when running in a terminal. Honors:

| Variable | Effect |
|---|---|
| `NO_COLOR=1` | disable color (https://no-color.org) |
| `JAX_QC_FORCE_COLOR=1` | force color even when stdout is not a TTY (useful for piping to `less -R`) |
