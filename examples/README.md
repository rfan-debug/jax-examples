# jax_qc examples

Runnable examples that exercise each layer of jax_qc — from Step 1
basis-set construction through Step 3 RHF SCF energies.

## Running

From the repo root:

```bash
pip install -e ".[test]"
python examples/01_h2_basis.py
python examples/02_heh_cation.py
python examples/03_water_basis_sets.py
python examples/04_xyz_roundtrip.py
python examples/05_profiling.py
python examples/06_integrals_h2.py   # Step 2: S/T/V/ERI/E_nuc
python examples/07_rhf_scf.py        # Step 3: RHF total energies
```

Every script is self-contained and uses only functions exported from
`jax_qc` itself. Examples 06 and 07 use PySCF as a reference when it is
installed (`pip install -e ".[full]"`); otherwise they fall back to
hard-coded literature values so the comparison still runs on a minimal
install.

## Colored output

The examples print with ANSI colors when running in a terminal. Honors:

| Variable | Effect |
|---|---|
| `NO_COLOR=1` | disable color (https://no-color.org) |
| `JAX_QC_FORCE_COLOR=1` | force color even when stdout is not a TTY (useful for piping to `less -R`) |
