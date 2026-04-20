"""Build S, T, V, ERI, and E_nuc for H2 / STO-3G and H3+ / STO-3G.

This is what Step 2 adds on top of Step 1: analytical s-type molecular
integrals. The script wraps the build in a StageTimer so you can see
where time is spent, and compares every numerical result against a
reference value (PySCF when installed, otherwise hard-coded literature
values for these tiny cases).
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import jax_qc
from jax_qc.core.types import make_molecule
from examples._colors import banner, compare, error, header, label, ok, value

try:
    from pyscf import gto

    _HAS_PYSCF = True
except ImportError:  # pragma: no cover
    _HAS_PYSCF = False


# ---- Reference energies ------------------------------------------------
#
# For s-type-only systems we can produce every integral matrix
# analytically. The reference values below come from PySCF when
# available; the literature / exactly-computed fall-backs are enough to
# keep the example self-contained on a minimal install.

H2_REF = {
    "E_nuc": 1.0 / 1.4,  # 1/R for two protons at 1.4 Bohr
    "S_off": 0.6593182201,  # <1s_A|1s_B> at R=1.4 (STO-3G, from PySCF)
    "T_diag": 0.7600318839,
    "V_diag": -1.8804408895,
    "ERI_aaaa": 0.7746059441,
    "ERI_aabb": 0.5696759265,
    "ERI_abab": 0.2970285412,
}

H3_REF = {
    # Equilateral triangle with bond length 1.65 Bohr: R_ij = 1.65 for
    # every pair; E_nuc = 3 * 1/1.65 = 1.81818...
    "E_nuc": 3.0 / 1.65,
}


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


def _pyscf_reference(mol) -> dict | None:
    """Return PySCF-computed S, T, V, ERI, E_nuc for the given molecule."""
    if not _HAS_PYSCF:
        return None
    coords = np.asarray(mol.coords)
    atoms = [[sym, tuple(xyz.tolist())] for sym, xyz in zip(mol.elements, coords)]
    m = gto.M(
        atom=atoms,
        basis="sto-3g",
        unit="bohr",
        charge=int(mol.charge),
        spin=int(mol.spin),
    )
    return {
        "S": m.intor("int1e_ovlp"),
        "T": m.intor("int1e_kin"),
        "V": m.intor("int1e_nuc"),
        "ERI": m.intor("int2e"),
        "E_nuc": float(m.energy_nuc()),
    }


def _print_matrix_diff(name: str, got: np.ndarray, ref: np.ndarray, tol: float) -> None:
    diff = float(np.max(np.abs(got - ref)))
    passed = diff <= tol
    tag = ok("PASS") if passed else error("FAIL")
    print(
        f"  {label(f'{name} max |Δ|')}: {value(f'{diff:.2e}')}  "
        f"(tol {tol:.0e}) [{tag}]"
    )


def _run(name: str, mol) -> None:
    print()
    print(banner(f"{name} / STO-3G — Integrals + E_nuc"))
    basis = jax_qc.build_basis_set(mol, "sto-3g")
    timer = jax_qc.StageTimer(sync_device=False)
    ints = jax_qc.compute_integrals(mol, basis, timer=timer)

    np.set_printoptions(precision=6, suppress=True)
    print(f"{label('n_basis')} = {value(str(basis.n_basis))}")
    print()
    print(header("S ="))
    print(np.asarray(ints.S))
    print(header("T ="))
    print(np.asarray(ints.T))
    print(header("V ="))
    print(np.asarray(ints.V))
    print(header("H_core ="))
    print(np.asarray(ints.H_core))
    print(
        f"{label('ERI shape')}: {ints.ERI.shape}   "
        f"{label('ERI[0,0,0,0]')} = {value(f'{float(ints.ERI[0,0,0,0]):.10f}')}"
    )
    print(f"{label('E_nuc')}  = {value(f'{ints.E_nuc:.10f}')}")

    # ---- Reference comparisons -----------------------------------------
    print()
    print(header("Reference comparison", color="bright_green"))
    ref_source = "pyscf" if _HAS_PYSCF else "literature"
    print(f"  {label('source')}: {value(ref_source)}")

    ref = _pyscf_reference(mol)
    if ref is not None:
        S = np.asarray(ints.S)
        T = np.asarray(ints.T)
        V = np.asarray(ints.V)
        ERI = np.asarray(ints.ERI)
        _print_matrix_diff("S", S, ref["S"], tol=1e-8)
        _print_matrix_diff("T", T, ref["T"], tol=1e-8)
        _print_matrix_diff("V", V, ref["V"], tol=1e-8)
        _print_matrix_diff("ERI", ERI, ref["ERI"], tol=1e-8)
        print(
            "  "
            + compare(
                ints.E_nuc,
                ref["E_nuc"],
                label_got="E_nuc (jax_qc)",
                label_ref="E_nuc (pyscf)",
                tol=1e-12,
            )
        )
    else:
        if name == "H2":
            r = H2_REF
            S = np.asarray(ints.S)
            T = np.asarray(ints.T)
            V = np.asarray(ints.V)
            ERI = np.asarray(ints.ERI)
            print(
                "  "
                + compare(
                    float(S[0, 1]), r["S_off"], label_got="S[0,1]",
                    label_ref="ref", tol=1e-8,
                )
            )
            print(
                "  "
                + compare(
                    float(T[0, 0]), r["T_diag"], label_got="T[0,0]",
                    label_ref="ref", tol=1e-8,
                )
            )
            print(
                "  "
                + compare(
                    float(V[0, 0]), r["V_diag"], label_got="V[0,0]",
                    label_ref="ref", tol=1e-8,
                )
            )
            print(
                "  "
                + compare(
                    float(ERI[0, 0, 0, 0]), r["ERI_aaaa"],
                    label_got="(aa|aa)", label_ref="ref", tol=1e-8,
                )
            )
            print(
                "  "
                + compare(
                    ints.E_nuc, r["E_nuc"],
                    label_got="E_nuc", label_ref="1/R=1/1.4", tol=1e-12,
                )
            )
        elif name == "H3+":
            print(
                "  "
                + compare(
                    ints.E_nuc, H3_REF["E_nuc"],
                    label_got="E_nuc", label_ref="3/R=3/1.65", tol=1e-12,
                )
            )

    print()
    print(header("Profiling report:"))
    print(timer.report())


def main() -> None:
    _run("H2", _h2())
    _run("H3+", _h3_plus())


if __name__ == "__main__":
    main()
