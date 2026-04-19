"""Physical constants and unit conversion factors used across jax_qc.

All internal computations use atomic units (Bohr, Hartree, electron mass = 1,
electron charge = 1, hbar = 1). These constants convert to/from common units
and encode element data needed for basis construction.

FP: Pure constants — no side effects.
"""

from __future__ import annotations

# --- Length ---
# CODATA 2018 recommended value.
BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM

# --- Energy ---
HARTREE_TO_EV: float = 27.211386245988
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

HARTREE_TO_KCAL_PER_MOL: float = 627.5094740631
KCAL_PER_MOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL_PER_MOL

HARTREE_TO_KJ_PER_MOL: float = 2625.4996394798254
KJ_PER_MOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KJ_PER_MOL

# --- Misc ---
# Speed of light in atomic units
SPEED_OF_LIGHT_AU: float = 137.035999084

# --- Elements ---
# Index is atomic number Z (1..118); Z=0 is the placeholder 'X' (dummy atom).
ELEMENT_SYMBOLS: tuple[str, ...] = (
    "X",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

# Symbol -> atomic number. Case-insensitive via .title() on lookup.
SYMBOL_TO_Z: dict[str, int] = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS)}


def symbol_to_z(symbol: str) -> int:
    """Convert an element symbol to atomic number. Case-insensitive."""
    key = symbol.strip().title()
    if key not in SYMBOL_TO_Z:
        raise ValueError(f"Unknown element symbol: {symbol!r}")
    return SYMBOL_TO_Z[key]


def z_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    if z < 0 or z >= len(ELEMENT_SYMBOLS):
        raise ValueError(f"Atomic number out of range: {z}")
    return ELEMENT_SYMBOLS[z]
