"""Fetch basis sets from Basis Set Exchange.

The BSE library ships a local copy of all basis set data (~37 MB), so no
network access is needed after install. This is NOT a web API call.

Install: pip install basis-set-exchange

FP: Pure functions — deterministic lookups against a local database.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import basis_set_exchange as bse


def fetch_basis(basis_name: str, elements: Iterable[int]) -> dict:
    """Fetch basis set data from BSE as a Python dict.

    Args:
        basis_name: e.g. 'sto-3g', '6-31G*', 'cc-pVDZ', 'aug-cc-pVTZ'.
                    Case-insensitive; BSE handles name normalization.
        elements:   iterable of atomic numbers, e.g. [1, 8] for H and O.

    Returns:
        BSE dict with the usual 'elements' / 'electron_shells' structure.
        Shells with combined sp angular momenta are split into separate
        s and p shells (``uncontract_spdf=True``) so downstream code can
        assume one angular momentum per shell.
    """
    elements = sorted({int(z) for z in elements})
    if not elements:
        raise ValueError("fetch_basis: elements must be non-empty.")
    return bse.get_basis(
        basis_name,
        elements=elements,
        uncontract_spdf=True,
        header=False,
    )


def list_available_bases(elements: Optional[Iterable[int]] = None) -> List[str]:
    """List all basis sets available for the given elements.

    Args:
        elements: if provided, filter to basis sets that define all listed
                  atomic numbers. Otherwise return every name BSE knows.
    """
    all_bases = list(bse.get_all_basis_names())
    if elements is None:
        return all_bases
    wanted = sorted({int(z) for z in elements})
    out: List[str] = []
    for name in all_bases:
        try:
            data = bse.get_basis(name, elements=wanted, header=False)
        except Exception:
            continue
        defined = {int(z) for z in data.get("elements", {}).keys()}
        if all(z in defined for z in wanted):
            out.append(name)
    return out


def get_basis_info(basis_name: str) -> dict:
    """Return metadata for a basis set (family, description, versions)."""
    return {
        "family": bse.get_basis_family(basis_name),
        "description": bse.get_basis_notes(basis_name),
        "versions": list(bse.get_basis_versions(basis_name)),
    }
