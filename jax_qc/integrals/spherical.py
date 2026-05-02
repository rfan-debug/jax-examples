"""Cartesian-to-spherical-harmonic transformation for shell components.

The McMurchie-Davidson primitives in ``obara_saika.py`` produce dense
Cartesian blocks of size ``(l+1)(l+2)/2``. For a "real solid harmonic"
spherical basis (the default in PySCF / libint / libcint) each shell with
angular momentum l contributes only ``2 l + 1`` basis functions; the
transformation between the two is a fixed linear map ``c_spherical_cart``
of shape ``(2 l + 1, (l+1)(l+2)/2)``.

We need to match libcint's row ordering (which PySCF uses by default):

    l=0: m = 0   -> [s]
    l=1: m = -1, 0, +1   -> [p_y, p_z, p_x]   (real spherical harmonics)
    l=2: m = -2, -1, 0, +1, +2  -> [d_xy, d_yz, d_z^2, d_xz, d_(x^2-y^2)]

Cartesian column ordering (libint, used by ``cartesian_components``):
    l=0: (0,0,0)
    l=1: (1,0,0), (0,1,0), (0,0,1)
    l=2: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)

The non-trivial transformation is l=2; for l<=1 the two bases coincide up
to a permutation. We hand-code matrices up to l=2 (sufficient for cc-pVDZ
and 6-31G* on first-row atoms).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def cart_to_spherical(l: int) -> np.ndarray:
    """Return the (2 l + 1, (l+1)(l+2)/2) Cartesian-to-spherical matrix."""
    if l == 0:
        return np.array([[1.0]])
    if l == 1:
        # Both bases agree (px, py, pz); transformation is identity. (PySCF
        # / libcint label spherical p as px, py, pz despite the m-quantum
        # number convention.)
        return np.eye(3)
    if l == 2:
        # Cartesian columns: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
        # i.e.                xx,     xy,      xz,      yy,      yz,      zz
        # Spherical rows (libcint): d_xy, d_yz, d_z^2, d_xz, d_(x^2-y^2)
        # The Cartesian Gaussians use the (l, 0, 0)-style normalization
        # (every component shares N_l). Real solid-harmonic conversion
        # picks up extra sqrt(3) / sqrt(12) factors to compensate for the
        # double-factorial mismatch between (2,0,0) and (1,1,0). The
        # libcint-used matrix is:
        sqrt3 = math.sqrt(3.0)
        return np.array(
            [
                # xx       xy     xz     yy        yz     zz
                [0.0, sqrt3, 0.0, 0.0, 0.0, 0.0],         # d_xy
                [0.0, 0.0, 0.0, 0.0, sqrt3, 0.0],         # d_yz
                [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],         # d_z^2
                [0.0, 0.0, sqrt3, 0.0, 0.0, 0.0],         # d_xz
                [sqrt3 / 2.0, 0.0, 0.0, -sqrt3 / 2.0, 0.0, 0.0],  # d_x^2-y^2
            ]
        )
    raise NotImplementedError(
        f"Cart-to-spherical transformation only implemented for l <= 2 "
        f"(got l={l})."
    )
