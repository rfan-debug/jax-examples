"""McMurchie-Davidson recurrences for general angular momentum integrals.

Every one- and two-electron integral over Cartesian Gaussian primitives
can be expressed as a sum of Hermite-Gaussian expansion coefficients
``E_ij^t`` times auxiliary functions ``R_{tuv}^n``. This module implements
both with plain NumPy — they are invoked from a Python-level orchestration
loop that iterates over shell pairs/quartets, so we do not need JAX
traceability here. The outputs are packed into ``jnp.ndarray`` matrices
by the callers in ``overlap.py``, ``kinetic.py``, ``nuclear.py``, and
``eri.py``.

References:
    * McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978).
    * Helgaker, Jorgensen, Olsen, "Molecular Electronic Structure Theory",
      Section 9.3 (Hermite Gaussian product functions) and Section 9.9
      (auxiliary integrals).

Conventions:
    * Cartesian angular momentum tuples ``(lx, ly, lz)`` are enumerated
      for total ``l`` in canonical libint order:
          l=0: (0,0,0)
          l=1: (1,0,0), (0,1,0), (0,0,1)
          l=2: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
      (See ``cartesian_components``.)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

from jax_qc.integrals.boys import boys_fn


# --- Cartesian enumeration --------------------------------------------

@lru_cache(maxsize=32)
def cartesian_components(l: int) -> Tuple[Tuple[int, int, int], ...]:
    """Enumerate (lx, ly, lz) for total angular momentum ``l`` in libint order.

    Outer loop over lx (descending), inner over ly; lz = l - lx - ly.
    """
    out: List[Tuple[int, int, int]] = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            out.append((lx, ly, lz))
    return tuple(out)


def n_cartesian(l: int) -> int:
    """Number of Cartesian components for angular momentum ``l``: (l+1)(l+2)/2."""
    return (l + 1) * (l + 2) // 2


# --- Hermite E-coefficients (1D) --------------------------------------

def hermite_e(
    la: int,
    lb: int,
    Xpa: float,
    Xpb: float,
    p: float,
) -> np.ndarray:
    """Hermite Gaussian expansion coefficients E_{ij}^t for one dimension.

    Returns an array ``E`` of shape ``(la+1, lb+1, la+lb+1)`` such that

        x_A^i x_B^j K_{AB}(x) = sum_t E[i, j, t] Lambda_t(x)

    where ``Lambda_t(x)`` are Hermite Gaussians centered at the Gaussian
    product center.

    Recurrence (McMurchie-Davidson; Helgaker 9.5):
        E[0,0,0]  = K_AB (stored by caller as exp(-mu (A-B)^2))
        E[i+1,j,t] = (1/(2p)) E[i,j,t-1] + Xpa E[i,j,t] + (t+1) E[i,j,t+1]
        E[i,j+1,t] = (1/(2p)) E[i,j,t-1] + Xpb E[i,j,t] + (t+1) E[i,j,t+1]

    We store the prefactor K_AB separately so that callers can easily
    contract over primitives. Here ``E[0,0,0] = 1``.
    """
    E = np.zeros((la + 1, lb + 1, la + lb + 1), dtype=np.float64)
    E[0, 0, 0] = 1.0
    inv_2p = 0.5 / p

    # Increment i index.
    for i in range(la):
        for t in range(i + 1):  # support for E[i, 0, 0..i]
            e_tm1 = E[i, 0, t - 1] if t - 1 >= 0 else 0.0
            e_t = E[i, 0, t]
            e_tp1 = E[i, 0, t + 1] if t + 1 <= i + 1 else 0.0
            E[i + 1, 0, t] = inv_2p * e_tm1 + Xpa * e_t + (t + 1) * e_tp1
        # t = i + 1 special case (comes only from the inv_2p term)
        E[i + 1, 0, i + 1] = inv_2p * E[i, 0, i]

    # Increment j index.
    for j in range(lb):
        for i in range(la + 1):
            for t in range(i + j + 1):
                e_tm1 = E[i, j, t - 1] if t - 1 >= 0 else 0.0
                e_t = E[i, j, t]
                e_tp1 = E[i, j, t + 1] if t + 1 <= i + j else 0.0
                E[i, j + 1, t] = inv_2p * e_tm1 + Xpb * e_t + (t + 1) * e_tp1
            E[i, j + 1, i + j + 1] = inv_2p * E[i, j, i + j]

    return E


# --- Hermite auxiliary R_{tuv}^n --------------------------------------

def hermite_r_aux(
    t_max: int,
    u_max: int,
    v_max: int,
    n_max: int,
    p: float,
    PC: np.ndarray,
) -> np.ndarray:
    """Compute R_{tuv}^n (Helgaker 9.9.18-9.9.20) for all t<=t_max, u<=u_max,
    v<=v_max, n<=n_max.

    R_{000}^n = (-2 p)^n F_n(p |PC|^2)
    R_{t+1, u, v}^n = t R_{t-1, u, v}^{n+1} + Xpc R_{t, u, v}^{n+1}
    R_{t, u+1, v}^n = u R_{t, u-1, v}^{n+1} + Ypc R_{t, u, v}^{n+1}
    R_{t, u, v+1}^n = v R_{t, u, v-1}^{n+1} + Zpc R_{t, u, v}^{n+1}

    To compute R at n=0 we recurse through higher n up to ``n_max +
    t_max + u_max + v_max``. Returns shape ``(t_max+1, u_max+1, v_max+1,
    n_max+1)``.
    """
    L_max = t_max + u_max + v_max + n_max
    T = p * float(PC[0] ** 2 + PC[1] ** 2 + PC[2] ** 2)
    # F_n(T) for n = 0..L_max
    Fn = np.array([float(boys_fn(n, T)) for n in range(L_max + 1)], dtype=np.float64)

    # R[t][u][v][n]
    R = np.zeros(
        (t_max + 1, u_max + 1, v_max + 1, L_max + 1), dtype=np.float64
    )
    minus_2p = -2.0 * p
    pow_minus_2p = 1.0
    for n in range(L_max + 1):
        R[0, 0, 0, n] = pow_minus_2p * Fn[n]
        pow_minus_2p *= minus_2p

    Xpc, Ypc, Zpc = float(PC[0]), float(PC[1]), float(PC[2])

    # Increment t (consumes one level of n).
    for t in range(t_max):
        for n in range(L_max - t):
            r_tm1 = R[t - 1, 0, 0, n + 1] if t - 1 >= 0 else 0.0
            R[t + 1, 0, 0, n] = t * r_tm1 + Xpc * R[t, 0, 0, n + 1]

    # Increment u.
    for u in range(u_max):
        for t in range(t_max + 1):
            for n in range(L_max - t - u):
                r_um1 = R[t, u - 1, 0, n + 1] if u - 1 >= 0 else 0.0
                R[t, u + 1, 0, n] = u * r_um1 + Ypc * R[t, u, 0, n + 1]

    # Increment v.
    for v in range(v_max):
        for u in range(u_max + 1):
            for t in range(t_max + 1):
                for n in range(L_max - t - u - v):
                    r_vm1 = R[t, u, v - 1, n + 1] if v - 1 >= 0 else 0.0
                    R[t, u, v + 1, n] = v * r_vm1 + Zpc * R[t, u, v, n + 1]

    # Trim to the requested n range.
    return R[: t_max + 1, : u_max + 1, : v_max + 1, : n_max + 1]
