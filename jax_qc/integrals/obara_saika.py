"""Primitive integral blocks over arbitrary-l Cartesian Gaussian pairs.

Although the filename says "Obara-Saika" (the classical name for the
recurrence family), we implement the closely-related McMurchie-Davidson
scheme — see ``mcmurchie_davidson.py`` for the 1D Hermite expansion
coefficients ``E_{ij}^t`` and the auxiliary ``R_{tuv}^n`` used by the
two-electron and nuclear primitives.

Every function below returns a dense Cartesian block over the ``nc_a =
(la+1)(la+2)/2`` and ``nc_b = (lb+1)(lb+2)/2`` components of a shell pair.
The blocks are NumPy arrays (not JAX) — this code is invoked at
integral-build time from a Python-level loop, not inside a jit. The
enclosing matrix builders pack the blocks into a final ``jnp.ndarray``.

Formulas:

* Primitive overlap (MMD, 1D):
      S_{ij}^{x} = E^{(x)}_{ij, 0} * sqrt(pi / p)
  3D: S_abc = S_{ij}^x * S_{kl}^y * S_{mn}^z * exp(-mu |A-B|^2)  (the
  ``K_AB`` prefactor is combined with the 3D product here).

* Primitive kinetic (1D):
      T_{ij}^{x} = 4 alpha beta S_{i+1, j+1}^{x}
                   + i*j*S_{i-1, j-1}^{x}
                   - 2 alpha * j S_{i+1, j-1}^{x}
                   - 2 beta * i S_{i-1, j+1}^{x}
  (Derived from <x^i | -1/2 d^2/dx^2 | x^j>; see Helgaker 9.3.39.)
  3D: T = T_x S_y S_z + S_x T_y S_z + S_x S_y T_z.

* Primitive nuclear attraction at nucleus C with charge Z:
      V = -Z * (2 pi / p) * K_AB * sum_{tuv} E_x_t E_y_u E_z_v * R_{tuv}^0(p, P-C)

* Primitive ERI (ab|cd):
      (ab|cd) = (2 pi^(5/2) / (p q sqrt(p+q))) * K_AB * K_CD
                * sum_{tuv,t'u'v'} E^(AB)_x_t E^(AB)_y_u E^(AB)_z_v
                                   (-1)^{t'+u'+v'}
                                   E^(CD)_x_t' E^(CD)_y_u' E^(CD)_z_v'
                                   R_{t+t', u+u', v+v'}^0(alpha=rho, PQ)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from jax_qc.integrals.mcmurchie_davidson import (
    cartesian_components,
    hermite_e,
    hermite_r_aux,
    n_cartesian,
)


def _K_AB(alpha: float, beta: float, A: np.ndarray, B: np.ndarray) -> float:
    """Prefactor exp(-mu |A-B|^2) with mu = alpha beta / (alpha + beta)."""
    p = alpha + beta
    mu = alpha * beta / p
    AB = A - B
    return float(np.exp(-mu * float(AB @ AB)))


def _P_center(alpha: float, beta: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (alpha * A + beta * B) / (alpha + beta)


# ----- one-electron primitive blocks -----------------------------------

def _overlap_1d(la: int, lb: int, Xpa: float, Xpb: float, p: float) -> np.ndarray:
    """1D overlap matrix S[i, j] = sqrt(pi/p) * E[i, j, 0] for i<=la, j<=lb.

    Returns shape (la+1, lb+1). Does NOT include K_AB.
    """
    E = hermite_e(la, lb, Xpa, Xpb, p)
    return math.sqrt(math.pi / p) * E[:, :, 0]


def _overlap_1d_extended(
    la: int, lb: int, Xpa: float, Xpb: float, p: float
) -> np.ndarray:
    """Same as ``_overlap_1d`` but for indices up to (la+1, lb+1).

    Used by the kinetic primitive, which needs S at shifted indices.
    """
    E = hermite_e(la + 1, lb + 1, Xpa, Xpb, p)
    return math.sqrt(math.pi / p) * E[:, :, 0]


def primitive_overlap_block(
    alpha: float,
    A: np.ndarray,
    la: int,
    beta: float,
    B: np.ndarray,
    lb: int,
) -> np.ndarray:
    """Primitive overlap over all Cartesian components; shape (nc_a, nc_b)."""
    p = alpha + beta
    P = _P_center(alpha, beta, A, B)
    Sx = _overlap_1d(la, lb, float(P[0] - A[0]), float(P[0] - B[0]), p)
    Sy = _overlap_1d(la, lb, float(P[1] - A[1]), float(P[1] - B[1]), p)
    Sz = _overlap_1d(la, lb, float(P[2] - A[2]), float(P[2] - B[2]), p)
    K = _K_AB(alpha, beta, A, B)

    comps_a = cartesian_components(la)
    comps_b = cartesian_components(lb)
    out = np.empty((n_cartesian(la), n_cartesian(lb)), dtype=np.float64)
    for i, (ax, ay, az) in enumerate(comps_a):
        for j, (bx, by, bz) in enumerate(comps_b):
            out[i, j] = K * Sx[ax, bx] * Sy[ay, by] * Sz[az, bz]
    return out


def _kinetic_1d(
    la: int,
    lb: int,
    Xpa: float,
    Xpb: float,
    alpha: float,
    beta: float,
    p: float,
) -> np.ndarray:
    """1D kinetic matrix T[i, j] for i<=la, j<=lb. Does NOT include K_AB."""
    # Extended 1D overlap up to (la+1, lb+1).
    S = _overlap_1d_extended(la, lb, Xpa, Xpb, p)
    T = np.zeros((la + 1, lb + 1), dtype=np.float64)
    for i in range(la + 1):
        for j in range(lb + 1):
            t_val = 4.0 * alpha * beta * S[i + 1, j + 1]
            if i > 0 and j > 0:
                t_val += i * j * S[i - 1, j - 1]
            if j > 0:
                t_val -= 2.0 * alpha * j * S[i + 1, j - 1]
            if i > 0:
                t_val -= 2.0 * beta * i * S[i - 1, j + 1]
            T[i, j] = 0.5 * t_val
    return T


def primitive_kinetic_block(
    alpha: float,
    A: np.ndarray,
    la: int,
    beta: float,
    B: np.ndarray,
    lb: int,
) -> np.ndarray:
    """Primitive kinetic over all Cartesian components; shape (nc_a, nc_b)."""
    p = alpha + beta
    P = _P_center(alpha, beta, A, B)
    K = _K_AB(alpha, beta, A, B)
    Xpa = float(P[0] - A[0]); Xpb = float(P[0] - B[0])
    Ypa = float(P[1] - A[1]); Ypb = float(P[1] - B[1])
    Zpa = float(P[2] - A[2]); Zpb = float(P[2] - B[2])
    Sx = _overlap_1d_extended(la, lb, Xpa, Xpb, p)
    Sy = _overlap_1d_extended(la, lb, Ypa, Ypb, p)
    Sz = _overlap_1d_extended(la, lb, Zpa, Zpb, p)
    Tx = _kinetic_1d(la, lb, Xpa, Xpb, alpha, beta, p)
    Ty = _kinetic_1d(la, lb, Ypa, Ypb, alpha, beta, p)
    Tz = _kinetic_1d(la, lb, Zpa, Zpb, alpha, beta, p)

    comps_a = cartesian_components(la)
    comps_b = cartesian_components(lb)
    out = np.empty((n_cartesian(la), n_cartesian(lb)), dtype=np.float64)
    for i, (ax, ay, az) in enumerate(comps_a):
        for j, (bx, by, bz) in enumerate(comps_b):
            val = (
                Tx[ax, bx] * Sy[ay, by] * Sz[az, bz]
                + Sx[ax, bx] * Ty[ay, by] * Sz[az, bz]
                + Sx[ax, bx] * Sy[ay, by] * Tz[az, bz]
            )
            out[i, j] = K * val
    return out


def primitive_nuclear_block(
    alpha: float,
    A: np.ndarray,
    la: int,
    beta: float,
    B: np.ndarray,
    lb: int,
    nuc_coords: np.ndarray,
    nuc_charges: np.ndarray,
) -> np.ndarray:
    """Primitive nuclear attraction summed over all nuclei; shape (nc_a, nc_b)."""
    p = alpha + beta
    P = _P_center(alpha, beta, A, B)
    K = _K_AB(alpha, beta, A, B)

    Ex = hermite_e(la, lb, float(P[0] - A[0]), float(P[0] - B[0]), p)
    Ey = hermite_e(la, lb, float(P[1] - A[1]), float(P[1] - B[1]), p)
    Ez = hermite_e(la, lb, float(P[2] - A[2]), float(P[2] - B[2]), p)

    comps_a = cartesian_components(la)
    comps_b = cartesian_components(lb)
    out = np.zeros((n_cartesian(la), n_cartesian(lb)), dtype=np.float64)
    two_pi_over_p = 2.0 * math.pi / p

    for Cvec, Z in zip(nuc_coords, nuc_charges):
        PC = np.asarray(P - Cvec, dtype=np.float64)
        R = hermite_r_aux(la + lb, la + lb, la + lb, 0, p, PC)  # shape (...,1)
        prefac = -float(Z) * two_pi_over_p * K
        for i, (ax, ay, az) in enumerate(comps_a):
            for j, (bx, by, bz) in enumerate(comps_b):
                acc = 0.0
                Ex_ij = Ex[ax, bx, : ax + bx + 1]
                Ey_ij = Ey[ay, by, : ay + by + 1]
                Ez_ij = Ez[az, bz, : az + bz + 1]
                for t in range(ax + bx + 1):
                    ex = Ex_ij[t]
                    if ex == 0.0:
                        continue
                    for u in range(ay + by + 1):
                        ey = Ey_ij[u]
                        if ey == 0.0:
                            continue
                        for v in range(az + bz + 1):
                            acc += ex * ey * Ez_ij[v] * R[t, u, v, 0]
                out[i, j] += prefac * acc
    return out


# ----- two-electron primitive block ------------------------------------

def primitive_eri_block(
    alpha: float, A: np.ndarray, la: int,
    beta: float, B: np.ndarray, lb: int,
    gamma: float, C: np.ndarray, lc: int,
    delta: float, D: np.ndarray, ld: int,
) -> np.ndarray:
    """Primitive (ab|cd) ERI over all Cartesian components.

    Returns shape (nc_a, nc_b, nc_c, nc_d).
    """
    p = alpha + beta
    q = gamma + delta
    rho = p * q / (p + q)
    P = _P_center(alpha, beta, A, B)
    Q = _P_center(gamma, delta, C, D)
    K_AB = _K_AB(alpha, beta, A, B)
    K_CD = _K_AB(gamma, delta, C, D)

    Ex_ab = hermite_e(la, lb, float(P[0] - A[0]), float(P[0] - B[0]), p)
    Ey_ab = hermite_e(la, lb, float(P[1] - A[1]), float(P[1] - B[1]), p)
    Ez_ab = hermite_e(la, lb, float(P[2] - A[2]), float(P[2] - B[2]), p)
    Ex_cd = hermite_e(lc, ld, float(Q[0] - C[0]), float(Q[0] - D[0]), q)
    Ey_cd = hermite_e(lc, ld, float(Q[1] - C[1]), float(Q[1] - D[1]), q)
    Ez_cd = hermite_e(lc, ld, float(Q[2] - C[2]), float(Q[2] - D[2]), q)

    PQ = np.asarray(P - Q, dtype=np.float64)
    t_max = la + lb
    u_max = la + lb
    v_max = la + lb
    t2_max = lc + ld
    u2_max = lc + ld
    v2_max = lc + ld
    # R[t][u][v][0] for t+t' <= la+lb+lc+ld etc.
    R = hermite_r_aux(
        t_max + t2_max,
        u_max + u2_max,
        v_max + v2_max,
        0,
        rho,
        PQ,
    )  # shape (T+1, U+1, V+1, 1)

    prefac = (
        2.0 * math.pi ** 2.5
        / (p * q * math.sqrt(p + q))
        * K_AB
        * K_CD
    )

    comps_a = cartesian_components(la)
    comps_b = cartesian_components(lb)
    comps_c = cartesian_components(lc)
    comps_d = cartesian_components(ld)
    out = np.zeros(
        (n_cartesian(la), n_cartesian(lb), n_cartesian(lc), n_cartesian(ld)),
        dtype=np.float64,
    )

    for i, (ax, ay, az) in enumerate(comps_a):
        for j, (bx, by, bz) in enumerate(comps_b):
            ex_ab = Ex_ab[ax, bx, : ax + bx + 1]
            ey_ab = Ey_ab[ay, by, : ay + by + 1]
            ez_ab = Ez_ab[az, bz, : az + bz + 1]
            for k, (cx, cy, cz) in enumerate(comps_c):
                for l, (dx, dy, dz) in enumerate(comps_d):
                    ex_cd = Ex_cd[cx, dx, : cx + dx + 1]
                    ey_cd = Ey_cd[cy, dy, : cy + dy + 1]
                    ez_cd = Ez_cd[cz, dz, : cz + dz + 1]
                    acc = 0.0
                    for t in range(ax + bx + 1):
                        e_t = ex_ab[t]
                        if e_t == 0.0:
                            continue
                        for u in range(ay + by + 1):
                            e_u = ey_ab[u]
                            if e_u == 0.0:
                                continue
                            for v in range(az + bz + 1):
                                e_v = ez_ab[v]
                                if e_v == 0.0:
                                    continue
                                for tp in range(cx + dx + 1):
                                    f_tp = ex_cd[tp]
                                    if f_tp == 0.0:
                                        continue
                                    sign_t = -1.0 if (tp & 1) else 1.0
                                    for up in range(cy + dy + 1):
                                        f_up = ey_cd[up]
                                        if f_up == 0.0:
                                            continue
                                        sign_tu = -sign_t if (up & 1) else sign_t
                                        for vp in range(cz + dz + 1):
                                            f_vp = ez_cd[vp]
                                            if f_vp == 0.0:
                                                continue
                                            sign = (
                                                -sign_tu if (vp & 1) else sign_tu
                                            )
                                            acc += (
                                                sign
                                                * e_t * e_u * e_v
                                                * f_tp * f_up * f_vp
                                                * R[t + tp, u + up, v + vp, 0]
                                            )
                    out[i, j, k, l] = prefac * acc
    return out


# ----- contracted shell-pair / quartet wrappers -------------------------

def contracted_overlap_block(sa, sb) -> np.ndarray:
    """Sum ``primitive_overlap_block`` over all (i, j) primitive pairs."""
    A = np.asarray(sa.center, dtype=np.float64)
    B = np.asarray(sb.center, dtype=np.float64)
    la = int(sa.angular_momentum)
    lb = int(sb.angular_momentum)
    exps_a = np.asarray(sa.exponents, dtype=np.float64)
    exps_b = np.asarray(sb.exponents, dtype=np.float64)
    ca = np.asarray(sa.coefficients, dtype=np.float64)
    cb = np.asarray(sb.coefficients, dtype=np.float64)
    block = np.zeros((n_cartesian(la), n_cartesian(lb)), dtype=np.float64)
    for i, a in enumerate(exps_a):
        for j, b in enumerate(exps_b):
            block += ca[i] * cb[j] * primitive_overlap_block(
                float(a), A, la, float(b), B, lb
            )
    return block


def contracted_kinetic_block(sa, sb) -> np.ndarray:
    A = np.asarray(sa.center, dtype=np.float64)
    B = np.asarray(sb.center, dtype=np.float64)
    la = int(sa.angular_momentum)
    lb = int(sb.angular_momentum)
    exps_a = np.asarray(sa.exponents, dtype=np.float64)
    exps_b = np.asarray(sb.exponents, dtype=np.float64)
    ca = np.asarray(sa.coefficients, dtype=np.float64)
    cb = np.asarray(sb.coefficients, dtype=np.float64)
    block = np.zeros((n_cartesian(la), n_cartesian(lb)), dtype=np.float64)
    for i, a in enumerate(exps_a):
        for j, b in enumerate(exps_b):
            block += ca[i] * cb[j] * primitive_kinetic_block(
                float(a), A, la, float(b), B, lb
            )
    return block


def contracted_nuclear_block(
    sa, sb, nuc_coords: np.ndarray, nuc_charges: np.ndarray
) -> np.ndarray:
    A = np.asarray(sa.center, dtype=np.float64)
    B = np.asarray(sb.center, dtype=np.float64)
    la = int(sa.angular_momentum)
    lb = int(sb.angular_momentum)
    exps_a = np.asarray(sa.exponents, dtype=np.float64)
    exps_b = np.asarray(sb.exponents, dtype=np.float64)
    ca = np.asarray(sa.coefficients, dtype=np.float64)
    cb = np.asarray(sb.coefficients, dtype=np.float64)
    block = np.zeros((n_cartesian(la), n_cartesian(lb)), dtype=np.float64)
    for i, a in enumerate(exps_a):
        for j, b in enumerate(exps_b):
            block += ca[i] * cb[j] * primitive_nuclear_block(
                float(a), A, la, float(b), B, lb, nuc_coords, nuc_charges
            )
    return block


def contracted_eri_block(sa, sb, sc, sd) -> np.ndarray:
    A = np.asarray(sa.center, dtype=np.float64)
    B = np.asarray(sb.center, dtype=np.float64)
    C = np.asarray(sc.center, dtype=np.float64)
    D = np.asarray(sd.center, dtype=np.float64)
    la = int(sa.angular_momentum)
    lb = int(sb.angular_momentum)
    lc = int(sc.angular_momentum)
    ld = int(sd.angular_momentum)
    exps_a = np.asarray(sa.exponents, dtype=np.float64)
    exps_b = np.asarray(sb.exponents, dtype=np.float64)
    exps_c = np.asarray(sc.exponents, dtype=np.float64)
    exps_d = np.asarray(sd.exponents, dtype=np.float64)
    ca = np.asarray(sa.coefficients, dtype=np.float64)
    cb = np.asarray(sb.coefficients, dtype=np.float64)
    cc = np.asarray(sc.coefficients, dtype=np.float64)
    cd = np.asarray(sd.coefficients, dtype=np.float64)
    block = np.zeros(
        (n_cartesian(la), n_cartesian(lb), n_cartesian(lc), n_cartesian(ld)),
        dtype=np.float64,
    )
    for i, a in enumerate(exps_a):
        for j, b in enumerate(exps_b):
            for k, c in enumerate(exps_c):
                for m, d in enumerate(exps_d):
                    w = ca[i] * cb[j] * cc[k] * cd[m]
                    block += w * primitive_eri_block(
                        float(a), A, la,
                        float(b), B, lb,
                        float(c), C, lc,
                        float(d), D, ld,
                    )
    return block
