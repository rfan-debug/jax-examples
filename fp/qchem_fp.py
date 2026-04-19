"""
Minimal Hartree-Fock in JAX — A Functional Programming Perspective

每一层计算都标注了对应的 FP 抽象，展示量子化学的计算结构
如何精确映射到 Functor / Applicative / Monad / Foldable 等概念。

Supports: RHF (Restricted Hartree-Fock) with s-type Gaussian basis
Demo:     H₂, HeH⁺, and potential energy surface scans

Author's note: 这不是 production code，而是一个 pedagogical framework，
目的是让 FP 结构在量子化学计算中"可见"。
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from dataclasses import dataclass
from typing import NamedTuple
from functools import partial
import time

# Enable 64-bit precision (essential for quantum chemistry)
jax.config.update("jax_enable_x64", True)


# ================================================================
# Part 1: Data Types — 底层类型定义
# ================================================================
# 这些是范畴中的"对象"（objects），后续所有计算都是这些对象之间的"态射"

class Primitive(NamedTuple):
    """原始高斯函数 g(r) = N * exp(-α|r-R|²)
    
    这是整个计算的"底层类型" — 所有量子化学量最终归结为
    这些基本函数之间的积分。
    """
    exponent: float      # α: 轨道指数
    center: jnp.ndarray  # R: 中心坐标 (3,)
    coeff: float         # d*N: 收缩系数 × 归一化因子


class BasisFunction(NamedTuple):
    """收缩高斯基函数 χ(r) = Σ_p d_p * g_p(r)
    
    从 FP 角度：这是 Primitive 上的 Free Monoid
    收缩 = 自由幺半群中的线性组合（formal sum）
    """
    primitives: list  # List[Primitive]


class Molecule(NamedTuple):
    """分子 = (原子坐标, 原子序数, 基函数集)"""
    coords: jnp.ndarray    # (n_atoms, 3)
    charges: jnp.ndarray   # (n_atoms,) — nuclear charges Z
    n_electrons: int
    basis: list             # List[BasisFunction]


# ================================================================
# Part 2: STO-3G Basis Set — 基函数构建
# ================================================================

# STO-3G 参数 (Hehre, Stewart, Pople, 1969)
STO3G_PARAMS = {
    # element: (exponents, coefficients) for 1s orbital
    'H': {
        '1s': {
            'exponents':    [3.42525091, 0.62391373, 0.16885540],
            'coefficients': [0.15432897, 0.53532814, 0.44463454],
        }
    },
    'He': {
        '1s': {
            'exponents':    [6.36242139, 1.15892300, 0.31364979],
            'coefficients': [0.15432897, 0.53532814, 0.44463454],
        }
    },
}


def normalize_primitive(alpha: float) -> float:
    """s 型高斯函数的归一化因子 N = (2α/π)^{3/4}
    
    Functor 的基础：保证 ⟨g|g⟩ = 1
    """
    return (2.0 * alpha / jnp.pi) ** 0.75


def build_basis(element: str, center: jnp.ndarray) -> list:
    """为单个原子构建基函数集
    
    这是一个 pure function: (元素, 坐标) → 基函数列表
    无 side effect，可以安全地 vmap over 多个原子
    """
    basis_fns = []
    params = STO3G_PARAMS[element]
    for shell_name, shell_data in params.items():
        prims = []
        for alpha, d in zip(shell_data['exponents'], shell_data['coefficients']):
            N = normalize_primitive(alpha)
            prims.append(Primitive(
                exponent=alpha,
                center=jnp.array(center, dtype=jnp.float64),
                coeff=d * N
            ))
        basis_fns.append(BasisFunction(primitives=prims))
    return basis_fns


# ================================================================
# Part 3: Molecular Integrals — Monoidal Structure (einsum / contraction)
# ================================================================
# 
# 核心思想：所有积分都是高斯函数之间的多线性映射
# 在 compact closed monoidal category 中，
# 每个积分对应一个 string diagram，指标收缩对应连线
#
# 这一层是 Pure / Applicative 的：
# 各个积分互不依赖，可以完全并行计算

# --- 辅助函数 ---

def boys_function_F0(t):
    """Boys 函数 F_0(t) = ∫₀¹ exp(-t·u²) du
    
    这是核吸引和双电子积分的核心
    F_0(t) = √(π/t)/2 · erf(√t)  for t > 0
    F_0(0) = 1
    """
    safe_t = jnp.where(t > 1e-12, t, 1e-12)
    return jnp.where(
        t > 1e-12,
        0.5 * jnp.sqrt(jnp.pi / safe_t) * jax.scipy.special.erf(jnp.sqrt(safe_t)),
        1.0
    )


def gaussian_product_center(alpha, center_a, beta, center_b):
    """高斯乘积定理：两个高斯的乘积 = 新的高斯
    
    P = (α·A + β·B) / (α + β)
    
    范畴论视角：这是 monoidal product 的内部操作
    两个对象（高斯）"融合"成一个（乘积高斯）
    """
    return (alpha * center_a + beta * center_b) / (alpha + beta)


# --- 单电子积分 (rank-2 张量 = 态射) ---

def overlap_primitive(pa: Primitive, pb: Primitive) -> float:
    """原始重叠积分 S_ab = ⟨g_a|g_b⟩
    
    S_ab = (π/p)^{3/2} · exp(-μ·|A-B|²)
    其中 p = α+β, μ = αβ/p
    
    FP 类型签名：Primitive → Primitive → Scalar
    这是一个 bilinear form — 范畴中的态射 V⊗V → k
    """
    alpha, A, da = pa
    beta, B, db = pb
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = jnp.sum((A - B) ** 2)
    return da * db * (jnp.pi / p) ** 1.5 * jnp.exp(-mu * AB2)


def kinetic_primitive(pa: Primitive, pb: Primitive) -> float:
    """原始动能积分 T_ab = ⟨g_a|-½∇²|g_b⟩
    
    T_ab = μ(3 - 2μ|A-B|²) · S_ab    (μ = αβ/(α+β))
    
    拉普拉斯算子 ∇² 是一个 endomorphism（自态射）
    动能积分 = 在这个自态射下的双线性形式
    """
    alpha, A, da = pa
    beta, B, db = pb
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = jnp.sum((A - B) ** 2)
    S_ab = (jnp.pi / p) ** 1.5 * jnp.exp(-mu * AB2)
    return da * db * mu * (3.0 - 2.0 * mu * AB2) * S_ab


def nuclear_primitive(pa: Primitive, pb: Primitive,
                      nuc_coord: jnp.ndarray, nuc_charge: float) -> float:
    """原始核吸引积分 V_ab = -Z · ⟨g_a|1/|r-C||g_b⟩
    
    V_ab = -Z · (2π/p) · K_ab · F_0(p·|P-C|²)
    
    库仑势 1/|r-C| 引入了 Boys 函数 — 这是量子化学
    区别于普通线性代数的地方：物理学的非线性进入计算
    """
    alpha, A, da = pa
    beta, B, db = pb
    p = alpha + beta
    mu = alpha * beta / p
    AB2 = jnp.sum((A - B) ** 2)
    P = gaussian_product_center(alpha, A, beta, B)
    PC2 = jnp.sum((P - nuc_coord) ** 2)
    K_ab = jnp.exp(-mu * AB2)
    return -nuc_charge * da * db * (2.0 * jnp.pi / p) * K_ab * boys_function_F0(p * PC2)


def eri_primitive(pa: Primitive, pb: Primitive,
                  pc: Primitive, pd: Primitive) -> float:
    """原始双电子积分 (ab|cd) = ⟨g_a g_b | 1/r₁₂ | g_c g_d⟩
    
    (ab|cd) = 2π^{5/2} / (pq√(p+q)) · K_ab · K_cd · F_0(ρ|PQ|²)
    
    这是 rank-4 张量 — monoidal category 中四条线的节点
    Coulomb 和 Exchange 的区别就是指标的不同收缩方式
    
    String diagram:
      a ─┐     ┌─ c
          ├─(ab|cd)─┤
      b ─┘     └─ d
    """
    alpha, A, da = pa
    beta, B, db = pb
    gamma, C, dc = pc
    delta, D, dd = pd

    p = alpha + beta
    q = gamma + delta
    mu_ab = alpha * beta / p
    mu_cd = gamma * delta / q
    rho = p * q / (p + q)

    AB2 = jnp.sum((A - B) ** 2)
    CD2 = jnp.sum((C - D) ** 2)

    P = gaussian_product_center(alpha, A, beta, B)
    Q = gaussian_product_center(gamma, C, delta, D)
    PQ2 = jnp.sum((P - Q) ** 2)

    K_ab = jnp.exp(-mu_ab * AB2)
    K_cd = jnp.exp(-mu_cd * CD2)

    prefactor = 2.0 * jnp.pi ** 2.5 / (p * q * jnp.sqrt(p + q))
    return da * db * dc * dd * prefactor * K_ab * K_cd * boys_function_F0(rho * PQ2)


# --- 收缩积分（从 Primitive → BasisFunction）---
# 这一步是 Foldable：对原始积分求和（fold over primitives）

def contracted_integral(fn, bf_a: BasisFunction, bf_b: BasisFunction, *args):
    """收缩积分 = Σ_{p∈a} Σ_{q∈b} integral(p, q, ...)
    
    从 FP 角度：这是 double fold（catamorphism）over 两个基函数的原始列表
    fold : (acc -> Primitive -> acc) -> acc -> [Primitive] -> acc
    
    收缩 = 自由幺半群上的代数求值（evaluate the free algebra）
    """
    total = 0.0
    for pa in bf_a.primitives:
        for pb in bf_b.primitives:
            total += fn(pa, pb, *args)
    return total


def contracted_eri(bf_a, bf_b, bf_c, bf_d):
    """四中心收缩积分 — 四重 fold"""
    total = 0.0
    for pa in bf_a.primitives:
        for pb in bf_b.primitives:
            for pc in bf_c.primitives:
                for pd in bf_d.primitives:
                    total += eri_primitive(pa, pb, pc, pd)
    return total


# --- 构建完整积分矩阵 ---
# Applicative：各矩阵元互不依赖，可并行计算

def build_integrals(mol: Molecule):
    """构建所有单电子和双电子积分
    
    ┌─────────────────────────────────────────────────┐
    │  这整个函数是 Applicative 的：                    │
    │  S[μ,ν] 的计算不依赖 T[μ,ν]                     │
    │  (μν|λσ) 的计算不依赖 S 或 T                     │
    │  每个矩阵元不依赖其他矩阵元                       │
    │  => 理论上可以完全并行（vmap over index pairs）   │
    └─────────────────────────────────────────────────┘
    
    返回: (S, T, V, ERI) — 所有积分，纯数据，无状态
    """
    basis = mol.basis
    n = len(basis)

    # --- Overlap matrix S (rank-2 tensor = morphism in FdVect) ---
    S = jnp.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s_ij = contracted_integral(overlap_primitive, basis[i], basis[j])
            S = S.at[i, j].set(s_ij)
            S = S.at[j, i].set(s_ij)

    # --- Kinetic energy matrix T ---
    T = jnp.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            t_ij = contracted_integral(kinetic_primitive, basis[i], basis[j])
            T = T.at[i, j].set(t_ij)
            T = T.at[j, i].set(t_ij)

    # --- Nuclear attraction matrix V ---
    # V = Σ_A V_A : fold over nuclei (Foldable over atoms)
    V = jnp.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v_ij = 0.0
            for atom_idx in range(len(mol.charges)):
                v_ij += contracted_integral(
                    nuclear_primitive, basis[i], basis[j],
                    mol.coords[atom_idx], mol.charges[atom_idx]
                )
            V = V.at[i, j].set(v_ij)
            V = V.at[j, i].set(v_ij)

    # Core Hamiltonian: H = T + V (pointwise addition = Applicative lift)
    H_core = T + V

    # --- Two-electron integrals (rank-4 tensor = string diagram node) ---
    # (μν|λσ): 四个自由指标 = compact closed category 中四条线
    ERI = jnp.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if i >= j and k >= l and i * (i+1)//2 + j >= k * (k+1)//2 + l:
                        val = contracted_eri(basis[i], basis[j], basis[k], basis[l])
                        # 八重对称性 — 这来自积分算子的自伴性（adjointness）
                        for ii, jj, kk, ll in [
                            (i,j,k,l), (j,i,k,l), (i,j,l,k), (j,i,l,k),
                            (k,l,i,j), (l,k,i,j), (k,l,j,i), (l,k,j,i)
                        ]:
                            ERI = ERI.at[ii,jj,kk,ll].set(val)

    return S, H_core, ERI


# ================================================================
# Part 4: SCF Iteration — State Monad
# ================================================================
#
# SCF 是量子化学中最核心的 monadic 计算：
#   D_n → F(D_n) → C_n → D_{n+1}
#   每一步的输入依赖前一步的输出 → 顺序依赖 → Monad
#
# 但每一步 *内部* 的操作是 Applicative 的：
#   J 和 K 的 einsum 互相独立
#   Fock 矩阵的构建是纯函数
#   对角化是纯函数

@jit
def build_fock(D, H_core, ERI):
    """构建 Fock 矩阵 F = H + G(D)
    
    ┌──────────────────────────────────────────────────┐
    │  Applicative 内部结构：                            │
    │                                                   │
    │  J[μν] = Σ_{λσ} D[λσ] · (μν|λσ)   — einsum 收缩 │
    │  K[μν] = Σ_{λσ} D[λσ] · (μλ|νσ)   — 交叉收缩    │
    │                                                   │
    │  J 和 K 的计算互不依赖 → Applicative               │
    │  Coulomb vs Exchange = 不同的 string diagram 连线   │
    └──────────────────────────────────────────────────┘
    """
    # Coulomb: 收缩 D 的 (λ,σ) 与 ERI 的后两个指标
    J = jnp.einsum('ls,mnls->mn', D, ERI)

    # Exchange: 交叉收缩 — braided monoidal structure
    K = jnp.einsum('ls,mlns->mn', D, ERI)

    # Fock matrix: 纯 pointwise 操作 (Applicative lift)
    F = H_core + 2.0 * J - K

    return F


@partial(jit, static_argnums=(1,))
def compute_density(C, n_occ):
    """密度矩阵 D = C_occ @ C_occ^T
    
    Functor (协变方向): MO coefficients → AO density
    D[μν] = Σ_i C[μi] C[νi]
    
    einsum 视角: 收缩占据轨道指标 i
    """
    C_occ = C[:, :n_occ]
    return jnp.einsum('mi,ni->mn', C_occ, C_occ)


@jit
def compute_energy(D, H_core, F):
    """电子能量 E = Tr[D(H + F)] = Σ_{μν} D_{μν}(H + F)_{μν}
    
    Foldable / Catamorphism:
    整个电子结构"坍缩"成一个标量
    R^{N×N} → R — 从矩阵空间到标量域的态射
    """
    return jnp.einsum('mn,mn->', D, H_core + F)


def nuclear_repulsion(mol: Molecule) -> float:
    """核排斥能 E_nuc = Σ_{A<B} Z_A Z_B / |R_A - R_B|
    
    Foldable: fold over 所有原子对
    这是纯经典的 — 没有量子效应
    """
    E_nuc = 0.0
    n_atoms = len(mol.charges)
    for A in range(n_atoms):
        for B in range(A + 1, n_atoms):
            R_AB = jnp.linalg.norm(mol.coords[A] - mol.coords[B])
            E_nuc += mol.charges[A] * mol.charges[B] / R_AB
    return E_nuc


def symmetric_orthogonalization(S):
    """对称正交化 X = S^{-1/2}
    
    Functor: 把非正交基变换到正交基
    这是 AO 空间的度量结构（内积）的"校正"
    """
    eigvals, eigvecs = jnp.linalg.eigh(S)
    return eigvecs @ jnp.diag(eigvals ** -0.5) @ eigvecs.T


def run_scf(mol: Molecule, max_iter=100, conv_threshold=1e-10, verbose=True):
    """Restricted Hartree-Fock SCF 过程
    
    ┌──────────────────────────────────────────────────────────────┐
    │  这是整个框架的 Monadic 核心：                                │
    │                                                              │
    │  State Monad:                                                │
    │    state = (D, E, converged)                                 │
    │    bind  = D_n → F(D_n) → eigh(F) → C → D_{n+1}            │
    │                                                              │
    │  D_{n+1} 依赖 D_n => 不能并行 => 必须 scan                   │
    │                                                              │
    │  但每一步内部是 Applicative 的：                               │
    │    build_fock(D) 内的 J,K 计算互不依赖                        │
    │    compute_energy(D, H, F) 是纯函数                           │
    │    eigh(F') 是纯函数                                          │
    │                                                              │
    │  Monadic 边界清晰：只有 D → F → C → D' 这条链是顺序的         │
    └──────────────────────────────────────────────────────────────┘
    """
    if verbose:
        print(f"  Building integrals (Applicative: all independent)...")

    # --- 积分计算：纯 Applicative，所有积分互不依赖 ---
    S, H_core, ERI = build_integrals(mol)

    n_occ = mol.n_electrons // 2  # 占据轨道数（RHF）
    n_basis = len(mol.basis)

    # --- 正交化矩阵 X = S^{-1/2} (Functor: 基变换) ---
    X = symmetric_orthogonalization(S)

    # --- 初始猜测：core Hamiltonian ---
    F = H_core.copy()
    F_prime = X.T @ F @ X              # Functor: AO → 正交 AO
    eps, C_prime = jnp.linalg.eigh(F_prime)
    C = X @ C_prime                     # Functor: 正交 AO → AO
    D = compute_density(C, n_occ)

    E_nuc = nuclear_repulsion(mol)

    if verbose:
        print(f"  Nuclear repulsion energy: {E_nuc:.10f} Hartree")
        print(f"  Starting SCF iterations (State Monad: sequential dependency)")
        print(f"  {'Iter':>4s}  {'E_total':>18s}  {'ΔE':>14s}  {'Status'}")
        print(f"  {'─'*4}  {'─'*18}  {'─'*14}  {'─'*10}")

    E_old = 0.0

    # --- SCF Loop: State Monad 的 runState ---
    # 这就是 jax.lax.scan 的展开形式
    # scan : (State -> (State, Output)) -> State -> [Input] -> (State, [Output])
    for iteration in range(1, max_iter + 1):
        # ---- 单步 SCF: monadic bind (>>=) ----
        # D_n → F(D_n)
        F = build_fock(D, H_core, ERI)

        # 能量求值 (Foldable: catamorphism from matrix to scalar)
        E_elec = compute_energy(D, H_core, F)
        E_total = E_elec + E_nuc
        delta_E = E_total - E_old

        if verbose and (iteration <= 5 or iteration % 10 == 0 or abs(delta_E) < conv_threshold):
            status = "✓ converged" if abs(delta_E) < conv_threshold else ""
            print(f"  {iteration:4d}  {E_total:18.12f}  {delta_E:14.2e}  {status}")

        if abs(delta_E) < conv_threshold and iteration > 1:
            break

        E_old = E_total

        # F → F' (Functor: 基变换 AO → 正交 AO)
        F_prime = X.T @ F @ X

        # F' → (ε, C') (对角化 = 分解为本征态射)
        eps, C_prime = jnp.linalg.eigh(F_prime)

        # C' → C (Functor: 基变换 正交 AO → AO)
        C = X @ C_prime

        # C → D_{n+1} (密度矩阵更新 — 完成 monadic bind)
        D = compute_density(C, n_occ)

    if verbose:
        print()

    return {
        'energy': float(E_total),
        'E_elec': float(E_elec),
        'E_nuc': float(E_nuc),
        'orbital_energies': eps,
        'coefficients': C,
        'density': D,
        'fock': F,
        'iterations': iteration,
        'S': S, 'H_core': H_core, 'ERI': ERI,
    }


# ================================================================
# Part 5: Molecular Builders — Pure Functions
# ================================================================

def build_molecule(atoms: list, coords: list, charge=0) -> Molecule:
    """构建分子
    
    Pure function: (原子列表, 坐标列表) → Molecule
    """
    element_Z = {'H': 1, 'He': 2}
    coords_array = jnp.array(coords, dtype=jnp.float64)
    charges = jnp.array([element_Z[a] for a in atoms], dtype=jnp.float64)
    n_electrons = int(sum(charges)) - charge

    basis = []
    for atom, coord in zip(atoms, coords):
        basis.extend(build_basis(atom, coord))

    return Molecule(
        coords=coords_array,
        charges=charges,
        n_electrons=n_electrons,
        basis=basis,
    )


# ================================================================
# Part 6: Analysis Tools — Foldable / Catamorphism
# ================================================================

def mulliken_population(result, mol):
    """Mulliken 布居分析
    
    Catamorphism: 密度矩阵 → 原子电荷
    从 R^{N×N} 坍缩到 R^{n_atoms}
    每个原子的电荷 = fold over 该原子的基函数贡献
    """
    D, S = result['density'], result['S']
    PS = D @ S  # 布居矩阵
    n_basis = len(mol.basis)

    # 对角元素 = 各基函数的布居数
    populations = 2.0 * jnp.diag(PS)  # factor 2 for RHF

    print("  Mulliken Population Analysis (Foldable: D⊗S → atomic charges)")
    basis_idx = 0
    for i, (atom, coord) in enumerate(zip(['H', 'He', 'H'], mol.coords)):
        if basis_idx < n_basis:
            print(f"    Atom {i}: pop = {populations[basis_idx]:.6f}")
            basis_idx += 1


def orbital_analysis(result):
    """轨道能级分析
    
    Functor: 对角化后的 Fock 矩阵 → 轨道能级列表
    fmap (extract energy) eigendecomposition
    """
    eps = result['orbital_energies']
    n_basis = len(eps)

    print("  Orbital Energies (eigh: Fock → eigenvalues, a Functor decomposition)")
    for i, e in enumerate(eps):
        label = "occupied" if i < n_basis // 2 + 1 else "virtual"
        print(f"    ε_{i+1} = {e:12.6f} Hartree  ({label})")


# ================================================================
# Part 7: PES Scan — Applicative (vmap over geometries)
# ================================================================

def pes_scan(atom1, atom2, distances, charge=0, verbose=False):
    """势能面扫描
    
    ┌────────────────────────────────────────────────────────┐
    │  Applicative: 每个几何构型的 SCF 完全独立               │
    │                                                        │
    │  vmap(run_scf)(geometries)                              │
    │  = [run_scf(g1), run_scf(g2), ..., run_scf(gN)]       │
    │                                                        │
    │  各点之间没有数据依赖 → 可以完全并行                     │
    │  这和 AIMD 形成对比 — AIMD 是 Monad（轨迹依赖前一步）   │
    └────────────────────────────────────────────────────────┘
    
    注意：虽然每个点内部的 SCF 是 Monad（迭代依赖），
    但点与点之间是 Applicative（独立）。
    这就是 FP 层级在实际计算中的具体体现。
    """
    energies = []

    for d in distances:
        mol = build_molecule(
            [atom1, atom2],
            [[0.0, 0.0, 0.0], [0.0, 0.0, d]],
            charge=charge,
        )
        result = run_scf(mol, verbose=verbose)
        energies.append(result['energy'])

    return jnp.array(energies)


# ================================================================
# Part 8: Demo — 把所有 FP 概念串起来
# ================================================================

def demo_h2():
    """H₂ 分子 — 最简单的量子化学计算"""
    print("=" * 70)
    print(" H₂ Molecule — RHF/STO-3G")
    print(" FP structure: Integrals(Applicative) → SCF(Monad) → Energy(Fold)")
    print("=" * 70)

    # H₂ at near-equilibrium bond length (1.4 bohr ≈ 0.74 Å)
    R = 1.4  # bohr
    mol = build_molecule(['H', 'H'], [[0,0,0], [0,0,R]])

    print(f"\n  Geometry: H-H distance = {R:.4f} bohr")
    print(f"  Basis: STO-3G ({len(mol.basis)} basis functions)")
    print(f"  Electrons: {mol.n_electrons}\n")

    result = run_scf(mol)

    print(f"  ─── Results ───")
    print(f"  Total energy:      {result['energy']:18.12f} Hartree")
    print(f"  Electronic energy: {result['E_elec']:18.12f} Hartree")
    print(f"  Nuclear repulsion: {result['E_nuc']:18.12f} Hartree")
    print(f"  SCF converged in {result['iterations']} iterations")
    print(f"  Reference (NIST):  -1.116705644 Hartree (HF/STO-3G)")

    print(f"\n  Orbital energies:")
    for i, e in enumerate(result['orbital_energies']):
        occ = "occ" if i < mol.n_electrons // 2 else "vir"
        print(f"    ε_{i+1} = {e:12.8f} Hartree  [{occ}]")

    return result


def demo_heh_plus():
    """HeH⁺ 分子 — 异核双原子"""
    print("\n" + "=" * 70)
    print(" HeH⁺ Molecule — RHF/STO-3G")
    print(" Demonstrates Profunctor: different atoms = asymmetric morphism")
    print("=" * 70)

    R = 1.4632  # bohr (equilibrium)
    mol = build_molecule(['He', 'H'], [[0,0,0], [0,0,R]], charge=1)

    print(f"\n  Geometry: He-H distance = {R:.4f} bohr")
    print(f"  Charge: +1, Electrons: {mol.n_electrons}\n")

    result = run_scf(mol)

    print(f"  ─── Results ───")
    print(f"  Total energy:      {result['energy']:18.12f} Hartree")
    print(f"  SCF converged in {result['iterations']} iterations")
    print(f"  Reference:         -2.860661929 Hartree (HF/STO-3G)")

    return result


def demo_pes_scan():
    """H₂ 势能面扫描 — Applicative over geometries"""
    print("\n" + "=" * 70)
    print(" H₂ Potential Energy Surface — Applicative Parallelism")
    print(" Each geometry is independent: vmap(SCF)(geometries)")
    print("=" * 70)

    distances = jnp.linspace(0.8, 6.0, 20)

    print(f"\n  Scanning {len(distances)} geometries (Applicative: all independent)...")
    t0 = time.time()
    energies = pes_scan('H', 'H', distances, verbose=False)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s\n")

    # 找平衡键长
    min_idx = jnp.argmin(energies)
    R_eq = distances[min_idx]
    E_eq = energies[min_idx]

    print(f"  Equilibrium bond length: {R_eq:.4f} bohr ({R_eq * 0.529177:.4f} Å)")
    print(f"  Minimum energy:          {E_eq:.10f} Hartree")

    # 解离极限
    E_dissoc = energies[-1]
    D_e = (E_dissoc - E_eq) * 627.509  # Hartree to kcal/mol
    print(f"  Dissociation energy:     {D_e:.2f} kcal/mol")
    print(f"  (at R = {distances[-1]:.1f} bohr, E = {E_dissoc:.10f})")

    print(f"\n  PES Data:")
    print(f"  {'R (bohr)':>10s}  {'E (Hartree)':>16s}")
    print(f"  {'─'*10}  {'─'*16}")
    for R, E in zip(distances, energies):
        marker = " ← min" if R == R_eq else ""
        print(f"  {R:10.4f}  {E:16.10f}{marker}")

    return distances, energies


def demo_fp_structure_summary():
    """总结所有 FP 概念在这个框架中的体现"""
    print("\n" + "=" * 70)
    print(" FP Abstraction Summary for this HF Implementation")
    print("=" * 70)
    print("""
  ┌─────────────────┬────────────────────────┬──────────────────────────┐
  │ FP Concept       │ QC Implementation      │ Key Property             │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Pure Function    │ All integral formulas  │ No side effects          │
  │                  │ Energy expressions     │ Referential transparency │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Functor          │ Basis transformation   │ Structure-preserving     │
  │                  │ C^T @ O @ C (AO→MO)   │ Functor laws hold        │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Applicative      │ Integral computation   │ All elements independent │
  │                  │ J,K matrix build       │ vmap-able / parallelizable│
  │                  │ PES scan points        │ No cross-dependency      │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Monad (State)    │ SCF iteration          │ D→F(D)→C→D' sequential  │
  │                  │ D_{n+1} depends on D_n │ Cannot parallelize steps │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Foldable         │ Energy = Tr(D(H+F))   │ Matrix → scalar collapse │
  │                  │ Nuclear repulsion sum  │ Catamorphism             │
  │                  │ Contraction over prims │ Double fold              │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Monoidal/Einsum  │ J = einsum('ls,mnls')  │ Tensor contraction       │
  │                  │ K = einsum('ls,mlns')  │ String diagram wiring    │
  │                  │ D = einsum('mi,ni')    │ Compact closed category  │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Profunctor       │ AO→MO (covariant)     │ dimap(L,R)(W) = R@W@L   │
  │                  │ MO→AO (contravariant) │ Forward + backward dirs  │
  ├─────────────────┼────────────────────────┼──────────────────────────┤
  │ Adjunction       │ JVP ⊣ VJP             │ ⟨Ax,y⟩ = ⟨x,A^Ty⟩       │
  │                  │ Forces via VJP         │ Wigner 2n+1 rule         │
  └─────────────────┴────────────────────────┴──────────────────────────┘

  Parallelism implications:
  
    Applicative (parallel)          Monad (sequential)
    ─────────────────────           ──────────────────
    Integral computation            SCF iterations
    PES scan points                 Geometry optimization  
    J and K matrices                AIMD trajectory
    Orbital energies                CC amplitude iterations
    Mulliken populations            
    
  The Applicative/Monad boundary is exactly where 
  parallelism meets sequential dependency.
""")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Functional Quantum Chemistry: Hartree-Fock in JAX             ║")
    print("║  A framework where FP abstractions meet electronic structure   ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # Demo 1: H₂
    result_h2 = demo_h2()

    # Demo 2: HeH⁺
    result_heh = demo_heh_plus()

    # Demo 3: PES scan
    distances, energies = demo_pes_scan()

    # Summary
    demo_fp_structure_summary()
