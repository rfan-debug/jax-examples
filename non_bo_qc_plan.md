# Quantum Chemistry Beyond the Born-Oppenheimer Approximation

## A Mathematical Framework and Implementation Plan for JAX-QC

---

## Table of Contents

1. [Motivation and Overview](#1-motivation-and-overview)
2. [The Full Molecular Problem](#2-the-full-molecular-problem)
3. [The Born-Oppenheimer Approximation as a Prior](#3-the-born-oppenheimer-approximation-as-a-prior)
4. [Exact Factorization (EF)](#4-exact-factorization-ef)
5. [Explicitly Correlated Gaussians (ECG)](#5-explicitly-correlated-gaussians-ecg)
6. [Path Integral Molecular Dynamics (PIMD)](#6-path-integral-molecular-dynamics-pimd)
7. [Nuclear Quantum Tunneling](#7-nuclear-quantum-tunneling)
8. [Survey of Existing Non-BO Codes](#8-survey-of-existing-non-bo-codes)
9. [Implementation Roadmap for JAX-QC](#9-implementation-roadmap-for-jax-qc)
10. [FP Abstraction Map](#10-fp-abstraction-map)

---

## 1. Motivation and Overview

Standard quantum chemistry (Steps 1-8 of jax_qc) operates entirely within
the Born-Oppenheimer (BO) approximation: electrons are solved for fixed
nuclear positions, producing a potential energy surface (PES) on which
nuclei move classically. This framework fails when:

- **Non-adiabatic effects** dominate (conical intersections, photochemistry,
  proton-coupled electron transfer).
- **Nuclear quantum effects** matter (zero-point energy, tunneling,
  delocalization of light nuclei like H/D/T).
- **Electron-nuclear correlation** is strong (e.g., positronic systems,
  muonic atoms, exotic matter).
- **Isotope effects** require explicit treatment of nuclear mass in the
  wavefunction.

This document develops three complementary approaches to go beyond BO,
each representing a different "prior" on the structure of the molecular
wavefunction:

| Approach | Prior on Psi(r,R) | Strengths | Limitations |
|----------|-------------------|-----------|-------------|
| **Exact Factorization** | Psi = phi_R(r) * chi(R) (exact) | Rigorous, single-surface picture | TDSE coupling, numerical cost |
| **Explicitly Correlated Gaussians** | Psi = Sum_k c_k G_k(all particles) | Highest accuracy, no BO at all | Few-body only (<=5 particles) |
| **Path Integral MD** | rho ~ integral Dq exp(-S[q]/hbar) | Nuclear quantum effects, scales | Classical nuclei + quantum corrections |

---

## 2. The Full Molecular Problem

### 2.1 The Non-Relativistic Molecular Hamiltonian

For a system of N_e electrons (mass m_e = 1 in a.u.) and N_n nuclei
(masses M_A, charges Z_A), the full Hamiltonian in atomic units is:

```
H = T_e + T_n + V_ee + V_nn + V_en
```

where:

```
T_e = -(1/2) Sum_i nabla_i^2            (electronic kinetic energy)

T_n = -Sum_A (1 / 2*M_A) nabla_A^2      (nuclear kinetic energy)

V_ee = Sum_{i<j} 1 / |r_i - r_j|        (electron-electron repulsion)

V_nn = Sum_{A<B} Z_A*Z_B / |R_A - R_B|  (nuclear-nuclear repulsion)

V_en = -Sum_{i,A} Z_A / |r_i - R_A|     (electron-nuclear attraction)
```

The full time-independent Schrodinger equation is:

```
H Psi(r, R) = E Psi(r, R)
```

where r = (r_1, ..., r_{N_e}) and R = (R_1, ..., R_{N_n}) denote the
collective electronic and nuclear coordinates, respectively.

### 2.2 Coordinate Systems

**Laboratory frame**: The Hamiltonian above is translationally invariant.
Separating the center-of-mass motion R_CM:

```
R_CM = (Sum_A M_A R_A + Sum_i m_e r_i) / (Sum_A M_A + N_e m_e)
```

yields an internal Hamiltonian H_int with 3(N_e + N_n) - 3 degrees of
freedom and a mass-polarization term:

```
H_int = H - (1 / 2*M_tot) (Sum_i nabla_i + Sum_A nabla_A)^2
```

**Body-fixed frame** (Jacobi or perimetric coordinates): For few-body
systems (ECG), one typically uses relative coordinates
rho_k = x_k - x_{k+1} after removing the center of mass, yielding the
internal Hamiltonian:

```
H_int = -(1/2) Sum_{k,l} (Lambda^{-1})_{kl} nabla_{rho_k} . nabla_{rho_l} + V({rho})
```

where Lambda is the mass matrix whose elements depend on the particle
masses and the chosen Jacobi tree.

### 2.3 Symmetry Requirements

- **Electrons**: Fermions. Psi must be antisymmetric under exchange of any
  two electronic coordinates (including spin). Enforced by Slater
  determinants or explicit antisymmetrization operators.
- **Identical nuclei**: If two nuclei are identical (same Z and M), the
  total wavefunction must satisfy bosonic (integer spin) or fermionic
  (half-integer spin) exchange symmetry for those nuclei. Example: the
  two protons in H2 are fermions (spin-1/2), so the spatial nuclear
  wavefunction must be combined with nuclear spin to give overall
  antisymmetry.

---

## 3. The Born-Oppenheimer Approximation as a Prior

### 3.1 The Standard BO Factorization

The BO approximation assumes the total wavefunction factorizes as:

```
Psi(r, R) ~ psi_R(r) * chi(R)
```

where:
- psi_R(r) is the electronic wavefunction, solved for each fixed R via
  H_elec(R) psi_R = E_elec(R) psi_R.
- chi(R) is the nuclear wavefunction, moving on the PES
  E_elec(R) + V_nn(R).

The key assumption is that T_n acting on psi_R is negligible:

```
nabla_A^2 [psi_R(r) chi(R)]
= psi_R nabla_A^2 chi                           (kept)
+ 2 (nabla_A psi_R) . (nabla_A chi)             (1st-order NACT, dropped)
+ (nabla_A^2 psi_R) chi                         (2nd-order NACT, dropped)
```

The last two terms are the **non-adiabatic coupling terms** (NACTs) that
the BO approximation discards.

### 3.2 Bayesian Interpretation: Factorization as a Prior

The BO approximation can be understood as a specific **prior** on the
joint probability amplitude Psi(r, R):

**Joint distribution view**: Define the probability density
|Psi(r, R)|^2 = P(r, R). Any joint distribution admits the exact
factorization:

```
P(r, R) = P(r | R) * P(R)
```

The BO approximation is the **mean-field prior** that assumes:

1. The conditional distribution P(r | R) is the ground-state electronic
   density at fixed R.
2. The marginal P(R) is determined by a nuclear Schrodinger equation on
   the adiabatic PES.
3. The coupling between the two (NACTs) is zero -- i.e., knowledge of R
   fully determines the electronic distribution without feedback.

**This is a conditional independence assumption**: the nuclear dynamics
are independent of the details of the electronic response beyond the
energy E_elec(R).

### 3.3 Alternative Priors and Their Physical Content

Different choices of prior on Psi(r, R) yield different approximation
schemes:

| Prior / Factorization | Mathematical Form | Physical Content |
|-----------------------|-------------------|------------------|
| **BO (adiabatic)** | Psi ~ psi_R^(0) chi | Electrons instantly follow nuclei; single PES |
| **Multi-state BO** | Psi ~ Sum_k psi_R^(k) chi_k | Multiple PES coupled by NACTs; surface hopping |
| **Exact Factorization** | Psi = phi_R * chi (exact) | No approximation in factorization; gauge freedom |
| **Crude BO** | Psi ~ psi_{R0} chi | Electronic wf frozen at reference geometry |
| **No factorization (ECG)** | Psi = Sum_k c_k G_k(r, R) | Flat prior; all correlations captured by basis |
| **Path integral** | rho ~ integral Dq exp(-S/hbar) | Sum over all paths; nuclei as quantum ring polymers |

**Key insight**: Moving from BO to non-BO is equivalent to relaxing the
prior from a restrictive factorization (conditional independence) to a
more flexible representation that captures electron-nuclear correlation.

### 3.4 The Non-Adiabatic Coupling as Prior Misspecification

When the BO prior is wrong (near conical intersections, for light nuclei),
the neglected NACTs quantify the **prior misspecification**:

**First-order NACT (derivative coupling)**:
```
d_kl^A(R) = < psi_k(R) | nabla_A | psi_l(R) >_r
```

**Second-order NACT (scalar coupling)**:
```
G_kl^A(R) = < psi_k(R) | nabla_A^2 | psi_l(R) >_r
```

These are precisely the terms that the BO prior sets to zero. The exact
nuclear equation with NACTs is:

```
[ -Sum_A (1/2M_A) (nabla_A + d^A)^2 + E_elec ] chi = E chi
```

where d^A and E_elec are matrices in the electronic state index. The
BO limit is the diagonal, single-state approximation: chi_k = 0 for
k != 0.

### 3.5 The Variational Perspective

All the priors above correspond to different **variational ansatze** for
the total wavefunction:

```
E_0 <= <Psi_trial | H | Psi_trial> / <Psi_trial | Psi_trial>
```

- BO: The trial wavefunction is constrained to the product form psi_R chi.
  The variational space is restricted, so E_BO >= E_exact.
- ECG: The trial wavefunction is a linear combination of correlated
  Gaussians with no factorization constraint. As the basis grows, it
  converges to E_exact.
- The "quality of the prior" is measured by how far E_trial is from E_exact.

---

## 4. Exact Factorization (EF)

### 4.1 Mathematical Foundation

The exact factorization of Abedi, Maitra, and Gross (2010) writes:

```
Psi(r, R, t) = phi_R(r, t) * chi(R, t)
```

subject to the **partial normalization condition**:

```
integral dr |phi_R(r, t)|^2 = 1    for all R, t
```

This is *not* an approximation -- it is an exact rewriting. The uniqueness
is guaranteed (up to a gauge transformation) by the partial normalization.

### 4.2 Exact Equations of Motion

Substituting into the TDSE i d/dt Psi = H Psi and projecting yields two
coupled equations:

**Electronic equation** (conditional):
```
[ H_BO(R) + U_en[chi, phi_R] ] phi_R = epsilon(R, t) phi_R
```

where epsilon(R, t) is the time-dependent potential energy surface (TDPES)
and U_en is the electron-nuclear coupling operator:

```
U_en = Sum_A (1 / 2M_A) [
    ((-i nabla_A chi) / chi) . (-i nabla_A)
    + ((-i nabla_A)^2 chi) / chi
]
```

**Nuclear equation** (marginal):
```
i d/dt chi(R, t) = [ Sum_A (-i nabla_A + A_A)^2 / (2 M_A) + epsilon(R, t) ] chi(R, t)
```

where the **Berry connection** (vector potential) and **TDPES** (scalar
potential) are:

```
A_A(R, t) = < phi_R | -i nabla_A | phi_R >_r

epsilon(R, t) = < phi_R | H_BO + U_en - i d/dt | phi_R >_r
```

### 4.3 Gauge Freedom and Gauge-Fixing

The factorization has a U(1) gauge freedom:
phi_R -> exp(i theta(R,t)) phi_R and chi -> exp(-i theta(R,t)) chi
leave Psi invariant.

Common gauge choices:
- **Real gauge**: Im(chi) = 0 (useful for time-independent problems).
- **Coulomb gauge**: nabla_A . A_A = 0.
- **Natural gauge**: A_A = 0 (absorbs the Berry phase into phi).

### 4.4 Time-Independent Limit

For stationary states, Psi(r, R) = phi_R(r) chi(R), and the equations
become:

```
[ H_BO(R) + U_en^{stat} ] phi_R = epsilon(R) phi_R

[ Sum_A (-i nabla_A + A_A)^2 / (2M_A) + epsilon(R) ] chi = E chi
```

where E is the exact total energy and epsilon(R) is now the exact static
PES (which includes the diagonal BO correction and the geometric phase).

### 4.5 Connection to BO

In the BO limit (M_A -> infinity), U_en -> 0, A_A -> 0, and
epsilon(R) -> E_elec(R). The exact factorization reduces to the standard
BO picture. The corrections are O(1/M_A).

### 4.6 Computational Approach

Practical EF calculations require:

1. **Representation of phi_R**: Linear combination of electronic basis
   functions (e.g., adiabatic states, Slater determinants) at each R.
2. **Representation of chi(R)**: Grid-based (DVR), Gaussian wavepacket,
   or trajectory-based (conditional trajectory).
3. **Self-consistent iteration**: The two equations are coupled (each
   depends on the other's solution). Iterate to convergence.

**Conditional trajectory approach** (CT-EF): Represent chi as a swarm
of classical trajectories guided by the quantum potential, similar to
Bohmian mechanics:

```
dR_A^(alpha)/dt = (1/M_A) Im (nabla_A chi / chi) |_{R=R^(alpha)}
```

This avoids the exponential scaling of grid-based nuclear propagation.

---

## 5. Explicitly Correlated Gaussians (ECG)

### 5.1 Mathematical Foundation

ECGs bypass the BO approximation entirely by expanding the total
wavefunction in a basis of correlated Gaussian functions that depend
on *all* inter-particle coordinates simultaneously.

For a system of n particles (after center-of-mass removal, n-1
relative coordinates rho in R^{3(n-1)}):

```
Psi = Sum_{k=1}^{K} c_k A[ G_k(rho) (x) Xi_k^{spin} ]
```

where A is the appropriate (anti)symmetrizer for identical particles
and Xi_k^{spin} is a spin function.

### 5.2 ECG Basis Functions

**Spherical ECG** (s-wave, L=0):

```
G_k(rho) = exp( -rho^T (A_k (x) I_3) rho )
```

where A_k is an (n-1) x (n-1) symmetric positive-definite matrix of
nonlinear parameters and I_3 is the 3x3 identity. In component form:

```
G_k = exp( -Sum_{i,j=1}^{n-1} (A_k)_{ij} rho_i . rho_j )
```

**ECG with shifted centers** (for non-zero angular momentum):

```
G_k(rho) = exp( -(rho - s_k)^T (A_k (x) I_3) (rho - s_k) )
```

**ECG with polynomial prefactors** (for L > 0):

```
G_k(rho) = (u_k^T rho)^{l_k} exp( -rho^T (A_k (x) I_3) rho )
```

### 5.3 Matrix Elements

The power of ECGs is that all matrix elements are analytically computable.

**Overlap**:

```
S_{kl} = < G_k | G_l > = ( pi^{n-1} / |A_k + A_l| )^{3/2}
```

**Kinetic energy** (for the internal Hamiltonian with mass matrix Lambda):

```
T_{kl} = < G_k | T | G_l > = 3 Tr[ A_k Lambda^{-1} A_l (A_k + A_l)^{-1} ] S_{kl}
```

**Coulomb interaction** (V = q_i q_j / r_{ij}, where
r_{ij}^2 = rho^T P_{ij} rho for an appropriate projection matrix P_{ij}):

```
V_{kl}^{(ij)} = (2 q_i q_j / sqrt(pi))
    * ( pi^{n-1} / |A_k + A_l| )^{3/2}
    * 1 / sqrt( Tr[ P_{ij} (A_k + A_l)^{-1} ] )
```

### 5.4 Generalized Eigenvalue Problem

The total energy is obtained by solving the generalized eigenvalue problem:

```
H c = E S c
```

where H_{kl} = T_{kl} + V_{kl} and S_{kl} are the Hamiltonian and
overlap matrix elements.

### 5.5 Nonlinear Parameter Optimization

The ECG exponent matrices {A_k} are **nonlinear variational parameters**
that must be optimized. The energy is a function of both the linear
coefficients c and the nonlinear parameters alpha = {A_k^{(ij)}}:

```
E(alpha) = min_c  c^T H(alpha) c / c^T S(alpha) c
```

The gradient with respect to the nonlinear parameters (crucial for JAX
autodiff) is:

```
dE/d alpha_m = [ c^T (dH/d alpha_m) c - E c^T (dS/d alpha_m) c ] / (c^T S c)
```

This is where **JAX autodiff is transformative**: dH_{kl}/dA_k^{(ij)}
and dS_{kl}/dA_k^{(ij)} can be computed automatically from the analytic
matrix element expressions, enabling gradient-based optimization (L-BFGS,
Adam) of thousands of nonlinear parameters.

### 5.6 Stochastic Variational Method (SVM)

The standard ECG optimization uses the **stochastic variational method**:

1. Start with K basis functions with random A_k.
2. For each step:
   a. Propose a new random A_{trial}.
   b. Solve the generalized eigenvalue problem with K+1 functions.
   c. If E_{K+1} < E_K, accept the new function.
3. Periodically refine all A_k via gradient descent.

With JAX, step 3 can use `jax.grad` on the energy w.r.t. all ECG
parameters simultaneously, which is far more efficient than the
traditional one-at-a-time approach.

### 5.7 Antisymmetrization for Fermions

For systems with identical fermions (electrons), the antisymmetrization
operator A generates N_e! permutations:

```
A = (1/N_e!) Sum_{P in S_{N_e}} (-1)^P P_hat
```

For ECGs, permuting particles corresponds to a linear transformation of
the A_k matrix:

```
P_hat G_k(rho) = G_k(P_rho rho)
    = exp( -rho^T (P_rho^T A_k P_rho (x) I_3) rho )
```

where P_rho is the permutation matrix in relative coordinates.
All matrix elements involving permuted Gaussians remain analytically
computable.

### 5.8 State of the Art

Current ECG calculations achieve:
- H2 (4 particles): energy to 10^{-12} Hartree accuracy
- Ps2 (positronium molecule, 4 particles): first prediction of stability
- LiH (6 particles at the edge of tractability): ~10^{-6} Hartree
- Scaling: O(K^3 * N!) per eigensolve, where K is basis size and N!
  is the permutation count

---

## 6. Path Integral Molecular Dynamics (PIMD)

### 6.1 Mathematical Foundation

The path integral formulation replaces the nuclear quantum mechanics with
an equivalent classical problem in an extended phase space. For a system
at thermal equilibrium (temperature T, beta = 1/(k_B T)), the quantum
partition function is:

```
Z = Tr[ exp(-beta H) ]
  = integral dR < R | exp(-beta H) | R >
```

Using the Trotter factorization with P imaginary-time slices
(Delta_tau = beta*hbar / P):

```
exp(-beta H) ~ ( exp(-Delta_tau V/2) exp(-Delta_tau T) exp(-Delta_tau V/2) )^P
```

The partition function becomes:

```
Z = lim_{P->inf} (M P / (2 pi beta hbar^2))^{3NP/2}
    integral Prod_{s=1}^{P} dR^{(s)}
    exp( -beta_P Sum_{s=1}^{P} [
        M P / (2 beta^2 hbar^2) |R^{(s+1)} - R^{(s)}|^2 + V(R^{(s)})
    ] )
```

where beta_P = beta/P and R^{(P+1)} = R^{(1)} (cyclic boundary
conditions -- ring polymer).

### 6.2 Ring Polymer Isomorphism

The key result: the quantum partition function is **exactly** equal to
the classical partition function of a **ring polymer** -- a system of P
replicas (beads) connected by harmonic springs:

```
Z_quantum^{nuclear} = Z_classical^{ring-polymer}
```

Each nucleus A is replaced by P beads {R_A^{(1)}, ..., R_A^{(P)}}
connected by springs with force constant:

```
k_spring = M_A P / (beta^2 hbar^2) = M_A omega_P^2

where omega_P = P / (beta hbar) = P k_B T / hbar
```

### 6.3 PIMD Equations of Motion

Ring Polymer Molecular Dynamics (RPMD) propagates the ring polymer using
classical equations of motion:

```
M_A R_A^{(s)}_ddot = -M_A omega_P^2 (2 R_A^{(s)} - R_A^{(s-1)} - R_A^{(s+1)})
                     - nabla_A V(R^{(s)})
```

where the first term is the intra-bead spring force and the second is
the physical force from the potential (which is the BO PES from jax_qc).

**Thermostatted RPMD** (for canonical ensemble sampling): Apply a
Langevin thermostat to each bead:

```
M_A R_A^{(s)}_ddot = F_A^{(s)}
                     - gamma M_A R_A^{(s)}_dot
                     + sqrt(2 gamma M_A k_B T) eta_A^{(s)}(t)
```

where gamma is the friction coefficient and eta is Gaussian white noise.

### 6.4 Normal Mode Transformation

The ring polymer Hamiltonian is diagonalized by the discrete Fourier
transform (normal mode transformation):

```
R_A_tilde^{(k)} = (1/sqrt(P)) Sum_{s=1}^{P} R_A^{(s)} exp(-2 pi i k s / P)
```

In normal mode coordinates, the spring potential becomes:

```
V_spring = (1/2) M_A Sum_{k=0}^{P-1} omega_k^2 |R_A_tilde^{(k)}|^2

where omega_k = 2 omega_P sin(k pi / P)
```

The k=0 mode is the centroid (center of mass of the ring polymer),
and k > 0 are the internal fluctuation modes.

### 6.5 Estimators for Physical Observables

**Thermodynamic energy** (virial estimator -- lower variance than
primitive):

```
<E> = 3N / (2 beta)
    + (1/P) Sum_{s=1}^{P} [
        V(R^{(s)})
        + (1/2) Sum_A (R_A^{(s)} - R_A_bar) . nabla_A V(R^{(s)})
      ]
```

where R_A_bar = (1/P) Sum_s R_A^{(s)} is the centroid.

**Radius of gyration** (measures nuclear delocalization):

```
R_gyr_A^2 = (1/P) Sum_{s=1}^{P} |R_A^{(s)} - R_A_bar|^2
```

For classical nuclei, R_gyr = 0. For quantum nuclei at temperature T:

```
R_gyr ~ hbar / sqrt(12 M_A k_B T)    (free particle limit)
```

### 6.6 Integration with jax_qc

The connection to the existing jax_qc framework is natural:

1. **Each bead** R^{(s)} defines a molecular geometry.
2. The **energy** V(R^{(s)}) is computed by `jax_qc.energy()`.
3. The **force** -nabla_A V(R^{(s)}) is computed by
   `jax_qc.compute_gradient()`.
4. The **PIMD integrator** adds the spring forces and thermostat.

All P beads can be evaluated **in parallel** (Applicative structure),
making this a natural fit for JAX's `vmap` over beads.

### 6.7 Key Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| P (beads) | 8-64 | More beads = better Trotter convergence; P ~ 1/T |
| Delta_t (time step) | 0.5-1.0 fs | Limited by highest spring frequency omega_{P-1} |
| Thermostat | PILE-G or GLE | PILE (Path Integral Langevin Equation) is optimal for PIMD |
| Temperature | 100-1000 K | Nuclear quantum effects most important at low T |

---

## 7. Nuclear Quantum Tunneling

Tunneling -- the penetration of a quantum particle through a classically
forbidden energy barrier -- is one of the most important nuclear quantum
effects in chemistry. It governs proton transfer rates, enzyme catalysis,
hydrogen diffusion in solids, and tunneling splittings in molecular
spectroscopy. Each of the three non-BO methods treats tunneling
differently, with distinct mathematical foundations and accuracy/cost
tradeoffs.

### 7.1 The Tunneling Problem

Consider a particle of mass M in a one-dimensional double-well potential
V(x) with barrier height V_b at x = 0 and minima at x = +/- x_0:

```
V(x) = V_b (1 - x^2/x_0^2)^2    (symmetric double well)
```

**Classical mechanics**: A particle with energy E < V_b is confined to
one well. The classically forbidden region is {x : V(x) > E}.

**Quantum mechanics**: The wavefunction has non-zero amplitude in the
forbidden region, decaying as:

```
psi(x) ~ exp( -integral_{x_1}^{x_2} kappa(x') dx' )

where kappa(x) = sqrt(2M(V(x) - E)) / hbar
```

The tunneling probability through a barrier of width L and height V_b
scales as:

```
T ~ exp( -2 integral_0^L sqrt(2M(V(x) - E)) dx / hbar )
  ~ exp( -2L sqrt(2M V_b) / hbar )
```

This exponential sensitivity to mass and barrier height is why tunneling
is critical for light nuclei (H, D, T) and irrelevant for heavy atoms.

### 7.2 Tunneling Splittings

For a symmetric double well, the ground state splits into a symmetric
(+) and antisymmetric (-) pair separated by the tunneling splitting:

```
Delta = E_- - E_+ = (hbar omega / pi) exp(-S_inst / hbar)
```

where omega is the harmonic frequency at the well minimum and S_inst is
the **instanton action** (the Euclidean action along the classical path
in imaginary time that connects the two wells):

```
S_inst = integral_{-inf}^{+inf} d tau [ (1/2) M x_dot^2 + V(x) ]
       = integral_{x_-}^{x_+} sqrt(2M V(x)) dx
```

The instanton (also called the "bounce") is the solution to Newton's
equation in the *inverted* potential -V(x):

```
M x_ddot = +dV/dx    (note: plus sign, inverted potential)
```

with boundary conditions x(tau -> -inf) = x_- and x(tau -> +inf) = x_+.

### 7.3 Tunneling in Path Integral Methods

#### 7.3.1 How Ring Polymers Capture Tunneling

In PIMD, tunneling manifests as the ring polymer **stretching across the
barrier**. At sufficiently low temperature (large P), the beads
{R^(1), ..., R^(P)} spread from one well to the other, sampling the
classically forbidden region:

```
Classical:  all beads in one well  -> R_gyr small
Tunneling:  beads span the barrier -> R_gyr ~ barrier width
```

The ring polymer free energy includes the tunneling contribution
automatically through the Boltzmann weight of configurations that
straddle the barrier.

#### 7.3.2 Ring Polymer Instanton (RPI) Theory

The Ring Polymer Instanton is the semiclassical limit of PIMD that gives
tunneling rates analytically. The key idea: find the **saddle point** of
the ring polymer potential energy surface.

**Ring polymer potential**:

```
U_RP({R^(s)}) = Sum_{s=1}^{P} [ (M P / 2 beta^2 hbar^2) |R^{(s+1)} - R^{(s)}|^2 + V(R^{(s)}) ]
```

The instanton is the stationary point of U_RP that is a first-order
saddle (one negative eigenvalue in the Hessian) corresponding to the
ring polymer stretched across the barrier.

**RPI tunneling rate** (thermal, semiclassical):

```
k_RPI = (1 / 2 pi hbar) sqrt( |lambda_1| / (2 pi)^{P-1} )
        * ( Prod_{k=2}^{P} lambda_k )^{-1/2}
        * exp(-beta_P U_RP^{inst})
```

where lambda_1 < 0 is the single negative Hessian eigenvalue and
lambda_k (k >= 2) are the positive eigenvalues of the ring polymer
Hessian at the instanton configuration.

**JAX implementation advantage**: The instanton search requires:
1. Gradient of U_RP: `jax.grad(U_RP)` -- trivial with autodiff
2. Hessian of U_RP: `jax.hessian(U_RP)` -- automatic second derivatives
3. Saddle-point optimization: use eigenvector-following with JAX gradients

This makes the RPI approach far more practical in JAX than in traditional
codes that require hand-coded Hessians.

#### 7.3.3 Limitations of RPMD for Tunneling

Standard RPMD has known deficiencies for deep tunneling:

- **Overestimates barrier recrossing**: The ring polymer can recross the
  dividing surface, leading to rate underestimation.
- **Spurious resonance**: At low T, the ring polymer internal modes can
  resonate with the physical barrier frequency.
- **Correct scaling but wrong prefactor**: RPMD gives the correct
  Arrhenius slope (dominated by exp(-S_inst/hbar)) but the prefactor
  can be off by factors of 2-3 for asymmetric barriers.

**Remedies available in the framework**:
- Thermostatted RPMD (T-RPMD): Applies friction only to internal modes,
  preserving centroid dynamics.
- Ring polymer instanton rate theory: Exact semiclassical rate from the
  saddle-point analysis (Section 7.3.2).

### 7.4 Tunneling in Explicitly Correlated Gaussians

#### 7.4.1 Variational Capture of Tunneling

ECG captures tunneling through the **variational principle**: the basis
set is flexible enough that the optimized wavefunction has non-zero
amplitude in the classically forbidden region.

For a double well, the ECG wavefunction:

```
Psi = Sum_k c_k exp(-alpha_k (x - s_k)^2)
```

can place Gaussian centers s_k on both sides of the barrier and inside
the forbidden region. The linear coefficients c_k automatically produce
the correct symmetric (+) and antisymmetric (-) combinations.

**Tunneling splitting from ECG**: Solve the generalized eigenvalue
problem for the two lowest eigenvalues E_0 and E_1:

```
Delta = E_1 - E_0
```

This is exact within the basis set completeness -- no semiclassical
approximation is needed.

#### 7.4.2 Basis Set Requirements for Tunneling

Accurate tunneling splittings require ECG functions in the barrier region
where |Psi|^2 is exponentially small. This demands:

- **Wide Gaussians** (small alpha_k) that extend into the barrier.
- **Functions centered in the forbidden region** (s_k near the barrier
  top) to capture the evanescent wavefunction.
- **High precision arithmetic**: the tunneling splitting can be 10^{-6}
  times the well depth, requiring many digits of accuracy in the
  eigensolve.

The stochastic variational method (SVM) with JAX gradient refinement is
well-suited: the gradient dE/d{s_k, alpha_k} will drive basis functions
into the barrier region where they most reduce the energy of the
antisymmetric state.

#### 7.4.3 Multi-Dimensional Tunneling with ECG

For molecular systems, tunneling occurs along a **minimum energy path**
(MEP) in the multi-dimensional coordinate space. The ECG basis naturally
handles this because the correlated Gaussian exponent matrices A_k
encode multi-dimensional correlations:

```
G_k(rho) = exp(-rho^T A_k rho)
```

A large off-diagonal element (A_k)_{ij} correlates coordinates i and j,
allowing the Gaussian to be oriented along the MEP rather than aligned
with the Cartesian axes. This is equivalent to using the "tunneling
path" as a natural coordinate, without explicitly constructing it.

### 7.5 Tunneling in Exact Factorization

#### 7.5.1 Tunneling Through the TDPES

In the EF framework, nuclear tunneling appears in the exact
time-dependent potential energy surface (TDPES) epsilon(R, t). During a
tunneling event:

```
Before tunneling:   chi(R) localized in well A
                    epsilon(R) ~ V_BO(R)  (standard BO surface)

During tunneling:   chi(R) developing amplitude in well B
                    epsilon(R) develops a "step" that lowers the
                    barrier, effectively guiding chi through

After tunneling:    chi(R) has amplitude in both wells
                    epsilon(R) shows the tunneling splitting
```

The key insight: the **exact** TDPES dynamically reshapes to facilitate
tunneling. The barrier in epsilon(R) is *lower* than the BO barrier
V_BO(R) because it includes the electron-nuclear correlation energy
that stabilizes the tunneling configuration.

#### 7.5.2 Quantum Potential and Tunneling Force

In the conditional trajectory formulation of EF, the nuclear trajectories
obey:

```
M_A R_A_ddot = -d epsilon/dR_A - (1/2M_A) d/dR_A ( nabla_A^2 |chi| / |chi| )
```

The second term is the **quantum potential** (Bohm potential):

```
Q(R) = -(hbar^2 / 2M) nabla^2 |chi| / |chi|
```

This quantum force is what drives trajectories through the classically
forbidden region. At the edges of the wavepacket (where |chi| is small
and rapidly varying), Q(R) becomes large and repulsive, effectively
"pushing" the trajectory through the barrier.

For a Gaussian wavepacket chi(R) = exp(-alpha(R - R_0)^2 + ikR):

```
Q(R) = (hbar^2 alpha / M) [ 1 - 2 alpha (R - R_0)^2 ]
```

Near the center (R ~ R_0), Q is positive (repulsive), broadening the
packet. At the tails, Q changes sign, and the interplay with the
classical potential V(R) determines whether the trajectory tunnels.

#### 7.5.3 Advantages for Tunneling

The EF approach to tunneling has unique strengths:

- **No semiclassical approximation**: The TDPES is exact (within
  numerical discretization), so all tunneling orders are captured.
- **Trajectory picture**: Individual trajectories either tunnel or
  reflect, giving mechanistic insight.
- **Branching ratio**: For asymmetric barriers, the EF naturally gives
  the transmission/reflection ratio from the fraction of trajectories
  that cross.

The main challenge is numerical: the quantum potential Q(R) diverges
where |chi| has nodes, requiring careful regularization or adaptive
grids.

### 7.6 Comparison of Tunneling Treatments

| Aspect | PIMD/RPMD | RPI (Instanton) | ECG | Exact Factorization |
|--------|-----------|-----------------|-----|---------------------|
| **Theory level** | Exact (thermo) / approximate (dynamics) | Semiclassical | Exact (variational) | Exact |
| **Tunneling splitting** | Indirect (from free energy) | Analytic formula | Direct eigenvalue difference | From TDPES dynamics |
| **Tunneling rate** | From long-time correlation functions | Analytic (saddle-point) | N/A (bound states only) | From transmission coefficient |
| **Deep tunneling** | Underestimates (RPMD) | Accurate | Exact | Exact |
| **Multi-dimensional** | Natural (ring polymer in full space) | Requires MEP | Natural (correlated Gaussians) | Grid scaling limits dimensions |
| **Temperature** | Finite T only | Low T (semiclassical) | T = 0 (ground state) | Any |
| **Computational cost** | O(P * N_SCF) per step | O(P^3) for Hessian eigenvalues | O(K^3 * N!) eigensolve | Grid-based: exponential in dim |
| **JAX advantage** | vmap over beads | jax.hessian for RPI | jax.grad for basis optimization | jax.grad for TDPES |

### 7.7 Benchmark Systems for Tunneling

| System | Tunneling Type | Observable | Method of Choice |
|--------|---------------|------------|-----------------|
| **Eckart barrier** | 1D scattering | Transmission coefficient T(E) | All (analytic reference) |
| **Symmetric double well** | 1D bound state | Tunneling splitting Delta | ECG, RPI |
| **H + H2 -> H2 + H** | Collinear reactive | Thermal rate k(T) | RPMD, RPI |
| **Malonaldehyde** | Intramolecular H-transfer | Tunneling splitting (21 cm^{-1}) | PIMD, RPI |
| **Formic acid dimer** | Double H-transfer (concerted) | Tunneling splitting | RPI |
| **H in Pd** | Diffusion through metal | Diffusion coefficient D(T) | PIMD |
| **Water hexamer prism** | Collective H-bond rearrangement | Tunneling splitting | RPI |

#### 7.7.1 The Eckart Barrier (Analytic Reference)

The Eckart barrier has an exact quantum transmission coefficient, making
it the standard test for tunneling methods:

```
V(x) = V_0 / cosh^2(x / a)
```

Exact transmission coefficient (for energy E):

```
T(E) = [ cosh(2 pi (alpha - beta)) - 1 ] / [ cosh(2 pi (alpha + beta)) - 1 ]

where alpha = (a / hbar) sqrt(2ME)
      beta  = (a / hbar) sqrt(2M(V_0 - E))    if E < V_0
```

The **crossover temperature** separating thermal activation from
tunneling-dominated kinetics is:

```
T_c = hbar omega_b / (2 pi k_B)

where omega_b = sqrt(2 V_0 / (M a^2))  is the barrier frequency
```

Below T_c, the thermal rate is dominated by tunneling. Above T_c,
classical over-barrier transitions dominate.

#### 7.7.2 Implementation Plan for Tunneling Benchmarks

```
tunneling/eckart.py
    - V_eckart(x, V0, a) -> potential
    - T_exact(E, V0, a, M) -> transmission coefficient
    - k_exact(T, V0, a, M) -> exact thermal rate (integral over T(E))

tunneling/double_well.py
    - V_double_well(x, V_b, x_0) -> potential
    - exact_splitting_dvr(V, M, n_grid) -> Delta (DVR reference)
    - instanton_action(V, M) -> S_inst
    - wkb_splitting(omega, S_inst) -> Delta_WKB

tunneling/rpi.py  (Ring Polymer Instanton)
    - ring_polymer_potential(beads, V, M, P, beta)
    - find_instanton(V, M, P, beta) -> saddle point of U_RP
      (uses jax.grad + eigenvector-following optimizer)
    - instanton_rate(V, M, P, beta) -> k_RPI
      (uses jax.hessian for the fluctuation determinant)
    - crossover_temperature(V, M) -> T_c

tests/test_tunneling.py
    - Eckart: RPMD rate vs exact within factor of 2
    - Eckart: RPI rate vs exact within 10% below T_c
    - Double well: ECG splitting vs DVR reference to 1%
    - Double well: instanton splitting vs WKB to 10%
    - Crossover temperature: T_c matches analytic formula
```

---

## 8. Survey of Existing Non-BO Codes

### 7.1 Explicitly Correlated Gaussians

| Code | Authors | Language | Particles | Features |
|------|---------|----------|-----------|----------|
| **LOWDIN** | Adamowicz group (Arizona) | Fortran | up to 6 | Highest accuracy non-BO; H2, LiH, Ps2; analytic gradients for SVM |
| **ECG codes** (Suzuki, Varga) | Suzuki & Varga | Fortran | up to 5 | Nuclear physics origins; SVM; few-body systems |
| **ATOM** | Pachucki group (Warsaw) | Fortran/C | 3-4 | Hylleraas + ECG; QED corrections; H, He, Li atoms |

**Limitations of existing ECG codes:**
- All are Fortran-based, hand-coded derivatives
- No GPU acceleration or automatic differentiation
- Basis optimization is stochastic (slow convergence)
- Hard-coded particle types and permutation symmetries

**jax_qc advantage**: JAX autodiff for gradient-based optimization of
ECG parameters; GPU vmap for parallel matrix element evaluation;
flexible particle types via configuration.

### 7.2 Exact Factorization

| Code | Authors | Language | Features |
|------|---------|----------|----------|
| **EF-CT** | Agostini, Gross, Min | Python/C | Conditional trajectory implementation; 1D models |
| **TDDFT-EF** | Gross group | Modified QE | Time-dependent EF with TDDFT for electrons |
| **SHARC** | Gonzalez group (Vienna) | Fortran | Surface hopping (multi-state BO, not full EF) |

**Limitations:**
- Full EF implementations are restricted to model systems (1D, 2D)
- 3D molecular EF is largely unexplored computationally
- Trajectory-based approaches have issues with wavefunction phase

### 7.3 Path Integral Methods

| Code | Authors | Language | Features |
|------|---------|----------|----------|
| **i-PI** | Ceriotti et al. (EPFL) | Python | Universal PIMD wrapper; interfaces to any force engine; GLE thermostats |
| **CP2K** | CP2K team | Fortran | Built-in PIMD; DFT forces; large-scale condensed phase |
| **LAMMPS** | Sandia | C++ | PIMD fix; empirical potentials; massive parallelism |
| **Plumed** | Bussi et al. | C++ | Plugin for enhanced sampling; can add PI to any MD code |
| **TorchMD-Net** | Various | Python/PyTorch | ML potentials with PIMD; differentiable |

**Limitations of existing PI codes:**
- i-PI requires an external force engine (communication overhead)
- CP2K/LAMMPS use non-differentiable force engines
- No code combines autodiff forces + PIMD natively

**jax_qc advantage**: Native `jax.grad` forces at each bead; `jax.vmap`
over beads for parallel evaluation; JIT compilation of the full
integrator; end-to-end differentiability enables learning/optimization.

---

## 9. Implementation Roadmap for JAX-QC

### Phase 1: PIMD (most practical, builds on existing code)

```
Step 9: PIMD integrator + thermostat (Day 13-14)
-------------------------------------------------
- [ ] pimd/ring_polymer.py
      - RingPolymer dataclass (beads, masses, P, T)
      - normal_mode_transform / inverse
      - spring_forces(ring_polymer) -> (N_atoms, 3, P)
      - centroid(ring_polymer) -> (N_atoms, 3)
      - radius_of_gyration(ring_polymer) -> (N_atoms,)
- [ ] pimd/integrator.py
      - velocity_verlet_step(state, forces, dt) -> state
      - PILE_thermostat(state, gamma, T, dt) -> state
- [ ] pimd/estimators.py
      - primitive_energy_estimator(ring_polymer, energies)
      - virial_energy_estimator(ring_polymer, energies, forces)
      - kinetic_energy_estimator(ring_polymer, T)
- [ ] pimd/interface.py
      - run_pimd(mol, basis, T, P, n_steps, dt)
      - Uses jax.vmap over beads for parallel energy+gradient
- [ ] tests/test_pimd.py
      - Harmonic oscillator: exact ZPE from PIMD matches hbar*omega/2
      - Free particle: R_gyr matches hbar/sqrt(12MkT)
      - H2 with STO-3G: nuclear delocalization at T=300K

Step 10: PIMD analysis + enhanced sampling (Day 15-16)
------------------------------------------------------
- [ ] pimd/analysis.py
      - compute_rdf(trajectory) -> g(r)
      - compute_momentum_distribution(trajectory)
      - isotope_fractionation(trajectory_H, trajectory_D)
- [ ] pimd/centroid_md.py (CMD)
      - Centroid constraint dynamics for approximate quantum dynamics
- [ ] pimd/ring_polymer_contraction.py
      - RPC: use fewer beads for expensive long-range forces
- [ ] examples/13_pimd_h2.py
      - H2 nuclear delocalization at various temperatures

Step 10b: Tunneling module + Ring Polymer Instanton (Day 16-17)
---------------------------------------------------------------
- [ ] tunneling/potentials.py
      - V_eckart(x, V0, a) -> 1D Eckart barrier
      - V_double_well(x, V_b, x_0) -> symmetric double well
      - T_eckart_exact(E, V0, a, M) -> exact transmission coefficient
      - k_eckart_exact(T, V0, a, M) -> exact thermal rate
      - crossover_temperature(omega_b) -> T_c
- [ ] tunneling/instanton.py
      - ring_polymer_potential(beads, V, M, P, beta) -> U_RP
      - ring_polymer_gradient(beads, V, M, P, beta) via jax.grad
      - ring_polymer_hessian(beads, V, M, P, beta) via jax.hessian
      - find_instanton(V, M, P, beta) -> saddle point (eigenvector-following)
      - instanton_rate(V, M, P, beta) -> k_RPI from fluctuation determinant
      - instanton_splitting(V, M, P, beta) -> Delta from instanton action
- [ ] tunneling/wkb.py
      - wkb_transmission(V, E, M, x_range) -> T_WKB
      - instanton_action(V, M, x_min, x_max) -> S_inst
      - wkb_splitting(omega, S_inst) -> Delta_WKB
- [ ] tests/test_tunneling.py
      - Eckart barrier: RPI rate vs exact (within factor 2 above T_c,
        within 10% below T_c)
      - Double well: instanton splitting vs DVR reference (within 10%)
      - Crossover temperature: T_c matches analytic formula
      - WKB: transmission vs exact for thin/thick barriers
- [ ] examples/14_tunneling.py
      - Eckart barrier: rate vs temperature (Arrhenius + tunneling crossover)
      - Double well: tunneling splitting vs barrier height
```

### Phase 2: ECG (highest accuracy, few-body)

```
Step 11: ECG basis functions + matrix elements (Day 17-19)
----------------------------------------------------------
- [ ] ecg/basis.py
      - ECGFunction dataclass (A_matrix, shift, spin_coupling)
      - validate_positive_definite(A)
      - permutation_matrix(particle_indices)
- [ ] ecg/matrix_elements.py
      - overlap_ecg(G_k, G_l, mass_matrix) -> S_kl
      - kinetic_ecg(G_k, G_l, mass_matrix) -> T_kl
      - coulomb_ecg(G_k, G_l, mass_matrix, charges, proj_matrix) -> V_kl
      - All implemented as pure JAX functions (jax.grad-compatible)
- [ ] ecg/antisymmetrize.py
      - generate_permutations(n_electrons)
      - antisymmetrized_matrix_element(G_k, G_l, perms, signs)
- [ ] ecg/coordinates.py
      - lab_to_jacobi(particle_coords, masses) -> relative_coords
      - construct_mass_matrix(masses, jacobi_tree)
      - construct_projection_matrix(i, j, jacobi_tree)
- [ ] tests/test_ecg.py
      - H atom (1 particle after CM removal): exact -0.5 Ha
      - H- (2 electrons, 1 proton -> 2 relative coords)
      - H2 vs BO: ECG energy < BO energy (captures nuclear correlation)

Step 12: ECG optimization + production (Day 20-22)
---------------------------------------------------
- [ ] ecg/solver.py
      - solve_generalized_eigenvalue(H, S) -> (energies, coefficients)
      - energy_functional(ecg_params, charges, masses)
      - jax.grad(energy_functional) for gradient-based optimization
- [ ] ecg/svm.py (stochastic variational method)
      - propose_random_function(n_particles, key)
      - trial_addition(basis, new_func) -> (accept, new_energy)
      - refine_all_parameters(basis, n_grad_steps) using jax.grad
- [ ] ecg/interface.py
      - run_ecg(particles, charges, masses, K_basis, method='svm+grad')
      - ECGResult(energy, wavefunction, basis_functions)
- [ ] benchmarks/ecg_benchmarks.py
      - H atom:      E = -0.5 Ha (exact)
      - H2+:         E vs exact (2-center Coulomb)
      - H2:          E vs BO (quantify BO error: ~1 cm^{-1})
      - Ps-:         (e+ e- e-) bound state
      - Positronium: (e+ e-): E = -0.25 Ha (exact)
```

### Phase 3: Exact Factorization (advanced, research-grade)

```
Step 13: EF framework + 1D models (Day 23-25)
----------------------------------------------
- [ ] ef/conditional.py
      - ConditionalWavefunction: phi_R(r) in electronic basis
      - compute_berry_connection(phi_R, dR) -> A_A
      - compute_tdpes(phi_R, chi, H_BO) -> epsilon(R)
      - compute_coupling_operator(chi, phi_R) -> U_en
- [ ] ef/marginal.py
      - MarginalWavefunction: chi(R) on a grid or as wavepackets
      - propagate_nuclear(chi, A, epsilon, dt)
      - nuclear_density(chi) -> |chi(R)|^2
      - nuclear_current(chi) -> j(R)
- [ ] ef/self_consistent.py
      - ef_scf_step(phi_R, chi) -> (phi_R_new, chi_new)
      - run_ef(mol, basis, grid_R, max_iter) -> EFResult
- [ ] ef/model_1d.py
      - 1D model systems: Shin-Metiu, double well
      - Exact diagonalization for reference
- [ ] tests/test_ef.py
      - 1D Shin-Metiu: TDPES matches reference
      - Gauge invariance: results independent of gauge choice

Step 14: EF for 3D molecules (Day 26-28)
-----------------------------------------
- [ ] ef/molecular_ef.py
      - Molecular EF with CAS-SCF for multi-state phi_R
      - Trajectory-based nuclear propagation
- [ ] ef/conical_intersection.py
      - Detect and handle conical intersections in the TDPES
      - Geometric phase (Berry phase) inclusion
- [ ] tests/test_ef_molecular.py
      - H2+ proton transfer: exact vs EF
```

### Phase 4: Integration, tunneling benchmarks, and cross-validation

```
Step 15: Cross-method benchmarks (Day 29-31)
---------------------------------------------
- [ ] benchmarks/non_bo_comparison.py
      - H2: BO energy vs ECG energy vs PIMD ZPE-corrected energy
      - Quantify BO error for each molecule
      - Compare computational cost scaling
- [ ] benchmarks/isotope_effects.py
      - H2 vs D2 vs T2: vibrational frequencies, bond lengths
      - PIMD captures mass-dependent quantum effects
- [ ] benchmarks/tunneling_rates.py
      - Eckart barrier: RPMD rate vs RPI rate vs exact T(E)
        at T = 0.5 T_c, T_c, 2 T_c, 5 T_c
      - Arrhenius plot: ln(k) vs 1/T showing tunneling crossover
      - H + H2 collinear: RPMD rate vs exact quantum rate
- [ ] benchmarks/tunneling_splittings.py
      - Symmetric double well (1D):
        ECG splitting vs RPI splitting vs DVR exact
        as function of barrier height (V_b / hbar*omega = 1, 2, 4, 8)
      - H2 inversion: tunneling splitting from ECG (non-BO)
        vs harmonic approximation
- [ ] benchmarks/proton_transfer.py
      - Malonaldehyde H-transfer: PIMD free energy profile at 300 K
        showing tunneling-assisted transfer
      - KIE (kinetic isotope effect): k_H / k_D from RPMD
        (expected ~3-10 for proton transfer reactions)
```

---

## 10. FP Abstraction Map

```
Component                 FP Abstraction      Parallelism        Hardware
------------------------  ------------------  -----------------  --------
BO energy (existing)      Applicative+Monad   per shell pair     GPU
BO gradient (existing)    Adjunction (VJP)    per coordinate     GPU
BO geom opt (existing)    Monad over Monad    sequential steps   CPU ctrl

PIMD ring polymer         Comonad             replicas share     GPU
  Spring forces           Applicative         vmap over beads    GPU
  Physical forces         Applicative         vmap over beads    GPU x P
  Normal mode transform   Functor (DFT)       FFT over beads     GPU
  Thermostat              State Monad         per normal mode    GPU
  Estimators              Foldable            reduce over beads  GPU

ECG matrix elements       Applicative         vmap over pairs    GPU
  Overlap S_kl            Functor             det + trace        GPU
  Kinetic T_kl            Functor             trace formula      GPU
  Coulomb V_kl            Functor             1/sqrt(trace)      GPU
  Antisymmetrize          Foldable            sum over perms     GPU
  Eigensolve              Pure function       single call        CPU/GPU
  Parameter optimization  Adjunction (VJP)    jax.grad of E      GPU
  SVM step                State Monad         propose + accept   CPU ctrl

Exact Factorization       Costate Monad       coupled equations  GPU
  Conditional phi_R       Applicative         per R grid point   GPU
  Marginal chi(R)         State Monad         time propagation   GPU
  Berry connection        Foldable            integral over r    GPU
  TDPES                   Foldable            integral over r    GPU
  Self-consistent loop    Fix-point (Monad)   iterate phi, chi   CPU ctrl

Tunneling (RPI)           Adjunction          saddle-point opt   GPU
  Ring polymer potential  Applicative         vmap over beads    GPU
  Instanton search        Monad (optim)       eigvec-following   GPU
  Fluctuation Hessian     Adjunction^2        jax.hessian        GPU
  Rate prefactor          Foldable            det of Hessian     CPU/GPU
  WKB integrals           Foldable            quadrature         CPU
```

**Key JAX features exploited:**

| JAX Feature | Non-BO Application |
|-------------|-------------------|
| `jax.grad` | ECG parameter optimization; PIMD virial estimator; EF coupling terms; RPI instanton gradient |
| `jax.hessian` | Ring polymer instanton: fluctuation determinant for tunneling rate prefactor |
| `jax.vmap` | PIMD: evaluate energy at P beads in parallel; ECG: batch matrix elements |
| `jax.jit` | Compile full PIMD step; JIT the ECG eigensolve loop; JIT instanton search |
| `jax.lax.scan` | PIMD trajectory accumulation without Python loop overhead |
| `jax.random` | SVM stochastic proposals; Langevin thermostat noise |
| `jax.lax.cond` | SVM accept/reject without Python branching |

---

## Appendix A: Key Physical Constants for Non-BO

| Quantity | Symbol | Value (a.u.) |
|----------|--------|-------------|
| Proton mass | M_p | 1836.15267 |
| Deuteron mass | M_d | 3670.48297 |
| Triton mass | M_t | 5496.92154 |
| Positron mass | m_{e+} | 1.0 |
| Muon mass | m_mu | 206.768 |
| Boltzmann constant | k_B | 3.16681e-6 Ha/K |
| hbar | hbar | 1.0 |

## Appendix B: BO Error Estimates

The leading-order BO correction (diagonal BO correction, DBOC) scales as:

```
Delta_E_DBOC = Sum_A (1 / 2M_A) < psi | nabla_A^2 | psi >
```

Typical magnitudes:

| System | Delta_E_DBOC (cm^{-1}) | Delta_E_DBOC (uHa) |
|--------|------------------------|---------------------|
| H2 | ~1.0 | ~4.6 |
| H2O | ~0.3 | ~1.4 |
| LiH | ~0.1 | ~0.5 |
| CH4 | ~0.2 | ~0.9 |

For comparison, the BO energy of H2/STO-3G is ~-1.117 Ha, so the
DBOC correction is ~4e-6 relative -- below the 1 uHa accuracy target
of the current jax_qc implementation but relevant for spectroscopic
accuracy.

## Appendix C: References

1. Born, M.; Oppenheimer, R. Ann. Phys. 1927, 389, 457-484.
2. Abedi, A.; Maitra, N.T.; Gross, E.K.U. PRL 2010, 105, 123002. (Exact Factorization)
3. Suzuki, Y.; Varga, K. "Stochastic Variational Approach to Quantum-Mechanical Few-Body Problems" Springer, 1998. (ECG/SVM)
4. Adamowicz, L.; Bubin, S.; Pavanello, M.; et al. Chem. Rev. 2013, 113, 36-79. (Non-BO ECG review)
5. Ceriotti, M.; Parrinello, M.; Markland, T.E.; Manolopoulos, D.E. JCP 2010, 133, 124104. (PIMD/PILE)
6. Habershon, S.; Manolopoulos, D.E.; Markland, T.E.; Miller, T.F. Annu. Rev. Phys. Chem. 2013, 64, 387-413. (RPMD review)
7. Agostini, F.; Gross, E.K.U. Eur. Phys. J. B 2021, 94, 179. (EF review)
8. Pachucki, K.; Komasa, J. JCP 2009, 130, 164113. (High-precision ECG)
9. Kapil, V.; et al. Comp. Phys. Comm. 2019, 236, 214-223. (i-PI)
10. Markland, T.E.; Ceriotti, M. Nature Reviews Chemistry 2018, 2, 0109. (Nuclear quantum effects review)
11. Richardson, J.O.; Althorpe, S.C. JCP 2009, 131, 214106. (Ring polymer instanton rate theory)
12. Richardson, J.O. JCP 2016, 144, 114106. (RPI for tunneling splittings)
13. Mil'nikov, G.V.; Nakamura, H. JCP 2001, 115, 6881. (Instanton approach to tunneling splittings)
14. Rommel, J.B.; Kastner, J. JCP 2011, 134, 184107. (Practical instanton calculations)
15. Andersson, S.; Nyman, G.; Arnaldsson, A.; Manthe, U.; Jonsson, H. JPCA 2009, 113, 4468. (Instanton rate for H+H2)
