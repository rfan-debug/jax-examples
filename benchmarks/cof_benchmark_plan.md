# Benchmark Progression: From Atoms to Covalent Organic Frameworks

## Roadmap for Systematic Validation of JAX-QC

This document defines a graded series of benchmark systems, progressing
from single atoms to fragments of Covalent Organic Frameworks (COFs).
Each level introduces new chemical complexity that stress-tests a
specific capability of jax_qc.

---

## Why COFs?

Covalent Organic Frameworks are crystalline porous polymers built from
organic building blocks connected by strong covalent bonds. They are
important for gas storage/separation, catalysis, optoelectronics, and
energy storage. The chemistry involves:

- **Aromatic cores**: benzene, triazine, porphyrin (pi-conjugation)
- **Linkage bonds**: imine (C=N), boroxine (B-O), hydrazone (C=N-NH),
  amide (C(=O)-NH), beta-ketoenamine
- **Pore chemistry**: guest-host interactions, confined reactions
- **Stacking**: 2D COF layers interact via pi-pi and van der Waals forces

Accurately modeling COFs requires handling aromatic systems, heteroatoms
(B, N, O), conjugation, and eventually periodic boundary conditions.

---

## Level 1: Atoms and Diatomics (DONE)

**Already validated in Steps 1-5. Purpose: baseline integral + SCF
correctness.**

| # | System | Atoms | e- | Basis | Method | PySCF E (Ha) | Status |
|---|--------|-------|----|-------|--------|-------------|--------|
| 1.1 | H2 | 2 | 2 | STO-3G | RHF | -1.1167 | PASS |
| 1.2 | He | 1 | 2 | STO-3G | RHF | -2.8552 | PASS |
| 1.3 | HeH+ | 2 | 2 | STO-3G | RHF | -2.8606 | PASS |
| 1.4 | H3+ | 3 | 2 | STO-3G | RHF | -1.2024 | PASS |
| 1.5 | LiH | 2 | 4 | STO-3G | RHF | -7.8633 | PASS |
| 1.6 | HF | 2 | 10 | STO-3G | RHF | -98.5728 | PASS |
| 1.7 | H (UHF) | 1 | 1 | STO-3G | UHF | -0.4666 | PASS |
| 1.8 | O2 (UHF) | 2 | 16 | STO-3G | UHF | -147.6340 | PASS |

**COF relevance**: None directly, but validates the computational engine.

---

## Level 2: Small Polyatomics (MOSTLY DONE)

**Purpose: validate p-shell integrals, multi-atom SCF, geometry
optimization. These are functional-group precursors to COF chemistry.**

| # | System | Atoms | e- | nao | Basis | PySCF E (Ha) | Status | COF Relevance |
|---|--------|-------|----|-----|-------|-------------|--------|---------------|
| 2.1 | H2O | 3 | 10 | 7 | STO-3G | -74.9659 | PASS | Hydrolysis product |
| 2.2 | NH3 | 4 | 10 | 8 | STO-3G | -55.4542 | PASS | Amine precursor |
| 2.3 | CH4 | 5 | 10 | 9 | STO-3G | -39.7269 | PASS | C-H bond baseline |
| 2.4 | CO | 2 | 14 | 10 | STO-3G | -111.2246 | TODO | Triple bond, carbonyl |
| 2.5 | N2 | 2 | 14 | 10 | STO-3G | -107.4960 | TODO | N-N bond baseline |
| 2.6 | HCN | 3 | 14 | 11 | STO-3G | -91.6750 | TODO | Nitrile, C-N triple bond |
| 2.7 | H2CO | 4 | 16 | 12 | STO-3G | -112.3539 | TODO | Aldehyde, C=O bond |
| 2.8 | C2H4 | 6 | 16 | 14 | STO-3G | -77.0727 | TODO | C=C pi bond |
| 2.9 | N2H2 | 4 | 16 | 12 | STO-3G | -108.2669 | TODO | Diazene, N=N double bond |

**New chemistry tested**: C=O, C=N, C=C double bonds; triple bonds;
pi-electron systems. All are building blocks of COF linkages.

---

## Level 3: COF-Relevant Functional Groups

**Purpose: validate the specific bond types and heteroatom combinations
that appear in COF linkages. Each molecule isolates one chemical motif.**

| # | System | Formula | Atoms | e- | nao | PySCF E (Ha) | COF Motif |
|---|--------|---------|-------|----|-----|-------------|-----------|
| 3.1 | Methanimine | CH2=NH | 5 | 16 | 13 | -92.8153 | Imine (C=N) bond -- the most common COF linkage |
| 3.2 | Formamide | HCONH2 | 6 | 24 | 18 | -166.5931 | Amide (C(=O)-NH) -- amide-linked COFs |
| 3.3 | Formic acid | HCOOH | 5 | 24 | 17 | -186.1831 | Carboxylic acid -- boronate ester precursor |
| 3.4 | Methylamine | CH3NH2 | 7 | 18 | 15 | -94.0297 | Primary amine -- COF monomer end group |
| 3.5 | Boric acid | B(OH)3 | 7 | 32 | 23 | -247.7221 | Boronic acid -- boron COF precursor |
| 3.6 | Glyoxal | OHCCHO | 6 | 30 | 20 | (compute) | Dialdehyde -- cross-linker model |

### Why these matter for COFs:

**Imine COFs** (e.g., COF-300, LZU-1, TAPB-DMTA): Formed by condensation
of an amine (R-NH2) with an aldehyde (R-CHO):
```
R-NH2 + OHC-R' --> R-N=CH-R' + H2O
```
Methanimine (3.1) is the simplest imine. Formamide (3.2) tests the amide
variant. Methylamine (3.4) tests the amine precursor.

**Boronate ester COFs** (e.g., COF-1, COF-5): Formed by condensation of
boronic acid with a diol:
```
R-B(OH)2 + HO-R'-OH --> R-B-O-R'-O (boronate ester) + 2 H2O
```
Boric acid (3.5) is the simplest boron-containing benchmark.

---

## Level 4: Aromatic Building Blocks

**Purpose: validate aromatic pi-systems, heteroaromatic rings, and
larger electron counts. These are the actual cores and nodes of COFs.**

| # | System | Formula | Atoms | e- | nao | PySCF E (Ha) | COF Role |
|---|--------|---------|-------|----|-----|-------------|----------|
| 4.1 | Benzene | C6H6 | 12 | 42 | 36 | -227.8906 | Universal aromatic linker core |
| 4.2 | Pyridine | C5H5N | 11 | 42 | 35 | -243.6241 | N-heterocycle in CTF nodes |
| 4.3 | 1,3,5-Triazine | C3N3H3 | 9 | 42 | 33 | -275.1120 | CTF (Covalent Triazine Framework) node |
| 4.4 | Phenol | C6H5OH | 13 | 50 | 41 | -301.7123 | Boronate ester fragment (diol side) |
| 4.5 | Aniline | C6H5NH2 | 14 | 50 | 42 | -282.1471 | Amine-functionalized benzene |
| 4.6 | Benzaldehyde | C6H5CHO | 14 | 56 | 46 | (compute) | Aldehyde-functionalized benzene |
| 4.7 | Boroxine | B3O3H3 | 9 | 42 | 33 | -296.7479 | Boroxine ring in boron COFs |

### Why these matter:

- **Benzene** (4.1): The aromatic ring is the core structural unit of
  nearly all 2D COFs. Linkers like BDBA, HHTP, and PDA are all
  benzene derivatives.
- **Triazine** (4.3): The node of Covalent Triazine Frameworks (CTFs),
  which are among the most stable COFs.
- **Boroxine** (4.7): The B3O3 ring that forms when three boronic acids
  self-condense -- this is the linkage in COF-1, the first COF ever
  reported (Yaghi, 2005).
- **Aniline** (4.5): A model for the amine monomers (e.g., TAPB =
  1,3,5-tri(4-aminophenyl)benzene) used in imine COFs.

### Scaling note:

At Level 4, we reach 42-56 electrons and 33-46 basis functions (STO-3G).
The ERI tensor has ~10^6 elements. This is where jax_qc's integral
computation becomes the bottleneck (~minutes per single-point).

---

## Level 5: COF Linkage Motifs (Dimer Fragments)

**Purpose: model the actual covalent bond formed during COF synthesis.
Each system is a minimal cluster containing one COF linkage connecting
two aromatic units.**

| # | System | Description | Atoms | e- (est.) | COF Type |
|---|--------|-------------|-------|-----------|----------|
| 5.1 | PhCH=NPh-H | Benzylideneaniline (imine dimer) | ~22 | ~86 | Imine COF linkage model |
| 5.2 | PhB(OH)OPh-H | Boronate ester dimer | ~24 | ~100 | Boronate COF linkage model |
| 5.3 | PhC(=O)NHPh-H | Benzanilide (amide dimer) | ~24 | ~96 | Amide COF linkage model |
| 5.4 | PhCH=N-NHCOPh-H | Hydrazone dimer | ~26 | ~104 | Hydrazone COF linkage model |

### Simplified models (computationally tractable):

For practical jax_qc calculations, use methyl groups instead of full
phenyl rings:

| # | System | Description | Atoms | e- | nao (est.) | Feasibility |
|---|--------|-------------|-------|----|------------|-------------|
| 5.1s | CH3CH=NHCH3 | N-methylethylideneamine | 12 | 32 | ~24 | Feasible |
| 5.2s | CH3B(OH)OCH3 | Methyl boronate ester | 10 | 34 | ~25 | Feasible |
| 5.3s | CH3C(=O)NHCH3 | N-methylacetamide (NMA) | 12 | 40 | ~30 | Feasible |
| 5.4s | CH3CH=N-NHCOCH3 | Methyl hydrazone | 14 | 46 | ~34 | Feasible |

### Key properties to compute at each linkage:

1. **Bond lengths**: C=N (imine ~1.27 A), B-O (boronate ~1.37 A),
   C-N (amide ~1.35 A), N-N (hydrazone ~1.38 A)
2. **Rotation barrier**: Energy as function of dihedral angle around
   the linkage bond (tests pi-conjugation across the link)
3. **Bond dissociation energy**: E(dimer) - E(fragment_1) - E(fragment_2)
4. **Mulliken charges**: Charge transfer across the linkage

---

## Level 6: COF Node-Linker Clusters

**Purpose: model the local coordination environment at a COF node.
These are the largest systems tractable with molecular (non-periodic)
calculations.**

| # | System | Description | Atoms (est.) | e- (est.) | COF |
|---|--------|-------------|-------------|-----------|-----|
| 6.1 | Boroxine-3(CH3) | Boroxine ring + 3 methyl | ~18 | ~66 | COF-1 node model |
| 6.2 | Triazine-3(NH2) | Triazine + 3 amino groups | ~18 | ~66 | CTF node model |
| 6.3 | Benzene-1,3,5-tri(CH=NH) | Tri-imine star | ~21 | ~72 | Imine COF node model |

### Scaling estimate:

| Level | Atoms | Electrons | nao (STO-3G) | ERI elements | Est. time |
|-------|-------|-----------|-------------|-------------|-----------|
| 1 | 1-2 | 1-16 | 1-10 | 10^1 - 10^4 | < 1 sec |
| 2 | 3-6 | 10-16 | 7-14 | 10^3 - 10^4 | 1-10 sec |
| 3 | 5-7 | 16-32 | 13-23 | 10^4 - 10^5 | 10-60 sec |
| 4 | 9-14 | 42-56 | 33-46 | 10^5 - 10^6 | 1-10 min |
| 5 | 10-26 | 32-104 | 24-80 | 10^6 - 10^7 | 10 min - 1 hr |
| 6 | 18-30 | 66-120 | 50-90 | 10^7 - 10^8 | 1-10 hr |

---

## Level 7: Toward Periodic COFs (Future)

Full COF simulation requires periodic boundary conditions (PBC), which
jax_qc does not yet support. The roadmap for PBC:

### 7.1 What's Needed

1. **k-space integration**: Replace molecular integrals with crystal
   integrals over the Brillouin zone.
2. **Ewald summation**: Long-range Coulomb interactions under PBC.
3. **Translational symmetry**: Reduce the unit cell calculation via
   Bloch's theorem.
4. **Band structure**: Eigenvalues as a function of k-vector.

### 7.2 Bridging Strategy: Cluster Models

Until PBC is implemented, COF properties can be approximated using
finite cluster models with increasing size:

```
Monomer  -->  Dimer  -->  Trimer  -->  Hexamer  -->  Periodic limit
(Level 4)   (Level 5)   (Level 6)    (Level 7)     (PBC)
```

Key observables that converge with cluster size:
- **HOMO-LUMO gap**: converges toward the band gap
- **Linkage bond length**: converges within 2-3 repeat units
- **Mulliken charges on the node**: converges within the first shell

### 7.3 Representative COF Targets

| COF | Linkage | Node | Linker | Atoms/UC | Practical Level |
|-----|---------|------|--------|----------|-----------------|
| **COF-1** | Boroxine (B-O) | B3O3 ring | 1,4-phenylene diboronic acid | 24 | Level 5-6 (fragment) |
| **COF-5** | Boronate ester | B3O3 ring + catechol | HHTP | 72 | Level 6 (fragment) |
| **LZU-1** | Imine (C=N) | 1,3,5-triformylbenzene | p-phenylenediamine | 42 | Level 5-6 |
| **TAPB-DMTA** | Imine | TAPB (triamine) | DMTA (dialdehyde) | 90 | Level 6+ |
| **CTF-1** | Triazine | Triazine ring | 1,4-dicyanobenzene | 24 | Level 5-6 |
| **COF-300** | Imine | Tetrahedral | Linear diamine | 120+ | PBC required |

---

## Test Implementation Plan

### Immediate (add to existing test suite):

```python
# tests/test_benchmark_levels.py

# --- Level 2 gaps (should be fast, STO-3G) ---
@pytest.mark.parametrize("name,atoms,e_ref", [
    ("CO",   "C 0 0 0; O 0 0 2.132", -111.2246),
    ("N2",   "N 0 0 0; N 0 0 2.074", -107.4960),
    ("HCN",  "H 0 0 0; C 0 0 2.014; N 0 0 4.181", -91.6750),
    ("H2CO", "...", -112.3539),
    ("C2H4", "...", -77.0727),
])
def test_level2_energy(name, atoms, e_ref): ...

# --- Level 3: functional groups ---
@pytest.mark.slow
@pytest.mark.parametrize("name,atoms,e_ref", [
    ("methanimine",  "...", -92.8153),
    ("formamide",    "...", -166.5931),
    ("boric_acid",   "...", -247.7221),
    ("methylamine",  "...", -94.0297),
])
def test_level3_energy(name, atoms, e_ref): ...

# --- Level 4: aromatics ---
@pytest.mark.slow
@pytest.mark.parametrize("name,atoms,e_ref", [
    ("benzene",   "...", -227.8906),
    ("pyridine",  "...", -243.6241),
    ("triazine",  "...", -275.1120),
    ("boroxine",  "...", -296.7479),
    ("phenol",    "...", -301.7123),
    ("aniline",   "...", -282.1471),
])
def test_level4_energy(name, atoms, e_ref): ...

# --- Level 5: linkage motifs ---
@pytest.mark.slow
@pytest.mark.parametrize("name,atoms,e_ref", [
    ("imine_model",     "...", ...),  # CH3CH=NHCH3
    ("boronate_model",  "...", ...),  # CH3B(OH)OCH3
    ("amide_model",     "...", ...),  # NMA
])
def test_level5_energy(name, atoms, e_ref): ...
```

### Properties to compute at each level:

| Property | Level 1-2 | Level 3 | Level 4 | Level 5 | Level 6 |
|----------|-----------|---------|---------|---------|---------|
| Total energy vs PySCF | Yes | Yes | Yes | Yes | Yes |
| Mulliken charges | -- | Yes | Yes | Yes | Yes |
| HOMO-LUMO gap | -- | Yes | Yes | Yes | Yes |
| Dipole moment | -- | Yes | Yes | Yes | -- |
| Geometry optimization | Yes | Yes | Some | Key bonds | Node only |
| Bond dissociation | -- | -- | -- | Yes | -- |
| Dihedral scan | -- | -- | -- | Yes | -- |

---

## Appendix: PySCF Reference Energies (RHF/STO-3G)

All energies computed with PySCF 2.13.0, conv_tol=1e-12, unit=Angstrom.

```
Level 2:
  CO                     -111.2245586956 Ha
  N2                     -107.4959750306 Ha
  HCN                     -91.6750435584 Ha
  H2CO                   -112.3538642288 Ha
  C2H4                    -77.0726949814 Ha
  N2H2 (diazene)         -108.2668608802 Ha

Level 3:
  Methanimine (CH2=NH)    -92.8152674834 Ha
  Formamide (HCONH2)     -166.5931059249 Ha
  Formic acid (HCOOH)    -186.1830615897 Ha
  Methylamine (CH3NH2)    -94.0297342135 Ha
  Boric acid (B(OH)3)    -247.7221201478 Ha

Level 4:
  Benzene (C6H6)         -227.8905655327 Ha
  Pyridine (C5H5N)       -243.6241020674 Ha
  1,3,5-Triazine         -275.1119984331 Ha
  Boroxine (B3O3H3)      -296.7479300793 Ha
  Phenol (C6H5OH)        -301.7123398552 Ha
  Aniline (C6H5NH2)      -282.1470606823 Ha
```
