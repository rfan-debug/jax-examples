"""Post-SCF property analysis.

All property functions are Foldable — reductions over density-matrix
and integral data. They are pure functions that take an ``SCFResult``
(and optionally the ``Molecule`` / ``BasisSet``) and return scalar or
array values.
"""
