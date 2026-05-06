"""Geometry optimization.

FP: Monad over Monad — the outer loop threads the geometry through
gradient evaluations (each of which is itself a Monadic SCF loop).
The optimizer state (coordinates, Hessian approximation, step count)
is an immutable value threaded through pure update functions.
"""
