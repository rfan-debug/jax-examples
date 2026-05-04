"""Analytic and numerical molecular gradients.

FP: Adjunction — gradients are the right adjoint (VJP) of the
energy computation. ``jax.grad`` computes dE/dR in one backward
pass through the integral + SCF pipeline.
"""
