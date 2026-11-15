"""Tolerance policy for IK solving."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TolerancePolicy", "DEFAULT_TOLERANCE_POLICY"]


@dataclass
class TolerancePolicy:
    axis_parallel: float = 1e-8
    axis_intersect: float = 1e-8
    subproblem_feasibility: float = 1e-9
    subproblem_numerical: float = 1e-5
    subproblem_degeneracy: float = 1e-12
    subproblem_dedup: float = 1e-3


DEFAULT_TOLERANCE_POLICY = TolerancePolicy()
