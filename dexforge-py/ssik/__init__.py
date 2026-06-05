"""ssik compatibility layer for dexforge_py.

This package provides ssik-compatible APIs wrapping dexforge_py, allowing
ssik tests to run unmodified against the Rust dexforge implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["TolerancePolicy", "DEFAULT_TOLERANCE_POLICY", "Solution", "RefinementMode"]

# ---------------------------------------------------------------------------
# Tolerance Policy
# ---------------------------------------------------------------------------


@dataclass
class TolerancePolicy:
    """Numeric tolerances for kinematic predicates and subproblem solvers.

    Field semantics match ssik.core.tolerances.TolerancePolicy for test compatibility.
    """

    axis_parallel: float = 1e-8
    axis_intersect: float = 1e-8
    subproblem_feasibility: float = 1e-9
    subproblem_numerical: float = 1e-5
    subproblem_degeneracy: float = 1e-12
    subproblem_dedup: float = 1e-3


DEFAULT_TOLERANCE_POLICY = TolerancePolicy()


# ---------------------------------------------------------------------------
# Solution Type
# ---------------------------------------------------------------------------

RefinementMode = Literal["none", "lm"]


@dataclass(frozen=True)
class Solution:
    """A single analytical IK solution.

    :param q: joint-angle vector. Length matches the chain's DOF.
    :param fk_residual: ``||FK(q) - T_target||_F`` at the moment the solver
        returned the candidate. The caller can compare against any target
        tolerance; the solver's own ``fk_atol`` was a *filter*, not a
        contract on the value reported here.
    :param refinement_used: ``"none"`` if the solution came directly from
        the algebraic / closed-form path; ``"lm"`` if Levenberg-Marquardt
        refinement polished it (when ``allow_refinement=True``).
    """

    q: NDArray[np.float64]
    fk_residual: float
    refinement_used: RefinementMode = "none"
