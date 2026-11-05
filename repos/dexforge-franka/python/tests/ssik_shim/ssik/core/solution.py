"""Standardized solver return type.

The public :class:`Solution` dataclass carries the joint vector and just
enough provenance for the caller to reason about precision:

- ``q`` is the joint configuration; length matches the chain's DOF.
- ``fk_residual`` reports the actual FK closure -- not the user's
  tolerance, the measured value.
- ``refinement_used`` reports whether numerical polish fired (``"lm"``
  vs ``"none"``). Closed-form solvers are always ``"none"``; numeric
  solvers (Raghavan-Roth, Husty-Pfurner) may polish near-miss candidates
  when ``allow_refinement=True``.

Adapted from ssik.core.solution for dexforge_franka compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["RefinementMode", "Solution"]

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

