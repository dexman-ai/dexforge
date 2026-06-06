"""Flexiv Rizon 10 IK driver - dexforge_py compatibility wrapper.

This module provides an ssik-compatible API wrapping dexforge_py, allowing
ssik tests to run unmodified against the Rust dexforge implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import dexforge_py
from ssik.core import Solution

__all__ = [
    "BASE_LINK",
    "DISPATCH_REASON",
    "DOF",
    "EE_LINK",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "T_HOME",
    "fk",
    "solve",
]

# --- Metadata constants ---

BASE_LINK = "base_link"
EE_LINK = "ee_link"
DOF = 7
SOLVER_NAME = "jointlock.seven_r"
SOLVER_TIER = 1
EXPECTED_MS_MEDIAN = 50.0
FLOP_BUDGET = 30274
DISPATCH_REASON = "7R revolute chain. Locking one joint reduces to 6R IK."

# Home pose: computed at module import via FK(zeros)

# Joint limits (from src/drivers/rizon10/consts.rs)
_JOINT_LIMITS = [
    (-2.8798, 2.8798),
    (-2.7576, 2.7576),
    (-2.8798, 2.8798),
    (-2.7925, 2.7925),
    (-3.0543, 3.0543),
    (-1.4835, 4.6251),
    (-3.0543, 3.0543),
]


def fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward kinematics.
    
    :param q: 7-element joint-angle vector in radians
    :returns: 4×4 SE(3) pose of ee_link in base_link
    """
    q_list = q.tolist() if hasattr(q, 'tolist') else list(q)
    T = dexforge_py.rizon10.fk(q_list)
    return np.array(T, dtype=np.float64)


def solve(
    T_target: NDArray[np.float64],
    max_solutions: int | None = None,
    q_seed: NDArray[np.float64] | None = None,
    respect_limits: bool = True,
    allow_refinement: bool = False,
    policy=None,
    refinement_max_iters: int = 15,
) -> list[Solution]:
    """Inverse kinematics.
    
    :param T_target: 4×4 SE(3) target pose of ee_link in base_link
    :param max_solutions: cap on returned solutions (post-dedup, post-limits)
    :param q_seed: optional seed configuration for nearest-first sorting
    :param respect_limits: when True, filter solutions outside joint limits
    :param allow_refinement: when True, polish candidates with LM refinement
    :param policy: tolerance policy (uses default if None)
    :param refinement_max_iters: max LM iterations per candidate
    :returns: list of Solution objects
    """
    # Convert inputs
    q_seed_list = q_seed.tolist() if q_seed is not None else None
    
    # Extract tolerance parameters from policy if provided
    if policy is not None:
        fk_atol = policy.subproblem_numerical
        dedup_atol = policy.subproblem_dedup
    else:
        fk_atol = 1e-5
        dedup_atol = 1e-3
    
    # Call dexforge_py
    solutions_raw = dexforge_py.rizon10.solve(
        T_target,
        max_solutions=max_solutions,
        q_seed=q_seed_list,
        respect_limits=respect_limits,
        allow_refinement=allow_refinement,
        fk_atol=fk_atol,
        dedup_atol=dedup_atol,
        refinement_max_iters=refinement_max_iters,
    )
    
    # Wrap in Solution objects with FK residuals
    solutions = []
    for q_list in solutions_raw:
        q_arr = np.array(q_list, dtype=np.float64)
        T_fk = fk(q_arr)
        fk_residual = np.linalg.norm(T_fk - T_target, ord='fro')
        refinement_used = "lm" if allow_refinement else "none"
        solutions.append(Solution(
            q=q_arr,
            fk_residual=fk_residual,
            refinement_used=refinement_used,
        ))
    
    return solutions


T_HOME = fk(np.zeros(DOF))
