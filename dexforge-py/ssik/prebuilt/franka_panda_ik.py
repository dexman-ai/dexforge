"""Franka Emika Panda IK driver - dexforge_py compatibility wrapper.

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

# --- Metadata constants (from ssik.prebuilt.franka_panda_ik) ---

BASE_LINK = "base_link"
EE_LINK = "ee_link"
DOF = 7
SOLVER_NAME = "jointlock.seven_r"
SOLVER_TIER = 1
EXPECTED_MS_MEDIAN = 50.0
FLOP_BUDGET = 30274
DISPATCH_REASON = "7R revolute chain (non-SRS). Locking one joint reduces to 6R IK."

# Home pose: FK at q = np.zeros(DOF)
T_HOME = np.array([
    [-0.7071068058785943, -0.7071067564945002, 0.0, 0.088],
    [-0.7071067564944993, 0.7071068058785934, -4.440892098500621e-16, -3.241851231905455e-17],
    [3.1401848077128287e-16, -3.140185027022262e-16, -0.9999999999999987, 0.9259999999999998],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

# Joint limits (from Franka Panda URDF)
_JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]


def fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward kinematics.
    
    :param q: 7-element joint-angle vector in radians
    :returns: 4×4 SE(3) pose of ee_link in base_link
    """
    q_list = q.tolist() if hasattr(q, 'tolist') else list(q)
    T = dexforge_py.franka_panda_fk(q_list)
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
    solutions_raw = dexforge_py.franka_panda_solve(
        "franka_panda",
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
