"""UFactory xArm7 IK driver - direct dexforge_py access.

This module provides a common interface to dexforge_py.xarm7_solve and dexforge_py.xarm7_fk.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import dexforge_py

__all__ = ["DOF", "_JOINT_LIMITS", "fk", "solve"]

# --- Metadata constants ---

DOF = 7

# Joint limits (from xArm7)
_JOINT_LIMITS = [
    (-6.28318530717959, 6.28318530717959),
    (-2.059, 2.0944),
    (-6.28318530717959, 6.28318530717959),
    (-0.19198, 3.927),
    (-6.28318530717959, 6.28318530717959),
    (-1.69297, 3.14159265358979),
    (-6.28318530717959, 6.28318530717959),
]


def fk(q: NDArray[np.float64] | list) -> NDArray[np.float64]:
    """Forward kinematics.
    
    :param q: 7-element joint-angle vector in radians
    :returns: 4×4 SE(3) pose of ee_link in base_link
    """
    q_list = q.tolist() if hasattr(q, 'tolist') else list(q)
    T = dexforge_py.xarm7_fk(q_list)
    return np.array(T, dtype=np.float64)


def solve(
    T_target: NDArray[np.float64] | list,
    max_solutions: int | None = None,
    q_seed: NDArray[np.float64] | list | None = None,
    respect_limits: bool = True,
    allow_refinement: bool = False,
    fk_atol: float = 1e-5,
    dedup_atol: float = 1e-3,
    refinement_max_iters: int = 15,
) -> list[list[float]]:
    """Inverse kinematics.
    
    :param T_target: 4×4 SE(3) target pose of ee_link in base_link
    :param max_solutions: cap on returned solutions (post-dedup, post-limits)
    :param q_seed: optional seed configuration for nearest-first sorting
    :param respect_limits: when True, filter solutions outside joint limits
    :param allow_refinement: when True, polish candidates with LM refinement
    :param fk_atol: FK closure tolerance
    :param dedup_atol: deduplication tolerance
    :param refinement_max_iters: max LM iterations per candidate
    :returns: list of joint-angle solutions (plain Python lists)
    """
    # Convert inputs
    q_seed_list = q_seed.tolist() if q_seed is not None and hasattr(q_seed, 'tolist') else q_seed
    
    # Call dexforge_py
    solutions = dexforge_py.xarm7_solve(
        T_target,
        max_solutions=max_solutions,
        q_seed=q_seed_list,
        respect_limits=respect_limits,
        allow_refinement=allow_refinement,
        fk_atol=fk_atol,
        dedup_atol=dedup_atol,
        refinement_max_iters=refinement_max_iters,
    )
    
    return solutions
