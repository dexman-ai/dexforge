"""Unitree Z1 IK driver - dexforge_unitree compatibility wrapper."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import dexforge_unitree
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

BASE_LINK = "link00"
EE_LINK = "link06"
DOF = 6
SOLVER_NAME = "ikgeo.three_parallel"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 1.6
FLOP_BUDGET = 2519
DISPATCH_REASON = "Three consecutive parallel axes at joints (1, 2, 3) -- the UR-class structure (UR3 / UR5 / UR10).\nClosed-form via SP6 (joints 0+4) + SP1 + SP3."

T_HOME = np.array([
    [1.0, 0.0, 0.0, -0.012799999999999971],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.1605],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

_JOINT_LIMITS = [
    (-2.6179938779914944, 2.6179938779914944),
    (0.0, 2.9670597283903604),
    (-2.8797932657906435, 0.0),
    (-1.5184364492350666, 1.5184364492350666),
    (-1.3439035240356338, 1.3439035240356338),
    (-2.792526803190927, 2.792526803190927),
]


def fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
    q_list = q.tolist() if hasattr(q, "tolist") else list(q)
    T = dexforge_unitree.z1.fk(q_list)
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
    q_seed_list = q_seed.tolist() if q_seed is not None else None
    if policy is not None:
        fk_atol = policy.subproblem_numerical
        dedup_atol = policy.subproblem_dedup
    else:
        fk_atol = 1e-5
        dedup_atol = 1e-3

    solutions_raw = dexforge_unitree.z1.solve(
        T_target,
        max_solutions=max_solutions,
        q_seed=q_seed_list,
        respect_limits=respect_limits,
        allow_refinement=allow_refinement,
        fk_atol=fk_atol,
        dedup_atol=dedup_atol,
        refinement_max_iters=refinement_max_iters,
    )

    solutions = []
    for q_list in solutions_raw:
        q_arr = np.array(q_list, dtype=np.float64)
        T_fk = fk(q_arr)
        fk_residual = np.linalg.norm(T_fk - T_target, ord="fro")
        refinement_used = "lm" if allow_refinement else "none"
        solutions.append(
            Solution(
                q=q_arr,
                fk_residual=fk_residual,
                refinement_used=refinement_used,
            )
        )
    return solutions
