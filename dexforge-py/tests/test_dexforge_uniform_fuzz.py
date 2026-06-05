"""Uniform 500-pose Hypothesis bulletproof sweep across dexforge_py arms.

Each dexforge_py supported arm gets the same property-test shape: 500 random
non-singular ``q*`` configurations, FK to ``T_target``, solve, assert
every returned IK FK-closes within the arm's documented residual ceiling.

This mirrors ssik/tests/test_prebuilt_uniform_fuzz.py but for the Rust
dexforge_py bindings.

Per-arm FK residual ceilings (from SSIK manifest) reflect the analytical
paths' numerical floors:
- Closed-form 6R (UR5): ~1e-12 to 1e-13
- RR 6R (Piper): ~1e-6 (RR resultant has structural conditioning floor)
- 7R via jointlock + LM (Franka): ~1e-7 to 1e-9

The universal contract is max_fk < 1e-6; per-arm tighter ceilings come
from ARMS_METADATA.fk_ceiling_fuzz, which carries a 3-5x safety margin
above 200-pose sampling worst-case.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

import dexforge_py
from conftest import ARMS_METADATA
from _hypothesis_strategies import non_singular_q6r, non_singular_q7r


# Tuple: (arm_name, fk_ceiling).
_6R_ARMS = [(name, arm.fk_ceiling_fuzz) for name, arm in ARMS_METADATA.items() if arm.dof == 6]
_7R_ARMS = [(name, arm.fk_ceiling_fuzz) for name, arm in ARMS_METADATA.items() if arm.dof == 7]


@pytest.mark.parametrize(
    ("arm_name", "fk_ceiling"),
    _6R_ARMS,
    ids=[a[0] for a in _6R_ARMS],
)
@given(q_star=non_singular_q6r())
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture],
)
def test_dexforge_6r_random_q_roundtrip(
    arm_name: str, fk_ceiling: float, q_star: np.ndarray
) -> None:
    """For every random non-singular 6R configuration, dexforge_py solve
    must return candidates that all FK-close within the arm's ceiling.
    """
    arm = ARMS_METADATA[arm_name]
    assert arm.dof == 6

    q_list = q_star.tolist()
    T_target = np.array(dexforge_py.fk(arm_name, q_list))

    # Solve without limit filtering to test the analytical engine.
    sols = dexforge_py.solve(arm_name, T_target, respect_limits=False, allow_refinement=False)

    # Every returned candidate must FK-close within the ceiling.
    for sol_q in sols:
        T_check = np.array(dexforge_py.fk(arm_name, sol_q))
        residual = np.linalg.norm(T_check - T_target, ord="fro")
        assert (
            residual < fk_ceiling
        ), f"{arm_name} @q={q_star}: FK residual {residual:.2e} exceeds ceiling {fk_ceiling:.2e}"


@pytest.mark.parametrize(
    ("arm_name", "fk_ceiling"),
    _7R_ARMS,
    ids=[a[0] for a in _7R_ARMS],
)
@given(q_star=non_singular_q7r())
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture],
)
def test_dexforge_7r_random_q_roundtrip(
    arm_name: str, fk_ceiling: float, q_star: np.ndarray
) -> None:
    """For every random non-singular 7R configuration, dexforge_py solve
    must return candidates that all FK-close within the arm's ceiling.
    """
    arm = ARMS_METADATA[arm_name]
    assert arm.dof == 7

    q_list = q_star.tolist()
    T_target = np.array(dexforge_py.fk(arm_name, q_list))

    # Solve without limit filtering to test the analytical engine.
    sols = dexforge_py.solve(arm_name, T_target, respect_limits=False, allow_refinement=False)

    # Every returned candidate must FK-close within the ceiling.
    for sol_q in sols:
        T_check = np.array(dexforge_py.fk(arm_name, sol_q))
        residual = np.linalg.norm(T_check - T_target, ord="fro")
        assert (
            residual < fk_ceiling
        ), f"{arm_name} @q={q_star}: FK residual {residual:.2e} exceeds ceiling {fk_ceiling:.2e}"
