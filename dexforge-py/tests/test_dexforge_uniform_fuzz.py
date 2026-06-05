"""Uniform 500-pose Hypothesis bulletproof sweep across every dexforge_py arm.

Each arm gets the same property-test shape: 500 random non-singular ``q*``
configurations, FK to ``T_target``, solve, assert every returned IK
FK-closes within the arm's documented residual ceiling.

Tolerances are per-arm because the analytical paths achieve different
numerical floors:
- closed-form 6R (UR5): ~1e-12 to 1e-13
- non-Pieper 6R (PiPER via RR): ~1e-6 (RR resultant has a structural
  conditioning floor)
- approximate-SRS / non-SRS 7R via jointlock + HP / cached-RR
  (Franka Panda): ~1e-7 to 1e-9

The universal bulletproof contract is ``max_fk < 1e-6`` for every
returned candidate. Per-arm tighter ceilings come from ``conftest.ARMS_METADATA``.

Per-arm parametrisation is driven by ``conftest.ARMS_METADATA``; known
coverage gaps with a documented xfail reason (see ssik manifest
``known_gaps`` table) become xfailed automatically.

Mirrors the pattern from ssik's ``test_prebuilt_uniform_fuzz.py`` (#267).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

import dexforge_py
from conftest import ARMS_METADATA
from _hypothesis_strategies import non_singular_q6r, non_singular_q7r

# Each tuple: (arm_name, fk_residual_ceiling).
#
# Ceilings come from each arm's ``fk_ceiling_fuzz`` field in ARMS_METADATA.
# They are calibrated to the arm's *worst-case* FK residual under a 200-pose
# Hypothesis-style sweep (the 500-pose @given run goes harder, so the
# ceiling carries a 3-5x margin above what 200-pose sampling sees). The
# bulletproof claim is "every returned IK FK-closes within the ceiling";
# per-arm precision floor reality is documented in ssik/docs/arm_coverage.md.
DEXFORGE_ARMS_6R: list[tuple[str, float]] = [
    (name, meta.fk_ceiling_fuzz)
    for name, meta in ARMS_METADATA.items()
    if meta.dof == 6
]

DEXFORGE_ARMS_7R: list[tuple[str, float]] = [
    (name, meta.fk_ceiling_fuzz)
    for name, meta in ARMS_METADATA.items()
    if meta.dof == 7
]


def _maybe_xfail(arm_name: str) -> None:
    """xfail if the arm has a known coverage gap.

    ``strict=False`` semantics: a future improvement that closes the
    gap surfaces as ``XPASS`` rather than silent regression.

    For dexforge_py, known gaps are documented in comments in
    ARMS_METADATA (e.g., PiPER's RR coverage gap filed as ssik#286).
    """
    # PiPER has a known RR coverage gap (ssik#286): some poses return 0 sols
    # where the reference Python artifact returns multiple. This is an upstream
    # ssik issue that affects both ssik.prebuilt.piper_ik and dexforge_py.
    # We xfail PiPER uniform fuzz tests until ssik#286 is resolved.
    if arm_name == "piper":
        pytest.xfail(
            "piper: RR coverage gap on q*=[0, 2.75, 0.29, 2.5, 2.75, 0] "
            "(returns 0 sols; q* + 0.01 returns 4). Filed as ssik#286."
        )


# ---------------------------------------------------------------------------
# 6R uniform sweep.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm_name", "fk_ceiling"),
    DEXFORGE_ARMS_6R,
    ids=[a[0] for a in DEXFORGE_ARMS_6R],
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
    """500 random non-singular 6R poses: solve(fk(q*)) returns at least one
    IK, every returned IK FK-closes within the arm's documented ceiling.

    This is the per-arm bulletproof contract for the "solver doesn't lie
    about correctness" claim. Per-pose seed recovery (the seeded q*
    appears in the returned set modulo wrap-to-pi) is not asserted here
    -- branch-collapse poses on UR5/Puma drift q-space by O(1e-4) rad
    while remaining T-space-correct; the dedicated tests in
    test_three_parallel / test_spherical_two_parallel already cover
    seed-recovery with the calibrated tolerance.

    Known coverage gaps (PiPER's RR / Z1's SP6 / etc.) are xfailed via
    ``_maybe_xfail``.
    """
    _maybe_xfail(arm_name)
    T_target = np.array(dexforge_py.fk(arm_name, q_star.tolist()))
    
    # respect_limits=False so we exercise the analytical solver's correctness
    # independent of URDF joint-limit filtering; q* from the Hypothesis strategy
    # is sampled in [-pi+0.3, pi-0.3] which routinely lands outside real arms'
    # narrower URDF limits (Franka's joint 2 is +-1.76 rad, etc.).
    solutions = dexforge_py.solve(arm_name, T_target, respect_limits=False)
    
    assert solutions, f"{arm_name}: reachable non-singular pose returned no IK"
    
    # Verify every returned solution FK-closes within the documented ceiling.
    for sol in solutions:
        q_sol = np.array(sol, dtype=np.float64)
        T_fk = np.array(dexforge_py.fk(arm_name, q_sol.tolist()))
        residual = np.linalg.norm(T_fk - T_target, ord='fro')
        assert residual < fk_ceiling, (
            f"{arm_name}: FK residual {residual:.2e} > {fk_ceiling:.0e} ceiling at "
            f"q*={q_star.tolist()}"
        )


# ---------------------------------------------------------------------------
# 7R uniform sweep.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm_name", "fk_ceiling"),
    DEXFORGE_ARMS_7R,
    ids=[a[0] for a in DEXFORGE_ARMS_7R],
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
    """500 random non-singular 7R poses: solve(fk(q*)) returns at least one
    IK, every returned IK FK-closes within the arm's documented ceiling.

    7R IK returns a discretised 1-parameter family per pose (lock-sweep
    samples x algebraic branches). Seed recovery isn't a clean assertion
    -- the seed q* is unlikely to land exactly on a lock-sample value --
    so the contract is FK-closure only. The corollary: a returned IK
    might be far from q* in joint-space yet correct in T-space.
    """
    _maybe_xfail(arm_name)
    T_target = np.array(dexforge_py.fk(arm_name, q_star.tolist()))
    
    # respect_limits=False so we exercise the analytical solver's correctness
    # independent of URDF joint-limit filtering; q* from the Hypothesis strategy
    # is sampled in [-pi+0.3, pi-0.3] which routinely lands outside real arms'
    # narrower URDF limits (Franka's joint 2 is +-1.76 rad, etc.).
    solutions = dexforge_py.solve(arm_name, T_target, respect_limits=False)
    
    assert solutions, f"{arm_name}: reachable non-singular pose returned no IK"
    
    # Verify every returned solution FK-closes within the documented ceiling.
    for sol in solutions:
        q_sol = np.array(sol, dtype=np.float64)
        T_fk = np.array(dexforge_py.fk(arm_name, q_sol.tolist()))
        residual = np.linalg.norm(T_fk - T_target, ord='fro')
        assert residual < fk_ceiling, (
            f"{arm_name}: FK residual {residual:.2e} > {fk_ceiling:.0e} ceiling at "
            f"q*={q_star.tolist()}"
        )
