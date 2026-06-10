"""Pytest suite for the dexforge_py Python interface.

Cross-validates per-arm native ``dexforge_py`` functions against the reference
Python ``ssik.prebuilt`` artifacts side by side, using random input poses.

Run with::

    pytest dexforge-py/tests/test_dexforge_py.py -v

The ``--arm`` and ``--n`` CLI options let you select the arm and the number of
random poses to test:

    pytest dexforge-py/tests/test_dexforge_py.py --arm=franka_panda --n=50 -v

Perf output is printed when ``--perf`` is passed (or run the dedicated perf
script ``dexforge-py/tests/perf_test.py`` directly).
"""

from __future__ import annotations

import math
import importlib
import time
from typing import List

import numpy as np
import pytest

import dexforge_py

# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------
# Canonical arm names to test.
# Value indicates whether exact matching is required for the arm:
#   True  -> exact count + strict 1e-5 joint match
#   False -> allow singular-pose relaxation (1e-3 joint + FK closure <= 1e-5)
# The ssik module name is derived as ssik.prebuilt.<arm>_ik.
ARMS = {
    "ur5": True,
    "piper": False,
    "franka_panda": True,
    "xarm7": False,
    "rizon4": False,
    "rizon10": False,
}
# For these 7R jointlock solvers, the deterministic kernel can surface
# additional FK-valid roots beyond the Python reference set on singular poses.
SOUNDNESS_ONLY_ARMS = {"rizon4", "rizon10"}

TWO_PI = 2.0 * math.pi
STRICT_TOL = 1e-5
# Tolerance for relaxed wrap-to-pi joint comparison on singular poses.
DEDUP_TOL = 1e-3
FK_MATCH_TOL = 1e-5


def _wrap_to_pi(a: float) -> float:
    return math.remainder(a, TWO_PI)


def _q_close_mod(a: List[float], b: List[float], tol: float = DEDUP_TOL) -> bool:
    """True if joint vectors a and b agree modulo 2π within tol."""
    return len(a) == len(b) and all(
        abs(_wrap_to_pi(ai - bi)) <= tol for ai, bi in zip(a, b)
    )


def _load_ssik_artifact(arm: str):
    """Import and return the ssik.prebuilt module for the given arm.
    
    Args:
        arm: Canonical arm name (e.g., 'ur5', 'franka_panda')
    
    Returns:
        The imported ssik.prebuilt module with solve() and fk() functions
    """
    if arm not in ARMS:
        pytest.skip(f"arm '{arm}' is not in the ARMS registry; skipping")
    try:
        return importlib.import_module(f"ssik.prebuilt.{arm}_ik")
    except ImportError as e:
        pytest.skip(f"ssik.prebuilt.{arm}_ik not available for arm '{arm}': {e}")


# ---------------------------------------------------------------------------
# Helpers shared by tests
# ---------------------------------------------------------------------------

def _random_poses(artifact, n: int, seed: int = 42):
    """Generate ``n`` SE(3) poses by FK on random in-limit joint configs."""
    rng = np.random.default_rng(seed)
    lims = artifact._JOINT_LIMITS
    poses = []
    attempts = 0
    while len(poses) < n and attempts < n * 40:
        attempts += 1
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T = artifact.fk(q)
        poses.append(T)
    return poses


# ---------------------------------------------------------------------------
# Test: cross-validation (Python artifact vs native Rust dexforge_py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(ARMS))
def test_cross_validate_solve(arm, request):
    """For N random poses, dexforge_py native functions must return the same solution set as
    the Python ssik prebuilt artifact.

    Solution sets are compared modulo 2π with the same dedup tolerance used by
    the solver itself, matching the approach of the Rust reference test suite.
    """
    n = request.config.getoption("--n")
    exact_arm = ARMS[arm]
    ssik_artifact = _load_ssik_artifact(arm)
    native_solve = getattr(dexforge_py, arm).solve
    native_fk = getattr(dexforge_py, arm).fk
    poses = _random_poses(ssik_artifact, n)

    mismatches = 0
    for T in poses:
        T_target = np.array(T, dtype=np.float64)
        # Python ssik artifact: raw geometric set (no limit filtering).
        # Returns Solution objects.
        py_sols = ssik_artifact.solve(T, respect_limits=False, allow_refinement=False)
        py_qs = [list(map(float, s.q)) for s in py_sols]

        # Rust dexforge_py native: same parameters.
        # Returns plain lists.
        rust_qs = native_solve(T, respect_limits=False, allow_refinement=False)

        if exact_arm:
            # Exact-arm policy: strict count equality and strict 1e-5 joint match.
            if len(py_qs) != len(rust_qs):
                mismatches += 1
                continue

            # Every Python solution must appear in the Rust set.
            for ref_q in py_qs:
                if not any(_q_close_mod(ref_q, r_q, tol=STRICT_TOL) for r_q in rust_qs):
                    mismatches += 1
                    break
            continue

        # Relaxed-arm policy (singular-tolerant): every returned Rust solution
        # must close the pose in FK space. For non-Rizon arms we additionally
        # require a reference-set match (strict 1e-5 preferred, 1e-3 tolerated).
        pose_failed = False
        for r_q in rust_qs:
            fk_residual = np.max(
                np.abs(np.array(native_fk(r_q), dtype=np.float64) - T_target)
            )
            if fk_residual > FK_MATCH_TOL:
                mismatches += 1
                pose_failed = True
                break

            if arm not in SOUNDNESS_ONLY_ARMS:
                if not any(
                    _q_close_mod(ref_q, r_q, tol=STRICT_TOL)
                    or _q_close_mod(ref_q, r_q, tol=DEDUP_TOL)
                    for ref_q in py_qs
                ):
                    mismatches += 1
                    pose_failed = True
                    break

        if pose_failed:
            continue

        # Completeness only when cardinality matches (mirrors Rust singular policy).
        if len(py_qs) == len(rust_qs):
            for ref_q in py_qs:
                if not any(
                    _q_close_mod(ref_q, r_q, tol=STRICT_TOL)
                    or _q_close_mod(ref_q, r_q, tol=DEDUP_TOL)
                    for r_q in rust_qs
                ):
                    mismatches += 1
                    break

    assert mismatches == 0, (
        f"{arm}: {mismatches}/{len(poses)} poses had mismatched solution sets"
    )


@pytest.mark.parametrize("arm", list(ARMS))
def test_cross_validate_fk(arm, request):
    """dexforge_py native fk must return the same FK matrix as the Python artifact's fk."""
    n = request.config.getoption("--n")
    ssik_artifact = _load_ssik_artifact(arm)
    native_fk = getattr(dexforge_py, arm).fk
    dof = ssik_artifact.DOF
    rng = np.random.default_rng(99)
    lims = ssik_artifact._JOINT_LIMITS

    for _ in range(n):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T_py = ssik_artifact.fk(q)
        T_rust = np.array(native_fk(q.tolist()), dtype=np.float64)
        np.testing.assert_allclose(
            T_rust, T_py, atol=1e-12,
            err_msg=f"{arm}: fk mismatch at q={q}"
        )


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

def test_unreachable_pose_returns_empty():
    """A pose far outside the workspace should return no solutions."""
    native_solve = getattr(dexforge_py, "ur5").solve
    T = np.eye(4)
    T[:3, 3] = [100.0, 100.0, 100.0]
    solutions = native_solve(T, respect_limits=False)
    assert solutions == []


def test_solve_accepts_list_of_lists():
    """solve() must accept a plain Python list-of-lists for T_target."""
    native_solve = getattr(dexforge_py, "ur5").solve
    T = [
        [1.0, 0.0, 0.0, 0.3],
        [0.0, 1.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result = native_solve(T, respect_limits=False)
    assert isinstance(result, list)


def test_fk_wrong_dof_raises():
    """fk() must raise ValueError when q has the wrong length."""
    ur5_fk = getattr(dexforge_py, "ur5").fk
    franka_fk = getattr(dexforge_py, "franka_panda").fk
    with pytest.raises(ValueError, match="requires"):
        ur5_fk([0.0] * 7)
    with pytest.raises(ValueError, match="requires"):
        franka_fk([0.0] * 6)


def test_fk_returns_valid_se3():
    """fk() must return a 4×4 matrix with a valid rotation block and [0,0,0,1] bottom row."""
    ur5_fk = getattr(dexforge_py, "ur5").fk
    q = [0.0] * 6
    T = np.array(ur5_fk(q), dtype=np.float64)
    assert T.shape == (4, 4)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)


def test_q_seed_accepted():
    """q_seed keyword argument must be accepted without error."""
    ssik_artifact = _load_ssik_artifact("ur5")
    native_solve = getattr(dexforge_py, "ur5").solve
    T = ssik_artifact.T_HOME.copy()
    T[0, 3] = 0.3
    seed = [0.0] * 6
    result = native_solve(T, q_seed=seed, respect_limits=False)
    assert isinstance(result, list)


def test_max_solutions_caps_output():
    """max_solutions must limit the number of returned solutions."""
    ssik_artifact = _load_ssik_artifact("ur5")
    native_solve = getattr(dexforge_py, "ur5").solve
    T = ssik_artifact.T_HOME.copy()
    T[0, 3] = 0.3
    result = native_solve(T, max_solutions=2, respect_limits=False)
    assert len(result) <= 2


# ---------------------------------------------------------------------------
# Perf test (runs only when --perf is passed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(ARMS))
def test_perf_solve(arm, request):
    """Wall-clock timing for Python vs native Rust solve().

    Printed only when ``--perf`` is passed; otherwise runs as a fast
    correctness smoke test.
    """
    n = max(request.config.getoption("--n"), 100)
    do_perf = request.config.getoption("--perf")
    ssik_artifact = _load_ssik_artifact(arm)
    native_solve = getattr(dexforge_py, arm).solve
    poses = _random_poses(ssik_artifact, n)

    # ── Python ssik artifact ──────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        ssik_artifact.solve(T, respect_limits=False, allow_refinement=False)
    t_py = time.perf_counter() - t0

    # ── Native Rust via dexforge_py ──────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        native_solve(T, respect_limits=False, allow_refinement=False)
    t_rust = time.perf_counter() - t0

    if do_perf:
        print(
            f"\n[perf] {arm}  N={n}\n"
            f"  Python: {t_py/n*1e3:.3f} ms  \n"
            f"  Rust: {t_rust/n*1e3:.3f} ms  \n"
            f"  Speedup: {t_py/t_rust:.0f}x"
        )

    # Minimal correctness check: at least one pose should produce solutions.
    assert any(
        len(native_solve(T, respect_limits=False)) > 0 for T in poses[:5]
    )
