"""Pytest suite for the dexforge_py Python interface.

Cross-validates per-arm ``dexforge_py.prebuilt`` modules against the reference
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
# Map canonical arm name -> prebuilt module name (without codebase prefix).
# To add a new arm once its Rust driver is implemented:
#   1. Add an entry here.
#   2. Make sure both ssik.prebuilt and dexforge_py.prebuilt have modules.
ARMS = {
    "ur5": "prebuilt.ur5_ik",
    "piper": "prebuilt.piper_ik",
    "franka_panda": "prebuilt.franka_panda_ik",
    "xarm7": "prebuilt.xarm7_ik",
    "rizon4": "prebuilt.rizon4_ik",
    "rizon10": "prebuilt.rizon10_ik",
}

TWO_PI = 2.0 * math.pi
# Tolerance for wrap-to-pi joint comparison (matches ssik DEFAULT_TOLERANCE_POLICY).
DEDUP_TOL = 1e-3


def _wrap_to_pi(a: float) -> float:
    return math.remainder(a, TWO_PI)


def _q_close_mod(a: List[float], b: List[float], tol: float = DEDUP_TOL) -> bool:
    """True if joint vectors a and b agree modulo 2π within tol."""
    return len(a) == len(b) and all(
        abs(_wrap_to_pi(ai - bi)) <= tol for ai, bi in zip(a, b)
    )


def _load_artifact(arm: str, code_base: str):
    """Import and return the prebuilt module for the given arm from the specified codebase.
    
    Args:
        arm: Canonical arm name (e.g., 'ur5', 'franka_panda')
        code_base: Either 'ssik' or 'dexforge_py'
    
    Returns:
        The imported module with solve() and fk() functions
    """
    module_name = ARMS.get(arm)
    if module_name is None:
        pytest.skip(f"arm '{arm}' is not in the ARMS registry; skipping")
    try:
        return importlib.import_module(f"{code_base}.{module_name}")
    except ImportError as e:
        pytest.skip(f"{code_base}.{module_name} not available for arm '{arm}': {e}")


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
# Test: cross-validation (Python artifact vs Rust dexforge_py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(ARMS))
def test_cross_validate_solve(arm, request):
    """For N random poses, dexforge_py must return the same solution set as
    the Python ssik prebuilt artifact.

    Solution sets are compared modulo 2π with the same dedup tolerance used by
    the solver itself, matching the approach of the Rust reference test suite.
    """
    n = request.config.getoption("--n")
    ssik_artifact = _load_artifact(arm, "ssik")
    dexforge_artifact = _load_artifact(arm, "dexforge_py")
    poses = _random_poses(ssik_artifact, n)

    mismatches = 0
    for T in poses:
        # Python ssik artifact: raw geometric set (no limit filtering).
        # Returns Solution objects.
        py_sols = ssik_artifact.solve(T, respect_limits=False, allow_refinement=False)
        py_qs = [list(map(float, s.q)) for s in py_sols]

        # Rust dexforge_py: same parameters.
        # Returns plain lists.
        rust_qs = dexforge_artifact.solve(T, respect_limits=False, allow_refinement=False)

        # Compare solution counts.
        if len(py_qs) != len(rust_qs):
            mismatches += 1
            continue

        # Every Python solution must appear in the Rust set.
        for ref_q in py_qs:
            if not any(_q_close_mod(ref_q, r_q) for r_q in rust_qs):
                mismatches += 1
                break

    assert mismatches == 0, (
        f"{arm}: {mismatches}/{len(poses)} poses had mismatched solution sets"
    )


@pytest.mark.parametrize("arm", list(ARMS))
def test_cross_validate_fk(arm, request):
    """dexforge_py.fk must return the same FK matrix as the Python artifact's fk."""
    n = request.config.getoption("--n")
    ssik_artifact = _load_artifact(arm, "ssik")
    dexforge_artifact = _load_artifact(arm, "dexforge_py")
    dof = ssik_artifact.DOF
    rng = np.random.default_rng(99)
    lims = ssik_artifact._JOINT_LIMITS

    for _ in range(n):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T_py = ssik_artifact.fk(q)
        T_rust = dexforge_artifact.fk(q)
        np.testing.assert_allclose(
            T_rust, T_py, atol=1e-12,
            err_msg=f"{arm}: fk mismatch at q={q}"
        )


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

def test_unreachable_pose_returns_empty():
    """A pose far outside the workspace should return no solutions."""
    from dexforge_py.prebuilt import ur5_ik
    T = np.eye(4)
    T[:3, 3] = [100.0, 100.0, 100.0]
    solutions = ur5_ik.solve(T, respect_limits=False)
    assert solutions == []


def test_solve_accepts_list_of_lists():
    """solve() must accept a plain Python list-of-lists for T_target."""
    from dexforge_py.prebuilt import ur5_ik
    T = [
        [1.0, 0.0, 0.0, 0.3],
        [0.0, 1.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result = ur5_ik.solve(T, respect_limits=False)
    assert isinstance(result, list)


def test_fk_wrong_dof_raises():
    """fk() must raise ValueError when q has the wrong length."""
    from dexforge_py.prebuilt import ur5_ik, franka_panda_ik
    with pytest.raises(ValueError, match="requires"):
        ur5_ik.fk([0.0] * 7)
    with pytest.raises(ValueError, match="requires"):
        franka_panda_ik.fk([0.0] * 6)


def test_fk_returns_valid_se3():
    """fk() must return a 4×4 matrix with a valid rotation block and [0,0,0,1] bottom row."""
    from dexforge_py.prebuilt import ur5_ik
    q = [0.0] * 6
    T = ur5_ik.fk(q)
    assert T.shape == (4, 4)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)


def test_q_seed_accepted():
    """q_seed keyword argument must be accepted without error."""
    from dexforge_py.prebuilt import ur5_ik
    from ssik.prebuilt import ur5_ik as ssik_ur5_ik
    T = ssik_ur5_ik.T_HOME.copy()
    T[0, 3] = 0.3
    seed = [0.0] * 6
    result = ur5_ik.solve(T, q_seed=seed, respect_limits=False)
    assert isinstance(result, list)


def test_max_solutions_caps_output():
    """max_solutions must limit the number of returned solutions."""
    from dexforge_py.prebuilt import ur5_ik
    from ssik.prebuilt import ur5_ik as ssik_ur5_ik
    T = ssik_ur5_ik.T_HOME.copy()
    T[0, 3] = 0.3
    result = ur5_ik.solve(T, max_solutions=2, respect_limits=False)
    assert len(result) <= 2


# ---------------------------------------------------------------------------
# Perf test (runs only when --perf is passed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(ARMS))
def test_perf_solve(arm, request):
    """Wall-clock timing for Python vs Rust solve().

    Printed only when ``--perf`` is passed; otherwise runs as a fast
    correctness smoke test.
    """
    n = max(request.config.getoption("--n"), 100)
    do_perf = request.config.getoption("--perf")
    ssik_artifact = _load_artifact(arm, "ssik")
    dexforge_artifact = _load_artifact(arm, "dexforge_py")
    poses = _random_poses(ssik_artifact, n)

    # ── Python ssik artifact ──────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        ssik_artifact.solve(T, respect_limits=False, allow_refinement=False)
    t_py = time.perf_counter() - t0

    # ── Rust via dexforge_py ──────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        dexforge_artifact.solve(T, respect_limits=False, allow_refinement=False)
    t_rust = time.perf_counter() - t0

    if do_perf:
        print(
            f"\n[perf] {arm}  N={n}\n"
            f"  Python total={t_py*1e3:.1f} ms  per-IK={t_py/n*1e6:.1f} us\n"
            f"  Rust   total={t_rust*1e3:.1f} ms  per-IK={t_rust/n*1e6:.1f} us\n"
            f"  Speedup: {t_py/t_rust:.1f}x"
        )

    # Minimal correctness check: at least one pose should produce solutions.
    assert any(
        len(dexforge_artifact.solve(T, respect_limits=False)) > 0 for T in poses[:5]
    )
