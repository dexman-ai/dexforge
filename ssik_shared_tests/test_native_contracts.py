"""Shared native API and parity contracts for OEM Python bindings.

These tests verify the direct Python binding surface (``dexforge_<oem>``)
against the ssik-compatible wrappers exposed through ``ssik.prebuilt``.
"""

from __future__ import annotations

import importlib
import math
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

from ssik.prebuilt._manifest import load_manifest
from codegen.ik_test_data import get_locked_ssik_version, load_arm_test_data

_MANIFEST = load_manifest()

# Resolve which OEM-native package should be imported for this manifest.
_NATIVE_PACKAGE_BY_ARM = {
    "ur5_ik": "dexforge_ur",
    "piper_ik": "dexforge_agilex",
    "franka_panda_ik": "dexforge_franka",
    "xarm7_ik": "dexforge_ufactory",
    "rizon4_ik": "dexforge_flexiv",
    "rizon10_ik": "dexforge_flexiv",
    "z1_ik": "dexforge_unitree",
}

TWO_PI = 2.0 * math.pi
STRICT_TOL = 1e-5
DEDUP_TOL = 1e-3
FK_MATCH_TOL = 1e-5

# Arms where deterministic branch policy should be exact against reference.
EXACT_BASE_ARMS = {"ur5", "franka_panda", "z1"}

# 7R jointlock implementations that may return FK-valid supersets near singularities.
SOUNDNESS_ONLY_ARMS = {"rizon4", "rizon10"}


def _iter_manifest_arm_names(request: pytest.FixtureRequest) -> list[str]:
    selected = request.config.getoption("--arm")
    names = list(_MANIFEST)
    if selected in (None, "", "all"):
        return names
    if selected.endswith("_ik"):
        if selected not in _MANIFEST:
            pytest.skip(f"arm '{selected}' is not in this OEM manifest")
        return [selected]
    candidate = f"{selected}_ik"
    if candidate not in _MANIFEST:
        pytest.skip(f"arm '{selected}' is not in this OEM manifest")
    return [candidate]


def _base_name(arm_name: str) -> str:
    return arm_name[:-3] if arm_name.endswith("_ik") else arm_name


def _wrap_to_pi(a: float) -> float:
    return math.remainder(a, TWO_PI)


def _q_close_mod(a: list[float], b: list[float], tol: float = DEDUP_TOL) -> bool:
    return len(a) == len(b) and all(
        abs(_wrap_to_pi(ai - bi)) <= tol for ai, bi in zip(a, b)
    )


def _load_ssik_artifact(arm_name: str) -> ModuleType:
    return importlib.import_module(f"ssik.prebuilt.{arm_name}")


def _native_package_name() -> str:
    pkgs = {_NATIVE_PACKAGE_BY_ARM[a] for a in _MANIFEST if a in _NATIVE_PACKAGE_BY_ARM}
    if len(pkgs) != 1:
        pytest.skip(
            "Could not resolve a unique OEM native package for this manifest: "
            f"arms={list(_MANIFEST)}"
        )
    return next(iter(pkgs))


def _load_native_module(arm_name: str):
    base = _base_name(arm_name)
    pkg_name = _native_package_name()
    try:
        native_pkg = importlib.import_module(pkg_name)
    except ImportError as e:
        pytest.skip(f"{pkg_name} not importable in this OEM test env: {e}")
    try:
        return getattr(native_pkg, base)
    except AttributeError:
        pytest.skip(f"{pkg_name}.{base} is not exposed in this OEM package")


def _random_poses(artifact: ModuleType, n: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lims = artifact._JOINT_LIMITS
    poses: list[np.ndarray] = []
    attempts = 0
    while len(poses) < n and attempts < n * 40:
        attempts += 1
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        poses.append(artifact.fk(q))
    return poses


def _first_arm_by_dof(dof: int) -> str | None:
    for name, arm in _MANIFEST.items():
        if arm.dof == dof:
            return name
    return None


def test_reference_data_ssik_version(request: pytest.FixtureRequest) -> None:
    locked_version = get_locked_ssik_version()
    for arm_name in _iter_manifest_arm_names(request):
        base = _base_name(arm_name)
        dataset = load_arm_test_data(base)
        assert dataset.ssik_version == locked_version, (
            f"{arm_name}: reference dataset ssik_version={dataset.ssik_version!r} "
            f"does not match SSIK_VERSION.txt ({locked_version!r})"
        )


def test_cross_validate_solve(request: pytest.FixtureRequest) -> None:
    n = int(request.config.getoption("--n"))
    locked_version = get_locked_ssik_version()

    for arm_name in _iter_manifest_arm_names(request):
        base = _base_name(arm_name)
        exact_arm = base in EXACT_BASE_ARMS

        dataset = load_arm_test_data(base)
        assert dataset.ssik_version == locked_version
        native = _load_native_module(arm_name)
        native_solve = native.solve
        native_fk = native.fk
        cases = dataset.cases[:n] if n > 0 else dataset.cases

        mismatches = 0
        for case in cases:
            T_target = np.asarray(case.tf, dtype=np.float64).reshape(4, 4)
            py_qs = case.solutions

            rust_qs = native_solve(
                T_target,
                respect_limits=False,
                allow_refinement=dataset.allow_refinement,
                refinement_max_iters=dataset.refinement_max_iters,
            )

            if exact_arm:
                if len(py_qs) != len(rust_qs):
                    mismatches += 1
                    continue
                for ref_q in py_qs:
                    if not any(_q_close_mod(ref_q, r_q, tol=STRICT_TOL) for r_q in rust_qs):
                        mismatches += 1
                        break
                continue

            if not rust_qs:
                mismatches += 1
                continue

            pose_failed = False
            for r_q in rust_qs:
                fk_residual = np.max(
                    np.abs(np.array(native_fk(r_q), dtype=np.float64) - T_target)
                )
                if fk_residual > FK_MATCH_TOL:
                    mismatches += 1
                    pose_failed = True
                    break

                if base not in SOUNDNESS_ONLY_ARMS:
                    tol = DEDUP_TOL if (case.singular or base == "xarm7") else STRICT_TOL
                    if not any(_q_close_mod(ref_q, r_q, tol=tol) for ref_q in py_qs):
                        mismatches += 1
                        pose_failed = True
                        break

            if pose_failed:
                continue

            if not case.singular:
                if base != "xarm7" and len(py_qs) != len(rust_qs):
                    mismatches += 1
                    continue
                if len(rust_qs) > len(py_qs):
                    mismatches += 1
                    continue
                if len(py_qs) == len(rust_qs):
                    tol = DEDUP_TOL if base == "xarm7" else STRICT_TOL
                    for ref_q in py_qs:
                        if not any(_q_close_mod(ref_q, r_q, tol=tol) for r_q in rust_qs):
                            mismatches += 1
                            break

        assert mismatches == 0, (
            f"{arm_name}: {mismatches}/{len(cases)} poses had mismatched solution sets"
        )


def test_cross_validate_fk(request: pytest.FixtureRequest) -> None:
    n = int(request.config.getoption("--n"))
    locked_version = get_locked_ssik_version()

    for arm_name in _iter_manifest_arm_names(request):
        base = _base_name(arm_name)
        dataset = load_arm_test_data(base)
        assert dataset.ssik_version == locked_version
        native = _load_native_module(arm_name)
        native_fk = native.fk
        cases = dataset.cases[:n] if n > 0 else dataset.cases

        for case in cases:
            q = case.q_fk
            T_py = np.asarray(case.tf, dtype=np.float64).reshape(4, 4)
            T_rust = np.array(native_fk(q), dtype=np.float64)
            np.testing.assert_allclose(
                T_rust, T_py, atol=1e-12,
                err_msg=f"{arm_name}: fk mismatch at q={q}",
            )


def test_unreachable_pose_returns_empty() -> None:
    arm_name = _first_arm_by_dof(6) or next(iter(_MANIFEST), None)
    if arm_name is None:
        pytest.skip("No arms available in this OEM manifest")
    native = _load_native_module(arm_name)
    T = np.eye(4)
    T[:3, 3] = [100.0, 100.0, 100.0]
    solutions = native.solve(T, respect_limits=False)
    assert solutions == []


def test_solve_accepts_list_of_lists() -> None:
    arm_name = _first_arm_by_dof(6) or next(iter(_MANIFEST), None)
    if arm_name is None:
        pytest.skip("No arms available in this OEM manifest")
    native = _load_native_module(arm_name)
    T = [
        [1.0, 0.0, 0.0, 0.3],
        [0.0, 1.0, 0.0, 0.1],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result = native.solve(T, respect_limits=False)
    assert isinstance(result, list)


def test_fk_wrong_dof_raises() -> None:
    arm_6 = _first_arm_by_dof(6)
    arm_7 = _first_arm_by_dof(7)

    if arm_6 is not None:
        fk_6 = _load_native_module(arm_6).fk
        with pytest.raises(ValueError, match="requires"):
            fk_6([0.0] * 7)

    if arm_7 is not None:
        fk_7 = _load_native_module(arm_7).fk
        with pytest.raises(ValueError, match="requires"):
            fk_7([0.0] * 6)

    if arm_6 is None and arm_7 is None:
        pytest.skip("No arms available in this OEM manifest")


def test_fk_returns_valid_se3() -> None:
    arm_name = next(iter(_MANIFEST), None)
    if arm_name is None:
        pytest.skip("No arms available in this OEM manifest")
    native = _load_native_module(arm_name)
    dof = _MANIFEST[arm_name].dof
    q = [0.0] * dof

    T = np.array(native.fk(q), dtype=np.float64)
    assert T.shape == (4, 4)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-15)


def test_q_seed_accepted() -> None:
    arm_name = next(iter(_MANIFEST), None)
    if arm_name is None:
        pytest.skip("No arms available in this OEM manifest")
    artifact = _load_ssik_artifact(arm_name)
    native = _load_native_module(arm_name)
    dof = _MANIFEST[arm_name].dof

    T = np.array(artifact.T_HOME, dtype=np.float64)
    T[0, 3] += 0.05
    seed = [0.0] * dof
    result = native.solve(T, q_seed=seed, respect_limits=False)
    assert isinstance(result, list)


def test_max_solutions_caps_output() -> None:
    arm_name = _first_arm_by_dof(6) or next(iter(_MANIFEST), None)
    if arm_name is None:
        pytest.skip("No arms available in this OEM manifest")
    artifact = _load_ssik_artifact(arm_name)
    native = _load_native_module(arm_name)

    T = np.array(artifact.T_HOME, dtype=np.float64)
    T[0, 3] += 0.05
    result = native.solve(T, max_solutions=2, respect_limits=False)
    assert len(result) <= 2


def test_perf_solve(request: pytest.FixtureRequest) -> None:
    do_perf = bool(request.config.getoption("--perf"))
    n = max(int(request.config.getoption("--n")), 100)

    for arm_name in _iter_manifest_arm_names(request):
        ssik_artifact = _load_ssik_artifact(arm_name)
        native = _load_native_module(arm_name)
        poses = _random_poses(ssik_artifact, n)

        t0 = time.perf_counter()
        for T in poses:
            ssik_artifact.solve(T, respect_limits=False, allow_refinement=False)
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        for T in poses:
            native.solve(T, respect_limits=False, allow_refinement=False)
        t_rust = time.perf_counter() - t0

        if do_perf:
            print(
                f"\\n[perf] {arm_name}  N={n}\\n"
                f"  Python: {t_py/n*1e3:.3f} ms  \\n"
                f"  Rust: {t_rust/n*1e3:.3f} ms  \\n"
                f"  Speedup: {t_py/t_rust:.0f}x"
            )

        assert any(
            len(native.solve(T, respect_limits=False)) > 0 for T in poses[:5]
        )
