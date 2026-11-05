#!/usr/bin/env python3
"""Standalone benchmark: ssik artifact wrappers vs native OEM modules.

This script is shared across OEM repos and discovers available arms from
``ssik.prebuilt._manifest``.
"""

from __future__ import annotations

import argparse
import importlib
import time

import numpy as np

from ssik.prebuilt._manifest import load_manifest

_NATIVE_PACKAGE_BY_ARM = {
    "ur5_ik": "dexforge_ur",
    "piper_ik": "dexforge_agilex",
    "franka_panda_ik": "dexforge_franka",
    "xarm7_ik": "dexforge_ufactory",
    "rizon4_ik": "dexforge_flexiv",
    "rizon10_ik": "dexforge_flexiv",
}


def _base_name(arm_name: str) -> str:
    return arm_name[:-3] if arm_name.endswith("_ik") else arm_name


def _resolve_arms(user_arm: str | None) -> list[str]:
    manifest = load_manifest()
    names = list(manifest)
    if user_arm in (None, "", "all"):
        return names

    if user_arm.endswith("_ik"):
        if user_arm not in manifest:
            raise ValueError(f"unknown arm '{user_arm}', available: {', '.join(names)}")
        return [user_arm]

    candidate = f"{user_arm}_ik"
    if candidate not in manifest:
        raise ValueError(f"unknown arm '{user_arm}', available: {', '.join(names)}")
    return [candidate]


def _load_artifact(arm_name: str):
    return importlib.import_module(f"ssik.prebuilt.{arm_name}")


def _native_package_name() -> str:
    manifest = load_manifest()
    pkgs = {_NATIVE_PACKAGE_BY_ARM[a] for a in manifest if a in _NATIVE_PACKAGE_BY_ARM}
    if len(pkgs) != 1:
        raise RuntimeError(
            "Could not resolve a unique OEM native package for this manifest: "
            f"arms={list(manifest)}"
        )
    return next(iter(pkgs))


def _load_native(arm_name: str):
    native_pkg = importlib.import_module(_native_package_name())

    return getattr(native_pkg, _base_name(arm_name))


def _random_poses(artifact, n: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lims = artifact._JOINT_LIMITS
    poses: list[np.ndarray] = []
    attempts = 0
    while len(poses) < n and attempts < n * 40:
        attempts += 1
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        poses.append(artifact.fk(q))
    return poses


def bench(arm_name: str, n: int) -> None:
    artifact = _load_artifact(arm_name)
    native = _load_native(arm_name)
    poses = _random_poses(artifact, n)
    actual_n = len(poses)

    # Warm up imports / first-call overhead.
    artifact.solve(poses[0], respect_limits=False, allow_refinement=False)
    native.solve(poses[0], respect_limits=False, allow_refinement=False)

    t0 = time.perf_counter()
    for T in poses:
        artifact.solve(T, respect_limits=False, allow_refinement=False)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for T in poses:
        native.solve(T, respect_limits=False, allow_refinement=False)
    t_rust = time.perf_counter() - t0

    sep = "-" * 56
    speedup = t_py / t_rust if t_rust > 0.0 else float("inf")
    print(sep)
    print(f"Arm      : {arm_name}  N={actual_n}")
    print(sep)
    print(f"ssik shim : {t_py/actual_n*1e3:8.3f} ms")
    print(f"native    : {t_rust/actual_n*1e3:8.3f} ms")
    print(f"Speedup   : {speedup:.0f}x (native vs shim)")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ssik wrappers vs native OEM Python module(s)."
    )
    parser.add_argument(
        "--arm",
        default="all",
        help=(
            "Arm name (e.g. ur5, ur5_ik) or 'all' for all arms in"
            " this OEM manifest."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of solve() calls per arm. (default: 100)",
    )
    args = parser.parse_args()

    for arm_name in _resolve_arms(args.arm):
        bench(arm_name, args.n)


if __name__ == "__main__":
    main()
