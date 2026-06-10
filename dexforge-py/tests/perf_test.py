#!/usr/bin/env python3
"""Standalone performance benchmark: Python ssik artifact vs Rust dexforge_py.

Usage::

    # Run with defaults (ur5, 100 iterations):
    python dexforge-py/tests/perf_test.py

    # Specify arm and iteration count:
    python dexforge-py/tests/perf_test.py --arm franka_panda --n 200

    # Run all known arms:
    python dexforge-py/tests/perf_test.py --arm all

The script prints total and per-IK wall-clock time for both the Python
ssik prebuilt artifact and the Rust dexforge_py bindings.

To add a new arm to the benchmark, add an entry to ARMS below (once the
corresponding Rust driver and dexforge_py dispatch arm are implemented).
"""

from __future__ import annotations

import argparse
import importlib
import time
from typing import List

import numpy as np

import dexforge_py

# ---------------------------------------------------------------------------
# Canonical arm names to benchmark.
# The ssik artifact module name is derived as ssik.prebuilt.<arm>_ik.
# ---------------------------------------------------------------------------
ARMS = ("ur5", "piper", "franka_panda", "xarm7", "rizon4", "rizon10")


def _load_artifact(arm: str):
    module_name = f"ssik.prebuilt.{arm}_ik"
    return importlib.import_module(module_name)


def _random_poses(artifact, n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate n SE(3) poses via FK on random in-limit configurations."""
    rng = np.random.default_rng(seed)
    lims = artifact._JOINT_LIMITS
    poses = []
    attempts = 0
    while len(poses) < n and attempts < n * 40:
        attempts += 1
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        poses.append(artifact.fk(q))
    return poses


def bench(arm: str, n: int) -> None:
    artifact = _load_artifact(arm)
    poses = _random_poses(artifact, n)
    actual_n = len(poses)

    # Warm up JIT / module import overhead with one call each.
    artifact.solve(poses[0], respect_limits=False, allow_refinement=False)
    getattr(dexforge_py, arm).solve(
        poses[0], respect_limits=False, allow_refinement=False
    )

    # ── Python artifact ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        artifact.solve(T, respect_limits=False, allow_refinement=False)
    t_py = time.perf_counter() - t0

    # ── Rust via dexforge_py ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        getattr(dexforge_py, arm).solve(
            T, respect_limits=False, allow_refinement=False
        )
    t_rust = time.perf_counter() - t0

    sep = "-" * 50
    print(sep)
    print(f"Arm      : {arm}   N={actual_n}")
    print(sep)
    print(
        f"Python   : {t_py/actual_n*1e3:8.3f} ms"
    )
    print(
        f"Rust     : {t_rust/actual_n*1e3:8.3f} ms"
    )
    speedup_str = f"{t_py/t_rust:.0f}x"
    speedup_str = f"\033[92m{speedup_str}\033[0m" if t_py > t_rust else f"\033[91m{speedup_str}\033[0m"
    print(f"Speedup  : {speedup_str} (Rust vs Python)")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Python ssik artifact vs Rust dexforge_py bindings."
    )
    parser.add_argument(
        "--arm",
        default="ur5",
        choices=list(ARMS) + ["all"],
        help="Arm to benchmark, or 'all' for every known arm. (default: ur5)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of solve() calls per arm. (default: 100)",
    )
    args = parser.parse_args()

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        bench(arm, args.n)


if __name__ == "__main__":
    main()
