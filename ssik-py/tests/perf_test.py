#!/usr/bin/env python3
"""Standalone performance benchmark: Python ssik artifact vs Rust ssik_py.

Usage::

    # Run with defaults (ur5, 100 iterations):
    python ssik-py/tests/perf_test.py

    # Specify arm and iteration count:
    python ssik-py/tests/perf_test.py --arm franka_panda --n 200

    # Run all known arms:
    python ssik-py/tests/perf_test.py --arm all

The script prints total and per-IK wall-clock time for both the Python
ssik prebuilt artifact and the Rust ssik_py bindings.

To add a new arm to the benchmark, add an entry to ARMS below (once the
corresponding Rust driver and ssik_py dispatch arm are implemented).
"""

from __future__ import annotations

import argparse
import importlib
import time
from typing import List

import numpy as np

import ssik_py

# ---------------------------------------------------------------------------
# Arm registry: canonical name -> ssik.prebuilt module name.
# Add new arms here once their Rust drivers are implemented.
# ---------------------------------------------------------------------------
ARMS = {
    "ur5": "ssik.prebuilt.ur5_ik",
    "franka_panda": "ssik.prebuilt.franka_panda_ik",
}


def _load_artifact(arm: str):
    module_name = ARMS[arm]
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
    ssik_py.solve(arm, poses[0], respect_limits=False, allow_refinement=False)

    # ── Python artifact ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        artifact.solve(T, respect_limits=False, allow_refinement=False)
    t_py = time.perf_counter() - t0

    # ── Rust via ssik_py ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    for T in poses:
        ssik_py.solve(arm, T, respect_limits=False, allow_refinement=False)
    t_rust = time.perf_counter() - t0

    sep = "-" * 50
    print(sep)
    print(f"Arm      : {arm}   N={actual_n}")
    print(sep)
    print(
        f"Python   : total={t_py*1e3:8.1f} ms   per-IK={t_py/actual_n*1e6:7.1f} µs"
    )
    print(
        f"Rust     : total={t_rust*1e3:8.1f} ms   per-IK={t_rust/actual_n*1e6:7.1f} µs"
    )
    print(f"Speedup  : {t_py/t_rust:.2f}x  (Rust vs Python)")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Python ssik artifact vs Rust ssik_py bindings."
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
