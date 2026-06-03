"""Benchmark: Python ssik vs Rust ssik-py IK solver.

Reads the saved input poses from ``src/drivers/<arm>/test_data.rs`` (the same
dataset used in ``cargo test``) and times both:

  * **Python** – ``ssik.prebuilt.<arm>_ik.solve(T, respect_limits=False)``
  * **Rust**   – ``ssik_py.compute_ik(x, y, z, rotation, arm)``

Usage::

    # Install dependencies (once):
    pip install ssik scipy numpy
    # Build and install the Rust bindings (from ssik-py/):
    maturin build --release && pip install target/wheels/ssik_py-*.whl

    # Run the benchmark from the repo root:
    python scripts/bench_ik.py           # defaults to --arm ur5
    python scripts/bench_ik.py --arm ur5

The script prints per-solver wall-clock statistics and the Rust speedup factor.
"""

import argparse
import importlib
import os
import re
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Parse poses from the Rust test-data file
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _test_data_path(arm: str) -> str:
    """Return the path to the Rust test-data file for the given arm."""
    return os.path.join(_REPO_ROOT, "src", "drivers", arm, "test_data.rs")


def _load_poses(path: str) -> list:
    """Return a list of 4×4 numpy arrays parsed from test_data.rs."""
    with open(path) as f:
        content = f.read()

    # Each data line looks like:
    #   (&[-0.690984..., ..., 1.0], &[[q0..q5], ...]),
    # The static type line has `(&[f64; 16]` which we must skip.
    # Match `(&[` followed immediately by a digit, minus sign or dot (not a letter).
    pattern = re.compile(r"\(&\[([-\d][^\]]*)\]")
    poses = []
    for m in pattern.finditer(content):
        raw = m.group(1)
        # Skip Rust type annotations like "f64; 16"
        if ";" in raw:
            continue
        floats = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(floats) == 16:
            poses.append(np.array(floats, dtype=np.float64).reshape(4, 4))
    return poses


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

def _solve_python(poses, arm: str):
    """Run ssik.prebuilt.<arm>_ik.solve on every pose; return list of solution lists."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm}_ik")

    results = []
    for T in poses:
        sols = mod.solve(T, respect_limits=False)
        results.append([list(map(float, s.q)) for s in sols])
    return results


def _solve_rust(poses, arm: str):
    """Run ssik_py.compute_ik on every pose; return list of solution lists."""
    import ssik_py  # imported lazily so import time is excluded

    results = []
    for T in poses:
        x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        rot = Rotation.from_matrix(T[:3, :3])
        sols = ssik_py.compute_ik(x, y, z, rot, arm)
        results.append(sols)
    return results


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_WARMUP = 3   # number of warm-up passes (results discarded)
_REPS = 10    # number of timed passes


def _time_solver(label: str, fn, poses) -> tuple:
    """Time *fn(poses)*, return (mean_ms, min_ms, max_ms, stdev_ms, last_results)."""
    results = None
    for _ in range(_WARMUP):
        results = fn(poses)

    times = []
    for _ in range(_REPS):
        t0 = time.perf_counter()
        results = fn(poses)
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000.0 for t in times]
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
    stdev_ms = variance ** 0.5

    per_pose_us = mean_ms * 1000.0 / len(poses)
    print(
        f"  {label:<8s}  mean={mean_ms:8.2f} ms  stdev={stdev_ms:6.2f} ms"
        f"  min={min_ms:8.2f} ms  max={max_ms:8.2f} ms  ({per_pose_us:.2f} µs/pose)"
    )
    return mean_ms, min_ms, max_ms, stdev_ms, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python ssik vs Rust ssik-py IK solver."
    )
    parser.add_argument(
        "--arm",
        default="ur5",
        help="Robot arm identifier (default: ur5). Must match a ssik.prebuilt.<arm>_ik "
             "module and a src/drivers/<arm>/test_data.rs file.",
    )
    args = parser.parse_args()
    arm = args.arm

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    test_data_rs = _test_data_path(arm)
    if not os.path.exists(test_data_rs):
        sys.exit(
            f"ERROR: test-data file not found: {test_data_rs}\n"
            "Run scripts/generate_test_data.py first."
        )

    poses = _load_poses(test_data_rs)
    if not poses:
        sys.exit("ERROR: no poses parsed from test_data.rs")

    print(f"Arm: {arm}")
    print(f"Loaded {len(poses)} poses from {os.path.relpath(test_data_rs)}")
    print(
        f"Timing {_WARMUP} warm-up + {_REPS} timed passes "
        f"({len(poses) * _REPS:,} total IK calls per solver)\n"
    )

    # ------------------------------------------------------------------
    # Run both solvers
    # ------------------------------------------------------------------
    print(f"=== IK benchmark: Python ssik vs Rust ssik-py  [{arm}] ===")
    py_mean, py_min, _, _py_stdev, py_results = _time_solver(
        "Python", lambda p: _solve_python(p, arm), poses
    )
    rs_mean, rs_min, _, _rs_stdev, rs_results = _time_solver(
        "Rust  ", lambda p: _solve_rust(p, arm), poses
    )

    # ------------------------------------------------------------------
    # Speedup
    # ------------------------------------------------------------------
    speedup_mean = py_mean / rs_mean
    speedup_min = py_min / rs_min
    print(f"\n  Speedup (mean): {speedup_mean:.1f}×  (min: {speedup_min:.1f}×)")

    # ------------------------------------------------------------------
    # Quick sanity check: solution counts must agree
    # ------------------------------------------------------------------
    mismatches = 0
    for i, (py_sols, rs_sols) in enumerate(zip(py_results, rs_results)):
        if len(py_sols) != len(rs_sols):
            mismatches += 1
            if mismatches <= 3:
                print(
                    f"  [warn] pose {i}: Python={len(py_sols)} sols, "
                    f"Rust={len(rs_sols)} sols"
                )
    if mismatches:
        print(f"  [warn] {mismatches}/{len(poses)} poses had mismatched solution counts")
    else:
        print(f"\n  Solution counts agree for all {len(poses)} poses. ✓")


if __name__ == "__main__":
    main()
