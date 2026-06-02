"""Benchmark: Python ssik vs Rust ssik-py IK solver on the UR5.

Reads the saved input poses from ``src/drivers/ur5/test_data.rs`` (the same
dataset used in ``cargo test``) and times both:

  * **Python** – ``ssik.prebuilt.ur5_ik.solve(T, respect_limits=False)``
  * **Rust**   – ``ssik_py.compute_ik(x, y, z, rotation, "ur5")``

Usage::

    # Install dependencies (once):
    pip install ssik scipy numpy
    # Build and install the Rust bindings (from ssik-py/):
    maturin build --release && pip install target/wheels/ssik_py-*.whl

    # Run the benchmark from the repo root:
    python scripts/bench_ik.py

The script prints per-solver wall-clock statistics and the Rust speedup factor.
"""

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
_TEST_DATA_RS = os.path.join(_REPO_ROOT, "src", "drivers", "ur5", "test_data.rs")


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

def _solve_python(poses):
    """Run ssik.prebuilt.ur5_ik.solve on every pose; return list of solution lists."""
    from ssik.prebuilt import ur5_ik  # imported lazily so import time is excluded

    results = []
    for T in poses:
        sols = ur5_ik.solve(T, respect_limits=False)
        results.append([list(map(float, s.q)) for s in sols])
    return results


def _solve_rust(poses):
    """Run ssik_py.compute_ik on every pose; return list of solution lists."""
    import ssik_py  # imported lazily so import time is excluded

    results = []
    for T in poses:
        x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        rot = Rotation.from_matrix(T[:3, :3])
        sols = ssik_py.compute_ik(x, y, z, rot, "ur5")
        results.append(sols)
    return results


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_WARMUP = 3   # number of warm-up passes (results discarded)
_REPS = 10    # number of timed passes


def _time_solver(label: str, fn, poses) -> tuple:
    """Time *fn(poses)*, return (mean_ms, min_ms, max_ms, last_results)."""
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

    per_pose_us = mean_ms * 1000.0 / len(poses)
    print(
        f"  {label:<8s}  mean={mean_ms:8.2f} ms  min={min_ms:8.2f} ms"
        f"  max={max_ms:8.2f} ms  ({per_pose_us:.2f} µs/pose)"
    )
    return mean_ms, min_ms, max_ms, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    if not os.path.exists(_TEST_DATA_RS):
        sys.exit(
            f"ERROR: test-data file not found: {_TEST_DATA_RS}\n"
            "Run scripts/generate_test_data.py first."
        )

    poses = _load_poses(_TEST_DATA_RS)
    if not poses:
        sys.exit("ERROR: no poses parsed from test_data.rs")

    print(f"Loaded {len(poses)} poses from {os.path.relpath(_TEST_DATA_RS)}")
    print(
        f"Timing {_WARMUP} warm-up + {_REPS} timed passes "
        f"({len(poses) * _REPS:,} total IK calls per solver)\n"
    )

    # ------------------------------------------------------------------
    # Run both solvers
    # ------------------------------------------------------------------
    print("=== IK benchmark: Python ssik vs Rust ssik-py ===")
    py_mean, py_min, _, py_results = _time_solver("Python", _solve_python, poses)
    rs_mean, rs_min, _, rs_results = _time_solver("Rust  ", _solve_rust, poses)

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
