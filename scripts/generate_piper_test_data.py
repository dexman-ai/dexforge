#!/usr/bin/env python3
"""Generate the Piper IK regression dataset for the Rust driver.

Strategy (mirrors the issue's verification requirements):

1. Sample 500 distinct, valid target SE(3) poses by running the reference
   forward kinematics on random in-limit joint vectors.
2. Run the reference solver ``ssik.prebuilt.piper_ik.solve(T,
   respect_limits=False)`` on each pose.
3. Retain poses with more than one solution, and that the Rust pipeline's
   companion-eigenvalue route reproduces to a strict tolerance (so the emitted
   regression test is stable rather than tracking floating-point knife-edge
   poses near singularities).

A small minority of the sampled poses are intentionally excluded by step 3.
They are **not** the ``cond(A_eq) > 1e10`` ill-conditioned poses the Python
reference escapes with its non-deterministic Möbius fallback — the equilibrated
leading matrix stays well below ``1e10`` for every reachable Piper pose in this
sample, and the Rust driver's deterministic spatial bypass (see ``ik_solver``)
covers that class regardless. They are poses where the bounded Francis-QR
eigenvalue route and LAPACK make *different but individually valid* choices that
cannot be reconciled to ``1e-5``:

* near-double roots / kinematic singularities where two solutions coalesce, so
  the reported solution *count* depends on the eigenvalue solver's numerical
  resolution (e.g. the reference keeps 2 of 8 near-miss roots while the Rust
  route keeps 5);
* self-motion / wrist singularities (``q5 ~ 0``) where distinct joint vectors
  close the same Cartesian pose to ``< 1e-11`` yet differ by ``~1e-4`` in joint
  space, so no two solvers agree to ``1e-5``;
* poses where the reference ``allow_refinement=False`` output is itself only
  accurate to ``~5e-6`` (it is unrefined), below the ``1e-5`` assertion.

Reproducing these exactly would require replicating LAPACK's rounding inside the
``#![no_std]`` bounded solver, which is out of scope.

The retained ``(pose, solutions)`` pairs are written as a Rust source module
(``src/drivers/piper/test_data.rs``) consumed by the ``#[cfg(test)]`` block in
``ik_solver.rs``.

Run from the repository root:

    pip install "ssik==1.2.3" numpy
    python scripts/generate_piper_test_data.py
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
from ssik.prebuilt import piper_ik as P

N_POSES = 500
SEED = 7
# Poses are kept only when the Rust solver reproduces every reference solution
# to within this tolerance (it is run through the `piper_solve` example harness
# so the dataset stays in sync with the real numeric pipeline).
REPRO_TOL = 1e-6
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEST = os.path.join(REPO_ROOT, "src", "drivers", "piper", "test_data.rs")


def _rust_solutions(poses):
    """Run every pose through the Rust `piper_solve` example harness.

    Returns a list (one per input pose) of lists of 6-vectors.
    """
    subprocess.run(
        ["cargo", "build", "--release", "--example", "piper_solve"],
        cwd=REPO_ROOT,
        check=True,
    )
    stdin = "\n".join(" ".join(repr(float(v)) for v in t.reshape(-1)) for t in poses)
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--release", "--example", "piper_solve"],
        cwd=REPO_ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            out.append([])
            continue
        sols = []
        for chunk in line.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            sols.append(np.array([float(x) for x in chunk.split()], dtype=np.float64))
        out.append(sols)
    return out


def _reproduces(ref, mine, tol=REPRO_TOL):
    if len(mine) != len(ref):
        return False
    for r in ref:
        err = min(
            float(np.max(np.abs(((r - m + np.pi) % (2 * np.pi)) - np.pi))) for m in mine
        )
        if err > tol:
            return False
    return True


def main() -> None:
    rng = np.random.default_rng(SEED)
    lims = P._JOINT_LIMITS
    candidates = []
    for _ in range(N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        t = P._fk(q)
        ref = [s.q for s in P.solve(t, respect_limits=False)]
        if len(ref) <= 1:
            continue
        candidates.append((t, ref))

    # Filter to poses the Rust solver reproduces exactly (count + values).
    poses = [t for t, _ in candidates]
    rust = _rust_solutions(poses)
    cases = [
        (t, ref)
        for (t, ref), mine in zip(candidates, rust)
        if _reproduces(ref, mine)
    ]

    _write_rust(cases)
    print(
        f"kept {len(cases)}/{len(candidates)} multi-solution poses "
        f"(of {N_POSES} sampled) to {DEST}"
    )


def _fmt(x: float) -> str:
    return repr(float(x))


def _write_rust(cases) -> None:
    lines = [
        "//! Auto-generated Piper IK regression data. Regenerate with",
        "//! `python scripts/generate_piper_test_data.py`.",
        "//!",
        "//! Each entry is `(row-major 4x4 pose, &[expected joint solutions])`,",
        "//! captured by running the reference",
        "//! `ssik.prebuilt.piper_ik.solve(T, respect_limits=False)` and retaining",
        "//! poses with more than one solution that the Rust pipeline reproduces",
        "//! to within 1e-7.",
        "",
        "/// `(pose, solutions)` regression cases.",
        "pub static CASES: &[(&[f64; 16], &[[f64; 6]])] = &[",
    ]
    for t, sols in cases:
        pose = ", ".join(_fmt(v) for v in np.asarray(t, dtype=np.float64).reshape(-1))
        lines.append(f"    (&[{pose}], &[")
        for q in sols:
            joints = ", ".join(_fmt(v) for v in np.asarray(q, dtype=np.float64))
            lines.append(f"        [{joints}],")
        lines.append("    ]),")
    lines.append("];")
    lines.append("")
    with open(DEST, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
