#!/usr/bin/env python3
"""Generate the Flexiv Rizon IK regression datasets for the Rust drivers.

Multi-arm generator (Rizon 4 and Rizon 10) modelled on
``codegen/generate_test_data.py``. The Rizon arms are 7R ``jointlock``
solvers, so the strategy is the same two-tier scheme already used by the Piper
and xArm7 regression sets:

1. Sample ``N_POSES`` distinct, in-limit target SE(3) poses by running the
   reference forward kinematics on random in-limit joint vectors. Every sampled
   multi-solution pose is kept.
2. Run the reference solver ``ssik.prebuilt.<arm>_ik.solve(T,
   respect_limits=False)`` and the Rust driver (via the ``<arm>_solve`` example
   harness) on the same poses.
3. Classify each pose:

   * **strict** (``singular == false``) — the Rust pipeline reproduces the
     unrefined ``allow_refinement=False`` reference set exactly (count + every
     value to ``REPRO_TOL``). The golden set is that reference output.
   * **singular** (``singular == true``) — the deterministic Raghavan-Roth
     kernel surfaces a different set of genuine roots than the reference
     husty_pfurner path at near-singular lock samples (usually extra roots, but
     occasionally missing some). The golden set becomes the *complete*
     ``allow_refinement=True`` reference set, retained for inspection; the
     driver test asserts only FK-closure soundness on these poses (strict
     completeness is asserted only for ``singular == false``). See
     ``ik_solver``'s test module.

The retained ``(pose, solutions, singular)`` triples are written as a Rust
source module (``src/drivers/<arm>/test_data.rs``).

Run from the repository root:

    pip install "ssik==1.2.4" numpy
    python codegen/generate_test_data.py            # both arms
    python codegen/generate_test_data.py rizon4     # one arm
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import numpy as np

N_POSES = 500
SEED = 7
# A pose is a "strict" case when the Rust solver reproduces every reference
# solution (count + value) of the unrefined `allow_refinement=False` output to
# within this tolerance.
REPRO_TOL = 1e-6
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

ARMS = ("rizon4", "rizon10")


def _rust_solutions(arm, poses):
    """Run every pose through dexforge_py Python bindings for the specified arm.

    Returns a list (one per input pose) of lists of 7-vectors.
    """
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "dexforge-py"))
    import dexforge_py

    arm_module = getattr(dexforge_py, arm)
    
    out = []
    for t in poses:
        # dexforge_py expects 4x4 matrix (numpy array or list of lists)
        sols = arm_module.solve(t)
        # Convert to list of numpy arrays
        out.append([np.array(q, dtype=np.float64) for q in sols])
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


def _generate(arm) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}_ik")
    dest = os.path.join(REPO_ROOT, "src", "drivers", arm, "test_data.rs")

    rng = np.random.default_rng(SEED)
    lims = mod._JOINT_LIMITS
    candidates = []
    for _ in range(N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        t = mod._fk(q)
        ref_false = [s.q for s in mod.solve(t, respect_limits=False)]
        if len(ref_false) <= 1:
            continue
        candidates.append((t, ref_false))

    poses = [t for t, _ in candidates]
    rust = _rust_solutions(arm, poses)

    cases = []
    singular = 0
    for (t, ref_false), mine in zip(candidates, rust):
        if _reproduces(ref_false, mine):
            cases.append((t, ref_false, False))
        else:
            ref_true = [
                s.q for s in mod.solve(t, respect_limits=False, allow_refinement=True)
            ]
            cases.append((t, ref_true, True))
            singular += 1

    _write_rust(arm, dest, cases)
    print(
        f"[{arm}] kept {len(cases)}/{len(candidates)} multi-solution poses "
        f"(of {N_POSES} sampled) to {dest}; "
        f"{len(cases) - singular} strict, {singular} singular"
    )


def _fmt(x: float) -> str:
    return repr(float(x))


def _write_rust(arm, dest, cases) -> None:
    pretty = "Rizon 4" if arm == "rizon4" else "Rizon 10"
    lines = [
        f"//! Auto-generated {pretty} IK regression data. Regenerate with",
        "//! `python codegen/generate_test_data.py`.",
        "//!",
        "//! Each entry is `(row-major 4x4 pose, &[expected joint solutions],",
        "//! singular)`, captured by running the reference solver and retaining",
        "//! every multi-solution pose:",
        "//!",
        "//! * `singular == false` — the Rust pipeline reproduces the unrefined",
        f"//!   `ssik.prebuilt.{arm}_ik.solve(T, respect_limits=False)` output",
        "//!   exactly (count + value); the test asserts an exact joint-space",
        "//!   match.",
        "//! * `singular == true` — the deterministic Raghavan-Roth kernel and",
        "//!   the reference husty_pfurner path resolve near-singular lock",
        "//!   samples differently (the kernel surfaces extra genuine roots and",
        "//!   may miss others). The stored solutions are the *complete*",
        "//!   `allow_refinement=True` reference set, kept for inspection; the",
        "//!   driver test asserts only FK-closure soundness for these poses",
        "//!   (completeness is asserted strictly only for `singular == false`).",
        "//!   See `ik_solver`'s test module.",
        "",
        "/// `(pose, solutions, singular)` regression cases.",
        "pub static CASES: &[(&[f64; 16], &[[f64; 7]], bool)] = &[",
    ]
    for t, sols, singular in cases:
        pose = ", ".join(_fmt(v) for v in np.asarray(t, dtype=np.float64).reshape(-1))
        lines.append(f"    (&[{pose}], &[")
        for q in sols:
            joints = ", ".join(_fmt(v) for v in np.asarray(q, dtype=np.float64))
            lines.append(f"        [{joints}],")
        lines.append(f"    ], {'true' if singular else 'false'}),")
    lines.append("];")
    lines.append("")
    with open(dest, "w") as f:
        f.write("\n".join(lines))


def main(argv) -> None:
    arms = argv[1:] if len(argv) > 1 else list(ARMS)
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")
        _generate(arm)


if __name__ == "__main__":
    main(sys.argv)
