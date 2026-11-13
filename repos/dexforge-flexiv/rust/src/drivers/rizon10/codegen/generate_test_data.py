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
from pathlib import Path
import sys

import numpy as np

_DRIVERS_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _DRIVERS_ROOT.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codegen.ik_test_data import IKCase, write_ik_test_data  # noqa: E402

N_POSES = 500
SEED = 7
# A pose is a "strict" case when the Rust solver reproduces every reference
# solution (count + value) of the unrefined `allow_refinement=False` output to
# within this tolerance.
REPRO_TOL = 1e-6

ARMS = ("rizon4", "rizon10")


def _rust_solutions(arm, poses):
    """Run every pose through dexforge_flexiv Python bindings for the specified arm.

    Returns a list (one per input pose) of lists of 7-vectors.
    """
    try:
        import dexforge_flexiv
    except ImportError as e:
        raise RuntimeError(
            "dexforge_flexiv is not importable. Install the OEM bindings first, "
            "for example: cd repos/dexforge-flexiv/python && maturin develop --release"
        ) from e

    arm_module = getattr(dexforge_flexiv, arm)
    
    out = []
    for t in poses:
        # dexforge_flexiv expects 4x4 matrix (numpy array or list of lists)
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
    driver_dir = _DRIVERS_ROOT / arm

    rng = np.random.default_rng(SEED)
    lims = mod._JOINT_LIMITS
    candidates = []
    for _ in range(N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        t = mod._fk(q)
        ref_false = [s.q for s in mod.solve(t, respect_limits=False)]
        if len(ref_false) <= 1:
            continue
        candidates.append((q, t, ref_false))

    poses = [t for _, t, _ in candidates]
    rust = _rust_solutions(arm, poses)

    cases = []
    singular = 0
    for (q, t, ref_false), mine in zip(candidates, rust):
        if _reproduces(ref_false, mine):
            sols = [list(map(float, s)) for s in ref_false]
            cases.append(
                IKCase(
                    q_fk=q.tolist(),
                    tf=np.asarray(t, dtype=np.float64).reshape(-1).tolist(),
                    solutions=sols,
                    singular=False,
                )
            )
        else:
            ref_true = [
                s.q for s in mod.solve(t, respect_limits=False, allow_refinement=True)
            ]
            sols = [list(map(float, s)) for s in ref_true]
            cases.append(
                IKCase(
                    q_fk=q.tolist(),
                    tf=np.asarray(t, dtype=np.float64).reshape(-1).tolist(),
                    solutions=sols,
                    singular=True,
                )
            )
            singular += 1

    bin_path, rs_path = write_ik_test_data(
        driver_dir=driver_dir,
        arm=arm,
        dof=7,
        cases=cases,
        has_singular_tuple=True,
        allow_refinement=False,
        refinement_max_iters=15,
    )
    print(
        f"[{arm}] kept {len(cases)}/{len(candidates)} multi-solution poses "
        f"(of {N_POSES} sampled) to {bin_path} and {rs_path}; "
        f"{len(cases) - singular} strict, {singular} singular"
    )


def main(argv) -> None:
    arms = argv[1:] if len(argv) > 1 else list(ARMS)
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")
        _generate(arm)


if __name__ == "__main__":
    main(sys.argv)
