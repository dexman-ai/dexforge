"""Generate the Rust reference dataset for the UR5 IK port.

Provenance for ``src/drivers/ur5/test_data.rs``. Reproduces the test data
required by the issue:

  1. Sample 500 valid SE(3) target poses (forward kinematics of random,
     in-limit joint configurations, guaranteeing reachability).
  2. Run the reference Python solver
     ``codegen/ur5_ik.py``'s ``solve(T, respect_limits=False)`` on each pose.
  3. Retain poses with >1 solution and emit ``(pose, solutions)`` pairs as a
     Rust ``static`` slice consumed by the ``#[cfg(test)]`` module.

Usage::

    pip install ssik   # solver runtime (sp6) used by the artifact
    python repos/dexforge-ur/rust/src/drivers/ur5/codegen/generate_test_data.py
    # writes …/ur5/test_data.rs (sibling of ik_solver.rs)

The RNG seed is fixed so regenerating the file is deterministic.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np

_DRIVER_DIR = Path(__file__).resolve().parent.parent
# The reference solver is this driver's own artifact (built from ur5.urdf), not the
# generic UR5 prebuilt shipped with ssik.
_spec = importlib.util.spec_from_file_location("ur5_ik", _DRIVER_DIR / "codegen" / "ur5_ik.py")
ur5_ik = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ur5_ik)
_REPO_ROOT = _DRIVER_DIR.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codegen.ik_test_data import IKCase, write_ik_test_data  # noqa: E402

rng = np.random.default_rng(12345)
# joint limits from artifact
lims = list(ur5_ik._JOINT_LIMITS)

cases=[]
attempts=0
while len(cases) < 500 and attempts < 20000:
    attempts+=1
    q = np.array([rng.uniform(lo,hi) for lo,hi in lims])
    T = ur5_ik.fk(q)
    sols = ur5_ik.solve(T, respect_limits=False)
    if len(sols) > 1:
        qs = [list(map(float, s.q)) for s in sols]
        cases.append(IKCase(q_fk=q.tolist(), tf=T.reshape(-1).tolist(), solutions=qs))

print("attempts",attempts,"cases",len(cases))
ncounts={}
for c in cases:
    ncounts[len(c.solutions)]=ncounts.get(len(c.solutions),0)+1
print("solution-count distribution:",ncounts)

bin_path, rs_path = write_ik_test_data(
    driver_dir=_DRIVER_DIR,
    arm="ur5",
    dof=6,
    cases=cases,
    has_singular_tuple=False,
    allow_refinement=False,
    refinement_max_iters=15,
)
print("wrote", bin_path, "bytes", bin_path.stat().st_size, "and", rs_path)
