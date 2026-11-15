"""Generate the shared ssik reference dataset for the Unitree Z1 IK port."""

from pathlib import Path
import sys

import numpy as np
from ssik.prebuilt import z1_ik

_DRIVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _DRIVER_DIR.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codegen.ik_test_data import IKCase, write_ik_test_data  # noqa: E402

rng = np.random.default_rng(12345)
lims = z1_ik._JOINT_LIMITS

cases = []
attempts = 0
while len(cases) < 500 and attempts < 20000:
    attempts += 1
    q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
    T = z1_ik.fk(q)
    sols = z1_ik.solve(T, respect_limits=False)
    if len(sols) > 1:
        qs = [list(map(float, s.q)) for s in sols]
        cases.append(IKCase(q_fk=q.tolist(), tf=T.reshape(-1).tolist(), solutions=qs))

print("attempts", attempts, "cases", len(cases))
ncounts = {}
for c in cases:
    ncounts[len(c.solutions)] = ncounts.get(len(c.solutions), 0) + 1
print("solution-count distribution:", ncounts)

bin_path, rs_path = write_ik_test_data(
    driver_dir=_DRIVER_DIR,
    arm="z1",
    dof=6,
    cases=cases,
    has_singular_tuple=False,
    allow_refinement=False,
    refinement_max_iters=15,
)
print("wrote", bin_path, "bytes", bin_path.stat().st_size, "and", rs_path)
