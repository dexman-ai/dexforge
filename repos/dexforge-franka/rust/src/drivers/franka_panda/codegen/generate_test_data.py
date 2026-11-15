"""Generate the Rust reference dataset for the Franka Panda IK port.

Provenance for ``src/drivers/franka_panda/test_data.rs``. Reproduces the test
data required by the issue:

  1. Sample 500 valid SE(3) target poses (forward kinematics of random,
     in-limit joint configurations, guaranteeing reachability).
  2. Run the reference Python solver
     ``ssik.prebuilt.franka_panda_ik.solve(T, respect_limits=False)`` on each
     pose.
  3. Retain poses with >1 solution and emit ``(pose, solutions)`` pairs as a
     Rust ``static`` slice consumed by the ``#[cfg(test)]`` module.

Usage::

    pip install ssik
    python codegen/generate_test_data.py

The RNG seed is fixed so regenerating the file is deterministic.
"""

from pathlib import Path
import sys

import numpy as np
from ssik.prebuilt import franka_panda_ik as fp

_DRIVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _DRIVER_DIR.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codegen.ik_test_data import IKCase, write_ik_test_data  # noqa: E402

rng = np.random.default_rng(12345)
lims = fp._JOINT_LIMITS

# Shared Gauss-Newton pose polish. The Rust driver applies the identical
# finite-difference Newton step (see `refine7` in
# src/drivers/franka_panda/ik_solver.rs), so genuinely-identical solutions
# converge to the same isolated pose root well within the 1e-5 test tolerance,
# even at the near-singular configurations where the raw algebraic joint values
# are ill-conditioned (FK residual < 1e-5 yet joint values differ by ~1e-3
# between NumPy and libm arithmetic).
REFINE_ITERS = 60
REFINE_EPS = 1e-7


def _residual12(q, T):
    Tc = fp._fk(np.asarray(q, dtype=np.float64))
    r = np.empty(12)
    r[:9] = (Tc[:3, :3] - T[:3, :3]).reshape(-1)
    r[9:] = Tc[:3, 3] - T[:3, 3]
    return r


def refine7(q, T):
    # Hold the locked joint 4 fixed; refine only the six free joints with
    # adaptive Levenberg-Marquardt damping so the redundant 7R cannot drift
    # along its self-motion manifold and the step stays bounded at near-singular
    # configurations. This mirrors the Rust `refine7` exactly.
    free = [0, 1, 2, 3, 5, 6]
    q = np.array(q, dtype=np.float64)
    lam = 1e-6
    r = _residual12(q, T)
    rn = np.linalg.norm(r)
    for _ in range(REFINE_ITERS):
        if rn < 1e-13:
            break
        J = np.empty((12, 6))
        for c, jidx in enumerate(free):
            qp = q.copy()
            qp[jidx] += REFINE_EPS
            J[:, c] = (_residual12(qp, T) - r) / REFINE_EPS
        JtJ = J.T @ J
        Jtr = J.T @ r
        accepted = False
        for _ in range(12):
            damped = JtJ + lam * np.eye(6)
            try:
                dq = np.linalg.solve(damped, Jtr)
            except np.linalg.LinAlgError:
                lam *= 8.0
                continue
            q_trial = q.copy()
            for c, jidx in enumerate(free):
                q_trial[jidx] -= dq[c]
            r_trial = _residual12(q_trial, T)
            rn_trial = np.linalg.norm(r_trial)
            if rn_trial < rn:
                q = q_trial
                r = r_trial
                rn = rn_trial
                lam = max(lam * 0.5, 1e-12)
                accepted = True
                break
            lam *= 8.0
            if lam > 1e8:
                break
        if not accepted:
            break
    return q


cases = []
attempts = 0
while len(cases) < 500 and attempts < 20000:
    attempts += 1
    q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
    T = fp.fk(q)
    sols = fp.solve(T, respect_limits=False)
    if len(sols) > 1:
        qs = [list(map(float, refine7(s.q, T))) for s in sols]
        cases.append(IKCase(q_fk=q.tolist(), tf=T.reshape(-1).tolist(), solutions=qs))

print("attempts", attempts, "cases", len(cases))
ncounts = {}
for c in cases:
    ncounts[len(c.solutions)] = ncounts.get(len(c.solutions), 0) + 1
print("solution-count distribution:", dict(sorted(ncounts.items())))

bin_path, rs_path = write_ik_test_data(
    driver_dir=_DRIVER_DIR,
    arm="franka_panda",
    dof=7,
    cases=cases,
    has_singular_tuple=False,
    allow_refinement=True,
    refinement_max_iters=REFINE_ITERS,
)
print("wrote", bin_path, "bytes", bin_path.stat().st_size, "and", rs_path)
