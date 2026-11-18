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
from ssik.solvers.ikgeo import _raghavan_roth as RR

_CODEGEN_DIR = Path(__file__).resolve().parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

import _rizon_codegen as cg  # noqa: E402

_DRIVERS_ROOT = _CODEGEN_DIR.parents[1]
_REPO_ROOT = _DRIVERS_ROOT.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codegen.ik_test_data import IKCase, write_ik_test_data  # noqa: E402

N_POSES = 500
SEED = 7
ARMS = ("rizon4", "rizon10")


def _solve_composed_rr(mod, samples, nss, t):
    """Run the 16 composed RR sub-chains in pure Python (`ssik`) on *t*.

    Returns ``(deduped_solutions, is_singular)`` where ``is_singular`` is True
    whenever any sub-chain candidate drifted above ``2e-7`` or was dropped at
    the ``1e-5`` FK gate.
    """
    cands = []
    is_singular = False
    for q_lock, ns in zip(samples, nss):
        t_dh = ns["_T_PRE_INV"] @ t @ ns["_T_POST_INV"]
        p_sin, p_cos, p_one, q_mat = ns["_build_pq_matrices"](
            t_dh[0, 0], t_dh[0, 1], t_dh[0, 2], t_dh[0, 3],
            t_dh[1, 0], t_dh[1, 1], t_dh[1, 2], t_dh[1, 3],
            t_dh[2, 0], t_dh[2, 1], t_dh[2, 2], t_dh[2, 3],
        )
        e_sin, e_cos, e_one = RR.eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
        e_quad, e_lin, e_const = RR.weierstrass_eliminate_trig(e_sin, e_cos, e_one)
        m_quad, m_lin, m_const = RR.build_m_matrix(e_quad, e_lin, e_const)
        roots, vecs = RR.solve_x2_roots_mobius(m_quad, m_lin, m_const)
        q_pinv = np.linalg.pinv(q_mat).astype(np.float64)
        dh_tuple = ns["_DH_TUPLE"]
        meta = ns["_RR_META"]
        theta_off = ns["_DH_THETA_OFFSET"]
        for r, v in zip(roots, vecs):
            bs = RR._back_substitute_inner(
                r, v, p_sin, p_cos, p_one, q_pinv, dh_tuple, t_dh, meta
            )
            if bs is None:
                continue
            if bs[1] > 2e-7:
                is_singular = True
            sub_q = np.asarray(bs[0], dtype=np.float64) - theta_off
            fq = np.empty(7, dtype=np.float64)
            fq[:2] = sub_q[:2]
            fq[2] = float(q_lock)
            fq[3:] = sub_q[2:]
            res = float(np.linalg.norm(mod._fk(fq) - t))
            if res <= 1e-5:
                cands.append((fq, res))
            else:
                is_singular = True

    dedup = []
    for fq, res in cands:
        dup = None
        for j, (eq, _eres) in enumerate(dedup):
            if mod._q_close_wrap(fq, eq, 1e-3):
                dup = j
                break
        if dup is None:
            dedup.append((fq, res))
        elif res < dedup[dup][1]:
            dedup[dup] = (fq, res)

    return [list(map(float, fq)) for fq, _ in dedup], is_singular


def _generate(arm: str) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}_ik")
    driver_dir = _DRIVERS_ROOT / arm
    samples = cg.lock_samples(mod)
    nss = [cg.composed_namespace(mod, s) for s in samples]

    rng = np.random.default_rng(SEED)
    lims = mod._JOINT_LIMITS
    cases = []
    singular = 0
    for _ in range(N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        t = mod._fk(q)
        sols, is_sing = _solve_composed_rr(mod, samples, nss, t)
        if len(sols) <= 1:
            continue
        if is_sing:
            singular += 1
        cases.append(
            IKCase(
                q_fk=q.tolist(),
                tf=np.asarray(t, dtype=np.float64).reshape(-1).tolist(),
                solutions=sols,
                singular=is_sing,
            )
        )

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
        f"[{arm}] kept {len(cases)} multi-solution poses "
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
