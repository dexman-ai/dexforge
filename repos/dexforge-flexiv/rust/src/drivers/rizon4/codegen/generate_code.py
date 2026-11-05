"""Generate ``consts.rs`` and ``pq.rs`` for the Flexiv Rizon IK drivers.

For each arm (``rizon4``, ``rizon10``) this composes the 16 locked 6R
sub-chains (one per ``_LOCK_SAMPLES`` entry, lock joint ``_LOCK_IDX = 2``, no
chain reversal) through the same RR codegen ssik uses for its tier-2 artifacts
and emits:

* ``src/drivers/<arm>/consts.rs`` — per-sample DH (alpha/a/d/theta_offset),
  ``T_PRE_INV`` / ``T_POST_INV`` bridges, RR back-substitution metadata
  (linearity / left- & right-bilinear / drop joint), plus the full 7R POE
  chain (axes / T_left / T_right), joint limits and the home pose.
* ``src/drivers/<arm>/pq.rs`` — the 16 per-sample ``(P_sin, P_cos, P_one, Q)``
  builders (``sympy.cse``'d, reproduced bit-for-bit) and a function-pointer
  table indexed by lock sample.

Usage::

    pip install "ssik==1.2.4" numpy sympy
    python codegen/generate_code.py            # both arms
    python codegen/generate_code.py rizon4     # one arm

Composition is cached on disk (see ``_rizon_codegen``), so re-runs are cheap.
"""

from __future__ import annotations

import os
import sys

import numpy as np

import _rizon_codegen as cg


def _emit_consts(arm: str) -> None:
    ref = cg.load_ref(arm)
    samples = cg.lock_samples(ref)
    n = len(samples)

    dh_alpha, dh_a, dh_d, dh_off = [], [], [], []
    t_pre_inv, t_post_inv = [], []
    linearity, left_bi, right_bi, drop = [], [], [], []
    for i in range(n):
        ns = cg.composed_namespace(ref, samples[i])
        dh_alpha.append(ns["_DH_ALPHA"])
        dh_a.append(ns["_DH_A"])
        dh_d.append(ns["_DH_D"])
        dh_off.append(ns["_DH_THETA_OFFSET"])
        t_pre_inv.append(ns["_T_PRE_INV"])
        t_post_inv.append(ns["_T_POST_INV"])
        linearity.append(int(ns["_LINEARITY_JOINT"]))
        left_bi.append(tuple(int(x) for x in ns["_LEFT_BILINEAR"]))
        right_bi.append(tuple(int(x) for x in ns["_RIGHT_BILINEAR"]))
        drop.append(int(ns["_DROP_JOINT"]))

    axes7 = [np.asarray(a, dtype=np.float64) for a in ref._JOINT_AXES]
    tl7 = [np.asarray(m, dtype=np.float64) for m in ref._JOINT_T_LEFTS]
    tr7 = [np.asarray(m, dtype=np.float64) for m in ref._JOINT_T_RIGHTS]
    limits = ref._JOINT_LIMITS
    t_home = np.asarray(ref.T_HOME, dtype=np.float64)

    o: list[str] = []
    o.append("//! AUTO-GENERATED baked constants for the Flexiv " + arm.title() + ".")
    o.append("//! Do not edit by hand. Regenerate with")
    o.append("//! `python codegen/generate_code.py " + arm + "`.")
    o.append("//!")
    o.append("//! The arm is solved by `jointlock.seven_r` (lock joint " + str(ref._LOCK_IDX) + ",")
    o.append("//! no chain reversal) wrapping a deterministic Raghavan-Roth 6R inner")
    o.append("//! solver. The 16 per-sample DH / `T_PRE_INV` / `T_POST_INV` / RR-meta")
    o.append("//! entries are the inner-solver geometry; `AXES7` / `T_LEFTS7` /")
    o.append("//! `T_RIGHTS7` are the full 7R chain used for FK closure verification.")
    o.append("")
    o.append("#![allow(clippy::unreadable_literal)]")
    o.append("#![allow(clippy::approx_constant)]")
    o.append("")
    o.append(f"/// Locked joint index (the joint pinned across the {n}-sample sweep).")
    o.append(f"pub(crate) const LOCK_IDX: usize = {ref._LOCK_IDX};")
    o.append("")
    o.append("/// Locked-joint sample schedule (np.linspace over the joint range, endpoint excluded).")
    o.append(f"pub(crate) const LOCK_SAMPLES: [f64; {n}] = [")
    o.append("    " + ", ".join(cg.rfloat(x) for x in samples) + ",")
    o.append("];")
    o.append("")

    def emit_mat6(name, data):
        o.append(f"/// Per lock-sample 6R sub-chain {name.split('_',1)[1].lower()}.")
        o.append(f"pub(crate) const {name}: [[f64; 6]; {n}] = [")
        for row in data:
            o.append("    " + cg.rvec(row) + ",")
        o.append("];")
        o.append("")

    emit_mat6("DH_ALPHA", dh_alpha)
    emit_mat6("DH_A", dh_a)
    emit_mat6("DH_D", dh_d)
    emit_mat6("DH_THETA_OFFSET", dh_off)

    def emit_t(name, data):
        o.append(f"/// Per lock-sample {name} bridge (row-major 4x4).")
        o.append(f"pub(crate) const {name}: [[[f64; 4]; 4]; {n}] = [")
        for m in data:
            o.append("    " + cg.rmat4(m) + ",")
        o.append("];")
        o.append("")

    emit_t("T_PRE_INV", t_pre_inv)
    emit_t("T_POST_INV", t_post_inv)

    o.append("/// Per lock-sample RR linearity joint.")
    o.append(f"pub(crate) const LINEARITY_JOINT: [usize; {n}] = [" + ", ".join(str(x) for x in linearity) + "];")
    o.append("")
    o.append("/// Per lock-sample RR left-bilinear joint pair.")
    o.append(f"pub(crate) const LEFT_BILINEAR: [(usize, usize); {n}] = [")
    o.append("    " + ", ".join(f"({a}, {b})" for a, b in left_bi) + ",")
    o.append("];")
    o.append("")
    o.append("/// Per lock-sample RR right-bilinear joint pair.")
    o.append(f"pub(crate) const RIGHT_BILINEAR: [(usize, usize); {n}] = [")
    o.append("    " + ", ".join(f"({a}, {b})" for a, b in right_bi) + ",")
    o.append("];")
    o.append("")
    o.append("/// Per lock-sample RR dropped joint (recovered from the FK residual).")
    o.append(f"pub(crate) const DROP_JOINT: [usize; {n}] = [" + ", ".join(str(x) for x in drop) + "];")
    o.append("")

    o.append("/// Full 7R joint axes (base frame at q = 0).")
    o.append("pub(crate) const AXES7: [[f64; 3]; 7] = [")
    for a in axes7:
        o.append("    " + cg.rvec(a[:3]) + ",")
    o.append("];")
    o.append("")
    o.append("/// Full 7R per-joint T_left (row-major 4x4).")
    o.append("pub(crate) const T_LEFTS7: [[[f64; 4]; 4]; 7] = [")
    for m in tl7:
        o.append("    " + cg.rmat4(m) + ",")
    o.append("];")
    o.append("")
    o.append("/// Full 7R per-joint T_right (row-major 4x4).")
    o.append("pub(crate) const T_RIGHTS7: [[[f64; 4]; 4]; 7] = [")
    for m in tr7:
        o.append("    " + cg.rmat4(m) + ",")
    o.append("];")
    o.append("")
    o.append("/// Joint angle limits `(lo, hi)` in radians, baked from `_JOINT_LIMITS`.")
    o.append("pub(crate) const JOINT_LIMITS: [(f64, f64); 7] = [")
    for lo, hi in limits:
        o.append(f"    ({cg.rfloat(lo)}, {cg.rfloat(hi)}),")
    o.append("];")
    o.append("")
    o.append("/// Home pose: FK at q = [0; 7] (row-major 4x4). Sanity-check vs the URDF.")
    o.append("#[cfg_attr(not(test), allow(dead_code))]")
    o.append("pub(crate) const T_HOME: [[f64; 4]; 4] = [")
    for i in range(4):
        o.append("    " + cg.rvec(t_home[i]) + ",")
    o.append("];")
    o.append("")

    dest = os.path.join(cg.REPO_ROOT, "src", "drivers", arm, "consts.rs")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("\n".join(o))
    print("wrote", os.path.relpath(dest, cg.REPO_ROOT))


def _emit_pq(arm: str) -> None:
    ref = cg.load_ref(arm)
    samples = cg.lock_samples(ref)
    n = len(samples)

    o: list[str] = []
    o.append("//! Per-sample Raghavan-Roth `(P_sin, P_cos, P_one, Q)` builders for the")
    o.append("//! Flexiv " + arm.title() + ", one per locked 6R sub-chain.")
    o.append("//!")
    o.append("//! Machine-generated (sympy `cse`) and reproduced without manual edits so")
    o.append("//! the floating-point accumulation order matches the Python reference")
    o.append("//! bit-for-bit. Regenerate with `codegen/generate_code.py`.")
    o.append("")
    o.append("// Do not edit by hand. See codegen/generate_code.py.")
    o.append("")
    o.append("/// Signature of a per-sample PQ builder.")
    o.append("pub(crate) type PqBuilder = fn(")
    o.append("    f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64,")
    o.append(") -> ([[f64; 9]; 14], [[f64; 9]; 14], [[f64; 9]; 14], [[f64; 8]; 14]);")
    o.append("")
    for i in range(n):
        src = cg.composed_source(ref, samples[i])
        o.append(cg.render_pq_builder(src, f"build_pq_s{i}"))
        o.append("")
    o.append("/// PQ builder table indexed by lock sample.")
    o.append(f"pub(crate) const BUILD_PQ: [PqBuilder; {n}] = [")
    o.append("    " + ", ".join(f"build_pq_s{i}" for i in range(n)) + ",")
    o.append("];")
    o.append("")

    dest = os.path.join(cg.REPO_ROOT, "src", "drivers", arm, "pq.rs")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        fh.write("\n".join(o))
    print("wrote", os.path.relpath(dest, cg.REPO_ROOT))


def main() -> None:
    arms = sys.argv[1:] or list(cg.ARMS)
    for arm in arms:
        if arm not in cg.ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {list(cg.ARMS)}")
        print(f"=== {arm} ===")
        _emit_consts(arm)
        _emit_pq(arm)


if __name__ == "__main__":
    main()
