#!/usr/bin/env python3
"""Pinocchio ground truth for the five-joint SO-101 (`model.rs` tests), from the UNMODIFIED URDF.

Reference model: the upstream LeRobot / SO-ARM100 URDF (`so101_new_calib.urdf`, kept next to
this script), 6 actuated joints with the gripper jaw LOCKED at q = 0 (qd = qdd = 0). Only the five
arm joints are sampled. This is deliberately independent of the derived URDFs (snapped joint
origins, jaw folded into `gripper_link`, virtual wrist joint), so the comparison includes any
error those edits introduce.

Per case (little-endian f64): q[5], qd[5], qdd[5], T[16 row-major], gravity[5],
M[25 row-major], coriolis[5], tau[5]  -> 71 floats. Gravity is (0, 0, -9.81).

Usage:
    pip install pin numpy
    python repos/dexforge-lerobot/rust/src/so101/codegen/generate_reference_data.py
    # writes ../reference_data.bin and ../reference_data.rs (next to model.rs)
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
LIMITS = np.array(
    [(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69), (-1.65806, 1.65806), (-2.74385, 2.84121)]
)
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
N_CASES = 300
FLOATS = 71


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=HERE / "so101_new_calib.urdf")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent)
    ap.add_argument("--ee", default="gripper_frame_link")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    m = pin.buildModelFromUrdf(str(a.urdf))
    d = m.createData()
    assert m.nq == 6 and m.nv == 6, f"expected 5 arm joints + jaw, got nq={m.nq}"
    names = [m.names[i] for i in range(1, m.njoints)]
    assert names[:5] == ARM, names
    fid = m.getFrameId(a.ee)
    rng = np.random.default_rng(a.seed)
    mid, half = LIMITS.mean(axis=1), 0.5 * (LIMITS[:, 1] - LIMITS[:, 0]) * 0.9

    buf = bytearray()
    for _ in range(N_CASES):
        q5 = mid + half * rng.uniform(-1, 1, 5)
        qd5 = rng.uniform(-2, 2, 5)
        qdd5 = rng.uniform(-5, 5, 5)
        q, qd, qdd = (np.r_[x, 0.0] for x in (q5, qd5, qdd5))  # jaw locked
        pin.framesForwardKinematics(m, d, q)
        T = d.oMf[fid].homogeneous
        g = pin.computeGeneralizedGravity(m, d, q)
        M = pin.crba(m, d, q)
        M = np.triu(M) + np.triu(M, 1).T
        c = pin.rnea(m, d, q, qd, np.zeros(6)) - g
        tau = pin.rnea(m, d, q, qd, qdd)
        row = np.concatenate([q5, qd5, qdd5, T.reshape(-1), g[:5], M[:5, :5].reshape(-1), c[:5], tau[:5]])
        assert row.size == FLOATS
        buf += struct.pack(f"<{FLOATS}d", *row)

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_data.bin").write_bytes(bytes(buf))
    (out / "reference_data.rs").write_text(
        f'''//! Auto-generated Pinocchio ground truth for the five-joint SO-101.
//!
//! Reference: the unmodified upstream `so101_new_calib.urdf`, gripper jaw locked at 0, gravity
//! (0, 0, -9.81). Do not edit by hand; regenerate with `codegen/generate_reference_data.py`.
//! Layout per case (f64 little-endian, {FLOATS} values):
//! q[5], qd[5], qdd[5], T[16], gravity[5], M[25], coriolis[5], tau[5].
#![allow(dead_code)]

pub const N_CASES: usize = {N_CASES};
const FLOATS: usize = {FLOATS};
static RAW: &[u8] = include_bytes!("reference_data.bin");

fn f(i: usize, k: usize) -> f64 {{
    let o = (i * FLOATS + k) * 8;
    let mut b = [0u8; 8];
    b.copy_from_slice(&RAW[o..o + 8]);
    f64::from_le_bytes(b)
}}

fn arr<const N: usize>(i: usize, k0: usize) -> [f64; N] {{
    let mut out = [0.0; N];
    for (j, v) in out.iter_mut().enumerate() {{
        *v = f(i, k0 + j);
    }}
    out
}}

pub fn q(i: usize) -> [f64; 5] {{ arr(i, 0) }}
pub fn qd(i: usize) -> [f64; 5] {{ arr(i, 5) }}
pub fn qdd(i: usize) -> [f64; 5] {{ arr(i, 10) }}
pub fn t(i: usize) -> [f64; 16] {{ arr(i, 15) }}
pub fn gravity_torque(i: usize) -> [f64; 5] {{ arr(i, 31) }}
pub fn mass(i: usize) -> [f64; 25] {{ arr(i, 36) }}
pub fn coriolis_torque(i: usize) -> [f64; 5] {{ arr(i, 61) }}
pub fn tau(i: usize) -> [f64; 5] {{ arr(i, 66) }}
'''
    )
    print(f"wrote {out/'reference_data.bin'} ({len(buf)} bytes, {N_CASES} cases) and reference_data.rs")


if __name__ == "__main__":
    main()
