//! Auto-generated Pinocchio ground truth for the five-joint SO-101.
//!
//! Reference: the unmodified upstream `so101_new_calib.urdf`, gripper jaw locked at 0, gravity
//! (0, 0, -9.81). Do not edit by hand; regenerate with `codegen/generate_reference_data.py`.
//! Layout per case (f64 little-endian, 71 values):
//! q[5], qd[5], qdd[5], T[16], gravity[5], M[25], coriolis[5], tau[5].
#![allow(dead_code)]

pub const N_CASES: usize = 300;
const FLOATS: usize = 71;
static RAW: &[u8] = include_bytes!("reference_data.bin");

fn f(i: usize, k: usize) -> f64 {
    let o = (i * FLOATS + k) * 8;
    let mut b = [0u8; 8];
    b.copy_from_slice(&RAW[o..o + 8]);
    f64::from_le_bytes(b)
}

fn arr<const N: usize>(i: usize, k0: usize) -> [f64; N] {
    let mut out = [0.0; N];
    for (j, v) in out.iter_mut().enumerate() {
        *v = f(i, k0 + j);
    }
    out
}

pub fn q(i: usize) -> [f64; 5] {
    arr(i, 0)
}
pub fn qd(i: usize) -> [f64; 5] {
    arr(i, 5)
}
pub fn qdd(i: usize) -> [f64; 5] {
    arr(i, 10)
}
pub fn t(i: usize) -> [f64; 16] {
    arr(i, 15)
}
pub fn gravity_torque(i: usize) -> [f64; 5] {
    arr(i, 31)
}
pub fn mass(i: usize) -> [f64; 25] {
    arr(i, 36)
}
pub fn coriolis_torque(i: usize) -> [f64; 5] {
    arr(i, 61)
}
pub fn tau(i: usize) -> [f64; 5] {
    arr(i, 66)
}
