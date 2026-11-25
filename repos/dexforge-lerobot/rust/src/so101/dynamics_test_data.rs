//! Auto-generated Pinocchio dynamics ground truth for the so101.
//!
//! Binary layout of `dynamics_test_data.bin` (little-endian `f64`):
//! for each of [`N_CASES`] cases, concatenated:
//!   `q[5]`, `qd[5]`, `qdd[5]`, `gravity_torque[5]`,
//!   `mass_matrix[5*5]` (row-major), `coriolis_torque[5]`,
//!   `inverse_dynamics_torque[5]`.
//!
//! ## Conventions
//!
//! * **Joint order** (Pinocchio `nv` / URDF actuated): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
//! * **Gravity** base-frame vector: [0.0, 0.0, -9.81] — must match the argument passed to dexforge `gravity` / `inverse_dynamics`.
//!
//! Do not edit by hand. Regenerate with:
//! ```text
//! pip install pin numpy
//! python codegen/generate_dynamics_test_data.py \
//!     --urdf <path/to/so101.urdf> --output-dir <driver-dir> \
//!     --arm so101 --samples 1000 --seed 0
//! ```

#![allow(clippy::all)]

/// Degrees of freedom of the so101.
pub const DOF: usize = 5;

/// Number of Pinocchio ground-truth samples in the binary blob.
pub const N_CASES: usize = 1000;

/// `f64` count per case (see module docs for field layout).
pub const FLOATS_PER_CASE: usize = 55;

/// Gravity vector baked into this dataset (base frame).
pub const GRAVITY: [f64; 3] = [0.0, 0.0, -9.81];

/// Pinocchio / URDF actuated joint names in index order.
pub const JOINT_NAMES: [&str; 5] = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"];

/// Raw little-endian `f64` payload (`N_CASES * FLOATS_PER_CASE` values).
pub static RAW: &[u8] = include_bytes!("dynamics_test_data.bin");

/// One decoded ground-truth sample.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub q: [f64; 5],
    pub qd: [f64; 5],
    pub qdd: [f64; 5],
    pub gravity_torque: [f64; 5],
    pub mass_matrix: [[f64; 5]; 5],
    pub coriolis_torque: [f64; 5],
    pub inverse_dynamics_torque: [f64; 5],
}

#[inline]
fn read_f64(bytes: &[u8], idx: usize) -> f64 {
    let start = idx * 8;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[start..start + 8]);
    f64::from_le_bytes(buf)
}

#[inline]
fn read_vec(bytes: &[u8], offset: usize) -> [f64; 5] {
    let mut out = [0.0f64; 5];
    for i in 0..5 {
        out[i] = read_f64(bytes, offset + i);
    }
    out
}

/// Decode ground-truth case `index` (`0 .. N_CASES`).
pub fn case(index: usize) -> Case {
    assert!(index < N_CASES, "dynamics_test_data case index out of range");
    let base = index * FLOATS_PER_CASE;
    // Sanity: blob length matches the declared layout.
    debug_assert_eq!(RAW.len(), N_CASES * FLOATS_PER_CASE * 8);
    let q = read_vec(RAW, base);
    let qd = read_vec(RAW, base + 5);
    let qdd = read_vec(RAW, base + 10);
    let gravity_torque = read_vec(RAW, base + 15);
    let mut mass_matrix = [[0.0f64; 5]; 5];
    let m_off = base + 20;
    for r in 0..5 {
        for c in 0..5 {
            mass_matrix[r][c] = read_f64(RAW, m_off + r * 5 + c);
        }
    }
    let coriolis_torque = read_vec(RAW, base + 45);
    let inverse_dynamics_torque = read_vec(RAW, base + 50);
    Case { q, qd, qdd, gravity_torque, mass_matrix, coriolis_torque, inverse_dynamics_torque }
}
