//! Analytical inverse kinematics for the Unitree **Z1**.
//!
//! Line-for-line translation of the Python `ssik` prebuilt artifact
//! `ssik/prebuilt/z1_ik.py` (solver `ikgeo.three_parallel`, tier 0) into a
//! deterministic, `#![no_std]`, zero-allocation Rust implementation.
//!
//! The Z1 has the same UR-class structure as the UR5 — three consecutive
//! parallel axes at joints 2/3/4 — so it shares the exact same closed form:
//! subproblem **SP6** for the coupled pair `(q1, q5)` followed by the inlined
//! `SP1 + SP3 + SP1` chain. The arm-agnostic SP6 solver, FK chain and dedup
//! live in [`crate::math::ssik_core`] / [`crate::math::kinematics`]; only the
//! baked per-arm constants and the SP1/SP3 closed forms differ from the UR5.
//!
//! ## Equivalence to the Python reference
//!
//! [`compute_ik`] reproduces `z1_ik.solve(T, respect_limits=False)`: the raw
//! geometric solution set, FK-verified against the target pose and
//! de-duplicated modulo `2*pi`.

use nalgebra::SMatrix;

use crate::math::kinematics::{eye_with, fk_residual, q_close_wrap};
use crate::math::ssik_core::{sp6_solve, wrap_to_pi};

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the Z1.
pub const DOF: usize = 6;

// --- tolerances (ssik DEFAULT_TOLERANCE_POLICY) ---
const DEDUP_TOL: f64 = 1e-3; // subproblem_dedup
const FK_ATOL: f64 = 1e-5; // FK closure tolerance (== subproblem_numerical)

// --- algebraic-block degeneracy constants from z1_ik.py ---
const DEG_SQ: f64 = 1e-16;
const FEAS_TOL: f64 = 1e-8;

// --- baked KinBody constants (for FK verification) ---
const JOINT_AXES: [[f64; 3]; 6] = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
];

const T_LEFTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0585),
    eye_with(0.0, 0.0, 0.045000000000000005),
    eye_with(-0.35, 0.0, 0.0),
    eye_with(0.218, 0.0, 0.056999999999999995),
    eye_with(0.07, 0.0, 0.0),
    eye_with(0.0492, 0.0, 0.0),
];

const T_RIGHTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
];

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Computes the complete analytical inverse kinematics for the Z1.
///
/// `target_tf` is the 4x4 SE(3) pose of the end-effector expressed in the base
/// frame. Returns up to 16 candidate joint vectors `[q1, .., q6]` (the Z1
/// yields at most 8); empty slots are [`None`]. The set is FK-verified to
/// within [`FK_ATOL`] and de-duplicated modulo `2*pi`, matching
/// `z1_ik.solve(T, respect_limits=False)`.
pub fn compute_ik(target_tf: &SMatrix<f64, 4, 4>) -> [Option<[f64; 6]>; 16] {
    let mut solutions: [Option<[f64; 6]>; 16] = [None; 16];
    let mut solution_count = 0usize;

    // Raw algebraic candidates (stack array, at most 8 for the Z1).
    let mut cands = [[0.0f64; 6]; 8];
    let cand_count = solve_algebraic(target_tf, &mut cands);

    for ci in 0..cand_count {
        let q = cands[ci];
        // Reject non-finite candidates (mirrors np.isfinite filter).
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        // FK closure check.
        let residual = fk_residual(&q, &JOINT_AXES, &T_LEFTS, &T_RIGHTS, target_tf);
        if residual > FK_ATOL {
            continue;
        }
        // Wrap-to-pi de-duplication against accepted solutions.
        let mut dup = false;
        for s in solutions.iter().take(solution_count) {
            if let Some(existing) = s {
                if q_close_wrap(&q, existing, DEDUP_TOL) {
                    dup = true;
                    break;
                }
            }
        }
        if dup {
            continue;
        }
        if solution_count < 16 {
            solutions[solution_count] = Some(q);
            solution_count += 1;
        }
    }

    solutions
}

// ---------------------------------------------------------------------------
// Algebraic candidate generation (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Fills `out` with the raw algebraic IK candidates and returns the count.
fn solve_algebraic(t: &SMatrix<f64, 4, 4>, out: &mut [[f64; 6]; 8]) -> usize {
    let r_00 = t[(0, 0)];
    let r_01 = t[(0, 1)];
    let r_02 = t[(0, 2)];
    let r_10 = t[(1, 0)];
    let r_11 = t[(1, 1)];
    let r_12 = t[(1, 2)];
    let r_20 = t[(2, 0)];
    let p_x = t[(0, 3)];
    let p_y = t[(1, 3)];
    let p_z = t[(2, 3)];

    let mut count = 0usize;

    // T_target-derived SP6 inputs.
    let p_16 = [p_x, p_y, p_z - 0.0585];
    let r_06_axes5 = [1.0 * r_00, 1.0 * r_10, 1.0 * r_20];

    const H_SP_0: [f64; 3] = [0.0, 1.0, 0.0];
    const K_SP_0: [f64; 3] = [-0.0, -0.0, -1.0];
    const K_SP_1: [f64; 3] = [0.0, 0.0, 1.0];
    const NEG_P_5: [f64; 3] = [-0.0492, -0.0, -0.0];
    const NEG_AXES_5: [f64; 3] = [-1.0, -0.0, -0.0];

    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, 0.0, 0.0, &mut theta15);

    for si in 0..n15 {
        let (q1, q5) = theta15[si];

        // SP1 for theta14 = q1+q2+q3+q4 (sum of parallel-axis rotations).
        let th14_x0 = 1.0 * libm::cos(q5);
        let theta14 = libm::atan2(
            -r_20 * th14_x0,
            th14_x0 * (1.0 * r_00 * libm::cos(q1) + 1.0 * r_10 * libm::sin(q1)),
        );

        // SP1 for q6 (wrist roll-2): closed-form atan2.
        let q6_x0 = 1.0 * libm::sin(q1);
        let q6_x1 = libm::cos(q1);
        let q6_x2 = 1.0 * libm::cos(q5);
        let q6 = libm::atan2(
            -q6_x2 * (-q6_x0 * r_02 + 1.0 * q6_x1 * r_12),
            q6_x2 * (-q6_x0 * r_01 + 1.0 * q6_x1 * r_11),
        );

        // d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4].
        let dinr_x0 = libm::cos(theta14);
        let dinr_x1 = libm::cos(q1);
        let dinr_x2 = 1.0 * libm::sin(q1);
        let dinr_x3 = 0.0492 * libm::cos(q5);
        let dinr_x4 = libm::sin(theta14);
        let d_inner_x = -dinr_x0 * dinr_x3 - 0.07 * dinr_x0 + dinr_x1 * p_x + dinr_x2 * p_y;
        let d_inner_y = dinr_x1 * p_y - dinr_x2 * p_x - 0.0492 * libm::sin(q5);
        let d_inner_z = dinr_x3 * dinr_x4 + 0.07 * dinr_x4 + 1.0 * p_z - 0.1035;
        let d_elbow =
            libm::sqrt(d_inner_x * d_inner_x + d_inner_y * d_inner_y + d_inner_z * d_inner_z);

        // SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        let q3_r_sq = 0.00621969250000000_f64;
        let q3_rhs = 0.0866365 - 0.5 * d_elbow * d_elbow;
        let q3_phi = 0.255742510054401_f64;
        let theta_q3_plus;
        let theta_q3_minus;
        if q3_r_sq < DEG_SQ {
            theta_q3_plus = 0.0;
            theta_q3_minus = 0.0;
        } else {
            let q3_r = libm::sqrt(q3_r_sq);
            if libm::fabs(q3_rhs) > q3_r + FEAS_TOL {
                let p = if q3_rhs > 0.0 { q3_phi } else { q3_phi + PI };
                theta_q3_plus = p;
                theta_q3_minus = p;
            } else {
                let clipped = (q3_rhs / q3_r).clamp(-1.0, 1.0);
                let delta = libm::acos(clipped);
                theta_q3_plus = q3_phi + delta;
                theta_q3_minus = q3_phi - delta;
            }
        }

        for q3 in [theta_q3_plus, theta_q3_minus] {
            // SP1 for q2 (shoulder pitch): closed-form atan2.
            let q2_x0 = libm::sin(q3);
            let q2_x1 = libm::cos(q3);
            let q2_x2 = -0.218 * q2_x0 + 0.057 * q2_x1;
            let q2_x3 = 0.057 * q2_x0 + 0.218 * q2_x1 - 0.35;
            let q2 = libm::atan2(
                d_inner_x * q2_x2 - d_inner_z * q2_x3,
                d_inner_x * q2_x3 + d_inner_z * q2_x2,
            );
            let q4 = wrap_to_pi(theta14 - q2 - q3);

            if count < 8 {
                out[count] = [q1, q2, q3, q4, q5, q6];
                count += 1;
            }
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrix;
    use std::vec::Vec;

    fn tf_from_row_major(a: &[f64; 16]) -> SMatrix<f64, 4, 4> {
        let mut m = SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = a[i * 4 + j];
            }
        }
        m
    }

    /// Joint vectors equal modulo 2*pi within `tol`.
    fn q_eq_mod(a: &[f64; 6], b: &[f64; 6], tol: f64) -> bool {
        for i in 0..6 {
            let d = libm::remainder(a[i] - b[i], 2.0 * core::f64::consts::PI);
            if libm::fabs(d) > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn home_pose_round_trips() {
        // T_HOME = FK at q = 0 (from z1_ik.py).
        let t_home = [
            1.0, 0.0, 0.0, -0.012799999999999971,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.1605,
            0.0, 0.0, 0.0, 1.0,
        ];
        let tf = tf_from_row_major(&t_home);
        let sols = compute_ik(&tf);
        let zero = [0.0; 6];
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-6));
        assert!(found, "home configuration q=0 not recovered");
    }

    #[test]
    fn matches_ssik_reference_dataset() {
        let tol = 1e-5;
        let mut total_expected = 0usize;
        let mut case_idx = 0usize;
        for (tf_arr, expected) in test_data::CASES {
            let tf = tf_from_row_major(tf_arr);
            let sols = compute_ik(&tf);
            let got: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();

            assert_eq!(
                got.len(),
                expected.len(),
                "case {case_idx}: solution count {} != expected {}",
                got.len(),
                expected.len()
            );

            for exp in expected.iter() {
                let matched = got.iter().any(|g| q_eq_mod(g, exp, tol));
                assert!(
                    matched,
                    "case {case_idx}: expected solution {:?} not reproduced",
                    exp
                );
                total_expected += 1;
            }
            case_idx += 1;
        }
        assert!(total_expected > 0, "no expected solutions in dataset");
    }
}
