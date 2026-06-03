//! Analytical inverse kinematics for the **PUMA 560**.
//!
//! Line-for-line translation of the Python `ssik` prebuilt artifact
//! `ssik/prebuilt/puma560_ik.py` (solver `ikgeo.spherical_two_parallel`,
//! tier 0) into a deterministic, `#![no_std]`, zero-allocation Rust
//! implementation.
//!
//! The PUMA 560 is a Pieper arm: a spherical wrist at joints 4/5/6 with the
//! shoulder axes 2/3 parallel. The closed form is fully inlined — `SP4` for
//! `q1`, `SP3` for the elbow `q3`, then `SP1`/`SP4`/`SP1` for the wrist
//! triple `(q2, q5, q4, q6)` — so unlike the three-parallel arms it needs no
//! SP6/conic machinery. The arm-agnostic FK chain and dedup live in
//! [`crate::math::kinematics`]; only the baked per-arm constants and the
//! closed-form angle expressions are arm-specific.
//!
//! ## Equivalence to the Python reference
//!
//! [`compute_ik`] reproduces `puma560_ik.solve(T, respect_limits=False)`: the
//! raw geometric solution set, FK-verified against the target pose and
//! de-duplicated modulo `2*pi`.

use nalgebra::SMatrix;

use crate::math::kinematics::{eye_with, fk_residual, q_close_wrap};

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the PUMA 560.
pub const DOF: usize = 6;

// --- tolerances (ssik DEFAULT_TOLERANCE_POLICY) ---
const DEDUP_TOL: f64 = 1e-3; // subproblem_dedup
const FK_ATOL: f64 = 1e-5; // FK closure tolerance (== subproblem_numerical)

// --- algebraic-block degeneracy constants from puma560_ik.py ---
const DEG_SQ: f64 = 1e-16;
const FEAS_TOL: f64 = 1e-8;

// --- baked KinBody constants (for FK verification) ---
const JOINT_AXES: [[f64; 3]; 6] = [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, 0.0, 1.0],
];

const T_LEFTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.4318, 0.0, 0.0),
    eye_with(0.020299999999999985, -0.15005, 9.187912610603016e-18),
    eye_with(0.0, 0.0, 0.4318),
    eye_with(0.0, 0.0, 0.0),
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

/// Computes the complete analytical inverse kinematics for the PUMA 560.
///
/// `target_tf` is the 4x4 SE(3) pose of the end-effector expressed in the base
/// frame. Returns up to 16 candidate joint vectors `[q1, .., q6]` (the PUMA
/// yields at most 8); empty slots are [`None`]. The set is FK-verified to
/// within [`FK_ATOL`] and de-duplicated modulo `2*pi`, matching
/// `puma560_ik.solve(T, respect_limits=False)`.
pub fn compute_ik(target_tf: &SMatrix<f64, 4, 4>) -> [Option<[f64; 6]>; 16] {
    let mut solutions: [Option<[f64; 6]>; 16] = [None; 16];
    let mut solution_count = 0usize;

    // Raw algebraic candidates (stack array, at most 8 for the PUMA).
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

/// SP3/SP4 two-branch angle resolution shared by the `q1`, `q3` and `q5`
/// blocks (port of the repeated `theta = phi +/- acos(rhs / R)` pattern).
#[inline]
fn sp_branch(r_sq: f64, rhs: f64, phi: f64) -> (f64, f64) {
    if r_sq < DEG_SQ {
        (0.0, 0.0)
    } else {
        let r = libm::sqrt(r_sq);
        if libm::fabs(rhs) > r + FEAS_TOL {
            let p = if rhs > 0.0 { phi } else { phi + PI };
            (p, p)
        } else {
            let clipped = (rhs / r).clamp(-1.0, 1.0);
            let delta = libm::acos(clipped);
            (phi + delta, phi - delta)
        }
    }
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
    let r_21 = t[(2, 1)];
    let r_22 = t[(2, 2)];
    let p_x = t[(0, 3)];
    let p_y = t[(1, 3)];
    let p_z = t[(2, 3)];

    let mut count = 0usize;

    // SP4 for q1 (shoulder pan).
    let q1_r_sq = 1.0 * p_x * p_x + 1.0 * p_y * p_y;
    let q1_rhs = 0.15005 - 6.12323399573677e-17 * p_z;
    let q1_phi = libm::atan2(1.0 * p_x, -1.0 * p_y);
    let (theta_q1_plus, theta_q1_minus) = sp_branch(q1_r_sq, q1_rhs, q1_phi);

    for q1 in [theta_q1_plus, theta_q1_minus] {
        // SP3 for q3 (elbow): reduces to SP4 with target shift.
        let q3_x0 = 1.0 * libm::sin(q1);
        let q3_x1 = libm::cos(q1);
        let q3_r_sq = 0.0348408995890292_f64;
        let q3_a = p_x * q3_x0 - p_y * q3_x1;
        let q3_b = -p_x * q3_x1 - p_y * q3_x0;
        let q3_rhs = -0.5 * p_z * p_z - 0.5 * q3_a * q3_a - 0.5 * q3_b * q3_b + 0.19791478625;
        let q3_phi = -1.52381841044681 + PI;
        let (theta_q3_plus, theta_q3_minus) = sp_branch(q3_r_sq, q3_rhs, q3_phi);

        for q3 in [theta_q3_plus, theta_q3_minus] {
            // SP1 for q2 (shoulder pitch): closed-form atan2.
            let q2_x0 = libm::sin(q3);
            let q2_x1 = libm::cos(q3);
            let q2_x2 = 0.4318 * q2_x0 - 0.0203 * q2_x1 - 0.4318;
            let q2_x3 = 1.0 * p_z;
            let q2_x4 = 0.0203 * q2_x0 + 0.4318 * q2_x1;
            let q2_x5 = libm::cos(q1);
            let q2_x6 = libm::sin(q1);
            let q2_x7 = 1.0 * q2_x6;
            let q2_x8 = -p_x * q2_x5 - p_y * q2_x7;
            let q2_x9 = p_x * q2_x7 - p_y * q2_x5;
            let q2 = libm::atan2(
                -q2_x2 * q2_x3
                    + q2_x4 * q2_x8
                    + q2_x9
                        * (2.64401243935914e-17 * q2_x0 - 1.24301650113456e-18 * q2_x1
                            - 2.64401243935914e-17),
                -0.15005 * p_x * q2_x6 + 0.15005 * p_y * q2_x5 - 9.18791261060302e-18 * p_z
                    + q2_x2 * q2_x8
                    - q2_x3 * (-q2_x4 - 9.18791261060302e-18)
                    + q2_x9
                        * (-1.24301650113456e-18 * q2_x0 - 2.64401243935914e-17 * q2_x1
                            + 0.15005),
            );

            // SP4 for q5 (wrist pitch).
            let q5_x0 = libm::sin(q2);
            let q5_x1 = libm::sin(q3);
            let q5_x2 = libm::cos(q2);
            let q5_x3 = 6.12323399573677e-17 * q5_x2 - 6.12323399573677e-17;
            let q5_x4 = libm::cos(q3);
            let q5_x5 = 6.12323399573677e-17 * q5_x4 - 6.12323399573677e-17;
            let q5_x6 = 1.0 * q5_x2;
            let q5_x7 = 1.0 * q5_x4 + 3.74939945665464e-33;
            let q5_x8 = 1.0 * q5_x7;
            let q5_x9 = 6.12323399573677e-17 * q5_x0;
            let q5_x10 = -q5_x1 * q5_x9 + q5_x3 * q5_x7 + q5_x5;
            let q5_x11 = 1.0 * libm::sin(q1);
            let q5_x12 = libm::cos(q1);
            let q5_x13 = -q5_x0 * q5_x8 - q5_x1 * q5_x6 - q5_x5 * q5_x9;
            let q5_r_sq = 1.00000000000000_f64;
            let q5_rhs = 1.0 * r_02 * (-q5_x10 * q5_x11 + q5_x12 * q5_x13)
                + 1.0 * r_12 * (q5_x10 * q5_x12 + q5_x11 * q5_x13)
                + 1.0
                    * r_22
                    * (-1.0 * q5_x0 * q5_x1
                        + 1.0 * q5_x3 * q5_x5
                        + q5_x8 * (q5_x6 + 3.74939945665464e-33))
                - 3.74939945665464e-33;
            let q5_phi = 0.0;
            let (theta_q5_plus, theta_q5_minus) = sp_branch(q5_r_sq, q5_rhs, q5_phi);

            for q5 in [theta_q5_plus, theta_q5_minus] {
                // SP1 for q4 (wrist roll-1): closed-form atan2.
                let q4_x0 = 6.12323399573677e-17 * libm::cos(q5) - 6.12323399573677e-17;
                let q4_x1 = libm::cos(q3);
                let q4_x2 = libm::sin(q2);
                let q4_x3 = 1.0 * q4_x2;
                let q4_x4 = libm::cos(q2);
                let q4_x5 = 6.12323399573677e-17 * q4_x4 - 6.12323399573677e-17;
                let q4_x6 = libm::sin(q3);
                let q4_x7 = 6.12323399573677e-17 * q4_x6;
                let q4_x8 = 1.0 * q4_x4 + 3.74939945665464e-33;
                let q4_x9 = 1.0 * q4_x6;
                let q4_x10 = 1.0 * r_22;
                let q4_x11 = libm::cos(q1);
                let q4_x12 = q4_x1 * q4_x4 - q4_x2 * q4_x9;
                let q4_x13 = 6.12323399573677e-17 * q4_x2;
                let q4_x14 = q4_x1 * q4_x13 + q4_x5 * q4_x9 + q4_x7;
                let q4_x15 = 1.0 * libm::sin(q1);
                let q4_x16 = 1.0 * r_02;
                let q4_x17 = 1.0 * r_12;
                let q4_x18 = q4_x10 * (q4_x1 * q4_x3 + q4_x5 * q4_x7 + q4_x8 * q4_x9)
                    + q4_x16 * (q4_x11 * q4_x12 - q4_x14 * q4_x15)
                    + q4_x17 * (q4_x11 * q4_x14 + q4_x12 * q4_x15);
                let q4_x19 = 6.12323399573677e-17 * q4_x1 - 6.12323399573677e-17;
                let q4_x20 = q4_x19 * q4_x5 - 3.74939945665464e-33 * q4_x2 * q4_x6 + 1.0;
                let q4_x21 = -q4_x13 - q4_x19 * q4_x3 - q4_x4 * q4_x7;
                let q4_x22 = q4_x10 * (1.0 * q4_x19 * q4_x8 - q4_x2 * q4_x7 + q4_x5)
                    + q4_x16 * (q4_x11 * q4_x21 - q4_x15 * q4_x20)
                    + q4_x17 * (q4_x11 * q4_x20 + q4_x15 * q4_x21);
                let q4_x23 = 1.0 * libm::sin(q5);
                let q4 = libm::atan2(
                    -q4_x0 * q4_x18 - q4_x22 * q4_x23,
                    q4_x0 * q4_x22 - q4_x18 * q4_x23,
                );

                // SP1 for q6 (wrist roll-2): closed-form atan2.
                let q6_x0 = libm::sin(q2);
                let q6_x1 = libm::sin(q3);
                let q6_x2 = libm::cos(q2);
                let q6_x3 = 6.12323399573677e-17 * q6_x2 - 6.12323399573677e-17;
                let q6_x4 = libm::cos(q3);
                let q6_x5 = 6.12323399573677e-17 * q6_x4 - 6.12323399573677e-17;
                let q6_x6 = 1.0 * q6_x2;
                let q6_x7 = 1.0 * q6_x4 + 3.74939945665464e-33;
                let q6_x8 = 1.0 * q6_x7;
                let q6_x9 = -1.0 * q6_x0 * q6_x1
                    + 1.0 * q6_x3 * q6_x5
                    + 1.0 * q6_x8 * (q6_x6 + 3.74939945665464e-33);
                let q6_x10 = 6.12323399573677e-17 * q6_x0;
                let q6_x11 = -q6_x1 * q6_x10 + q6_x3 * q6_x7 + q6_x5;
                let q6_x12 = 1.0 * libm::sin(q1);
                let q6_x13 = libm::cos(q1);
                let q6_x14 = -q6_x0 * q6_x8 - q6_x1 * q6_x6 - q6_x10 * q6_x5;
                let q6_x15 = -1.0 * q6_x11 * q6_x12 + 1.0 * q6_x13 * q6_x14;
                let q6_x16 = 1.0 * q6_x11 * q6_x13 + 1.0 * q6_x12 * q6_x14;
                let q6_x17 = q6_x15 * r_01 + q6_x16 * r_11 + q6_x9 * r_21;
                let q6_x18 = 1.0 * libm::sin(q5);
                let q6_x19 = libm::cos(q5);
                let q6_x20 = 6.12323399573677e-17 * q6_x19 - 6.12323399573677e-17;
                let q6_x21 = q6_x15 * r_00 + q6_x16 * r_10 + q6_x9 * r_20;
                // q6_x22 / q6_x23 are emitted by the codegen but unused in the
                // final atan2; kept out to avoid dead bindings.
                let q6 = libm::atan2(
                    -q6_x17 * q6_x18 + q6_x20 * q6_x21,
                    q6_x17 * q6_x20 + q6_x18 * q6_x21,
                );

                if count < 8 {
                    out[count] = [q1, q2, q3, q4, q5, q6];
                    count += 1;
                }
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
        // T_HOME = FK at q = 0 (from puma560_ik.py).
        let t_home = [
            1.0, 0.0, 0.0, 0.4521,
            0.0, 1.0, 0.0, -0.15005,
            0.0, 0.0, 1.0, 0.4318,
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
