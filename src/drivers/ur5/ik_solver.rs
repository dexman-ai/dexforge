//! Analytical inverse kinematics for the Universal Robots **UR5**.
//!
//! This is a line-by-line translation of the Python `ssik` prebuilt artifact
//! `ssik/prebuilt/ur5_ik.py` (solver `ikgeo.three_parallel`, tier 0) into a
//! deterministic, `#![no_std]`, zero-allocation Rust implementation.
//!
//! The closed form is: subproblem **SP6** for the coupled wrist/shoulder angle
//! pair `(q1, q5)`, followed by the inlined `SP1 + SP3 + SP1` chain for the
//! remaining joints. SP6 reduces to intersecting two conics, which produces a
//! quartic solved through the companion-matrix eigenvalue routine in
//! [`crate::math::ssik_core`].
//!
//! ## Equivalence to the Python reference
//!
//! [`compute_ik`] reproduces `ur5_ik.solve(T, respect_limits=False)`: the raw
//! geometric solution set, FK-verified against the target pose and de-duplicated
//! modulo `2*pi`. Joint-limit clamping (`respect_limits=True`) is intentionally
//! *not* applied so the output is the pure algebraic set; callers can filter to
//! limits afterwards if desired.

use nalgebra::SMatrix;

use crate::math::kinematics::{eye_with, fk_residual, q_close_wrap};
use crate::math::ssik_core::{sp6_solve, TWO_PI};

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the UR5.
pub const DOF: usize = 6;

// --- tolerances (ssik DEFAULT_TOLERANCE_POLICY) ---
const DEDUP_TOL: f64 = 1e-3; // subproblem_dedup
const FK_ATOL: f64 = 1e-5; // FK closure tolerance (== subproblem_numerical)

// --- algebraic-block degeneracy constants from ur5_ik.py ---
const DEG_SQ: f64 = 1e-16;
const FEAS_TOL: f64 = 1e-8;

// --- baked KinBody constants (for FK verification) ---
const JOINT_AXES: [[f64; 3]; 6] = [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.2246467991473532e-16, -1.0],
    [0.0, -1.0, 6.123233995736766e-17],
];

const T_LEFTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.089159),
    eye_with(-0.425, 0.0, 0.0),
    eye_with(-0.39225000000000004, 0.0, 0.0),
    eye_with(0.0, -0.10915, 0.0),
    eye_with(0.0, -1.3877787807814457e-17, -0.09465),
];

const T_RIGHTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 6.123233995736766e-17, -1.0, -0.0823],
        [0.0, 1.0, 6.123233995736766e-17, 5.204170427930421e-18],
        [0.0, 0.0, 0.0, 1.0],
    ],
];

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Computes the complete analytical inverse kinematics for the UR5.
///
/// `target_tf` is the 4x4 SE(3) pose of `ee_link` expressed in `base_link`.
/// Returns up to 16 candidate joint vectors `[q1, .., q6]` (the UR5 yields at
/// most 8); empty slots are [`None`]. The set is FK-verified to within
/// [`FK_ATOL`] and de-duplicated modulo `2*pi`, matching
/// `ur5_ik.solve(T, respect_limits=False)`.
pub fn compute_ik(target_tf: &SMatrix<f64, 4, 4>) -> [Option<[f64; 6]>; 16] {
    let mut solutions: [Option<[f64; 6]>; 16] = [None; 16];
    let mut solution_count = 0usize;

    // Raw algebraic candidates (stack array, at most 8 for the UR5).
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
    let r_21 = t[(2, 1)];
    let r_22 = t[(2, 2)];
    let p_x = t[(0, 3)];
    let p_y = t[(1, 3)];
    let p_z = t[(2, 3)];

    let mut count = 0usize;

    // T_target-derived SP6 inputs.
    let p_16_x = p_x - 1.64748849439063e-19 * r_01 - 0.0823 * r_02;
    let p_16_y = p_y - 1.64748849439063e-19 * r_11 - 0.0823 * r_12;
    let p_16_z = p_z - 1.64748849439063e-19 * r_21 - 0.0823 * r_22;
    let r_06_axes5 = [1.0 * r_02, 1.0 * r_12, 1.0 * r_22];

    const H_SP_0: [f64; 3] = [0.0, -1.0, 6.123233995736766e-17];
    const K_SP_0: [f64; 3] = [-0.0, -0.0, -1.0];
    const K_SP_1: [f64; 3] = [0.0, -1.2246467991473532e-16, -1.0];
    const NEG_P_5: [f64; 3] = [-0.0, 1.3877787807814457e-17, 0.09465];
    const NEG_AXES_5: [f64; 3] = [-0.0, 1.0, -6.123233995736766e-17];

    let p_16 = [p_16_x, p_16_y, p_16_z];
    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, 0.10915, 0.0, &mut theta15);

    for si in 0..n15 {
        let (q1, q5) = theta15[si];

        // NOTE: the literal constants in the theta14 / q6 / d_inner blocks
        // below (e.g. `6.12323399573677e-17`, `7.49879891330929e-33`) are the
        // machine-generated symbolic coefficients emitted by `ssik build` and
        // are copied verbatim from `ur5_ik.py`. Most are rounding residue of
        // exact zeros (cos(pi/2)) preserved here so the floating-point
        // accumulation order matches the Python reference bit-for-bit.

        // SP1 for theta14 = q1+q2+q3+q4.
        let th14_x0 = libm::sin(q5);
        let th14_x1 = 1.0 * r_22;
        let th14_x2 = libm::sin(q1);
        let th14_x3 = 6.12323399573677e-17 * r_01 - 1.0 * r_02;
        let th14_x4 = th14_x2 * th14_x3;
        let th14_x5 = 1.0 * r_01 + 6.12323399573677e-17 * r_02;
        let th14_x6 = 6.12323399573677e-17 * th14_x2 * th14_x5;
        let th14_x7 = libm::cos(q1);
        let th14_x8 = 6.12323399573677e-17 * r_11 - 1.0 * r_12;
        let th14_x9 = th14_x7 * th14_x8;
        let th14_x10 = 1.0 * r_11 + 6.12323399573677e-17 * r_12;
        let th14_x11 = 6.12323399573677e-17 * th14_x10 * th14_x7;
        let th14_x12 = th14_x11 + th14_x4 - th14_x6 - th14_x9;
        let th14_x13 = libm::cos(q5);
        let th14_x14 = 6.12323399573677e-17 * th14_x10 * th14_x2 - th14_x2 * th14_x8
            - th14_x3 * th14_x7
            + 6.12323399573677e-17 * th14_x5 * th14_x7;
        let th14_x15 = 1.0 * th14_x13;
        let theta14 = libm::atan2(
            -th14_x0 * th14_x1 - 6.12323399573677e-17 * th14_x0 * th14_x12
                + th14_x14 * (6.12323399573677e-17 - 6.12323399573677e-17 * th14_x13),
            -1.0 * th14_x0 * th14_x14
                + th14_x1 * (1.22464679914735e-16 * th14_x13 - 6.12323399573677e-17)
                + th14_x12 * (-th14_x15 - 7.49879891330929e-33)
                - (th14_x15 + 3.74939945665464e-33)
                    * (6.12323399573677e-17 * r_22 - th14_x11 - th14_x4 + th14_x6 + th14_x9),
        );

        // SP1 for q6 (wrist roll-2).
        let q6_x0 = libm::cos(q5);
        let q6_x1 = libm::sin(q1);
        let q6_x2 = 1.0 * q6_x1;
        let q6_x3 = libm::cos(q1);
        let q6_x4 = 1.0 * q6_x3;
        let q6_x5 = q6_x2 * r_00 - q6_x4 * r_10 + 6.12323399573677e-17 * r_20;
        let q6_x6 = libm::sin(q5);
        let q6_x7 = 6.12323399573677e-17 * r_22;
        let q6_x8 = 6.12323399573677e-17 * r_01 - 1.0 * r_02;
        let q6_x9 = q6_x4 * (6.12323399573677e-17 * r_11 - 1.0 * r_12);
        let q6_x10 = q6_x2 * q6_x8 - q6_x7 - q6_x9 + 3.74939945665464e-33 * r_21;
        let q6_x11 = 1.0 * r_01 + 6.12323399573677e-17 * r_02;
        let q6_x12 = 1.0 * r_11 + 6.12323399573677e-17 * r_12;
        let q6_x13 = q6_x11 * q6_x2 - q6_x12 * q6_x4
            + 6.12323399573677e-17 * r_21
            + 3.74939945665464e-33 * r_22;
        let q6_x14 = 1.0 * q6_x6;
        let q6_x15 = 1.0 * q6_x0;
        let q6 = libm::atan2(
            -6.12323399573677e-17 * q6_x10 * q6_x6 - q6_x13 * q6_x14
                + q6_x5 * (6.12323399573677e-17 * q6_x0 - 6.12323399573677e-17),
            q6_x10 * (-q6_x15 - 7.49879891330929e-33)
                + q6_x13 * (1.22464679914735e-16 * q6_x0 - 6.12323399573677e-17)
                + q6_x14 * q6_x5
                - (-q6_x15 - 3.74939945665464e-33)
                    * (-6.12323399573677e-17 * q6_x1 * q6_x11
                        + 1.0 * q6_x1 * q6_x8
                        + 6.12323399573677e-17 * q6_x12 * q6_x3
                        - q6_x7
                        - q6_x9),
        );

        // d_inner vector.
        let dinr_x0 = libm::sin(theta14);
        let dinr_x1 = libm::sin(q5);
        let dinr_x2 = libm::cos(theta14);
        let dinr_x3 = libm::cos(q5);
        let dinr_x4 = 1.22464679914735e-16 - 1.22464679914735e-16 * dinr_x3;
        let dinr_x5 = 1.0 * dinr_x3 + 1.49975978266186e-32;
        let dinr_x6 = libm::sin(q1);
        let dinr_x7 = p_y - 1.64748849439063e-19 * r_11 - 0.0823 * r_12;
        let dinr_x8 = libm::cos(q1);
        let dinr_x9 = p_x - 1.64748849439063e-19 * r_01 - 0.0823 * r_02;
        let dinr_x10 = 6.12323399573677e-17 * dinr_x2 - 6.12323399573677e-17;
        let dinr_x11 = 1.38777878078145e-17 * dinr_x4;
        let d_inner_x = -1.96734287847793e-17 * dinr_x0 * dinr_x4
            - 8.49769420904307e-34 * dinr_x0 * dinr_x5
            - 0.09465 * dinr_x0
            + 2.28650585388476e-18 * dinr_x1 * dinr_x2
            + 1.0 * dinr_x6 * dinr_x7
            + dinr_x8 * dinr_x9;
        let d_inner_y = 1.40008103759583e-34 * dinr_x0 * dinr_x1
            + dinr_x10 * dinr_x11
            + 5.79564097696485e-18 * dinr_x2
            + 2.28650585388476e-18 * dinr_x3
            - 1.0 * dinr_x6 * dinr_x9
            + dinr_x7 * dinr_x8
            + 0.10915;
        let d_inner_z = 2.28650585388476e-18 * dinr_x0 * dinr_x1
            + 0.09465 * dinr_x10 * dinr_x4
            + 1.38777878078145e-17 * dinr_x10 * dinr_x5
            + dinr_x11 * (1.0 * dinr_x2 + 3.74939945665464e-33)
            + 0.09465 * dinr_x2
            + 1.0 * p_z
            - 1.64748849439063e-19 * r_21
            - 0.0823 * r_22
            - 0.089159;
        let d_elbow =
            libm::sqrt(d_inner_x * d_inner_x + d_inner_y * d_inner_y + d_inner_z * d_inner_z);

        // SP3 for q3 (elbow).
        let q3_r_sq = 0.0277909737890625_f64;
        let q3_rhs = 0.16724253125 - 0.5 * d_elbow * d_elbow;
        let q3_phi = PI;
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
            let q2_x0 = libm::sin(q3);
            let q2_x1 = 0.39225 * q2_x0;
            let q2_x2 = libm::cos(q3);
            let q2_x3 = -0.39225 * q2_x2 - 0.425;
            let q2 = libm::atan2(
                d_inner_x * q2_x1
                    + d_inner_y * (-2.40183853482775e-17 * q2_x2 - 2.60237444818813e-17)
                    + d_inner_z * q2_x3,
                d_inner_x * q2_x3 - 2.40183853482775e-17 * d_inner_y * q2_x0 - d_inner_z * q2_x1,
            );
            let q4 = libm::remainder(theta14 - q2 - q3, TWO_PI);

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
        let t_home = [
            1.0, 0.0, 0.0, -0.81725,
            0.0, 6.123233995736766e-17, -1.0, -0.19145,
            0.0, 1.0, 6.123233995736766e-17, -0.005490999999999996,
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
