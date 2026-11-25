//! Analytical inverse kinematics for the Unitree **Z1** (6-DOF).
//!
//! This is a line-by-line translation of the Python `ssik` prebuilt artifact
//! `ssik/prebuilt/z1_ik.py` (solver `ikgeo.three_parallel`, tier 0) into a
//! deterministic, `#![no_std]`, zero-allocation Rust implementation.
//!
//! The closed form is: subproblem **SP6** for the coupled wrist/shoulder angle
//! pair `(q1, q5)`, followed by the inlined `SP1 + SP3 + SP1` chain for the
//! remaining joints. The raw candidates go through the shared 6-DOF
//! post-process [`dexhal::math::closed_form::finish6`].

use nalgebra::SMatrix;

use dexhal::math::closed_form::finish6;
use dexhal::math::single::CANDIDATES;
use dexhal::math::ssik_core::{sp6_solve, wrap_to_pi};
pub use dexhal::math::tolerances::TolerancePolicy;

use super::arm::Z1;

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the Unitree Z1.
pub const DOF: usize = 6;

/// Maximum number of IK solutions returned by [`solve`].
pub use super::arm::MAX_SOLUTIONS;

/// Solver identifier — mirrors `SOLVER_NAME` in the Python artifact.
pub const SOLVER_NAME: &str = "ikgeo.three_parallel";

/// Solver tier — mirrors `SOLVER_TIER` in the Python artifact.
pub const SOLVER_TIER: u32 = 0;

/// Expected median runtime in milliseconds from the Python artifact.
pub const EXPECTED_MS_MEDIAN: f64 = 1.6;

/// Approximate FLOP count per solve from the Python artifact.
pub const FLOP_BUDGET: u32 = 2519;

/// Solver dispatch rationale from the Python artifact.
pub const DISPATCH_REASON: &str = "Three consecutive parallel axes at joints (1, 2, 3) -- the UR-class structure (UR3 / UR5 / UR10).\nClosed-form via SP6 (joints 0+4) + SP1 + SP3.";

/// Base link name in the URDF.
pub const BASE_LINK: &str = "link00";

/// End-effector link name in the URDF.
pub const EE_LINK: &str = "link06";

/// Home pose: FK at `q = [0.0; 6]`.
pub const T_HOME: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, -0.012799999999999971],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.1605],
    [0.0, 0.0, 0.0, 1.0],
];

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Inverse kinematics — mirrors `z1_ik.solve`; every solution (`ik_all`).
///
/// Raw candidates from [`solve_algebraic`] go through
/// [`finish6`] (FK closure / refinement, dedup, `Z1::JOINT_LIMITS` when
/// `respect_limits`, nearest-to-seed sort, `max_solutions` cap). Always
/// enumerates the full set; a single seeded solution is `Z1::flange_one`
/// (`dexhal::math::single::six_dof_one`).
///
/// Returns up to 16 candidate joint vectors `[q1, .., q6]` in radians;
/// unused slots are [`None`].
pub fn solve(
    t: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 6]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 6]>; MAX_SOLUTIONS] {
    let mut cands = [[0.0f64; 6]; 8];
    let n = solve_algebraic(t, &mut cands, policy);
    finish6::<Z1, MAX_SOLUTIONS>(
        &cands[..n],
        t,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        policy,
        refinement_max_iters,
    )
}

/// Raw (unfiltered, unpolished) closed-form IK candidates for `t`, padded to
/// [`CANDIDATES`]. The standard candidate source of `ArmModel::flange_one`:
/// refinement and acceptance are left to [`dexhal::math::single::six_dof_one`].
pub fn raw_candidates(t: &SMatrix<f64, 4, 4>, policy: TolerancePolicy) -> ([[f64; 6]; CANDIDATES], usize) {
    let mut raw = [[0.0; 6]; 8];
    let n = solve_algebraic(t, &mut raw, policy);
    let mut out = [[0.0; 6]; CANDIDATES];
    out[..n].copy_from_slice(&raw[..n]);
    (out, n)
}

// ---------------------------------------------------------------------------
// Algebraic candidate generation (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

pub(super) fn solve_algebraic(
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]; 8],
    policy: TolerancePolicy,
) -> usize {
    let r_00 = t[(0, 0)];
    let r_01 = t[(0, 1)];
    let r_02 = t[(0, 2)];
    let r_10 = t[(1, 0)];
    let r_11 = t[(1, 1)];
    let r_12 = t[(1, 2)];
    let r_20 = t[(2, 0)];
    let _r_21 = t[(2, 1)];
    let _r_22 = t[(2, 2)];
    let p_x = t[(0, 3)];
    let p_y = t[(1, 3)];
    let p_z = t[(2, 3)];

    let mut count = 0usize;

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
    let n15 = sp6_solve(
        &h_sp,
        &k_sp,
        &p_sp,
        0.0,
        0.0,
        policy.subproblem_numerical,
        policy.subproblem_degeneracy,
        &mut theta15,
    );

    for &(q1, q5) in theta15.iter().take(n15) {
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

        // d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
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
        if q3_r_sq < policy.axis_parallel * policy.axis_parallel {
            theta_q3_plus = 0.0;
            theta_q3_minus = 0.0;
        } else {
            let q3_r = libm::sqrt(q3_r_sq);
            if libm::fabs(q3_rhs) > q3_r + policy.subproblem_feasibility {
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
// Forward kinematics (for FK closure verification)
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics — mirrors `z1_ik.fk`.
pub fn fk(q: &[f64; 6]) -> SMatrix<f64, 4, 4> {
    let acc = super::generated_kinematics::fk(q);
    let mut out = SMatrix::<f64, 4, 4>::zeros();
    for i in 0..4 {
        for j in 0..4 {
            out[(i, j)] = acc[i][j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dexhal::math::closed_form::refine6;
    use dexhal::math::kinematics::mat4_frob_diff;
    use dexhal::math::postprocess::{within_limits, wrap_distance};
    use dexhal::math::{ArmModel, IkKind};
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
    fn refine6_recovers_a_perturbed_configuration() {
        let q_true = [0.4, -1.1, 0.9, -1.3, 0.7, 0.2];
        let target = fk(&q_true);
        let seed = [0.45, -1.05, 0.85, -1.35, 0.65, 0.25];
        let policy = TolerancePolicy::DEFAULT;
        let q = refine6::<Z1, MAX_SOLUTIONS>(&seed, &target, 30, policy.subproblem_numerical)
            .expect("refine6 should converge from a nearby seed");
        assert!(mat4_frob_diff(&fk(&q), &target) <= policy.subproblem_numerical);
    }

    #[test]
    fn flange_one_near_matches_first_seeded_solution() {
        let policy = TolerancePolicy::DEFAULT;
        let configs = [[0.3, -1.0, 1.2, -0.5, 0.8, 0.2], [0.4, -1.1, 0.9, -1.3, 0.7, 0.2]];
        let offsets = [-0.4, -0.1, 0.05, 0.3];
        for q_true in configs {
            let t = fk(&q_true);
            for off in offsets {
                let seed = [
                    q_true[0] + off,
                    q_true[1] - off,
                    q_true[2] + off,
                    q_true[3] - off,
                    q_true[4] + off,
                    q_true[5] - off,
                ];
                let single = Z1::flange_one(&t, &seed, IkKind::Near, policy, 15);
                let full = solve(&t, None, Some(&seed), true, true, policy, 15)[0];
                match (single, full) {
                    (Some(a), Some(b)) => assert!(q_eq_mod(&a, &b, 1e-9), "single {a:?} != full {b:?}"),
                    (None, None) => {}
                    _ => panic!("single/full disagree on existence: {single:?} vs {full:?}"),
                }
            }
        }
    }

    #[test]
    fn fk_home_pose() {
        let zero = [0.0f64; 6];
        let t = fk(&zero);
        for i in 0..4 {
            for j in 0..4 {
                let expected = T_HOME[i][j];
                let got = t[(i, j)];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "fk home pose mismatch at ({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn home_pose_round_trips() {
        let t_home: [f64; 16] = [
            T_HOME[0][0], T_HOME[0][1], T_HOME[0][2], T_HOME[0][3],
            T_HOME[1][0], T_HOME[1][1], T_HOME[1][2], T_HOME[1][3],
            T_HOME[2][0], T_HOME[2][1], T_HOME[2][2], T_HOME[2][3],
            T_HOME[3][0], T_HOME[3][1], T_HOME[3][2], T_HOME[3][3],
        ];
        let tf = tf_from_row_major(&t_home);
        let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let zero = [0.0; 6];
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-6));
        assert!(found, "home configuration q=0 not recovered");
    }

    #[test]
    fn matches_ssik_reference_dataset() {
        assert!(!test_data::ssik_version().is_empty());
        let tol = 1e-5;
        let mut total_expected = 0usize;
        let mut case_idx = 0usize;
        for (tf_arr, expected) in &test_data::cases() {
            let tf = tf_from_row_major(tf_arr);
            let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
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

    #[test]
    fn respect_limits_drops_out_of_range() {
        let t_home: [f64; 16] = [
            T_HOME[0][0], T_HOME[0][1], T_HOME[0][2], T_HOME[0][3],
            T_HOME[1][0], T_HOME[1][1], T_HOME[1][2], T_HOME[1][3],
            T_HOME[2][0], T_HOME[2][1], T_HOME[2][2], T_HOME[2][3],
            T_HOME[3][0], T_HOME[3][1], T_HOME[3][2], T_HOME[3][3],
        ];
        let tf = tf_from_row_major(&t_home);
        let sols_lim = solve(&tf, None, None, true, false, TolerancePolicy::DEFAULT, 15);
        for q in sols_lim.iter().flatten() {
            assert!(within_limits(q, &Z1::JOINT_LIMITS), "out-of-limits solution returned: {q:?}");
        }
    }

    #[test]
    fn max_solutions_cap() {
        let (tf_arr, _) = &test_data::cases()[0];
        let tf = tf_from_row_major(tf_arr);
        let all_sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let n_all = all_sols.iter().flatten().count();
        if n_all > 1 {
            let capped = solve(&tf, Some(1), None, false, false, TolerancePolicy::DEFAULT, 15);
            let n_capped = capped.iter().flatten().count();
            assert_eq!(n_capped, 1, "max_solutions=1 should return 1 solution");
        }
    }

    #[test]
    fn q_seed_sorts_nearest_first() {
        let (tf_arr, _) = &test_data::cases()[0];
        let tf = tf_from_row_major(tf_arr);
        let zero = [0.0f64; 6];
        let sols = solve(&tf, None, Some(&zero), false, false, TolerancePolicy::DEFAULT, 15);
        let solutions: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
        if solutions.len() > 1 {
            let d0 = wrap_distance(&solutions[0], &zero);
            for s in &solutions[1..] {
                assert!(
                    d0 <= wrap_distance(s, &zero) + 1e-12,
                    "q_seed sort: first solution is not the nearest to seed"
                );
            }
        }
    }
}
