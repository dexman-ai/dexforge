//! Analytical inverse kinematics for the Universal Robots **UR5e** (arm name `ur5`).
//!
//! This is a line-by-line translation of the Python `ssik` artifact
//! `codegen/ur5_ik.py`, built from `ur5.urdf` (the UR5e from Universal Robots'
//! own `ur_description`; solver `ikgeo.three_parallel`, tier 0), into a deterministic,
//! `#![no_std]`, zero-allocation Rust implementation.
//!
//! The closed form (SP6 for `(q1, q5)`, then `SP1 + SP3 + SP1`) is shared with
//! the other UR arms in [`crate::drivers::ur_common`]; this module only supplies
//! the baked UR5e geometry literals as a [`Model`].
//!
//! ## Equivalence to the Python reference
//!
//! [`solve`] mirrors `ur5_ik.solve` with the same argument list:
//! `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`, `policy`,
//! and `refinement_max_iters`. The algebraic solution set goes through the
//! shared 6-DOF post-process [`dexhal::math::closed_form::finish6`]: FK-verified
//! to [`TolerancePolicy::subproblem_numerical`] and de-duplicated modulo `2*pi`.
//!
//! [`fk`] mirrors `ur5_ik.fk` (alias of `_fk`): product-of-exponentials forward
//! kinematics using the baked chain constants.

use nalgebra::SMatrix;

use crate::drivers::ur_common::{self, Model, MAX_CANDIDATES};
pub use dexhal::math::tolerances::TolerancePolicy;

use super::arm::{Ur5, MAX_SOLUTIONS};

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the UR5.
pub const DOF: usize = 6;

/// Base link name in the URDF.
pub const BASE_LINK: &str = "base_link";

/// End-effector link name in the URDF.
pub const EE_LINK: &str = "ee_link";

/// Geometry literals of `ur5_ik.py`'s `_solve_algebraic`, verbatim.
const MODEL: Model = Model {
    d1: 0.1625,
    d4: 0.1333,
    d5: 0.0997,
    d6: 0.0996,
    neg_p5_y: 0.0996,
    a2: 0.425,
    a3: 0.3922,
    dinr_c10: 6.09874105975382e-18,
    dinr_cy: 6.10486429374955e-18,
    q3_r_sq: 0.0277838892250000_f64,
    q3_rhs0: 0.176107365,
    q3_phi0: -2.08115015714358e-17,
    q2_c0: 8.16227091631711e-18,
    q2_c1: 4.99794947572064e-34,
    q2_c2: 2.40153237312796e-17,
    q2_c3: 2.60237444818813e-17,
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Inverse kinematics — mirrors `ur5_ik.solve`; every solution (`ik_all`).
///
/// # Arguments
///
/// * `t` — 4×4 SE(3) target pose of `ee_link` in `base_link`.
/// * `max_solutions` — cap on returned solutions (post-dedup, post-limits).
///   `None` returns the full algebraic set (at most 8 for the UR5).
/// * `q_seed` — when `Some`, solutions are sorted by wrap-to-pi distance
///   from the seed (closest first). Has no effect on *which* solutions are
///   returned.
/// * `respect_limits` — when `true`, solutions outside the baked URDF joint
///   limits (`Ur5::JOINT_LIMITS`) are dropped (after a `±2π`
///   wrap-into-limits attempt). Pass `false` for the raw geometric set.
/// * `allow_refinement` — when `true`, algebraic candidates that exceed
///   `policy.subproblem_numerical` are polished with a bounded Levenberg-Marquardt step.
///   For the UR5 (tier-0 arm) this is rarely needed; default `false`.
/// * `policy` — FK closure and dedup tolerances; use
///   [`TolerancePolicy::DEFAULT`] for the Python defaults.
/// * `refinement_max_iters` — maximum LM iterations when
///   `allow_refinement = true`.
///
/// Always enumerates the full set; a single seeded solution is
/// `Ur5::flange_one` (`dexhal::math::single::six_dof_one`).
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
    ur_common::solve::<Ur5, MAX_SOLUTIONS>(
        &MODEL,
        t,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        policy,
        refinement_max_iters,
    )
}

/// Fills `out` with the raw algebraic IK candidates and returns the count.
pub(crate) fn solve_algebraic(
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]; MAX_CANDIDATES],
    policy: TolerancePolicy,
) -> usize {
    ur_common::solve_algebraic(&MODEL, t, out, policy)
}

// ---------------------------------------------------------------------------
// Forward kinematics (for FK closure verification)
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics — mirrors `ur5_ik.fk`.
///
/// `q` is a 6-element joint-angle vector in radians. Returns the 4x4 SE(3)
/// pose of `ee_link` in `base_link`.
///
/// Delegates to the SymPy-generated, branch-free symbolic FK in
/// [`generated_kinematics::fk`] (regenerate with `codegen/generate_dynamics.py`).
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
    use dexhal::math::postprocess::wrap_distance;
    use dexhal::math::single::within_max_joint_step;
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

    /// A generic (non-singular) configuration. q = 0 is a wrist singularity for
    /// the UR5e (q5 = 0), where the solver returns only one shoulder branch (see
    /// `wrist_singularity_still_solves`), so round-trip tests use this instead.
    const REF_Q: [f64; 6] = [0.3, -1.0, 1.2, -0.5, 0.8, 0.2];

    #[test]
    fn refine6_recovers_a_perturbed_configuration() {
        let q_true = [0.4, -1.1, 0.9, -1.3, 0.7, 0.2];
        let target = fk(&q_true);
        let seed = [0.45, -1.05, 0.85, -1.35, 0.65, 0.25];
        let q = refine6::<Ur5, MAX_SOLUTIONS>(&seed, &target, 30, TolerancePolicy::DEFAULT.subproblem_numerical)
            .expect("refine6 should converge from a nearby seed");
        assert!(mat4_frob_diff(&fk(&q), &target) <= TolerancePolicy::DEFAULT.subproblem_numerical);
    }

    #[test]
    fn flange_one_near_matches_first_seeded_solution() {
        let policy = TolerancePolicy::DEFAULT;
        let configs = [REF_Q, [0.4, -1.1, 0.9, -1.3, 0.7, 0.2], [-0.7, -0.6, 1.8, 0.4, -1.2, 1.0]];
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
                let single = Ur5::flange_one(&t, &seed, IkKind::Near, policy, 15);
                let full = solve(&t, None, Some(&seed), true, true, policy, 15);
                let full = full.iter().flatten().find(|b| within_max_joint_step(b, &seed)).copied();
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
        // UR5e at q = 0 in the controller base frame, straight from the datasheet dimensions:
        // x = -(a2 + a3), y = -(d4 + d6), z = d1 - d5.
        let t_home = [
            1.0, 0.0, 0.0, -(0.425 + 0.3922),
            0.0, 0.0, -1.0, -(0.1333 + 0.0996),
            0.0, 1.0, 0.0, 0.1625 - 0.0997,
            0.0, 0.0, 0.0, 1.0,
        ];
        for i in 0..4 {
            for j in 0..4 {
                let expected = t_home[i * 4 + j];
                let got = t[(i, j)];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "fk home pose mismatch at ({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn reference_pose_round_trips() {
        let tf = fk(&REF_Q);
        let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &REF_Q, 1e-6));
        assert!(found, "reference configuration not recovered");
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
        // REF_Q is within all joint limits.
        let tf = fk(&REF_Q);
        // With limits on, REF_Q should still survive (it's within all limits).
        let sols_lim = solve(&tf, None, None, true, false, TolerancePolicy::DEFAULT, 15);
        let found = sols_lim.iter().flatten().any(|s| q_eq_mod(s, &REF_Q, 1e-6));
        assert!(found, "REF_Q should survive respect_limits=true");
    }

    #[test]
    fn max_solutions_cap() {
        let tf = fk(&REF_Q);
        let all_sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let n_all: usize = all_sols.iter().flatten().count();
        if n_all > 1 {
            let capped = solve(&tf, Some(1), None, false, false, TolerancePolicy::DEFAULT, 15);
            let n_capped: usize = capped.iter().flatten().count();
            assert_eq!(n_capped, 1, "max_solutions=1 should return exactly 1 solution");
        }
    }

    #[test]
    fn q_seed_sorts_nearest_first() {
        let tf = fk(&REF_Q);
        let zero = [0.0f64; 6];
        let sols = solve(&tf, None, Some(&zero), false, false, TolerancePolicy::DEFAULT, 15);
        let solutions: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
        if solutions.len() > 1 {
            // First solution must be the closest to seed.
            let d0 = wrap_distance(&solutions[0], &zero);
            for s in &solutions[1..] {
                assert!(
                    d0 <= wrap_distance(s, &zero) + 1e-12,
                    "q_seed sort: first solution is not the nearest to seed"
                );
            }
        }
    }

    /// Wrist singularity (q5 = 0 or pi): joints 4 and 6 share an axis, so only their
    /// combination is defined and the original q4/q6 cannot be recovered.
    ///
    /// - At the singularity (q5 below about 5e-7, or pi) the solver returns 4 solutions on one
    ///   shoulder branch instead of 8, and that branch need not be the one the arm is in.
    /// - For q5 between about 1e-6 and 1e-5 the problem is ill-conditioned: the solution count
    ///   flips between 4 and 8 and the pose error reaches a few 1e-6. The Python reference
    ///   artifact does the same (the exact q5 values where it flips differ slightly).
    /// - Elsewhere the pose error is tight, and at q5 = 1e-3 all 8 solutions are back and the
    ///   original q1..q3, q5 are among them.
    ///
    /// Always required: solutions exist, are finite, and reproduce the pose to the tolerance below.
    #[test]
    fn wrist_singularity_still_solves() {
        use core::f64::consts::PI;
        // (q5, pose tolerance): tight at/above the singularity, loose in the ill-conditioned window.
        let cases = [
            (0.0, 1e-8),
            (1e-9, 1e-8),
            (5e-7, 1e-8),
            (1e-6, 1e-5),
            (1e-5, 1e-5),
            (1e-3, 1e-8),
            (PI, 1e-8),
        ];
        for (q5, tol) in cases {
            let q = [0.3, -1.0, 1.2, -0.5, q5, 0.2];
            let tf = fk(&q);
            let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            let all: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
            assert!(!all.is_empty(), "q5={q5}: no solution at/near the singularity");

            for s in &all {
                assert!(s.iter().all(|x| x.is_finite()), "q5={q5}: non-finite solution {s:?}");
                let t2 = fk(s);
                for i in 0..3 {
                    for j in 0..4 {
                        assert!(
                            (t2[(i, j)] - tf[(i, j)]).abs() < tol,
                            "q5={q5}: solution {s:?} does not reproduce the pose at ({i},{j}) to {tol}"
                        );
                    }
                }
            }
        }

        // Just off the singularity the full solution set (and the original arm joints) returns.
        let q = [0.3, -1.0, 1.2, -0.5, 1e-3, 0.2];
        let sols = solve(&fk(&q), None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let all: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
        assert_eq!(all.len(), 8, "expected the full 8 solutions just off the singularity");
        let ok = all.iter().any(|s| {
            q_eq_mod(&[s[0], s[1], s[2], 0.0, s[4], 0.0], &[q[0], q[1], q[2], 0.0, q[4], 0.0], 1e-6)
        });
        assert!(ok, "original q1..q3, q5 not among {all:?}");
    }
}
