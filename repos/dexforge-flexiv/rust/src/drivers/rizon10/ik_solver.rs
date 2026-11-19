//! Analytical inverse kinematics for the **Flexiv Rizon 10** (7-DOF).
//!
//! Deterministic, `#![no_std]`, zero-allocation translation of the Python
//! `ssik` prebuilt artifact `ssik/prebuilt/rizon10_ik.py`
//! (solver `jointlock.seven_r`, tier 1).
//!
//! The Rizon 10 is a redundant, non-SRS 7R arm. The `ssik` solver locks **joint
//! 2** (no chain reversal) to each of 16 fixed sample values and solves the
//! remaining six joints as a generic 6R sub-chain with the deterministic
//! **Raghavan-Roth** kernel. The whole sweep is implemented by the shared
//! [`crate::drivers::rizon_common`] solver; this module only supplies the baked
//! [`Model`] built from the generated [`consts`] / [`pq`] tables and re-exports
//! the public surface (`solve`, `fk`, [`TolerancePolicy`]).
//!
//! See [`crate::drivers::rizon_common`] for the algorithm and the
//! deterministic-execution note.

use nalgebra::SMatrix;

use crate::drivers::rizon_common::{self, Model};

pub use crate::drivers::rizon_common::{DOF, MAX_SOLUTIONS};
pub use dexhal::math::tolerances::TolerancePolicy;

use super::consts;
use super::pq;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Solver identifier — mirrors `SOLVER_NAME` in the Python artifact.
pub const SOLVER_NAME: &str = "jointlock.seven_r";

/// Solver tier — mirrors `SOLVER_TIER` in the Python artifact.
pub const SOLVER_TIER: u32 = 1;

/// Base link name in the URDF.
pub const BASE_LINK: &str = "base_link";

/// End-effector link name in the URDF.
pub const EE_LINK: &str = "flange";

/// Home pose: 4×4 SE(3) of the flange in `base_link` at `q = [0; 7]`.
/// Mirrors `T_HOME` in the Python artifact.
pub const T_HOME: [[f64; 4]; 4] = consts::T_HOME;

/// Baked model: per-sample RR derivation data + full 7R chain geometry.
const MODEL: Model = Model {
    lock_idx: consts::LOCK_IDX,
    lock_samples: &consts::LOCK_SAMPLES,
    dh_alpha: &consts::DH_ALPHA,
    dh_a: &consts::DH_A,
    dh_d: &consts::DH_D,
    dh_theta_offset: &consts::DH_THETA_OFFSET,
    t_pre_inv: &consts::T_PRE_INV,
    t_post_inv: &consts::T_POST_INV,
    linearity_joint: &consts::LINEARITY_JOINT,
    left_bilinear: &consts::LEFT_BILINEAR,
    right_bilinear: &consts::RIGHT_BILINEAR,
    drop_joint: &consts::DROP_JOINT,
    build_pq: &pq::BUILD_PQ,
    axes7: &consts::AXES7,
    t_lefts7: &consts::T_LEFTS7,
    t_rights7: &consts::T_RIGHTS7,
    joint_limits: &consts::JOINT_LIMITS,
};

/// Inverse kinematics — mirrors `rizon10_ik.solve`.
///
/// Returns up to [`MAX_SOLUTIONS`] candidate joint vectors `[q1, .., q7]` in
/// radians; unused slots are [`None`].
#[allow(clippy::too_many_arguments)]
pub fn solve(
    t: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 7]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    rizon_common::solve(
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

/// Forward kinematics — mirrors `rizon10_ik.fk`.
///
/// Delegates to the SymPy-generated, branch-free symbolic FK in
/// [`generated_kinematics::fk`] (regenerate with `codegen/generate_dynamics.py`).
/// The shared [`rizon_common`] solver keeps using its `Model`-based FK so the
/// algebraic root refinement reproduces the reference dataset bit-for-bit.
pub fn fk(q: &[f64; 7]) -> SMatrix<f64, 4, 4> {
    let acc = super::generated_kinematics::fk(q);
    SMatrix::<f64, 4, 4>::from_fn(|i, j| acc[i][j])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn tf_from_row_major(a: &[f64; 16]) -> SMatrix<f64, 4, 4> {
        SMatrix::<f64, 4, 4>::from_fn(|i, j| a[i * 4 + j])
    }

    fn q_eq_mod(a: &[f64; 7], b: &[f64; 7], tol: f64) -> bool {
        for i in 0..7 {
            let d = libm::remainder(a[i] - b[i], 2.0 * core::f64::consts::PI);
            if libm::fabs(d) > tol {
                return false;
            }
        }
        true
    }

    fn fk_residual(q: &[f64; 7], tf: &SMatrix<f64, 4, 4>) -> f64 {
        dexhal::math::kinematics::mat4_frob_diff(&fk(q), tf)
    }

    #[test]
    fn fk_home_pose() {
        let zero = [0.0f64; 7];
        let t = fk(&zero);
        for i in 0..4 {
            for j in 0..4 {
                let expected = T_HOME[i][j];
                let got = t[(i, j)];
                assert!(
                    (got - expected).abs() < 1e-9,
                    "fk home pose mismatch at ({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    // Strict joint-space tolerance for well-conditioned poses.
    const STRICT_TOL: f64 = 1e-5;
    // Cartesian FK-residual tolerance every returned solution must satisfy.
    const FK_RESIDUAL_TOL: f64 = 1e-5;

    #[test]
    fn matches_ssik_reference_dataset() {
        // The Rizon driver is validated against the Python `ssik` reference
        // dataset (codegen/generate_test_data.py) with the two-tier
        // tolerance mandated by the issue:
        //
        //  * Soundness (all poses): every returned configuration solves the
        //    target pose (FK closure < 1e-5). The deterministic Raghavan-Roth
        //    kernel can surface a few extra *genuine* roots beyond the
        //    reference set; each one still closes the pose, so soundness is the
        //    FK-residual check rather than a "no spurious" cardinality bound.
        //  * Completeness (strict poses only): for `singular == false` poses the
        //    Rust pipeline reproduces the reference set exactly (count + every
        //    value to < 1e-5). For `singular == true` poses the deterministic
        //    kernel returns a *sound subset* — it resolves near-double roots on
        //    the self-motion manifold to a different set than the reference
        //    husty_pfurner refinement (surfacing extra genuine roots and
        //    occasionally missing others), so completeness is not asserted
        //    there; only the global FK-closure soundness above applies. This
        //    mirrors the xArm7 / Franka near-singular policy.
        assert!(!test_data::ssik_version().is_empty());
        let cases = test_data::cases();
        let mut total_expected = 0usize;
        let mut strict_cases = 0usize;
        let mut singular_cases = 0usize;
        let mut case_idx = 0usize;
        for (tf_arr, expected, singular) in &cases {
            let tf = tf_from_row_major(tf_arr);
            let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            let got: Vec<[f64; 7]> = sols.iter().flatten().copied().collect();

            // Soundness: every returned solution closes the target pose.
            for g in got.iter() {
                let residual = fk_residual(g, &tf);
                assert!(
                    residual <= FK_RESIDUAL_TOL,
                    "case {case_idx}: returned solution {:?} does not close the pose \
                     (FK residual {residual:e} > {FK_RESIDUAL_TOL:e})",
                    g
                );
            }

            // Completeness: every reference solution is reproduced exactly for
            // strict (well-conditioned) poses. For singular poses the
            // deterministic kernel returns a *sound subset* (the reference's
            // husty_pfurner refinement can reach points off the deterministic
            // root set), so completeness is not asserted there — only the
            // global FK-closure soundness above applies. This mirrors the
            // xArm7 / Franka near-singular policy.
            if !*singular {
                for exp in expected.iter() {
                    let matched = got.iter().any(|g| q_eq_mod(g, exp, STRICT_TOL));
                    assert!(
                        matched,
                        "case {case_idx}: reference solution {:?} not reproduced",
                        exp
                    );
                }
            } else {
                // Soundness floor for singular poses: the kernel must still
                // return at least one valid configuration.
                assert!(
                    !got.is_empty(),
                    "case {case_idx}: singular pose produced no solutions"
                );
            }

            if *singular {
                singular_cases += 1;
            } else {
                strict_cases += 1;
                // For strict poses the deterministic kernel reproduces the
                // reference set exactly (no extra roots either).
                assert_eq!(
                    got.len(),
                    expected.len(),
                    "case {case_idx}: strict solution count {} != expected {}",
                    got.len(),
                    expected.len()
                );
            }
            total_expected += expected.len();
            case_idx += 1;
        }
        assert!(total_expected > 0, "no expected solutions in dataset");
        assert_eq!(case_idx, cases.len());
        assert!(strict_cases > 0 || singular_cases > 0);
    }

    #[test]
    fn respect_limits_subset() {
        for (tf_arr, _expected, _singular) in test_data::cases().iter().take(64) {
            let tf = tf_from_row_major(tf_arr);
            let raw = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            let lim = solve(&tf, None, None, true, false, TolerancePolicy::DEFAULT, 15);
            let n_raw = raw.iter().flatten().count();
            let n_lim = lim.iter().flatten().count();
            assert!(
                n_lim <= n_raw,
                "respect_limits returned more solutions ({n_lim}) than the raw set ({n_raw})"
            );
            for q in lim.iter().flatten() {
                for i in 0..7 {
                    let (lo, hi) = consts::JOINT_LIMITS[i];
                    assert!(
                        q[i] >= lo - 1e-9 && q[i] <= hi + 1e-9,
                        "respect_limits returned out-of-range joint {i}: {}",
                        q[i]
                    );
                }
            }
        }
    }

    #[test]
    fn max_solutions_cap() {
        let (tf_arr, _, _) = &test_data::cases()[0];
        let tf = tf_from_row_major(tf_arr);
        let all_sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let n_all: usize = all_sols.iter().flatten().count();
        if n_all > 1 {
            let capped = solve(&tf, Some(1), None, false, false, TolerancePolicy::DEFAULT, 15);
            let n_capped: usize = capped.iter().flatten().count();
            assert_eq!(n_capped, 1, "max_solutions=1 should return exactly 1 solution");
        }
    }

    /// Item 54: a seeded solve keeps the seed's locked joint 2 (off the sample
    /// grid) instead of snapping it to the nearest lock sample, also along a
    /// path.
    #[test]
    fn seeded_solve_keeps_the_seed_locked_joint() {
        let mut q = [0.3, -0.4, 0.37, 1.2, 0.2, 0.8, 0.5];
        let mut seed = q;
        for v in seed.iter_mut() {
            *v += 0.02;
        }
        for step in 0..20 {
            let target = fk(&q);
            let sols = solve(&target, Some(1), Some(&seed), true, true, TolerancePolicy::DEFAULT, 15);
            let sol = sols[0].expect("seeded solve finds a solution");
            assert_eq!(sol[2], seed[2], "step {step}");
            assert!(fk_residual(&sol, &target) < 1e-9, "step {step}");
            let jump = sol.iter().zip(&seed).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(jump < 0.1, "step {step}: jumped to {sol:?}");
            seed = sol;
            q[0] += 0.01;
            q[3] += 0.005;
        }
    }
}