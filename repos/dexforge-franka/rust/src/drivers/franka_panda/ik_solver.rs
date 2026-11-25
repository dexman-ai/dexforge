//! Analytical inverse kinematics for the **Franka Emika Panda** (7-DOF).
//!
//! This is a deterministic, `#![no_std]`, zero-allocation translation of the
//! Python `ssik` prebuilt artifact `ssik/prebuilt/franka_panda_ik.py`
//! (solver `jointlock.seven_r`, tier 1) into Rust.
//!
//! ## Algorithm
//!
//! The Franka is a redundant 7R arm with no closed-form 6R structure, so the
//! `ssik` solver collapses it to a series of 6R problems by **locking joint 4**
//! to each of 16 fixed sample values (`np.linspace` over the joint-4 range,
//! endpoint excluded). For each sample the remaining six joints form a 6R
//! sub-chain; reversing that chain (the EAIK `REVERSED` decomposition) places
//! its spherical wrist at the canonical position, where an inner ikgeo solver
//! applies:
//!
//! * `spherical` — for 15 of the 16 samples (SP5 shoulder + SP4/SP1 wrist), and
//! * `spherical_two_parallel` — for sample index 8 (SP4/SP3/SP1).
//!
//! The locked/reversed sub-chain geometry is a pure function of the arm
//! topology, so it is baked once by `codegen/generate_franka_consts.py` into
//! [`super::consts`]. At runtime each inner solve is verified against the
//! reversed sub-chain's forward kinematics, mapped back to original-chain
//! ordering, padded with the locked angle, FK-verified against the full 7R
//! chain, and de-duplicated modulo `2*pi`. The algorithm is shared with the
//! xArm7 port in [`dexhal::math::jointlock`]; this module supplies the
//! baked geometry and the public API.
//!
//! ## Equivalence to the Python reference
//!
//! [`solve`] reproduces `franka_panda_ik.solve(T, respect_limits=False)`:
//! the raw geometric solution set, FK-verified to `1e-5` and de-duplicated to
//! `1e-3` per joint. The exhaustive 16-sample sweep yields up to **128**
//! candidates (`16 * 8`); after dedup the count is typically 16–128, so unlike
//! the 6R drivers the output array holds 128 slots rather than 16. Unused
//! slots are `None`.
//!
//! Each de-duplicated root is finally polished by
//! [`refine7`](jointlock::refine7), a bounded analytic Levenberg-Marquardt
//! step that holds the locked joint 4 fixed and refines the six free joints
//! onto the exact pose. Distinct roots are far apart (`>1e-3`), so polishing
//! never merges or splits the solution set; it only removes the tiny `libm`-vs-`numpy` drift that the inner algebraic solvers accumulate at
//! near-singular configurations, guaranteeing the strict `1e-5` joint-space
//! agreement asserted by the reference test. The reference dataset is generated
//! with the byte-identical Levenberg-Marquardt polish, so both sides converge
//! to the same isolated root.

use nalgebra::SMatrix;

use dexhal::math::jointlock::{self, SphericalLockModel};
pub use dexhal::math::tolerances::TolerancePolicy;

use super::arm::FrankaPanda;
use super::consts;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the Franka Panda.
pub const DOF: usize = 7;

/// Upper bound on the number of distinct IK solutions (16 lock samples x up to
/// 8 inner solutions each). The Franka's exhaustive sweep can saturate this.
pub const MAX_SOLUTIONS: usize = 128;

/// Solver identifier — mirrors `SOLVER_NAME` in the Python artifact.
pub const SOLVER_NAME: &str = "jointlock.seven_r";

/// Solver tier — mirrors `SOLVER_TIER` in the Python artifact.
pub const SOLVER_TIER: u32 = 1;

/// Median solve time in milliseconds on commodity single-thread hardware —
/// mirrors `EXPECTED_MS_MEDIAN` in the Python artifact.
pub const EXPECTED_MS_MEDIAN: f64 = 50.0;

/// Approximate floating-point operation budget per solve —
/// mirrors `FLOP_BUDGET` in the Python artifact.
pub const FLOP_BUDGET: u32 = 30274;

/// Human-readable explanation of why this solver was chosen —
/// mirrors `DISPATCH_REASON` in the Python artifact.
pub const DISPATCH_REASON: &str = "7R revolute chain (non-SRS). Locking one joint\n\
    (auto-selected by topology rank of the resulting 6R\n\
    sub-chain) reduces this to a series of 6R IK problems.\n\
    Covers Franka Panda, FR3, uFactory xArm7, and any other\n\
    non-SRS 7R revolute arm.";

/// Name of the base (world) link — mirrors `BASE_LINK` in the Python artifact.
pub const BASE_LINK: &str = "base_link";

/// Name of the end-effector link — mirrors `EE_LINK` in the Python artifact.
pub const EE_LINK: &str = "ee_link";

/// Home pose: 4×4 SE(3) of `ee_link` in `base_link` at `q = [0; 7]`.
/// Mirrors `T_HOME` in the Python artifact; use as a sanity-check that the
/// baked geometry matches your URDF.
pub const T_HOME: [[f64; 4]; 4] = [
    [-0.7071068058785943, -0.7071067564945002, 0.0, 0.088],
    [
        -0.7071067564944993,
        0.7071068058785934,
        -4.440892098500621e-16,
        -3.241851231905455e-17,
    ],
    [
        3.1401848077128287e-16,
        -3.140185027022262e-16,
        -0.9999999999999987,
        0.9259999999999998,
    ],
    [0.0, 0.0, 0.0, 1.0],
];

/// The joint the sweep locks.
pub const LOCKED_JOINT: usize = 4;

/// Baked geometry of the 16 reversed, joint-4-locked sub-chains.
const MODEL: SphericalLockModel = SphericalLockModel {
    lock_samples: &consts::LOCK_SAMPLES,
    solver_kind: &consts::SOLVER_KIND,
    sub_axes: &consts::SUB_AXES,
    sub_tl: &consts::SUB_TL,
    sub_trr: &consts::SUB_TRR,
    sub_trt: &consts::SUB_TRT,
    axes7: &consts::AXES7,
    tl7: &consts::TL7,
    tr7_r: &consts::TR7_R,
    tr7_t: &consts::TR7_T,
    locked_joint: LOCKED_JOINT,
};

/// Iteration ceiling for the Levenberg-Marquardt pose-refinement polish.
#[cfg(test)]
const REFINE_ITERS: usize = 60;

/// Driver candidates with lock joint 4 pinned to `lock_value` (one of
/// [`consts::LOCK_SAMPLES`]), as a 7R configuration set. Unknown values give
/// no candidates. Used as the sample solver of [`seven_dof_one`].
///
/// [`seven_dof_one`]: dexhal::math::single::seven_dof_one
pub fn lock_sample_solutions(
    target_tf: &SMatrix<f64, 4, 4>,
    lock_value: f64,
    policy: TolerancePolicy,
) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    jointlock::lock_sample_solutions(&MODEL, target_tf, lock_value, policy)
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics — mirrors `franka_panda_ik.fk`.
///
/// `q` is a 7-element joint-angle vector in radians. Returns the 4×4 SE(3)
/// pose of `ee_link` in `base_link`.
///
/// Delegates to the SymPy-generated, branch-free symbolic FK in
/// [`super::generated_kinematics::fk`] (regenerate with `codegen/generate_dynamics.py`).
/// The solver's internal hot path keeps using [`jointlock::fk7`] so the
/// algebraic root refinement reproduces the reference dataset bit-for-bit.
pub fn fk(q: &[f64; 7]) -> SMatrix<f64, 4, 4> {
    let acc = super::generated_kinematics::fk(q);
    SMatrix::<f64, 4, 4>::from_fn(|i, j| acc[i][j])
}

/// Inverse kinematics — mirrors `franka_panda_ik.solve`.
///
/// # Arguments
///
/// * `target_tf` — 4×4 SE(3) target pose of `ee_link` in `base_link`.
/// * `max_solutions` — cap on returned solutions (post-dedup, post-limits).
///   `None` returns the full algebraic set (at most [`MAX_SOLUTIONS`]).
/// * `q_seed` — when `Some`, the solutions are sorted by wrap-to-pi distance
///   from the seed (closest first). The sweep always enumerates the full set;
///   seeded single-solution selection is handled by the caller (`flange_one`).
/// * `respect_limits` — when `true`, solutions outside the URDF joint limits
///   ([`FrankaPanda`]'s `JOINT_LIMITS`) are dropped (after a `±2π`
///   wrap-into-limits attempt). Pass `false` for the raw geometric set.
/// * `allow_refinement` — when `true`, each accepted algebraic candidate is
///   polished with a bounded Levenberg-Marquardt step (the same `refine7`
///   used by `codegen/generate_test_data.py`). Pass `true` to match
///   the reference dataset; `false` for raw algebraic roots.
/// * `policy` — FK closure and dedup tolerances; use
///   [`TolerancePolicy::DEFAULT`] for the Python defaults.
/// * `refinement_max_iters` — maximum LM iterations per candidate when
///   `allow_refinement = true`.
///
/// Returns up to [`MAX_SOLUTIONS`] candidate joint vectors `[q1, .., q7]`
/// in radians; unused slots are [`None`].
pub fn solve(
    target_tf: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 7]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    jointlock::solve::<FrankaPanda, MAX_SOLUTIONS>(
        &MODEL,
        target_tf,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        policy,
        refinement_max_iters,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dexhal::math::postprocess::wrap_distance;
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

    fn q_eq_mod(a: &[f64; 7], b: &[f64; 7], tol: f64) -> bool {
        for i in 0..7 {
            let d = libm::remainder(a[i] - b[i], 2.0 * core::f64::consts::PI);
            if libm::fabs(d) > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn fk_home_pose() {
        let zero = [0.0f64; 7];
        let t = fk(&zero);
        // T_HOME from the Python artifact (FK at q = zeros(7)).
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
        // T_HOME from the artifact (FK at q = 0).
        let t_home: [f64; 16] = [
            T_HOME[0][0], T_HOME[0][1], T_HOME[0][2], T_HOME[0][3],
            T_HOME[1][0], T_HOME[1][1], T_HOME[1][2], T_HOME[1][3],
            T_HOME[2][0], T_HOME[2][1], T_HOME[2][2], T_HOME[2][3],
            T_HOME[3][0], T_HOME[3][1], T_HOME[3][2], T_HOME[3][3],
        ];
        let tf = tf_from_row_major(&t_home);
        let sols = solve(&tf, None, None, false, true, TolerancePolicy::DEFAULT, REFINE_ITERS);
        let zero = [0.0; 7];
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-5));
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
            // Use allow_refinement=true with REFINE_ITERS=60 to match the
            // test data generated by codegen/generate_test_data.py.
            let sols = solve(&tf, None, None, false, true, TolerancePolicy::DEFAULT, REFINE_ITERS);
            let got: Vec<[f64; 7]> = sols.iter().flatten().copied().collect();

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
        // q=0 is the home pose; all joints are at 0 which is within limits
        // for joints 0,1,2,4,6.  Joint 3 limit is (-3.0718, -0.0698) which
        // does NOT include 0 — so q=0 should be dropped by respect_limits.
        let t_home: [f64; 16] = [
            T_HOME[0][0], T_HOME[0][1], T_HOME[0][2], T_HOME[0][3],
            T_HOME[1][0], T_HOME[1][1], T_HOME[1][2], T_HOME[1][3],
            T_HOME[2][0], T_HOME[2][1], T_HOME[2][2], T_HOME[2][3],
            T_HOME[3][0], T_HOME[3][1], T_HOME[3][2], T_HOME[3][3],
        ];
        let tf = tf_from_row_major(&t_home);
        // Without limits the raw geometric set should include q=0.
        let sols_raw =
            solve(&tf, None, None, false, true, TolerancePolicy::DEFAULT, REFINE_ITERS);
        let zero = [0.0f64; 7];
        let found_raw = sols_raw.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-5));
        assert!(found_raw, "q=0 should be present in the raw solution set");

        // With limits on, q=0 violates joint-3 limits, so it should be absent.
        let sols_lim =
            solve(&tf, None, None, true, true, TolerancePolicy::DEFAULT, REFINE_ITERS);
        let found_lim = sols_lim.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-5));
        assert!(
            !found_lim,
            "q=0 (joint 3 = 0, outside (-3.0718, -0.0698)) should be absent with respect_limits=true"
        );
    }

    #[test]
    fn max_solutions_cap() {
        let t_home: [f64; 16] = [
            T_HOME[0][0], T_HOME[0][1], T_HOME[0][2], T_HOME[0][3],
            T_HOME[1][0], T_HOME[1][1], T_HOME[1][2], T_HOME[1][3],
            T_HOME[2][0], T_HOME[2][1], T_HOME[2][2], T_HOME[2][3],
            T_HOME[3][0], T_HOME[3][1], T_HOME[3][2], T_HOME[3][3],
        ];
        let tf = tf_from_row_major(&t_home);
        let all_sols =
            solve(&tf, None, None, false, true, TolerancePolicy::DEFAULT, REFINE_ITERS);
        let n_all: usize = all_sols.iter().flatten().count();
        if n_all > 1 {
            let capped =
                solve(&tf, Some(1), None, false, true, TolerancePolicy::DEFAULT, REFINE_ITERS);
            let n_capped: usize = capped.iter().flatten().count();
            assert_eq!(n_capped, 1, "max_solutions=1 should return exactly 1 solution");
        }
    }

    #[test]
    fn q_seed_sorts_nearest_first() {
        let t_home: [f64; 16] = [
            T_HOME[0][0], T_HOME[0][1], T_HOME[0][2], T_HOME[0][3],
            T_HOME[1][0], T_HOME[1][1], T_HOME[1][2], T_HOME[1][3],
            T_HOME[2][0], T_HOME[2][1], T_HOME[2][2], T_HOME[2][3],
            T_HOME[3][0], T_HOME[3][1], T_HOME[3][2], T_HOME[3][3],
        ];
        let tf = tf_from_row_major(&t_home);
        let zero = [0.0f64; 7];
        let sols = solve(
            &tf,
            None,
            Some(&zero),
            false,
            true,
            TolerancePolicy::DEFAULT,
            REFINE_ITERS,
        );
        let solutions: Vec<[f64; 7]> = sols.iter().flatten().copied().collect();
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
}
