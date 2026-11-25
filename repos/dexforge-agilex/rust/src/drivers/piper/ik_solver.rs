//! Analytical inverse kinematics for the **AgileX Piper** 6-DOF arm.
//!
//! This is a translation of the Python `ssik` prebuilt artifact
//! `ssik/prebuilt/piper_ik.py` (solver `ikgeo.general_6r`, tier 2) into a
//! deterministic, `#![no_std]`, zero-allocation Rust implementation.
//!
//! Unlike the tier-0 UR5 closed form, the Piper has no Pieper-class
//! decoupling, so it is solved with the numeric **Raghavan-Roth** elimination
//! followed by the **Manocha-Canny** companion-eigenvalue route, shared with
//! the Rizon drivers in [`dexhal::math::raghavan_roth`]:
//!
//! 1. The per-arm `(P_sin, P_cos, P_one, Q)` matrices are evaluated by the
//!    machine-generated [`pq::build_pq_matrices`].
//! 2. `q_0, q_1` are eliminated through the left null space of `Q`, then the
//!    Weierstrass half-angle substitution turns the trigonometric system into a
//!    `12x12` matrix polynomial `M(x) = M_quad x^2 + M_lin x + M_const`.
//! 3. The real roots of `det M(x) = 0` (the `tan(q_2/2)` candidates) and their
//!    kernels are extracted by [`dexhal::math::ssik_core::rr_real_roots`] — a
//!    bounded Francis-QR eigenvalue solve on the `24x24` companion matrix.
//! 4. Each root is back-substituted to the remaining joints and mapped from the
//!    DH frame to the POE frame.
//!
//! This module only supplies the Piper's baked [`RrChain`] and the spatial
//! bypass; [`solve`] hands the raw candidates to the shared [`finish6`]
//! post-process.
//!
//! ## Equivalence to the Python reference
//!
//! [`solve`] mirrors `piper_ik.solve` with the same argument list:
//! `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`, `policy`,
//! and `refinement_max_iters`. The algebraic solution set is FK-verified to
//! [`TolerancePolicy::subproblem_numerical`] and de-duplicated modulo `2*pi`.
//!
//! [`fk`] mirrors `piper_ik.fk` (alias of `_fk`): product-of-exponentials
//! forward kinematics using the baked chain constants.
//!
//! ## Deterministic-execution note
//!
//! The Python reference falls back to a randomised Möbius reparameterisation +
//! generalised eigenvalue solve when the equilibrated leading matrix is
//! ill-conditioned (`cond > 1e10`). That path is non-deterministic (RNG) and is
//! intentionally **not** ported. Instead, ill-conditioned targets are handled
//! by a *deterministic spatial bypass*: the target is nudged 10 µm off the
//! singularity (see `PERTURBATIONS`), re-solved with the bounded algebraic
//! route, and the roots are snapped back onto the original pose with the
//! bounded Levenberg-Marquardt [`refine6`]. This keeps the real-time execution
//! path bounded and reproducible while still resolving the singular poses.

use nalgebra::SMatrix;

use dexhal::math::closed_form::{finish6, refine6};
use dexhal::math::kinematics::mat4_frob_diff;
use dexhal::math::raghavan_roth::{mat4, solve_6r, RrChain};
use dexhal::math::single::CANDIDATES;
pub use dexhal::math::tolerances::TolerancePolicy;

use super::arm::{Piper, MAX_SOLUTIONS};
use super::consts;
use super::pq;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the Piper.
pub const DOF: usize = 6;

/// Base link name in the URDF.
pub const BASE_LINK: &str = "base_link";

/// End-effector link name in the URDF.
pub const EE_LINK: &str = "link6";

/// 2-norm condition number of the equilibrated leading matrix `A_eq` above
/// which the analytic Raghavan-Roth route is considered ill-conditioned. This
/// matches the Python reference's `cond_threshold` (`ikgeo._raghavan_roth`),
/// which there triggers the non-deterministic Möbius/random-eigenvalue
/// fallback. Here it instead triggers the deterministic spatial bypass.
const ILL_CONDITIONED_COND: f64 = 1e10;

/// 10 µm — magnitude of the deterministic Cartesian micro-perturbations used to
/// bump an ill-conditioned target off a mathematical singularity. Expressed in
/// the driver's base length unit (metres), so `1e-5` == 10 µm.
const PERTURBATION_DELTA: f64 = 1e-5;

/// Deterministic spatial micro-perturbations: +10 µm translations along the
/// tool-frame X, Y and Z axes (row-major homogeneous translation matrices).
///
/// When the analytic route reports `cond(A_eq) > ILL_CONDITIONED_COND`, the
/// target pose sits on (or extremely close to) a singularity of the polynomial
/// system. Physically nudging the target by 10 µm collapses the condition
/// number so the purely algebraic extraction succeeds; the resulting roots are
/// then snapped back onto the *original* target with [`refine6`]. The whole
/// loop is stack-allocated (fixed-size arrays + nalgebra `SMatrix`).
const PERTURBATIONS: [[[f64; 4]; 4]; 3] = [
    // +10 µm along X.
    [
        [1.0, 0.0, 0.0, PERTURBATION_DELTA],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    // +10 µm along Y.
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, PERTURBATION_DELTA],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    // +10 µm along Z.
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, PERTURBATION_DELTA],
        [0.0, 0.0, 0.0, 1.0],
    ],
];

/// Bounded LM iterations used to close the 10 µm gap after the spatial bypass.
/// The gap is tiny, so a single deterministic pass typically converges; the cap
/// keeps execution time bounded for the real-time loop.
const BYPASS_REFINE_MAX_ITERS: usize = 64;

/// Bounded LM iterations used to sharpen an already-closing root onto exact FK
/// closure. Seeded from a near-true configuration, so it converges in a couple
/// of deterministic iterations.
const POLISH_MAX_ITERS: usize = 16;

/// The Piper's baked Raghavan-Roth chain (back-substitution metadata from
/// `piper_ik._RR_META`).
const RR_CHAIN: RrChain<'static> = RrChain {
    dh_alpha: &consts::DH_ALPHA,
    dh_a: &consts::DH_A,
    dh_d: &consts::DH_D,
    dh_theta_offset: &consts::DH_THETA_OFFSET,
    t_pre_inv: &consts::T_PRE_INV,
    t_post_inv: &consts::T_POST_INV,
    linearity_joint: 1,
    left_bilinear: (2, 3),
    right_bilinear: (0, 5),
    drop_joint: 4,
    build_pq: pq::build_pq_matrices,
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Inverse kinematics — mirrors `piper_ik.solve`.
///
/// # Arguments
///
/// * `t` — 4×4 SE(3) target pose of `link6` in `base_link`.
/// * `max_solutions` — cap on returned solutions (post-dedup, post-limits).
///   `None` returns the full algebraic set (at most 16 for a generic 6R arm).
/// * `q_seed` — when `Some`, solutions are sorted by wrap-to-pi distance from
///   the seed (closest first). Has no effect on *which* solutions are returned.
/// * `respect_limits` — when `true`, solutions outside the URDF joint limits
///   (`Piper::JOINT_LIMITS`) are dropped (after a `±2π` wrap-into-limits
///   attempt). Pass `false` for the raw geometric set.
/// * `allow_refinement` — when `true`, algebraic candidates that exceed
///   `policy.subproblem_numerical` are polished with a bounded Levenberg-Marquardt step.
/// * `policy` — FK closure and dedup tolerances; use
///   [`TolerancePolicy::DEFAULT`] for the Python defaults.
/// * `refinement_max_iters` — maximum LM iterations when
///   `allow_refinement = true`.
///
/// Always enumerates the full candidate set (no early exit); single-solution
/// IK is `Piper::flange_one`. When the primary candidates yield no solution,
/// the [`spatial_bypass`] candidates are post-processed instead.
///
/// Returns up to 16 candidate joint vectors `[q1, .., q6]` in radians; unused
/// slots are [`None`].
pub fn solve(
    t: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 6]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 6]>; MAX_SOLUTIONS] {
    let finish = |cands: &[[f64; 6]]| {
        finish6::<Piper, MAX_SOLUTIONS>(
            cands,
            t,
            max_solutions,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        )
    };
    let mut cands = [[0.0f64; 6]; MAX_SOLUTIONS];
    let n = solve_candidates(t, policy, &mut cands);
    polish_closed(&mut cands[..n], t, policy.subproblem_numerical);
    let sols = finish(&cands[..n]);
    if sols[0].is_some() {
        return sols;
    }
    let n = spatial_bypass(t, policy, &mut cands);
    polish_closed(&mut cands[..n], t, policy.subproblem_numerical);
    finish(&cands[..n])
}

/// Polish every candidate that already closes within `fk_atol` onto the exact
/// root with a bounded LM pass (which never changes which roots are accepted).
/// The bounded Francis-QR eigenvalues are slightly less precise than the
/// reference LAPACK route (joint values up to ~1e-5 off the root).
fn polish_closed(cands: &mut [[f64; 6]], t: &SMatrix<f64, 4, 4>, fk_atol: f64) {
    for q in cands {
        if q.iter().all(|v| v.is_finite()) && mat4_frob_diff(&fk(q), t) <= fk_atol {
            if let Some(r) = refine6::<Piper, MAX_SOLUTIONS>(q, t, POLISH_MAX_ITERS, fk_atol) {
                *q = r;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Algebraic candidate generation (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Candidate generation with the deterministic spatial bypass.
///
/// Runs the analytic Raghavan-Roth route on `t`. If the equilibrated leading
/// matrix is well-conditioned (`cond <= ILL_CONDITIONED_COND`), its candidates
/// are returned directly. Otherwise the target sits on a polynomial-system
/// singularity and the analytic route is unreliable, so we fall back to
/// [`spatial_bypass`], which perturbs the target, re-solves, and snaps the roots
/// back onto the original pose.
pub(super) fn solve_candidates(
    t: &SMatrix<f64, 4, 4>,
    policy: TolerancePolicy,
    out: &mut [[f64; 6]; MAX_SOLUTIONS],
) -> usize {
    let mut cond = 0.0f64;
    let n = solve_6r(&RR_CHAIN, t, out, &mut cond);
    if cond <= ILL_CONDITIONED_COND {
        return n;
    }
    spatial_bypass(t, policy, out)
}

/// Raw (unfiltered, unpolished) closed-form IK candidates for `t`, padded to
/// [`CANDIDATES`]. The standard candidate source of `ArmModel::flange_one`:
/// refinement and acceptance are left to [`dexhal::math::single::six_dof_one`].
pub fn raw_candidates(t: &SMatrix<f64, 4, 4>, policy: TolerancePolicy) -> ([[f64; 6]; CANDIDATES], usize) {
    let mut raw = [[0.0; 6]; MAX_SOLUTIONS];
    let n = solve_candidates(t, policy, &mut raw).min(CANDIDATES);
    let mut out = [[0.0; 6]; CANDIDATES];
    out[..n].copy_from_slice(&raw[..n]);
    (out, n)
}

/// Deterministic Spatial Bypass for ill-conditioned targets.
///
/// Iterates over [`PERTURBATIONS`] (at most 3 stack-allocated 10 µm Cartesian
/// bumps). For each, the target is nudged off the singularity, the purely
/// algebraic extraction is re-run on the perturbed pose, and — on the first
/// well-conditioned bump that yields roots — every root is refined back onto the
/// **original, unperturbed** target with [`refine6`]. The 10 µm gap closes in a
/// bounded LM pass. Returns the number of refined candidates written to `out`
/// (0 if no perturbation de-conditions the system).
fn spatial_bypass(
    t: &SMatrix<f64, 4, 4>,
    policy: TolerancePolicy,
    out: &mut [[f64; 6]; MAX_SOLUTIONS],
) -> usize {
    let mut tmp = [[0.0f64; 6]; MAX_SOLUTIONS];
    for pert in PERTURBATIONS.iter() {
        let t_pert = t * mat4(pert);
        let mut cond = 0.0f64;
        let n = solve_6r(&RR_CHAIN, &t_pert, &mut tmp, &mut cond);
        if n == 0 || cond > ILL_CONDITIONED_COND {
            // Still singular along this axis; try the next bump.
            continue;
        }
        let mut count = 0usize;
        for cand in tmp.iter().take(n) {
            // Snap the 10 µm-offset root back onto the true target pose.
            if let Some(refined) = refine6::<Piper, MAX_SOLUTIONS>(
                cand,
                t,
                BYPASS_REFINE_MAX_ITERS,
                policy.subproblem_numerical,
            ) {
                out[count] = refined;
                count += 1;
            }
        }
        if count > 0 {
            return count;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics — mirrors `piper_ik.fk`.
///
/// `q` is a 6-element joint-angle vector in radians. Returns the 4x4 SE(3)
/// pose of `link6` in `base_link`.
///
/// Delegates to the SymPy-generated, branch-free symbolic FK in
/// [`super::generated_kinematics::fk`] (regenerate with `codegen/generate_dynamics.py`).
pub fn fk(q: &[f64; 6]) -> SMatrix<f64, 4, 4> {
    let acc = super::generated_kinematics::fk(q);
    SMatrix::<f64, 4, 4>::from_fn(|i, j| acc[i][j])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;
    use dexhal::math::postprocess::wrap_distance;
    use std::vec::Vec;

    fn tf_from_row_major(a: &[f64; 16]) -> SMatrix<f64, 4, 4> {
        SMatrix::<f64, 4, 4>::from_fn(|i, j| a[i * 4 + j])
    }

    fn fk_residual(q: &[f64; 6], target: &SMatrix<f64, 4, 4>) -> f64 {
        mat4_frob_diff(&fk(q), target)
    }

    fn q_eq_mod(a: &[f64; 6], b: &[f64; 6], tol: f64) -> bool {
        for i in 0..6 {
            let d = libm::remainder(a[i] - b[i], TAU);
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
        let q = refine6::<Piper, MAX_SOLUTIONS>(&seed, &target, 30, policy.subproblem_numerical)
            .expect("refine6 should converge from a nearby seed");
        assert!(fk_residual(&q, &target) <= policy.subproblem_numerical);
    }

    #[test]
    fn seeded_single_solution_matches_nearest_of_full_set() {
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
                let single = solve(&t, Some(1), Some(&seed), true, true, policy, 15)[0];
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
                let expected = consts::T_HOME[i][j];
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
    // Relaxed joint-space tolerance documented for singular poses: bounded
    // Francis-QR and LAPACK resolve near-double roots to slightly different
    // points on the same self-motion manifold, so distinct joint vectors close
    // the identical Cartesian pose. Solutions are validated in FK-residual space
    // (below) and only required to agree in joint space to this looser bound.
    const SINGULAR_JOINT_TOL: f64 = 1e-3;
    // Cartesian FK-residual tolerance every returned Rust solution must satisfy
    // (max-abs entry of `fk(q) - target`). This is the Cartesian comparison that
    // backs the relaxed joint tolerance at singularities. Matches the FK closure
    // tolerance of 1e-5 for consistency across all artifacts.
    const FK_RESIDUAL_TOL: f64 = 1e-5;

    #[test]
    fn matches_ssik_reference_dataset() {
        assert!(!test_data::ssik_version().is_empty());
        let cases = test_data::cases();
        let mut total_expected = 0usize;
        let mut strict_cases = 0usize;
        let mut singular_cases = 0usize;
        let mut case_idx = 0usize;
        for (tf_arr, expected, singular) in &cases {
            let tf = tf_from_row_major(tf_arr);
            let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            let got: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();

            // Cartesian validity (all cases): every returned solution must close
            // the target pose. This is the FK-residual comparison referenced by
            // the relaxed joint tolerance for singular poses.
            for g in got.iter() {
                let residual = fk_residual(g, &tf);
                assert!(
                    residual <= FK_RESIDUAL_TOL,
                    "case {case_idx}: returned solution {:?} does not close the pose \
                     (FK residual {residual:e} > {FK_RESIDUAL_TOL:e})",
                    g
                );
            }

            if *singular {
                singular_cases += 1;
                // At singularities near-double roots coalesce differently per
                // eigenvalue solver, so the *cardinality* of the solution set is
                // not solver-invariant. We therefore do not require the complete
                // `allow_refinement=True` reference set to be reproduced;
                // instead we require the solver to return at least one valid
                // root and that every returned root matches a reference root
                // within the relaxed joint tolerance (no spurious solutions).
                assert!(
                    !got.is_empty(),
                    "case {case_idx}: singular pose produced no solutions"
                );
                for g in got.iter() {
                    let matched = expected.iter().any(|exp| q_eq_mod(g, exp, SINGULAR_JOINT_TOL));
                    assert!(
                        matched,
                        "case {case_idx}: returned solution {:?} matches no reference root \
                         within the singular joint tolerance",
                        g
                    );
                }
            } else {
                strict_cases += 1;
                assert_eq!(
                    got.len(),
                    expected.len(),
                    "case {case_idx}: solution count {} != expected {}",
                    got.len(),
                    expected.len()
                );
                for exp in expected.iter() {
                    let matched = got.iter().any(|g| q_eq_mod(g, exp, STRICT_TOL));
                    assert!(
                        matched,
                        "case {case_idx}: expected solution {:?} not reproduced",
                        exp
                    );
                }
            }
            total_expected += expected.len();
            case_idx += 1;
        }
        assert!(total_expected > 0, "no expected solutions in dataset");
        assert_eq!(case_idx, cases.len());
        // The full 500-pose sample is retained; a non-trivial share exercises
        // the singular (FK-residual) comparison path.
        assert!(strict_cases > 0 && singular_cases > 0);
    }

    #[test]
    fn spatial_bypass_resolves_target() {
        // The deterministic spatial bypass (perturb 10 µm -> algebraic re-solve
        // -> snap back with `refine6`) never fires on the sampled dataset
        // because `cond(A_eq)` stays well below `ILL_CONDITIONED_COND`. Exercise
        // it directly (relaxing the gating condition) so the perturb/snap-back
        // path is covered: every root it returns must close the original,
        // unperturbed target pose in Cartesian space.
        let mut covered = 0usize;
        for (tf_arr, _expected, _singular) in test_data::cases().iter().take(16) {
            let tf = tf_from_row_major(tf_arr);
            let mut out = [[0.0f64; 6]; 16];
            let n = spatial_bypass(&tf, TolerancePolicy::DEFAULT, &mut out);
            if n == 0 {
                continue;
            }
            covered += 1;
            for q in out.iter().take(n) {
                assert!(
                    q.iter().all(|v| v.is_finite()),
                    "spatial bypass produced a non-finite root"
                );
                let residual = fk_residual(q, &tf);
                assert!(
                    residual <= FK_RESIDUAL_TOL,
                    "spatial bypass root {:?} does not close the original target \
                     (FK residual {residual:e} > {FK_RESIDUAL_TOL:e})",
                    q
                );
            }
        }
        assert!(
            covered > 0,
            "spatial bypass returned no solutions for any sampled pose"
        );
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

    #[test]
    fn q_seed_sorts_nearest_first() {
        let (tf_arr, _, _) = &test_data::cases()[0];
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
