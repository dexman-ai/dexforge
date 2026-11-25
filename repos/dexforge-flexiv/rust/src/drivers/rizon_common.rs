//! Shared deterministic IK solver for the Flexiv **Rizon** arm family
//! (`jointlock.seven_r` outer sweep + Raghavan-Roth 6R inner solver).
//!
//! Both Rizon variants (Rizon 4 / Rizon 10) are 7R, non-SRS arms with no
//! closed-form 6R structure, so the `ssik` solver `jointlock.seven_r` collapses
//! them to a series of 6R problems by **locking joint 2** (no chain reversal) to
//! each of 16 fixed sample values. Unlike the xArm7 / Franka members of this
//! class — whose locked sub-chains are spherical — the Rizon sub-chains are
//! generic 6R, so each one is solved with the deterministic **Raghavan-Roth**
//! kernel that already backs the Piper driver
//! ([`crate::math::ssik_core`]'s `rr_*` functions).
//!
//! The two arms differ only in their baked geometry, so this module factors the
//! whole pipeline behind a [`Model`] of `&'static` per-sample constants and
//! per-sample [`PqBuilder`] function pointers. Each arm's `ik_solver` supplies a
//! `Model` built from its generated `consts` / `pq` modules.
//!
//! ## Algorithm
//!
//! For every lock sample `s`:
//!
//! 1. Bridge the target into the sub-chain DH frame:
//!    `T_dh = T_PRE_INV[s] · T · T_POST_INV[s]`.
//! 2. Evaluate the per-sample `(P_sin, P_cos, P_one, Q)` matrices with
//!    `build_pq[s]`, eliminate `q0, q1` through the left null space of `Q`, run
//!    the Weierstrass substitution, and extract the real `tan(q/2)` roots and
//!    kernels via the bounded `24×24` companion eigenvalue solve — all through
//!    the shared [`solve_6r`].
//! 3. Back-substitute each root to the six sub-chain joints, map DH→POE, and
//!    pad the locked angle back in at index 2
//!    (`full = [sub0, sub1, q_lock, sub2, sub3, sub4, sub5]`).
//!
//! The collected candidates are FK-verified against the full 7R chain
//! ([`fk`]'s closure `< subproblem_numerical`), de-duplicated modulo `2π`, and —
//! when `allow_refinement` is set — polished by the analytic masked
//! Levenberg-Marquardt step ([`refine`]) that holds the locked joint fixed.
//!
//! [`solve`] always enumerates the full sweep. Single solutions are produced by
//! the shared [`crate::math::single::seven_dof_one`] helper, which draws its
//! per-sample candidates from [`sample_candidates`].
//!
//! ## Deterministic-execution note
//!
//! As in the Piper port, the reference's randomised Möbius fallback for
//! ill-conditioned sub-systems is intentionally **not** ported: every numeric
//! loop carries a hard iteration ceiling and every buffer is a fixed-size,
//! stack-allocated array. The deterministic kernel reproduces every reference
//! solution and may additionally surface a few extra *genuine* roots (each
//! verified to close the pose to `< subproblem_numerical`).
//!
//! ## What remains Rizon-specific in this module
//!
//! After deduplication with the generic math utilities, the code in this module
//! is exclusively Rizon-specific and cannot be generalized:
//!
//! * **DH-based FK** ([`fk`]) — product-of-exponentials FK using
//!   the per-arm Model's axes and transforms. Cannot be moved to kinematics
//!   because it depends on per-arm geometry stored in the Model.
//! * **Joint-masked LM refinement** ([`refine`]) — builds the free-joint index
//!   set (all DOF except the locked joint) and delegates to the shared analytic
//!   [`crate::math::kinematics::lm_refine_masked_analytic`] loop. The masking
//!   strategy is specific to the `jointlock.seven_r` redundancy resolution.
//! * **Sample-level solver** ([`solve_sample`], [`pad_locked`]) — selects the
//!   per-sample [`RrChain`] and pads the locked angle back in.
//!
//! Generic utilities that were previously duplicated here have been moved to:
//! * [`crate::math::kinematics`]: `norm12`, LM constants, and the shared LM
//!   loops (`lm_refine`, `lm_refine_masked`, `lm_refine_masked_analytic`)
//! * [`crate::math::postprocess`]: `dedup_solutions`, `wrap_to_limits`,
//!   `respect_limits`, `nearest_to_seed` (on `A::JOINT_LIMITS`)
//! * [`dexhal::math::raghavan_roth`]: the whole 6R Raghavan-Roth candidate
//!   route (`solve_6r`, `back_substitute`, `ratio_from`, `embed_e`,
//!   `dh_matrix`, `W_TRIG_TO_X`, the ratio tables, matrix conversions)
//! * [`dexhal::math::closed_form`]: `pose_residual`, `rotation_of`

use nalgebra::SMatrix;

use dexhal::math::closed_form::{pose_residual, rotation_of};
use dexhal::math::kinematics::{
    lm_refine_masked_analytic, mat4_frob_diff, mat4_mul, residual_columns, rot_homogeneous,
};
use dexhal::math::postprocess::{
    dedup_solutions, nearest_to_seed, respect_limits as filter_limits, wrap_to_limits,
};
pub use dexhal::math::raghavan_roth::PqBuilder;
use dexhal::math::raghavan_roth::{solve_6r, RrChain};
pub use dexhal::math::tolerances::TolerancePolicy;
use dexhal::math::ArmModel;


/// Degrees of freedom of every Rizon arm.
pub const DOF: usize = 7;

/// Number of locked-joint samples in the sweep.
pub const N_SAMPLES: usize = 16;

/// Maximum inner Raghavan-Roth candidates kept per lock sample.
const INNER_MAX: usize = 16;

/// Upper bound on raw 7R candidates collected across the sweep
/// (`N_SAMPLES * INNER_MAX`) before de-duplication.
const CAND_CAP: usize = N_SAMPLES * INNER_MAX;

/// Upper bound on the number of distinct IK solutions returned. The Rizon
/// exhaustive sweep can surface more roots than the wrist-decoupled 7R arms
/// (the deterministic RR kernel adds genuine extra roots at some samples), so
/// the output array is sized to [`CAND_CAP`].
pub const MAX_SOLUTIONS: usize = CAND_CAP;


/// Per-arm baked geometry and per-sample Raghavan-Roth derivation data.
///
/// All fields are `&'static` references into the arm's generated `consts` / `pq`
/// modules, so a `Model` is a thin, copyable bundle of pointers with no owned
/// data.
#[derive(Clone, Copy)]
pub struct Model {
    /// Locked joint index (pinned across the sweep; `2` for both Rizon arms).
    pub lock_idx: usize,
    /// Locked-joint sample schedule.
    pub lock_samples: &'static [f64; N_SAMPLES],
    /// Per-sample 6R sub-chain DH `alpha`.
    pub dh_alpha: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample 6R sub-chain DH `a`.
    pub dh_a: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample 6R sub-chain DH `d`.
    pub dh_d: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample DH→POE `theta` offsets.
    pub dh_theta_offset: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample `T_PRE_INV` bridge (row-major 4×4).
    pub t_pre_inv: &'static [[[f64; 4]; 4]; N_SAMPLES],
    /// Per-sample `T_POST_INV` bridge (row-major 4×4).
    pub t_post_inv: &'static [[[f64; 4]; 4]; N_SAMPLES],
    /// Per-sample RR linearity joint.
    pub linearity_joint: &'static [usize; N_SAMPLES],
    /// Per-sample RR left-bilinear joint pair.
    pub left_bilinear: &'static [(usize, usize); N_SAMPLES],
    /// Per-sample RR right-bilinear joint pair.
    pub right_bilinear: &'static [(usize, usize); N_SAMPLES],
    /// Per-sample RR dropped joint.
    pub drop_joint: &'static [usize; N_SAMPLES],
    /// Per-sample PQ builder table.
    pub build_pq: &'static [PqBuilder; N_SAMPLES],
    /// Full 7R joint axes (base frame at `q = 0`).
    pub axes7: &'static [[f64; 3]; 7],
    /// Full 7R per-joint `T_left` (row-major 4×4).
    pub t_lefts7: &'static [[[f64; 4]; 4]; 7],
    /// Full 7R per-joint `T_right` (row-major 4×4).
    pub t_rights7: &'static [[[f64; 4]; 4]; 7],
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Every inverse-kinematics solution of a Rizon arm described by `model`.
///
/// Enumerates the full sweep (every lock sample, every 6R candidate) without
/// early exit. Returns up to [`MAX_SOLUTIONS`] joint vectors in radians; unused
/// slots are [`None`]. With `q_seed`, solutions are ordered nearest-first.
pub fn solve<A: ArmModel<DOF, MAX_SOLUTIONS>>(
    model: &Model,
    t: &SMatrix<f64, 4, 4>,
    q_seed: Option<&[f64; 7]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    let mut cand_q = [[0.0f64; 7]; CAND_CAP];
    let mut cand_res = [0.0f64; CAND_CAP];
    let mut n_cand = 0usize;
    let mut inner = [[0.0f64; 6]; INNER_MAX];

    for sample in 0..N_SAMPLES {
        let q_lock = model.lock_samples[sample];
        let n_inner = solve_sample(model, sample, t, &mut inner);
        for sub in inner.iter().take(n_inner) {
            let full = pad_locked(model.lock_idx, sub, q_lock);
            if !full.iter().all(|v| v.is_finite()) {
                continue;
            }
            let residual = fk_residual(model, &full, t);
            if residual > policy.subproblem_numerical {
                continue;
            }
            if n_cand < CAND_CAP {
                cand_q[n_cand] = full;
                cand_res[n_cand] = residual;
                n_cand += 1;
            }
        }
    }

    // De-duplicate modulo 2π, keeping the lower-residual representative.
    let mut d_q = [[0.0f64; 7]; CAND_CAP];
    let mut d_res = [0.0f64; CAND_CAP];
    let n_d = dedup_solutions(
        &cand_q,
        &cand_res,
        n_cand,
        &mut d_q,
        &mut d_res,
        policy.subproblem_dedup,
    );

    let mut sols = [[0.0f64; 7]; MAX_SOLUTIONS];
    for (slot, q) in sols.iter_mut().zip(d_q.iter()).take(n_d) {
        *slot = if allow_refinement {
            refine::<A>(model, q, t, refinement_max_iters)
        } else {
            *q
        };
    }
    let mut count = n_d;
    if respect_limits {
        wrap_to_limits(&mut sols, count, &A::JOINT_LIMITS);
        count = filter_limits(&mut sols, count, &A::JOINT_LIMITS);
    }
    if let Some(seed) = q_seed {
        nearest_to_seed(&mut sols, count, seed);
    }

    let mut solutions: [Option<[f64; 7]>; MAX_SOLUTIONS] = [None; MAX_SOLUTIONS];
    for (o, q) in solutions.iter_mut().zip(sols.iter()).take(count) {
        *o = Some(*q);
    }
    solutions
}

/// Raw 7R candidates of the lock sample equal to `lock_value` (one 6R solve,
/// not FK-verified), with the lock joint at `lock_value`. Empty when
/// `lock_value` is not one of `model.lock_samples`.
pub fn sample_candidates(
    model: &Model,
    lock_value: f64,
    t: &SMatrix<f64, 4, 4>,
) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    let mut out = [None; MAX_SOLUTIONS];
    let Some(sample) = model.lock_samples.iter().position(|&s| s == lock_value) else {
        return out;
    };
    let mut inner = [[0.0f64; 6]; INNER_MAX];
    let n = solve_sample(model, sample, t, &mut inner);
    for (slot, sub) in out.iter_mut().zip(inner.iter().take(n)) {
        *slot = Some(pad_locked(model.lock_idx, sub, lock_value));
    }
    out
}

/// Pad the six sub-chain joints with the locked angle at `lock_idx`.
///
/// For the Rizon arms (`lock_idx = 2`, no chain reversal) this is
/// `[sub0, sub1, q_lock, sub2, sub3, sub4, sub5]`.
fn pad_locked(lock_idx: usize, sub: &[f64; 6], q_lock: f64) -> [f64; 7] {
    let mut full = [0.0f64; 7];
    let mut k = 0usize;
    for (i, slot) in full.iter_mut().enumerate() {
        if i == lock_idx {
            *slot = q_lock;
        } else {
            *slot = sub[k];
            k += 1;
        }
    }
    full
}

// ---------------------------------------------------------------------------
// Per-sample Raghavan-Roth inner solver (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Solve the locked 6R sub-chain for lock sample `sample`, writing up to
/// [`INNER_MAX`] sub-chain joint vectors to `out`. Returns the count.
fn solve_sample(
    model: &Model,
    sample: usize,
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]; INNER_MAX],
) -> usize {
    let chain = RrChain {
        dh_alpha: &model.dh_alpha[sample],
        dh_a: &model.dh_a[sample],
        dh_d: &model.dh_d[sample],
        dh_theta_offset: &model.dh_theta_offset[sample],
        t_pre_inv: &model.t_pre_inv[sample],
        t_post_inv: &model.t_post_inv[sample],
        linearity_joint: model.linearity_joint[sample],
        left_bilinear: model.left_bilinear[sample],
        right_bilinear: model.right_bilinear[sample],
        drop_joint: model.drop_joint[sample],
        build_pq: model.build_pq[sample],
    };
    let mut cond = 0.0f64;
    solve_6r(&chain, t, out, &mut cond)
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics of the full 7R chain — mirrors
/// `rizon{4,10}_ik.fk`. Returns the 4×4 SE(3) pose of the flange in `base_link`.
///
/// **Rizon-specific rationale**: This FK uses per-arm geometry from the Model
/// (axes7, t_lefts7, t_rights7). It cannot be moved to a generic module because
/// each arm has different kinematic parameters stored in its Model instance.
pub fn fk(model: &Model, q: &[f64; 7]) -> SMatrix<f64, 4, 4> {
    let mut acc = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for i in 0..7 {
        let rot = rot_homogeneous(&model.axes7[i], q[i]);
        let m = mat4_mul(&model.t_lefts7[i], &rot);
        let n = mat4_mul(&m, &model.t_rights7[i]);
        acc = mat4_mul(&acc, &n);
    }
    SMatrix::<f64, 4, 4>::from_fn(|i, j| acc[i][j])
}

/// Frobenius-norm residual between FK(q) and the target pose.
fn fk_residual(model: &Model, q: &[f64; 7], target: &SMatrix<f64, 4, 4>) -> f64 {
    mat4_frob_diff(&fk(model, q), target)
}

// ---------------------------------------------------------------------------
// Levenberg-Marquardt refinement (holds the locked joint fixed)
// ---------------------------------------------------------------------------

/// Analytic Levenberg-Marquardt polish of a 7R configuration, **holding the
/// locked joint fixed** at its value in `q_in`. Refining only the six free
/// joints targets the isolated 6R root and keeps the redundant 7R from drifting
/// along its self-motion manifold.
fn refine<A: ArmModel<DOF, MAX_SOLUTIONS>>(
    model: &Model,
    q_in: &[f64; 7],
    target: &SMatrix<f64, 4, 4>,
    max_iters: usize,
) -> [f64; 7] {
    let free: [usize; 6] = core::array::from_fn(|i| if i < model.lock_idx { i } else { i + 1 });
    lm_refine_masked_analytic::<7, 6, _, _>(
        q_in,
        &free,
        |q| pose_residual(&A::flange_fk(q), target),
        |q| residual_columns(&rotation_of(&A::flange_fk(q)), &A::flange_jacobian(q)),
        max_iters,
    )
}
