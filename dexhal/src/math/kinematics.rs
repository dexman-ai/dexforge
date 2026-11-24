//! Generic kinematics helpers shared across all per-arm IK drivers.
//!
//! This module collects arm-agnostic linear-algebra primitives and the
//! shared bounded Levenberg-Marquardt (LM) refinement loop that were
//! previously duplicated inside each driver.
//!
//! # What lives here
//!
//! * **4×4 homogeneous utilities**: [`eye_with`], [`rot_homogeneous`], [`mat4_mul`]
//! * **2×2 matrix helpers**: [`mat2_ata`], [`vec2_mat2`]
//! * **Pose residual norm**: [`norm12`]
//! * **Rigid-transform type**: [`Transform`], [`IDENTITY`], [`compose`],
//!   [`se3_inv`], [`frob_diff`]
//! * **Shared LM tuning constants** and **generic const-DOF refinement loops**:
//!   [`lm_refine`], [`lm_refine6`] (refine all DOF) and [`lm_refine_masked`]
//!   (refine a subset of joints, holding the rest fixed)
//!
//! # What stays in each driver
//!
//! * Arm-specific FK (`fk`, `fk7`, `fk_sub`, …) and the pose-residual
//!   functions that call them — they encode per-arm kinematics constants.
//! * Each 7R `jointlock.seven_r` driver's `refine7` is now a thin wrapper that
//!   selects the free-joint set (excluding the locked joint) and delegates to
//!   the shared [`lm_refine_masked`] loop.
//! * Piper `spatial_bypass` and perturbation constants — the 10 µm
//!   deterministic singularity escape is specific to the Raghavan-Roth
//!   polynomial-system condition check of that driver.

use nalgebra::SMatrix;

use super::ssik_core::mat3_mul;

use crate::math::ssik_core::{add3, mat3_transpose, mat3_vec, neg3};


// ---------------------------------------------------------------------------
// 4×4 homogeneous-matrix utilities
// ---------------------------------------------------------------------------

/// 4×4 homogeneous identity with the given translation column.
#[inline]
pub const fn eye_with(x: f64, y: f64, z: f64) -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Homogeneous Rodrigues rotation about unit `axis` by `angle` radians.
///
/// Returns a 4×4 SE(3) matrix with zero translation.
#[inline]
pub fn rot_homogeneous(axis: &[f64; 3], angle: f64) -> [[f64; 4]; 4] {
    let c = libm::cos(angle);
    let s = libm::sin(angle);
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let oc = 1.0 - c;
    [
        [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s, 0.0],
        [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s, 0.0],
        [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// 4×4 matrix product `A * B`.
#[inline]
pub fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 2×2 matrix helpers (used by the SP6 LM Jacobian normal equations)
// ---------------------------------------------------------------------------

/// `M^T M` for a 2×2 matrix stored row-major.
#[inline]
pub fn mat2_ata(m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let c00 = m[0][0] * m[0][0] + m[1][0] * m[1][0];
    let c01 = m[0][0] * m[0][1] + m[1][0] * m[1][1];
    let c11 = m[0][1] * m[0][1] + m[1][1] * m[1][1];
    [[c00, c01], [c01, c11]]
}

/// `v^T M` (1×2) for 2-vector `v` and 2×2 matrix `M`.
#[inline]
pub fn vec2_mat2(v: &[f64; 2], m: &[[f64; 2]; 2]) -> [f64; 2] {
    [
        v[0] * m[0][0] + v[1] * m[1][0],
        v[0] * m[0][1] + v[1] * m[1][1],
    ]
}

// ---------------------------------------------------------------------------
// Pose residual norm
// ---------------------------------------------------------------------------

/// L2 norm of the 12-vector pose residual `[vec(ΔR); Δt]`.
#[inline]
pub fn norm12(r: &[f64; 12]) -> f64 {
    let mut s = 0.0;
    for v in r.iter() {
        s += v * v;
    }
    libm::sqrt(s)
}

// ---------------------------------------------------------------------------
// Rigid-transform type
// ---------------------------------------------------------------------------

/// A rigid-body transform stored as a rotation block and translation vector.
///
/// Equivalent to the upper 3×4 block of a 4×4 SE(3) homogeneous matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub r: [[f64; 3]; 3],
    pub t: [f64; 3],
}

/// The identity element of SE(3).
pub const IDENTITY: Transform = Transform {
    r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    t: [0.0, 0.0, 0.0],
};

/// Compose two transforms: `a ∘ b`.
#[inline]
pub fn compose(a: &Transform, b: &Transform) -> Transform {
    Transform {
        r: mat3_mul(&a.r, &b.r),
        t: add3(&mat3_vec(&a.r, &b.t), &a.t),
    }
}

/// SE(3) inverse: `[R^T | -R^T t]`.
#[inline]
pub fn se3_inv(tf: &Transform) -> Transform {
    let rt = mat3_transpose(&tf.r);
    Transform {
        r: rt,
        t: neg3(&mat3_vec(&rt, &tf.t)),
    }
}

/// Frobenius norm of the pose difference between two transforms.
///
/// Equivalent to the Frobenius norm of the 4×4 matrix difference (the
/// identical bottom row `[0, 0, 0, 1]` contributes nothing).
#[inline]
pub fn frob_diff(a: &Transform, b: &Transform) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = a.r[i][j] - b.r[i][j];
            s += d * d;
        }
        let d = a.t[i] - b.t[i];
        s += d * d;
    }
    libm::sqrt(s)
}

/// Frobenius norm of the difference between two 4×4 homogeneous matrices.
///
/// This is the `[f64; 12]`-free counterpart of [`frob_diff`] for callers that
/// already hold their pose as an `SMatrix<f64, 4, 4>` (the per-arm FK output
/// type). It sums the squared difference over **all 16 entries** — including
/// the bottom row — exactly reproducing each driver's former local
/// `fk_residual` kernel.
#[inline]
pub fn mat4_frob_diff(a: &SMatrix<f64, 4, 4>, b: &SMatrix<f64, 4, 4>) -> f64 {
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = a[(i, j)] - b[(i, j)];
            s += d * d;
        }
    }
    libm::sqrt(s)
}

/// 12-vector pose residual `[vec(a.R − b.R); a.t − b.t]` between two transforms.
///
/// The first 9 elements are the row-major flattening of `(a.R − b.R)` and
/// the last 3 are the translation difference `a.t − b.t`.  This is the
/// standard residual vector consumed by [`lm_refine`] and [`dls_ik`].
#[inline]
pub fn pose_residual12(a: &Transform, b: &Transform) -> [f64; 12] {
    let mut r = [0.0f64; 12];
    let mut k = 0usize;
    for i in 0..3 {
        for j in 0..3 {
            r[k] = a.r[i][j] - b.r[i][j];
            k += 1;
        }
    }
    for i in 0..3 {
        r[9 + i] = a.t[i] - b.t[i];
    }
    r
}

/// Columns of the `12 × DOF` Jacobian of [`pose_residual12`] from a geometric
/// Jacobian.
///
/// `rot` is the current rotation of the controlled frame and `jac` its base-frame
/// geometric Jacobian `[Jv; Jw]`. A spatial angular velocity `ω` gives
/// `∂R = [ω]× R`, and the linear rows are `∂t = Jv`. Column `j` of the result is
/// `∂residual/∂q_j`, in the layout of [`pose_residual12`].
pub fn residual_columns<const DOF: usize>(
    rot: &[[f64; 3]; 3],
    jac: &[[f64; DOF]; 6],
) -> [[f64; 12]; DOF] {
    let mut cols = [[0.0f64; 12]; DOF];
    for j in 0..DOF {
        let w = [jac[3][j], jac[4][j], jac[5][j]];
        let skew = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
        let dr = mat3_mul(&skew, rot);
        for row in 0..3 {
            for col in 0..3 {
                cols[j][3 * row + col] = dr[row][col];
            }
        }
        cols[j][9] = jac[0][j];
        cols[j][10] = jac[1][j];
        cols[j][11] = jac[2][j];
    }
    cols
}

// ---------------------------------------------------------------------------
// Shared Levenberg-Marquardt tuning constants
//
// Identical across UR5, Piper, and Franka. Centralised here so a single
// edit propagates to all arms.
// ---------------------------------------------------------------------------

/// Finite-difference step for the numerical Jacobian column.
pub const REFINE_EPS: f64 = 1e-7;
/// Residual below which the LM loop considers the solution converged.
pub const LM_CONVERGENCE_THRESHOLD: f64 = 1e-13;
/// Starting value for the Levenberg-Marquardt damping parameter λ.
pub const LM_INITIAL_LAMBDA: f64 = 1e-6;
/// Multiplicative growth applied to λ when a step is rejected.
pub const LM_LAMBDA_GROWTH: f64 = 8.0;
/// Multiplicative decay applied to λ when a step is accepted.
pub const LM_LAMBDA_DECAY: f64 = 0.5;
/// Lower clamp on λ (prevents underflow).
pub const LM_LAMBDA_MIN: f64 = 1e-12;
/// Upper clamp on λ; when exceeded the backtrack inner loop aborts.
pub const LM_LAMBDA_MAX: f64 = 1e8;
/// Maximum accept/reject backtrack attempts per outer LM iteration.
pub const LM_BACKTRACK_MAX: usize = 12;

// ---------------------------------------------------------------------------
// Generic const-DOF Levenberg-Marquardt refinement
// ---------------------------------------------------------------------------

/// Bounded Levenberg-Marquardt polish for a joint vector of **any** DOF.
///
/// This is the universal LM loop shared across all arm drivers.  The
/// arm-specific part — computing the 12-vector pose residual from the arm's
/// own FK — is supplied as a closure so the function remains free of per-arm
/// types and constants.  The joint-space dimension is a const generic
/// parameter `DOF`, so all matrix allocations (Jacobian `J : 12×DOF`,
/// normal-equations `JᵀJ : DOF×DOF`, step `dq : DOF×1`) live entirely on
/// the stack: no heap, no `Vec`, compatible with `#![no_std]`.
///
/// `residual_fn(q)` must return `[vec(R(q) − R_t); t(q) − t_t]` — the
/// standard 12-vector pose residual.  The closure typically captures the
/// target pose and delegates to the driver's local `pose_residual12`.
///
/// The const generic parameter is monomorphised at each call site, so the
/// compiled cost is identical to having the loop inlined in each driver.
///
/// Returns `Some(q_refined)` when the FK residual falls within `fk_atol`,
/// or `None` if the iteration budget is exhausted without convergence.
///
/// # Examples
///
/// ```ignore
/// // 6-DOF arm (same as calling lm_refine6 directly)
/// lm_refine::<6, _>(&q6, |q| residual12(q, &target), max_iters, 1e-10);
///
/// // 7-DOF arm — all seven joints are refined simultaneously
/// lm_refine::<7, _>(&q7, |q| residual12_7dof(q, &target), max_iters, 1e-10);
///
/// // 8+ DOF gantry
/// lm_refine::<8, _>(&q8, |q| residual12_8dof(q, &target), max_iters, 1e-10);
/// ```
pub fn lm_refine<const DOF: usize, F>(
    q_in: &[f64; DOF],
    residual_fn: F,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; DOF]>
where
    F: Fn(&[f64; DOF]) -> [f64; 12],
{
    // Refine all DOF joints (free = [0, 1, …, DOF−1]).
    let free: [usize; DOF] = core::array::from_fn(|i| i);
    let (q, rn) = lm_refine_core::<DOF, DOF, F, _>(
        q_in,
        &free,
        &residual_fn,
        |q: &[f64; DOF], r: &[f64; 12]| fd_jacobian(&residual_fn, &free, q, r),
        max_iters,
    );
    if rn <= fk_atol {
        Some(q)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Joint-masked Levenberg-Marquardt refinement
// ---------------------------------------------------------------------------

/// Bounded Levenberg-Marquardt polish that refines **only a subset of joints**.
///
/// This is the universal masked LM loop shared by every redundant 7R
/// `jointlock.seven_r` driver (Franka, xArm7, Rizon).  Those solvers collapse
/// a redundant arm to a 6R sub-problem by locking one joint to a sample value;
/// the polish must then refine the remaining free joints while **holding the
/// locked joint fixed**, otherwise the redundant arm would drift along its
/// self-motion manifold instead of converging onto the isolated root.
///
/// The set of free joints is supplied as `free` (a `[usize; FREE]` array of
/// joint indices into the `DOF`-vector); the locked joint is simply any index
/// absent from `free`.  Both the full joint count `DOF` and the free-joint
/// count `FREE` are const generic parameters, so every matrix
/// (Jacobian `J : 12×FREE`, normal equations `JᵀJ : FREE×FREE`, step
/// `dq : FREE×1`) lives entirely on the stack — no heap, `#![no_std]`-safe,
/// and hard-bounded by `max_iters`.
///
/// `residual_fn(q)` must return the standard 12-vector pose residual
/// `[vec(R(q) − R_t); t(q) − t_t]` of the **full** `DOF`-joint FK.
///
/// Unlike [`lm_refine`], this returns the refined joint vector unconditionally
/// (the redundant-arm drivers FK-verify and de-duplicate the result
/// downstream, so there is no convergence/`fk_atol` gate here).
///
/// # Examples
///
/// ```ignore
/// // 7R arm with joint 4 locked: refine joints {0,1,2,3,5,6}.
/// const FREE: [usize; 6] = [0, 1, 2, 3, 5, 6];
/// let q = lm_refine_masked::<7, 6, _>(&q7, &FREE, |q| residual12(q, &target), max_iters);
/// ```
pub fn lm_refine_masked<const DOF: usize, const FREE: usize, F>(
    q_in: &[f64; DOF],
    free: &[usize; FREE],
    residual_fn: F,
    max_iters: usize,
) -> [f64; DOF]
where
    F: Fn(&[f64; DOF]) -> [f64; 12],
{
    lm_refine_core::<DOF, FREE, F, _>(
        q_in,
        free,
        &residual_fn,
        |q: &[f64; DOF], r: &[f64; 12]| fd_jacobian(&residual_fn, free, q, r),
        max_iters,
    )
    .0
}

/// [`lm_refine_masked`] with a caller-supplied analytic Jacobian.
///
/// `jac_fn(q)` returns the full `12×DOF` Jacobian of `residual_fn` as an
/// array of columns (`jac_fn(q)[j]` is `∂residual/∂q_j`, in the same row
/// layout as [`pose_residual12`]). Only the columns listed in `free` are used,
/// so the locked joint's column is ignored.
pub fn lm_refine_masked_analytic<const DOF: usize, const FREE: usize, F, J>(
    q_in: &[f64; DOF],
    free: &[usize; FREE],
    residual_fn: F,
    jac_fn: J,
    max_iters: usize,
) -> [f64; DOF]
where
    F: Fn(&[f64; DOF]) -> [f64; 12],
    J: Fn(&[f64; DOF]) -> [[f64; 12]; DOF],
{
    lm_refine_core::<DOF, FREE, F, _>(
        q_in,
        free,
        &residual_fn,
        |q: &[f64; DOF], _r: &[f64; 12]| {
            let cols = jac_fn(q);
            let mut jac = SMatrix::<f64, 12, FREE>::zeros();
            for (c, &jidx) in free.iter().enumerate() {
                for row in 0..12 {
                    jac[(row, c)] = cols[jidx][row];
                }
            }
            jac
        },
        max_iters,
    )
    .0
}

/// Forward-difference Jacobian (12×FREE) over the columns listed in `free`.
fn fd_jacobian<const DOF: usize, const FREE: usize, F>(
    residual_fn: &F,
    free: &[usize; FREE],
    q: &[f64; DOF],
    r: &[f64; 12],
) -> SMatrix<f64, 12, FREE>
where
    F: Fn(&[f64; DOF]) -> [f64; 12],
{
    let mut jac = SMatrix::<f64, 12, FREE>::zeros();
    for (c, &jidx) in free.iter().enumerate() {
        let mut qp = *q;
        qp[jidx] += REFINE_EPS;
        let rp = residual_fn(&qp);
        for row in 0..12 {
            jac[(row, c)] = (rp[row] - r[row]) / REFINE_EPS;
        }
    }
    jac
}

/// Shared accept/reject LM core driving [`lm_refine`], [`lm_refine_masked`],
/// and [`lm_refine_masked_analytic`].
///
/// `jac_fn(q, r)` returns the `12×FREE` Jacobian over the joints in `free`.
/// Refines those joints and returns the refined joint vector together with
/// the final 12-vector residual norm so callers can apply their own
/// convergence gate (or ignore it).
fn lm_refine_core<const DOF: usize, const FREE: usize, F, J>(
    q_in: &[f64; DOF],
    free: &[usize; FREE],
    residual_fn: &F,
    jac_fn: J,
    max_iters: usize,
) -> ([f64; DOF], f64)
where
    F: Fn(&[f64; DOF]) -> [f64; 12],
    J: Fn(&[f64; DOF], &[f64; 12]) -> SMatrix<f64, 12, FREE>,
{
    let mut q = *q_in;
    let mut lambda = LM_INITIAL_LAMBDA;
    let mut r = residual_fn(&q);
    let mut rn = norm12(&r);

    for _ in 0..max_iters {
        if rn < LM_CONVERGENCE_THRESHOLD {
            break;
        }
        let jac = jac_fn(&q, &r);
        let rv = SMatrix::<f64, 12, 1>::from_column_slice(&r);
        let jt = jac.transpose();
        let jtj = jt * jac; // FREE×FREE
        let jtr = jt * rv; // FREE×1

        // Adaptive LM damping with accept/reject backtrack.
        let mut accepted = false;
        for _ in 0..LM_BACKTRACK_MAX {
            let mut damped = jtj;
            for d in 0..FREE {
                damped[(d, d)] += lambda;
            }
            let inv = match damped.try_inverse() {
                Some(m) => m,
                None => {
                    lambda *= LM_LAMBDA_GROWTH;
                    continue;
                }
            };
            let dq = inv * jtr;
            let mut q_trial = q;
            for (c, &jidx) in free.iter().enumerate() {
                q_trial[jidx] -= dq[(c, 0)];
            }
            let r_trial = residual_fn(&q_trial);
            let rn_trial = norm12(&r_trial);
            if rn_trial < rn {
                q = q_trial;
                r = r_trial;
                rn = rn_trial;
                lambda = (lambda * LM_LAMBDA_DECAY).max(LM_LAMBDA_MIN);
                accepted = true;
                break;
            }
            lambda *= LM_LAMBDA_GROWTH;
            if lambda > LM_LAMBDA_MAX {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    (q, rn)
}

/// Bounded Levenberg-Marquardt polish for a **6-DOF** joint vector.
///
/// This is a convenience wrapper around [`lm_refine::<6, _>`] preserved for
/// backward compatibility with the UR5 and Piper drivers.
///
/// `residual_fn(q)` must return `[vec(R(q) − R_t); t(q) − t_t]`.
///
/// Returns `Some(q_refined)` when the FK residual falls within `fk_atol`,
/// or `None` if the iteration budget is exhausted without convergence.
#[inline]
pub fn lm_refine6<F>(
    q_in: &[f64; 6],
    residual_fn: F,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 6]>
where
    F: Fn(&[f64; 6]) -> [f64; 12],
{
    lm_refine::<6, F>(q_in, residual_fn, max_iters, fk_atol)
}

// ---------------------------------------------------------------------------
// DLS (Dampened Least Squares) public refinement API
// ---------------------------------------------------------------------------

/// Default maximum LM iterations for [`dls_ik`].
///
/// 150 iterations is sufficient for convergence of any well-conditioned IK
/// problem starting from an analytical seed while keeping the worst-case
/// execution time bounded for real-time control loops.
pub const DLS_MAX_ITERS: usize = 150;

/// Default FK closure tolerance for [`dls_ik`] (Frobenius norm of the
/// residual between the actual FK output and the goal pose).
pub const DLS_FK_ATOL: f64 = 1e-5;

/// Dampened Least Squares (DLS) IK refinement for a **calibrated** arm with
/// **any** DOF.
///
/// Takes an initial joint configuration `q_init` (e.g., the analytical IK
/// solution computed for the *idealized* arm model) and the actual arm's
/// forward-kinematics function `fk_fn` (which may differ from the idealized
/// model due to calibration offsets, manufacturing tolerances, or a different
/// URDF), and iteratively refines the joint angles until the FK of the actual
/// arm satisfies the pose goal within `fk_atol`.
///
/// Each iteration uses the analytic residual Jacobian `jac_fn` (see
/// [`residual_columns`]) inside the shared LM core, so no finite differences
/// are taken. The residual is the 12-vector pose residual
/// `[vec(R(q) − R_goal); t(q) − t_goal]`. The loop is zero-allocation and
/// bounded by `max_iters`.
///
/// # Arguments
///
/// * `goal`      — Target end-effector pose as a [`Transform`].
/// * `q_init`    — Initial joint configuration (radians). Typically the
///                 analytical IK solution for the idealized arm.
/// * `fk_fn`     — Forward kinematics of the **actual** arm.  Must be
///                 deterministic and return a [`Transform`].
/// * `jac_fn`   — Analytic `12 × DOF` Jacobian of the pose residual of `fk_fn`
///                 at `q`, as columns (see [`residual_columns`]).
/// * `max_iters` — Hard iteration ceiling. Use [`DLS_MAX_ITERS`] unless a
///                 tighter real-time budget is needed.
/// * `fk_atol`   — FK closure tolerance (Frobenius norm). Use [`DLS_FK_ATOL`]
///                 (`1e-5`) to match the rest of the library.
///
/// # Returns
///
/// `Some(q_refined)` when `frob_diff(fk_fn(q_refined), goal) ≤ fk_atol`,
/// `None` if the iteration budget is exhausted without convergence.
///
/// # Examples
///
/// ```ignore
/// // Refine a 6-DOF analytical seed onto a calibrated arm.
/// let q_seed: [f64; 6] = analytical_ik(&idealized_model, &goal);
/// let q_refined = dls_ik(&goal, &q_seed, |q| calibrated_fk(q), |q| calibrated_jac(q), DLS_MAX_ITERS, DLS_FK_ATOL);
///
/// // Works for any DOF, e.g. a 7-DOF arm:
/// let q_seed7: [f64; 7] = analytical_ik7(&idealized_model, &goal);
/// let q_refined7 = dls_ik(&goal, &q_seed7, |q| calibrated_fk7(q), |q| calibrated_jac7(q), DLS_MAX_ITERS, DLS_FK_ATOL);
/// ```
pub fn dls_ik<const DOF: usize, F, J>(
    goal: &Transform,
    q_init: &[f64; DOF],
    fk_fn: F,
    jac_fn: J,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; DOF]>
where
    F: Fn(&[f64; DOF]) -> Transform,
    J: Fn(&[f64; DOF]) -> [[f64; 12]; DOF],
{
    let free: [usize; DOF] = core::array::from_fn(|i| i);
    let (q, rn) = lm_refine_core::<DOF, DOF, _, _>(
        q_init,
        &free,
        &|q: &[f64; DOF]| pose_residual12(&fk_fn(q), goal),
        |q: &[f64; DOF], _r: &[f64; 12]| {
            let cols = jac_fn(q);
            let mut jac = SMatrix::<f64, 12, DOF>::zeros();
            for (c, col) in cols.iter().enumerate() {
                for row in 0..12 {
                    jac[(row, c)] = col[row];
                }
            }
            jac
        },
        max_iters,
    );
    if rn <= fk_atol {
        Some(q)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    /// Central-difference residual Jacobian of `fk` against `goal`, in the
    /// column layout expected by [`dls_ik`]'s `jac_fn`.
    fn fd_residual_columns<const DOF: usize, F>(
        fk: &F,
        goal: &Transform,
        q: &[f64; DOF],
    ) -> [[f64; 12]; DOF]
    where
        F: Fn(&[f64; DOF]) -> Transform,
    {
        let h = 1e-7;
        let mut cols = [[0.0f64; 12]; DOF];
        for j in 0..DOF {
            let mut qp = *q;
            let mut qm = *q;
            qp[j] += h;
            qm[j] -= h;
            let rp = pose_residual12(&fk(&qp), goal);
            let rm = pose_residual12(&fk(&qm), goal);
            for row in 0..12 {
                cols[j][row] = (rp[row] - rm[row]) / (2.0 * h);
            }
        }
        cols
    }

    // -----------------------------------------------------------------------
    // Minimal 2-DOF planar arm for self-contained tests
    // -----------------------------------------------------------------------
    //
    // Two revolute joints in the XY-plane (both around Z), with link lengths
    // l1 and l2.  The full SE(3) FK is:
    //
    //   R = Rz(q[0] + q[1])
    //   t = [l1·cos(q[0]) + l2·cos(q[0]+q[1]),
    //        l1·sin(q[0]) + l2·sin(q[0]+q[1]),
    //        0]

    /// Nominal 2-DOF planar arm (l1 = l2 = 1.0).
    fn fk_2dof_nominal(q: &[f64; 2]) -> Transform {
        let a = q[0] + q[1];
        let (ca, sa) = (libm::cos(a), libm::sin(a));
        let (c0, s0) = (libm::cos(q[0]), libm::sin(q[0]));
        Transform {
            r: [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
            t: [c0 + ca, s0 + sa, 0.0],
        }
    }

    /// Calibrated 2-DOF planar arm (l1 = 1.01, l2 = 0.99).
    fn fk_2dof_calibrated(q: &[f64; 2]) -> Transform {
        const L1: f64 = 1.01;
        const L2: f64 = 0.99;
        let a = q[0] + q[1];
        let (ca, sa) = (libm::cos(a), libm::sin(a));
        let (c0, s0) = (libm::cos(q[0]), libm::sin(q[0]));
        Transform {
            r: [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
            t: [L1 * c0 + L2 * ca, L1 * s0 + L2 * sa, 0.0],
        }
    }

    /// Nominal 3-DOF spatial arm (revolute joints around Z, Y, Y; links of
    /// unit length).  Used to verify that `dls_ik` works for DOF > 2.
    fn fk_3dof_nominal(q: &[f64; 3]) -> Transform {
        // Joint 0: Rz(q[0]), translation [1,0,0] in body frame
        // Joint 1: Ry(q[1]), translation [1,0,0] in body frame
        // Joint 2: Ry(q[2]), no trailing translation (end-effector)
        let rz = |a: f64| -> [[f64; 3]; 3] {
            let (c, s) = (libm::cos(a), libm::sin(a));
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        };
        let ry = |a: f64| -> [[f64; 3]; 3] {
            let (c, s) = (libm::cos(a), libm::sin(a));
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
        };
        let t0 = Transform { r: rz(q[0]), t: [0.0, 0.0, 0.0] };
        let l0 = Transform { r: IDENTITY.r, t: [1.0, 0.0, 0.0] };
        let t1 = Transform { r: ry(q[1]), t: [0.0, 0.0, 0.0] };
        let l1 = Transform { r: IDENTITY.r, t: [1.0, 0.0, 0.0] };
        let t2 = Transform { r: ry(q[2]), t: [0.0, 0.0, 0.0] };
        compose(&compose(&compose(&compose(&t0, &l0), &t1), &l1), &t2)
    }

    // -----------------------------------------------------------------------
    // Test: converges immediately when q_init is already exact
    // -----------------------------------------------------------------------
    #[test]
    fn dls_ik_already_exact() {
        let q_true = [0.5f64, -0.3f64];
        let goal = fk_2dof_nominal(&q_true);
        let result = dls_ik(
            &goal,
            &q_true,
            fk_2dof_nominal,
            |q| fd_residual_columns(&fk_2dof_nominal, &goal, q),
            DLS_MAX_ITERS,
            DLS_FK_ATOL,
        );
        assert!(result.is_some(), "expected Some but got None");
        let residual = frob_diff(&fk_2dof_nominal(&result.unwrap()), &goal);
        assert!(
            residual <= DLS_FK_ATOL,
            "residual {:.2e} > atol {:.2e}",
            residual,
            DLS_FK_ATOL
        );
    }

    // -----------------------------------------------------------------------
    // Test: converges from a perturbed initial guess (same arm)
    // -----------------------------------------------------------------------
    #[test]
    fn dls_ik_perturbed_initial_guess() {
        let q_true = [1.0f64, -0.7f64];
        let goal = fk_2dof_nominal(&q_true);
        // Perturb the initial guess by 0.1 rad in each joint.
        let q_init = [q_true[0] + 0.1, q_true[1] - 0.1];
        let result = dls_ik(
            &goal,
            &q_init,
            fk_2dof_nominal,
            |q| fd_residual_columns(&fk_2dof_nominal, &goal, q),
            DLS_MAX_ITERS,
            DLS_FK_ATOL,
        );
        assert!(result.is_some(), "expected Some but got None");
        let residual = frob_diff(&fk_2dof_nominal(&result.unwrap()), &goal);
        assert!(
            residual <= DLS_FK_ATOL,
            "residual {:.2e} > atol {:.2e}",
            residual,
            DLS_FK_ATOL
        );
    }

    // -----------------------------------------------------------------------
    // Test: calibrated-arm scenario — q_init from nominal arm, FK from
    // calibrated arm.  This is the primary use-case described in the issue.
    // -----------------------------------------------------------------------
    #[test]
    fn dls_ik_calibrated_arm() {
        // True joint angles (valid for the calibrated arm).
        let q_true = [0.5f64, -0.3f64];
        // Goal: what the calibrated arm produces at q_true.
        let goal = fk_2dof_calibrated(&q_true);
        // q_init: analytical IK from the nominal arm would give angles close
        // to q_true (but not exact because the arms differ).  We simulate
        // this with a small perturbation.
        let q_init = [q_true[0] + 0.02, q_true[1] + 0.02];
        // DLS with the calibrated FK should converge to q_refined such that
        // calibrated_fk(q_refined) ≈ goal.
        let result = dls_ik(
            &goal,
            &q_init,
            fk_2dof_calibrated,
            |q| fd_residual_columns(&fk_2dof_calibrated, &goal, q),
            DLS_MAX_ITERS,
            DLS_FK_ATOL,
        );
        assert!(result.is_some(), "expected Some but got None");
        let residual = frob_diff(&fk_2dof_calibrated(&result.unwrap()), &goal);
        assert!(
            residual <= DLS_FK_ATOL,
            "residual {:.2e} > atol {:.2e}",
            residual,
            DLS_FK_ATOL
        );
    }

    // -----------------------------------------------------------------------
    // Test: DOF generality — 3-DOF arm
    // -----------------------------------------------------------------------
    #[test]
    fn dls_ik_3dof() {
        let q_true = [0.4f64, 0.6f64, -0.5f64];
        let goal = fk_3dof_nominal(&q_true);
        let q_init = [q_true[0] + 0.05, q_true[1] - 0.05, q_true[2] + 0.03];
        let result = dls_ik(
            &goal,
            &q_init,
            fk_3dof_nominal,
            |q| fd_residual_columns(&fk_3dof_nominal, &goal, q),
            DLS_MAX_ITERS,
            DLS_FK_ATOL,
        );
        assert!(result.is_some(), "expected Some but got None");
        let residual = frob_diff(&fk_3dof_nominal(&result.unwrap()), &goal);
        assert!(
            residual <= DLS_FK_ATOL,
            "residual {:.2e} > atol {:.2e}",
            residual,
            DLS_FK_ATOL
        );
    }

    // -----------------------------------------------------------------------
    // Test: mat4_frob_diff matches a hand-rolled 16-entry Frobenius distance
    // -----------------------------------------------------------------------
    #[test]
    fn mat4_frob_diff_matches_manual() {
        let a = SMatrix::<f64, 4, 4>::from_fn(|i, j| (i * 4 + j) as f64);
        let b = SMatrix::<f64, 4, 4>::from_fn(|i, j| (i * 4 + j) as f64 + 0.5 * (i as f64));
        // Reference: explicit sum over all 16 entries, exactly the kernel the
        // per-arm `fk_residual` helpers used before delegating here.
        let mut s = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                let d = a[(i, j)] - b[(i, j)];
                s += d * d;
            }
        }
        let expected = libm::sqrt(s);
        let got = mat4_frob_diff(&a, &b);
        assert_eq!(got, expected, "mat4_frob_diff diverged from manual kernel");
        // Identical matrices yield exactly zero.
        assert_eq!(mat4_frob_diff(&a, &a), 0.0);
    }

    // -----------------------------------------------------------------------
    // Test: lm_refine_masked refines only the free joints, leaving the locked
    // joint bit-for-bit unchanged while converging the free joint onto target.
    // -----------------------------------------------------------------------
    #[test]
    fn lm_refine_masked_holds_locked_joint() {
        let q_true = [0.5f64, -0.3f64];
        let goal = fk_2dof_nominal(&q_true);
        // Perturb only joint 0; joint 1 starts at its true (locked) value.
        let q_init = [q_true[0] + 0.1, q_true[1]];
        // Free set = {0}; joint 1 is locked.
        let free = [0usize];
        let q = lm_refine_masked::<2, 1, _>(
            &q_init,
            &free,
            |q| pose_residual12(&fk_2dof_nominal(q), &goal),
            DLS_MAX_ITERS,
        );
        // Locked joint must be untouched (bit-for-bit).
        assert_eq!(q[1], q_init[1], "locked joint 1 was modified");
        // Free joint converged onto the pose goal.
        let residual = frob_diff(&fk_2dof_nominal(&q), &goal);
        assert!(
            residual <= DLS_FK_ATOL,
            "residual {:.2e} > atol {:.2e}",
            residual,
            DLS_FK_ATOL
        );
    }

    #[test]
    fn lm_refine_masked_analytic_maps_reordered_free_columns() {
        const DOF: usize = 5;
        const FREE: [usize; 2] = [4, 1];
        let q_init = [0.125, 0.7, -0.375, 0.625, -0.5];
        let target = [q_init[0], -0.4, q_init[2], q_init[3], 1.0];
        let residual = |q: &[f64; DOF]| {
            let mut r = [0.0; 12];
            for i in 0..DOF {
                r[i] = q[i] - target[i];
            }
            r
        };
        let jacobian = |_q: &[f64; DOF]| {
            let mut columns = [[0.0; 12]; DOF];
            for i in 0..DOF {
                columns[i][i] = 1.0;
            }
            columns
        };

        let q = lm_refine_masked_analytic::<DOF, 2, _, _>(
            &q_init,
            &FREE,
            residual,
            jacobian,
            DLS_MAX_ITERS,
        );

        assert!(
            norm12(&residual(&q)) <= DLS_FK_ATOL,
            "analytic masked refinement did not converge"
        );
        assert_eq!(q[4], target[4]);
        assert_eq!(q[1], target[1]);
        for i in [0, 2, 3] {
            assert_eq!(
                q[i].to_bits(),
                q_init[i].to_bits(),
                "locked joint {i} changed"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: lm_refine (full DOF) and lm_refine_masked with every joint free
    // produce identical results — the two share a single core loop.
    // -----------------------------------------------------------------------
    #[test]
    fn lm_refine_equals_masked_all_free() {
        let q_true = [1.0f64, -0.7f64];
        let goal = fk_2dof_nominal(&q_true);
        let q_init = [q_true[0] + 0.1, q_true[1] - 0.1];
        let residual_fn = |q: &[f64; 2]| pose_residual12(&fk_2dof_nominal(q), &goal);

        let full = lm_refine::<2, _>(&q_init, residual_fn, DLS_MAX_ITERS, DLS_FK_ATOL)
            .expect("lm_refine should converge");
        let masked =
            lm_refine_masked::<2, 2, _>(&q_init, &[0, 1], residual_fn, DLS_MAX_ITERS);
        assert_eq!(full, masked, "full and all-free masked refine diverged");
    }
}
