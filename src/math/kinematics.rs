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
//! * **Shared LM tuning constants** and **generic const-DOF refinement loop**:
//!   [`lm_refine`], [`lm_refine6`]
//!
//! # What stays in each driver
//!
//! * Arm-specific FK (`fk`, `fk7`, `fk_sub`, …) and the pose-residual
//!   functions that call them — they encode per-arm kinematics constants.
//! * Franka `refine7` — its joint-masking strategy is specific to the
//!   `jointlock.seven_r` redundancy resolution and lives only there.
//! * Piper `spatial_bypass` and perturbation constants — the 10 µm
//!   deterministic singularity escape is specific to the Raghavan-Roth
//!   polynomial-system condition check of that driver.

use nalgebra::SMatrix;

use crate::math::ssik_core::{add3, mat3_mul, mat3_transpose, mat3_vec, neg3};

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
#[derive(Clone, Copy)]
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

/// 12-vector pose residual `[vec(a.R − b.R); a.t − b.t]` between two transforms.
///
/// The first 9 elements are the column-major flattening of `(a.R − b.R)` and
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
    let mut q = *q_in;
    let mut lambda = LM_INITIAL_LAMBDA;
    let mut r = residual_fn(&q);
    let mut rn = norm12(&r);

    for _ in 0..max_iters {
        if rn < LM_CONVERGENCE_THRESHOLD {
            break;
        }
        // Finite-difference Jacobian J (12×DOF) over all DOF joints.
        let mut jac = SMatrix::<f64, 12, DOF>::zeros();
        for col in 0..DOF {
            let mut qp = q;
            qp[col] += REFINE_EPS;
            let rp = residual_fn(&qp);
            for row in 0..12 {
                jac[(row, col)] = (rp[row] - r[row]) / REFINE_EPS;
            }
        }
        let rv = SMatrix::<f64, 12, 1>::from_column_slice(&r);
        let jt = jac.transpose();
        let jtj = jt * jac; // DOF×DOF
        let jtr = jt * rv; // DOF×1

        // Adaptive LM damping with accept/reject backtrack.
        let mut accepted = false;
        for _ in 0..LM_BACKTRACK_MAX {
            let mut damped = jtj;
            for d in 0..DOF {
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
            for col in 0..DOF {
                q_trial[col] -= dq[(col, 0)];
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

    if rn <= fk_atol {
        Some(q)
    } else {
        None
    }
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
/// Internally delegates to [`lm_refine`] with the standard 12-vector pose
/// residual `[vec(R(q) − R_goal); t(q) − t_goal]`, so all coding restrictions
/// are inherited: zero allocation, `#![no_std]`-compatible, and bounded
/// iteration count (hard-capped at `max_iters`).
///
/// # Arguments
///
/// * `goal`      — Target end-effector pose as a [`Transform`].
/// * `q_init`    — Initial joint configuration (radians). Typically the
///                 analytical IK solution for the idealized arm.
/// * `fk_fn`     — Forward kinematics of the **actual** arm.  Must be
///                 deterministic and return a [`Transform`].
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
/// let q_refined = dls_ik(&goal, &q_seed, |q| calibrated_fk(q), DLS_MAX_ITERS, DLS_FK_ATOL);
///
/// // Works for any DOF, e.g. a 7-DOF arm:
/// let q_seed7: [f64; 7] = analytical_ik7(&idealized_model, &goal);
/// let q_refined7 = dls_ik(&goal, &q_seed7, |q| calibrated_fk7(q), DLS_MAX_ITERS, DLS_FK_ATOL);
/// ```
pub fn dls_ik<const DOF: usize, F>(
    goal: &Transform,
    q_init: &[f64; DOF],
    fk_fn: F,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; DOF]>
where
    F: Fn(&[f64; DOF]) -> Transform,
{
    lm_refine::<DOF, _>(
        q_init,
        |q| pose_residual12(&fk_fn(q), goal),
        max_iters,
        fk_atol,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

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
        let result = dls_ik(&goal, &q_true, fk_2dof_nominal, DLS_MAX_ITERS, DLS_FK_ATOL);
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
        let result = dls_ik(&goal, &q_init, fk_2dof_nominal, DLS_MAX_ITERS, DLS_FK_ATOL);
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
        let result =
            dls_ik(&goal, &q_init, fk_2dof_calibrated, DLS_MAX_ITERS, DLS_FK_ATOL);
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
        let result = dls_ik(&goal, &q_init, fk_3dof_nominal, DLS_MAX_ITERS, DLS_FK_ATOL);
        assert!(result.is_some(), "expected Some but got None");
        let residual = frob_diff(&fk_3dof_nominal(&result.unwrap()), &goal);
        assert!(
            residual <= DLS_FK_ATOL,
            "residual {:.2e} > atol {:.2e}",
            residual,
            DLS_FK_ATOL
        );
    }
}
