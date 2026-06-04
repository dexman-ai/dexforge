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
//! * **Shared LM tuning constants** and **generic 6-DOF refinement loop**:
//!   [`lm_refine6`]
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
// Generic 6-DOF Levenberg-Marquardt refinement
// ---------------------------------------------------------------------------

/// Bounded Levenberg-Marquardt polish for a **6-DOF** joint vector.
///
/// This is the common LM loop shared by the UR5 and Piper drivers.  The
/// arm-specific part — computing the 12-vector pose residual from the arm's
/// own FK — is supplied as a closure so the function remains free of
/// per-arm types and constants.
///
/// `residual_fn(q)` must return `[vec(R(q) − R_t); t(q) − t_t]`.  The
/// closure typically captures the target pose and delegates to the driver's
/// local `pose_residual12`.
///
/// The generic type parameter is monomorphised at each call site, so the
/// compiled cost is identical to having the loop inlined in each driver.
///
/// Returns `Some(q_refined)` when the FK residual falls within `fk_atol`,
/// or `None` if the iteration budget is exhausted without convergence.
pub fn lm_refine6<F>(
    q_in: &[f64; 6],
    residual_fn: F,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 6]>
where
    F: Fn(&[f64; 6]) -> [f64; 12],
{
    let mut q = *q_in;
    let mut lambda = LM_INITIAL_LAMBDA;
    let mut r = residual_fn(&q);
    let mut rn = norm12(&r);

    for _ in 0..max_iters {
        if rn < LM_CONVERGENCE_THRESHOLD {
            break;
        }
        // Finite-difference Jacobian J (12×6) over all 6 joints.
        let mut jac = SMatrix::<f64, 12, 6>::zeros();
        for col in 0..6 {
            let mut qp = q;
            qp[col] += REFINE_EPS;
            let rp = residual_fn(&qp);
            for row in 0..12 {
                jac[(row, col)] = (rp[row] - r[row]) / REFINE_EPS;
            }
        }
        let rv = SMatrix::<f64, 12, 1>::from_column_slice(&r);
        let jt = jac.transpose();
        let jtj = jt * jac; // 6×6
        let jtr = jt * rv; // 6×1

        // Adaptive LM damping with accept/reject backtrack.
        let mut accepted = false;
        for _ in 0..LM_BACKTRACK_MAX {
            let mut damped = jtj;
            for d in 0..6 {
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
            for col in 0..6 {
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
