//! Generic, allocation-free forward-kinematics helpers shared by the per-arm
//! IK drivers.
//!
//! Every `ssik` prebuilt artifact verifies its algebraic candidates by running
//! product-of-exponentials forward kinematics built from the same per-joint
//! constants (`axis`, `T_left`, `T_right`) and then comparing against the
//! target pose, before de-duplicating the survivors modulo `2*pi`. That closure
//! check and the dense 4x4 chain product are completely arm-agnostic, so they
//! live here parameterised over the degrees of freedom `DOF`. The per-arm
//! drivers only supply their baked chain constants.

use nalgebra::SMatrix;

use crate::math::ssik_core::wrap_to_pi;

/// 4x4 homogeneous identity with the given translation column.
///
/// Convenience builder for the common `T_left` / `T_right` entries that are a
/// pure translation (the overwhelming majority of the baked chain constants).
pub const fn eye_with(x: f64, y: f64, z: f64) -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Dense 4x4 matrix product `a * b`.
pub fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for kk in 0..4 {
                s += a[i][kk] * b[kk][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Homogeneous rotation of `angle` about unit `axis` (Rodrigues), 4x4.
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

/// Product-of-exponentials forward kinematics for a serial revolute chain.
///
/// `q` are the joint angles and `axes` / `t_lefts` / `t_rights` are the baked
/// per-joint constants. The accumulated pose is
/// `prod_i (t_lefts[i] * R(axes[i], q[i]) * t_rights[i])`.
pub fn fk_chain<const DOF: usize>(
    q: &[f64; DOF],
    axes: &[[f64; 3]; DOF],
    t_lefts: &[[[f64; 4]; 4]; DOF],
    t_rights: &[[[f64; 4]; 4]; DOF],
) -> [[f64; 4]; 4] {
    let mut acc = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for i in 0..DOF {
        let rot = rot_homogeneous(&axes[i], q[i]);
        let m = mat4_mul(&t_lefts[i], &rot);
        let n = mat4_mul(&m, &t_rights[i]);
        acc = mat4_mul(&acc, &n);
    }
    acc
}

/// Frobenius-norm residual between `fk_chain(q, ..)` and the target pose.
pub fn fk_residual<const DOF: usize>(
    q: &[f64; DOF],
    axes: &[[f64; 3]; DOF],
    t_lefts: &[[[f64; 4]; 4]; DOF],
    t_rights: &[[[f64; 4]; 4]; DOF],
    target: &SMatrix<f64, 4, 4>,
) -> f64 {
    let t = fk_chain(q, axes, t_lefts, t_rights);
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = t[i][j] - target[(i, j)];
            s += d * d;
        }
    }
    libm::sqrt(s)
}

/// `true` if joint vectors `a` and `b` agree (mod `2*pi`) within `tol` per axis.
pub fn q_close_wrap<const DOF: usize>(a: &[f64; DOF], b: &[f64; DOF], tol: f64) -> bool {
    for i in 0..DOF {
        if libm::fabs(wrap_to_pi(a[i] - b[i])) > tol {
            return false;
        }
    }
    true
}
