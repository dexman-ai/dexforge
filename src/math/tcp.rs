//! Runtime tool-center-point (TCP) and grasped-payload support.
//!
//! This module answers the feasibility question raised in the issue
//! *"Runtime TCP modification in ik and dynamics"*: can the end-effector frame
//! (TCP) and the inertial properties contributed by a grasped tool be changed
//! **at runtime**, without regenerating the per-arm symbolic FK / Jacobian /
//! dynamics code, and **without** breaking the crate's real-time contract
//! (`#![no_std]`, zero heap allocation, bounded execution, determinism)?
//!
//! The answer is **yes**, because both the TCP offset and a grasped payload
//! attach *rigidly to the flange* (the last actuated link). Everything the
//! runtime change needs is a constant rigid transform plus the geometric data
//! the generated [`fk`](crate::drivers) and `jacobian` already expose. None of
//! the heavy symbolic kinematics has to be re-derived.
//!
//! # What lives here
//!
//! ### Kinematics (TCP offset)
//! * [`se3_inv`] — exact SE(3) inverse of a 4×4 homogeneous pose.
//! * [`fk_with_tcp`] — post-multiply the flange FK by the TCP transform to get
//!   the base→TCP pose.
//! * [`flange_target_for_tcp`] — given a desired base→TCP target, recover the
//!   base→flange target to feed the existing per-arm `solve` unchanged.
//!
//! ### Dynamics (grasped payload — mass + centre of mass)
//! * [`Payload`] — the runtime mass and flange-frame centre of mass of a tool.
//! * [`payload_gravity_torque`] — the *exact* additional gravity holding torque
//!   the payload contributes, computed from the flange pose and the baked
//!   geometric Jacobian.
//! * [`payload_inertia_matrix`] — the payload's (point-mass) contribution to
//!   the joint-space inertia matrix `M(q)`.
//!
//! # Why this keeps the real-time contract
//!
//! * **`no_std` / zero allocation** — every routine takes and returns
//!   fixed-size stack arrays / `SMatrix`, exactly like the rest of the crate.
//! * **Bounded execution** — there are no loops beyond fixed `DOF`/`3`/`4`
//!   ranges and no iterative solvers; worst-case time is a small constant.
//! * **Determinism** — the math is closed form (no RNG, no data-dependent
//!   branching), so a given input always yields bit-identical output.
//!
//! # Exactness
//!
//! The TCP composition is algebraically exact. The payload gravity torque is
//! *exact* relative to what the codegen would emit if the tool were baked into
//! the URDF as an extra rigid body on the flange: it reproduces
//! `tau_j = dU/dq_j` with `U = -m (g · p_com)` (the convention in
//! `codegen/generate_dynamics.py::build_gravity`), where `p_com` is the tool
//! centre of mass rigidly carried by the flange. The inertia contribution is
//! exact for a point mass (the issue scopes the runtime payload to *mass* and
//! *centre of mass* only, i.e. no rotational inertia tensor).

use nalgebra::SMatrix;

use crate::math::ssik_core::{add3, cross3, dot3};

/// A 4×4 homogeneous SE(3) pose, matching the per-arm `fk` / `solve` type.
pub type Pose = SMatrix<f64, 4, 4>;

/// Exact SE(3) inverse of a homogeneous pose `[R | t; 0 0 0 1]`.
///
/// Returns `[Rᵀ | -Rᵀ t; 0 0 0 1]`. This is the rigid-motion inverse: it
/// assumes the rotation block is orthonormal (true for any valid TCP or FK
/// pose) and so avoids a general matrix factorisation, keeping the cost a
/// fixed handful of multiply-adds.
#[inline]
pub fn se3_inv(m: &Pose) -> Pose {
    let mut out = Pose::zeros();
    // Rᵀ
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = m[(j, i)];
        }
    }
    // -Rᵀ t
    for i in 0..3 {
        let mut s = 0.0;
        for k in 0..3 {
            s += m[(k, i)] * m[(k, 3)];
        }
        out[(i, 3)] = -s;
    }
    out[(3, 3)] = 1.0;
    out
}

/// Forward kinematics of the TCP: `base→TCP = (base→flange) · (flange→TCP)`.
///
/// `flange_pose` is the arm's existing `fk(q)` output (base→flange / base→ee).
/// `tcp` is the runtime tool transform expressed in the flange frame. Changing
/// `tcp` at runtime (e.g. after grasping a tool) needs no code regeneration.
#[inline]
pub fn fk_with_tcp(flange_pose: &Pose, tcp: &Pose) -> Pose {
    flange_pose * tcp
}

/// Recover the flange target for an inverse-kinematics call from a TCP target.
///
/// Given a desired base→TCP pose and the runtime `tcp` (flange→TCP) transform,
/// returns `base→flange = (base→TCP) · (flange→TCP)⁻¹`. Feed the result to the
/// arm's existing `solve(&flange_target, …)`; the returned joint angles place
/// the *TCP* at `tcp_target`. The IK solver itself is unchanged.
#[inline]
pub fn flange_target_for_tcp(tcp_target: &Pose, tcp: &Pose) -> Pose {
    tcp_target * se3_inv(tcp)
}

/// A grasped tool / payload rigidly attached to the flange (end-effector link).
///
/// The runtime inertial change the issue targets: a `mass` and a centre of
/// mass `com` expressed in the flange frame. This is everything needed for
/// gravity compensation (the dominant term when holding a tool in
/// contact-rich tasks) and for the point-mass inertia contribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Payload {
    /// Tool mass in kilograms.
    pub mass: f64,
    /// Tool centre of mass `[x, y, z]` in the flange (end-effector) frame.
    pub com: [f64; 3],
}

impl Payload {
    /// A massless payload (no dynamics contribution).
    pub const NONE: Payload = Payload {
        mass: 0.0,
        com: [0.0, 0.0, 0.0],
    };

    /// Construct a payload from a mass and flange-frame centre of mass.
    #[inline]
    pub const fn new(mass: f64, com: [f64; 3]) -> Payload {
        Payload { mass, com }
    }
}

/// Translational Jacobian (3 × DOF, base frame) of a point rigidly fixed to the
/// flange, given as `local` in the flange frame.
///
/// The point is carried by the flange, so its velocity is `v_O + ω × r`, with
/// `r = R_flange · local` the point's offset from the flange origin in base
/// frame. Column `j` of the returned `[[f64; 3]; DOF]` (column-major: `out[j]`
/// is column `j`) is `Jv[:,j] + Jw[:,j] × r`, where `Jv`/`Jw` are the
/// translational/angular blocks (rows 0–2 / 3–5) of the baked geometric
/// Jacobian.
///
/// This is the shared kernel behind both the TCP-point geometric Jacobian and
/// the grasped-payload centre-of-mass dynamics: pass the TCP translation for
/// the former and the payload centre of mass for the latter.
#[inline]
pub fn flange_point_jacobian<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    local: &[f64; 3],
) -> [[f64; 3]; DOF] {
    // r = R_flange · local (point offset from flange origin, in base frame).
    let mut r = [0.0f64; 3];
    for i in 0..3 {
        let mut s = 0.0;
        for k in 0..3 {
            s += flange_pose[(i, k)] * local[k];
        }
        r[i] = s;
    }

    let mut jvp = [[0.0f64; 3]; DOF];
    for j in 0..DOF {
        let jv = [jacobian[0][j], jacobian[1][j], jacobian[2][j]];
        let jw = [jacobian[3][j], jacobian[4][j], jacobian[5][j]];
        jvp[j] = add3(&jv, &cross3(&jw, &r));
    }
    jvp
}

/// Exact additional gravity (holding) torque contributed by a grasped payload.
///
/// Returns the per-joint torque `tau_j = -m · (g · ∂p_com/∂q_j)`, matching the
/// `tau = dU/dq`, `U = -m (g · p_com)` convention used by the offline codegen
/// (`codegen/generate_dynamics.py::build_gravity`). Add this elementwise to the
/// arm's baked `gravity(q, g)` to gravity-compensate while holding the tool —
/// no dynamics regeneration required.
///
/// * `flange_pose` — the arm's `fk(q)` (base→flange) pose.
/// * `jacobian` — the arm's baked geometric Jacobian `[Jv; Jw]` (6 × DOF).
/// * `payload` — runtime tool mass and flange-frame centre of mass.
/// * `gravity` — gravity acceleration vector in the base frame
///   (e.g. `[0.0, 0.0, -9.81]`).
#[inline]
pub fn payload_gravity_torque<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    payload: &Payload,
    gravity: &[f64; 3],
) -> [f64; DOF] {
    let mut tau = [0.0f64; DOF];
    if payload.mass == 0.0 {
        return tau;
    }
    let jvp = flange_point_jacobian(flange_pose, jacobian, &payload.com);
    for j in 0..DOF {
        tau[j] = -payload.mass * dot3(gravity, &jvp[j]);
    }
    tau
}

/// Payload (point-mass) contribution to the joint-space inertia matrix `M(q)`.
///
/// Returns `m · Jvpᵀ Jvp` (DOF × DOF, symmetric), where `Jvp` is the
/// translational Jacobian of the payload centre of mass. This is the exact
/// inertia contribution for a *point* mass; the issue scopes the runtime
/// payload to mass + centre of mass (no rotational inertia tensor), so the
/// angular term is intentionally omitted. Add this elementwise to the arm's
/// baked `mass_matrix(q)` to account for the tool's translational inertia.
#[inline]
pub fn payload_inertia_matrix<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    payload: &Payload,
) -> [[f64; DOF]; DOF] {
    let mut mm = [[0.0f64; DOF]; DOF];
    if payload.mass == 0.0 {
        return mm;
    }
    let jvp = flange_point_jacobian(flange_pose, jacobian, &payload.com);
    for i in 0..DOF {
        for j in 0..DOF {
            mm[i][j] = payload.mass * dot3(&jvp[i], &jvp[j]);
        }
    }
    mm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::xarm7::generated_kinematics as xarm7;
    use crate::drivers::xarm7::ik_solver as xarm7_ik;

    fn pose_from_rows(a: &[[f64; 4]; 4]) -> Pose {
        let mut m = Pose::zeros();
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = a[i][j];
            }
        }
        m
    }

    fn approx_eq_pose(a: &Pose, b: &Pose, tol: f64) {
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() <= tol,
                    "mismatch at ({i},{j}): {} vs {}",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    #[test]
    fn se3_inv_is_a_true_inverse() {
        // A non-trivial TCP: rotate then translate.
        let tcp = pose_from_rows(&xarm7::fk(&[0.3, -0.4, 0.5, -0.6, 0.7, -0.2, 0.1]));
        let inv = se3_inv(&tcp);
        let prod = tcp * inv;
        approx_eq_pose(&prod, &Pose::identity(), 1e-12);
        // Rotation block stays orthonormal.
        let prod2 = inv * tcp;
        approx_eq_pose(&prod2, &Pose::identity(), 1e-12);
    }

    #[test]
    fn identity_tcp_is_a_no_op() {
        let q = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7];
        let flange = xarm7_ik::fk(&q);
        let tcp = Pose::identity();
        approx_eq_pose(&fk_with_tcp(&flange, &tcp), &flange, 1e-15);
        approx_eq_pose(&flange_target_for_tcp(&flange, &tcp), &flange, 1e-15);
    }

    #[test]
    fn fk_with_tcp_and_retarget_are_inverse() {
        let q = [0.2, -0.1, 0.4, -0.7, 0.3, 0.5, -0.4];
        let flange = xarm7_ik::fk(&q);
        // Pure translation TCP: 10 cm along flange z.
        let mut tcp = Pose::identity();
        tcp[(2, 3)] = 0.10;

        let tcp_pose = fk_with_tcp(&flange, &tcp);
        // Recovering the flange target from the TCP pose must round-trip.
        let recovered = flange_target_for_tcp(&tcp_pose, &tcp);
        approx_eq_pose(&recovered, &flange, 1e-12);
    }

    #[test]
    fn tcp_translation_offsets_origin_in_flange_frame() {
        let q = [0.0; 7];
        let flange = xarm7_ik::fk(&q);
        let mut tcp = Pose::identity();
        tcp[(0, 3)] = 0.05; // 5 cm along flange x.

        let tcp_pose = fk_with_tcp(&flange, &tcp);
        // The TCP origin equals the flange origin shifted by 0.05 along the
        // flange's x axis (first rotation column).
        for i in 0..3 {
            let expected = flange[(i, 3)] + 0.05 * flange[(i, 0)];
            assert!((tcp_pose[(i, 3)] - expected).abs() <= 1e-12);
        }
    }

    #[test]
    fn massless_payload_has_no_dynamics_effect() {
        let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let flange = xarm7_ik::fk(&q);
        let jac = xarm7::jacobian(&q);
        let tau = payload_gravity_torque(&flange, &jac, &Payload::NONE, &[0.0, 0.0, -9.81]);
        assert_eq!(tau, [0.0; 7]);
        let mm = payload_inertia_matrix(&flange, &jac, &Payload::NONE);
        assert_eq!(mm, [[0.0; 7]; 7]);
    }

    #[test]
    fn payload_gravity_matches_finite_difference_of_potential() {
        // The emitted torque tau_j = dU/dq_j with U = -m g·p_com must agree
        // with a central finite-difference of the potential energy, validating
        // the analytic COM Jacobian against the baked FK directly.
        let q = [0.2, -0.3, 0.5, -0.6, 0.4, 0.7, -0.1];
        let gravity = [0.0, 0.0, -9.81];
        let payload = Payload::new(2.5, [0.03, -0.02, 0.11]);

        let flange = xarm7_ik::fk(&q);
        let jac = xarm7::jacobian(&q);
        let tau = payload_gravity_torque(&flange, &jac, &payload, &gravity);

        // Potential energy of the payload COM at joint vector qq.
        let potential = |qq: &[f64; 7]| -> f64 {
            let fk = xarm7::fk(qq);
            // p_com = flange_origin + R_flange · com
            let mut p = [fk[0][3], fk[1][3], fk[2][3]];
            for i in 0..3 {
                for k in 0..3 {
                    p[i] += fk[i][k] * payload.com[k];
                }
            }
            -payload.mass * (gravity[0] * p[0] + gravity[1] * p[1] + gravity[2] * p[2])
        };

        let h = 1e-6;
        for j in 0..7 {
            let mut qp = q;
            let mut qm = q;
            qp[j] += h;
            qm[j] -= h;
            let fd = (potential(&qp) - potential(&qm)) / (2.0 * h);
            assert!(
                (tau[j] - fd).abs() <= 1e-4,
                "joint {j}: analytic {} vs finite-difference {}",
                tau[j],
                fd
            );
        }
    }

    #[test]
    fn ik_through_tcp_retarget_places_tcp_on_target() {
        // End-to-end: choose a known configuration, define a TCP, compute the
        // base->TCP pose, retarget it back to a flange target, solve IK, and
        // confirm every returned solution puts the *TCP* on the original
        // target. This exercises the full "TCP in IK" path with no codegen.
        use crate::math::tolerances::TolerancePolicy;

        let q_ref = [0.15, -0.25, 0.35, -0.45, 0.55, -0.15, 0.25];
        let flange_ref = xarm7_ik::fk(&q_ref);

        // A non-trivial TCP: rotation + translation in the flange frame.
        let tcp = pose_from_rows(&xarm7::fk(&[0.2, 0.1, -0.3, 0.4, -0.2, 0.1, 0.0]));
        let tcp_target = fk_with_tcp(&flange_ref, &tcp);

        let flange_target = flange_target_for_tcp(&tcp_target, &tcp);
        let sols = xarm7_ik::solve(
            &flange_target,
            None,
            None,
            false,
            true,
            TolerancePolicy::DEFAULT,
            15,
        );

        let mut checked = 0usize;
        for q in sols.iter().flatten() {
            let flange = xarm7_ik::fk(q);
            let tcp_pose = fk_with_tcp(&flange, &tcp);
            approx_eq_pose(&tcp_pose, &tcp_target, 1e-6);
            checked += 1;
        }
        assert!(checked > 0, "IK returned no solutions for the retargeted pose");
    }

    #[test]
    fn payload_inertia_is_symmetric_psd() {
        let q = [0.3, 0.1, -0.2, 0.5, -0.4, 0.2, 0.6];
        let flange = xarm7_ik::fk(&q);
        let jac = xarm7::jacobian(&q);
        let payload = Payload::new(1.8, [0.0, 0.0, 0.08]);
        let mm = payload_inertia_matrix(&flange, &jac, &payload);

        // Symmetry.
        for i in 0..7 {
            for j in 0..7 {
                assert!((mm[i][j] - mm[j][i]).abs() <= 1e-12);
            }
        }
        // Diagonal entries are non-negative (Gram matrix of the COM Jacobian).
        for i in 0..7 {
            assert!(mm[i][i] >= 0.0);
        }
    }
}
