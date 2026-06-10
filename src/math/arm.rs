//! Generic, arm-parameterised kinematics/dynamics facade with runtime TCP and
//! grasped-payload support.
//!
//! This module wraps the per-arm drivers behind a single [`Arm`] trait so the
//! same call shape works for every supported manipulator, with the runtime
//! tool-center-point (TCP) and grasped payload passed as plain arguments:
//!
//! ```
//! use dexforge_examples::math::arm::{Arm, Xarm7, Payload};
//! use dexforge_examples::math::arm::Pose;
//!
//! let q = [0.0; 7];
//! let mut tcp = Pose::identity();
//! tcp[(2, 3)] = 0.12; // 12 cm tool along the flange z axis.
//!
//! let tcp_pose = Xarm7::fk(&q, &tcp);          // base -> TCP forward kinematics
//! let j = Xarm7::jacobian(&q, &tcp);           // geometric Jacobian at the TCP
//! let payload = Payload::new(2.5, [0.0, 0.0, 0.06]);
//! let g = Xarm7::gravity(&q, &payload, &[0.0, 0.0, -9.81]); // payload holding torque
//! ```
//!
//! Each arm is a zero-sized marker type ([`Ur5`], [`Piper`], [`FrankaPanda`],
//! [`Xarm7`], [`Rizon4`], [`Rizon10`]) whose [`ArmModel`] implementation lives
//! in its driver module alongside the arm-specific solver code. The facade
//! methods ([`Arm::fk`], [`Arm::ik`], [`Arm::jacobian`],
//! [`Arm::inverse_jacobian`], [`Arm::gravity`], [`Arm::mass_matrix`],
//! [`Arm::coriolis`]) are default trait methods built directly in this module,
//! so adding TCP/payload support requires **no** code
//! regeneration and changes no per-arm driver code.
//!
//! Like the rest of the crate, every method is `#![no_std]`, allocates nothing,
//! runs in bounded time (fixed `DOF`/`3`/`6` loops, no iterative solvers beyond
//! the underlying `solve`), and is deterministic.

use nalgebra::SMatrix;

use crate::math::ssik_core::{add3, cross3, dot3};
use crate::math::tolerances::TolerancePolicy;

pub use crate::drivers::franka_panda::arm::FrankaPanda;
pub use crate::drivers::piper::arm::Piper;
pub use crate::drivers::rizon10::arm::Rizon10;
pub use crate::drivers::rizon4::arm::Rizon4;
pub use crate::drivers::ur5::arm::Ur5;
pub use crate::drivers::xarm7::arm::Xarm7;

/// Finite-difference step (rad) for the payload Coriolis directional derivative.
const CORIOLIS_FD_STEP: f64 = 1e-6;

/// A 4×4 homogeneous SE(3) pose, matching the per-arm `fk` / `solve` type.
pub type Pose = SMatrix<f64, 4, 4>;

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

/// Minimum-norm (Moore–Penrose) pseudo-inverse `J⁺ = Jᵀ (J Jᵀ)⁻¹` of a
/// `6 × DOF` geometric Jacobian, returned as a `DOF × 6` array.
///
/// For a non-redundant arm (`DOF == 6`) this is the exact inverse; for a
/// redundant arm (`DOF > 6`) it is the least-squares / minimum-norm inverse.
/// `J Jᵀ` is a fixed `6 × 6` matrix, so the inversion is bounded and
/// allocation-free. Returns `None` at a kinematic singularity (rank-deficient
/// `J`, where `J Jᵀ` is not invertible).
fn pseudo_inverse<const DOF: usize>(j: &[[f64; DOF]; 6]) -> Option<[[f64; 6]; DOF]> {
    // A = J Jᵀ  (6 × 6, symmetric).
    let mut a = SMatrix::<f64, 6, 6>::zeros();
    for i in 0..6 {
        for k in 0..6 {
            let mut s = 0.0;
            for c in 0..DOF {
                s += j[i][c] * j[k][c];
            }
            a[(i, k)] = s;
        }
    }
    let ainv = a.try_inverse()?;
    // J⁺ = Jᵀ A⁻¹  (DOF × 6).
    let mut out = [[0.0f64; 6]; DOF];
    for r in 0..DOF {
        for c in 0..6 {
            let mut s = 0.0;
            for i in 0..6 {
                s += j[i][r] * ainv[(i, c)];
            }
            out[r][c] = s;
        }
    }
    Some(out)
}

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

/// A manipulator trait exposed behind a uniform, TCP/payload-aware interface.
///
/// `DOF` is the number of actuated joints and `MAX` the capacity of the IK
/// solution array (both fixed per arm). Implementors only provide the three
/// *flange-frame* primitives (`flange_fk`, `flange_jacobian`, `flange_solve`);
/// every TCP/payload-aware method is a provided default built on
/// local helpers in [`crate::math::arm`].
pub trait ArmModel<const DOF: usize, const MAX: usize> {
    /// Forward kinematics of the flange (base → end-effector link), i.e. the
    /// driver's native `fk(q)`.
    fn flange_fk(q: &[f64; DOF]) -> Pose;

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) of the flange, i.e. the driver's
    /// native `generated_kinematics::jacobian(q)`.
    fn flange_jacobian(q: &[f64; DOF]) -> [[f64; DOF]; 6];

    /// Inverse kinematics for a flange target, i.e. the driver's native
    /// `solve(...)`.
    #[allow(clippy::too_many_arguments)]
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; DOF]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; DOF]>; MAX];

    // -----------------------------------------------------------------------
    // TCP-aware kinematics (default methods)
    // -----------------------------------------------------------------------

    /// Forward kinematics of the **TCP**: `base→TCP = fk(q) · tcp`.
    ///
    /// Pass [`Pose::identity`] for `tcp` to recover the plain flange FK.
    #[inline]
    fn fk(q: &[f64; DOF], tcp: &Pose) -> Pose {
        &Self::flange_fk(q) * tcp
    }

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) referenced at the **TCP origin**.
    ///
    /// The angular rows are unchanged; the linear rows are re-referenced to the
    /// TCP point (`Jv_tcp = Jv + Jw × (R_flange · t_tcp)`).
    #[inline]
    fn jacobian(q: &[f64; DOF], tcp: &Pose) -> [[f64; DOF]; 6] {
        let flange = Self::flange_fk(q);
        let jflange = Self::flange_jacobian(q);
        let p = [tcp[(0, 3)], tcp[(1, 3)], tcp[(2, 3)]];
        let lin = flange_point_jacobian(&flange, &jflange, &p);
        
        let mut j = [[0.0f64; DOF]; 6];
        for c in 0..DOF {
            j[0][c] = lin[c][0];
            j[1][c] = lin[c][1];
            j[2][c] = lin[c][2];
            j[3][c] = jflange[3][c];
            j[4][c] = jflange[4][c];
            j[5][c] = jflange[5][c];
        }
        j
    }

    /// Minimum-norm pseudo-inverse (`DOF × 6`) of the TCP geometric Jacobian.
    ///
    /// Maps a spatial twist at the TCP to joint rates. Returns `None` at a
    /// kinematic singularity.
    #[inline]
    fn inverse_jacobian(q: &[f64; DOF], tcp: &Pose) -> Option<[[f64; 6]; DOF]> {
        pseudo_inverse(&Self::jacobian(q, tcp))
    }

    /// Inverse kinematics for a desired **TCP** target pose.
    ///
    /// Retargets `tcp_target` to the corresponding flange target and forwards
    /// to the native solver, so the returned joint vectors place the *TCP* at
    /// `tcp_target`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn ik(
        tcp_target: &Pose,
        tcp: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; DOF]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; DOF]>; MAX] {
        let flange_target = tcp_target * se3_inv(tcp);
        Self::flange_solve(
            &flange_target,
            max_solutions,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        )
    }

    // -----------------------------------------------------------------------
    // Payload dynamics (default methods)
    // -----------------------------------------------------------------------

    /// Additional gravity (holding) torque contributed by a grasped `payload`.
    ///
    /// Add elementwise to the arm's own gravity torque. See
    /// [`payload_gravity_torque`].
    #[inline]
    fn gravity(q: &[f64; DOF], payload: &Payload, gravity: &[f64; 3]) -> [f64; DOF] {
        let mut tau = [0.0f64; DOF];
        if payload.mass == 0.0 {
            return tau;
        }
        let jvp = flange_point_jacobian(&Self::flange_fk(q), &Self::flange_jacobian(q), &payload.com);
        for j in 0..DOF {
            tau[j] = -payload.mass * dot3(gravity, &jvp[j]);
        }
        tau
    }

    /// Grasped `payload` (point-mass) contribution to the joint-space inertia
    /// matrix `M(q)`. See [`payload_inertia_matrix`].
    #[inline]
    fn mass_matrix(q: &[f64; DOF], payload: &Payload) -> [[f64; DOF]; DOF] {
        let mut mm = [[0.0f64; DOF]; DOF];
        if payload.mass == 0.0 {
            return mm;
        }
        let jvp = flange_point_jacobian(&Self::flange_fk(q), &Self::flange_jacobian(q), &payload.com);
        for i in 0..DOF {
            for j in 0..DOF {
                mm[i][j] = payload.mass * dot3(&jvp[i], &jvp[j]);
            }
        }
        mm
    }

    /// Coriolis & centrifugal torque `C(q, qd) qd` contributed by a grasped
    /// `payload`.
    ///
    /// A payload makes the joint-space inertia `M(q)` configuration-dependent,
    /// so it contributes a velocity-product torque whenever the arm moves —
    /// **including when the tool centre of mass is off the end-effector axis**,
    /// where this coupling is strongest. For a point mass the term is
    /// `τ = m Jvpᵀ (J̇vp qd)`, with `Jvp` the centre-of-mass translational
    /// Jacobian. The time derivative `J̇vp qd` is obtained from a bounded,
    /// deterministic central difference of `Jvp` along `qd` (two extra FK /
    /// Jacobian evaluations), so the routine stays allocation-free and
    /// constant-time. Add the result elementwise to the arm's own Coriolis
    /// torque.
    #[inline]
    fn coriolis(q: &[f64; DOF], qd: &[f64; DOF], payload: &Payload) -> [f64; DOF] {
        let mut tau = [0.0f64; DOF];
        if payload.mass == 0.0 {
            return tau;
        }
        let h = CORIOLIS_FD_STEP;
        // Centre-of-mass translational Jacobian columns at the base config.
        let jvp = flange_point_jacobian(
            &Self::flange_fk(q),
            &Self::flange_jacobian(q),
            &payload.com,
        );
        // Perturb the configuration along the joint-velocity direction.
        let mut qp = *q;
        let mut qm = *q;
        for j in 0..DOF {
            qp[j] += h * qd[j];
            qm[j] -= h * qd[j];
        }
        let jvp_p = flange_point_jacobian(
            &Self::flange_fk(&qp),
            &Self::flange_jacobian(&qp),
            &payload.com,
        );
        let jvp_m = flange_point_jacobian(
            &Self::flange_fk(&qm),
            &Self::flange_jacobian(&qm),
            &payload.com,
        );
        // a_bias = (d/dt Jvp) qd = Σ_j [(Jvp⁺ − Jvp⁻) / 2h]·column_j · qd_j.
        let mut a_bias = [0.0f64; 3];
        for j in 0..DOF {
            let w = qd[j] / (2.0 * h);
            for r in 0..3 {
                a_bias[r] += (jvp_p[j][r] - jvp_m[j][r]) * w;
            }
        }
        for i in 0..DOF {
            tau[i] = payload.mass * dot3(&jvp[i], &a_bias);
        }
        tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn facade_fk_matches_driver_when_tcp_identity() {
        let q = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7];
        let tcp = Pose::identity();
        let via_facade = Xarm7::fk(&q, &tcp);
        let direct = crate::drivers::xarm7::ik_solver::fk(&q);
        approx_eq_pose(&via_facade, &direct, 1e-15);
    }

    #[test]
    fn facade_jacobian_matches_driver_when_tcp_identity() {
        let q = [0.2, 0.1, -0.3, 0.4, -0.2, 0.1, 0.0];
        let tcp = Pose::identity();
        let jf = Xarm7::jacobian(&q, &tcp);
        let direct = crate::drivers::xarm7::generated_kinematics::jacobian(&q);
        for r in 0..6 {
            for c in 0..7 {
                assert!((jf[r][c] - direct[r][c]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn facade_ik_places_tcp_on_target_ur5() {
        // 6-DOF arm: exercises the non-redundant path end to end.
        let q_ref = [0.3, -0.5, 0.7, -0.4, 0.6, -0.2];
        let flange_ref = crate::drivers::ur5::ik_solver::fk(&q_ref);
        let mut tcp = Pose::identity();
        tcp[(2, 3)] = 0.08;
        let tcp_target = Ur5::fk(&q_ref, &tcp);

        let sols = Ur5::ik(
            &tcp_target,
            &tcp,
            None,
            None,
            false,
            true,
            TolerancePolicy::DEFAULT,
            15,
        );
        let _ = flange_ref;
        let mut checked = 0usize;
        for q in sols.iter().flatten() {
            let tcp_pose = Ur5::fk(q, &tcp);
            approx_eq_pose(&tcp_pose, &tcp_target, 1e-6);
            checked += 1;
        }
        assert!(checked > 0, "no IK solutions for retargeted TCP pose");
    }

    #[test]
    fn inverse_jacobian_reproduces_twist() {
        // J · (J⁺ v) == v for a full-rank configuration.
        let q = [0.3, -0.4, 0.5, -0.6, 0.7, -0.2, 0.1];
        let tcp = Pose::identity();
        let j = Xarm7::jacobian(&q, &tcp);
        let jpinv = Xarm7::inverse_jacobian(&q, &tcp).expect("non-singular");

        let v = [0.10, -0.05, 0.20, 0.01, -0.02, 0.03];
        // qd = J⁺ v   (length DOF)
        let mut qd = [0.0f64; 7];
        for r in 0..7 {
            let mut s = 0.0;
            for c in 0..6 {
                s += jpinv[r][c] * v[c];
            }
            qd[r] = s;
        }
        // v_hat = J qd   (length 6)
        for r in 0..6 {
            let mut s = 0.0;
            for c in 0..7 {
                s += j[r][c] * qd[c];
            }
            assert!((s - v[r]).abs() <= 1e-9, "twist row {r}: {s} vs {}", v[r]);
        }
    }

    #[test]
    fn inverse_jacobian_is_true_inverse_for_6dof() {
        let q = [0.2, -0.3, 0.6, -0.5, 0.4, -0.1];
        let tcp = Pose::identity();
        let j = Ur5::jacobian(&q, &tcp);
        let jinv = Ur5::inverse_jacobian(&q, &tcp).expect("non-singular");
        // J · J⁻¹ == I₆
        for a in 0..6 {
            for b in 0..6 {
                let mut s = 0.0;
                for c in 0..6 {
                    s += j[a][c] * jinv[c][b];
                }
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((s - expected).abs() <= 1e-9, "({a},{b}) = {s}");
            }
        }
    }

    #[test]
    fn coriolis_matches_christoffel_from_mass_matrix() {
        // Cross-check the τ = m Jvpᵀ J̇vp qd payload Coriolis against an
        // independent Christoffel-symbol evaluation that finite-differences the
        // payload mass matrix M(q) = m JvpᵀJvp. An off-axis COM is used so the
        // velocity coupling is non-trivial.
        let q = [0.2, -0.3, 0.5, -0.6, 0.4, 0.7, -0.1];
        let qd = [0.3, -0.2, 0.4, 0.1, -0.5, 0.2, 0.3];
        let payload = Payload::new(2.0, [0.04, -0.03, 0.10]);

        let tau = Xarm7::coriolis(&q, &qd, &payload);

        // Christoffel symbols from M(q): C_i = Σ_jk ½(∂M_ij/∂q_k + ∂M_ik/∂q_j
        //   − ∂M_jk/∂q_i) qd_k qd_j.
        let h = 1e-6;
        let mass = |qq: &[f64; 7]| Xarm7::mass_matrix(qq, &payload);
        let dm = |a: usize, b: usize, k: usize| -> f64 {
            let mut qp = q;
            let mut qm = q;
            qp[k] += h;
            qm[k] -= h;
            (mass(&qp)[a][b] - mass(&qm)[a][b]) / (2.0 * h)
        };
        let mut expected = [0.0f64; 7];
        for i in 0..7 {
            let mut acc = 0.0;
            for j in 0..7 {
                for k in 0..7 {
                    let c = 0.5 * (dm(i, j, k) + dm(i, k, j) - dm(j, k, i));
                    acc += c * qd[k] * qd[j];
                }
            }
            expected[i] = acc;
        }
        for i in 0..7 {
            assert!(
                (tau[i] - expected[i]).abs() <= 1e-3,
                "joint {i}: facade {} vs christoffel {}",
                tau[i],
                expected[i]
            );
        }
    }

    #[test]
    fn massless_payload_dynamics_are_zero() {
        let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let qd = [0.1; 7];
        assert_eq!(
            Xarm7::gravity(&q, &Payload::NONE, &[0.0, 0.0, -9.81]),
            [0.0; 7]
        );
        assert_eq!(Xarm7::coriolis(&q, &qd, &Payload::NONE), [0.0; 7]);
        assert_eq!(Xarm7::mass_matrix(&q, &Payload::NONE), [[0.0; 7]; 7]);
    }
}
