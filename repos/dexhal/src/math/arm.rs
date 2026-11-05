//! Generic, arm-parameterised kinematics/dynamics facade with runtime TCP and
//! grasped-payload support.
//!
//! This module wraps the per-arm drivers behind a single [`Arm`] trait so the
//! same call shape works for every supported manipulator, with the runtime
//! tool-center-point (TCP) and grasped payload passed as plain arguments:
//!
//! ```
//! use dexhal::math::arm::{ArmModel, Payload, Pose};
//! use dexhal::math::tolerances::TolerancePolicy;
//!
//! struct DemoArm;
//!
//! impl ArmModel<7, 1> for DemoArm {
//!     fn flange_fk(_q: &[f64; 7]) -> Pose {
//!         Pose::identity()
//!     }
//!
//!     fn flange_jacobian(_q: &[f64; 7]) -> [[f64; 7]; 6] {
//!         [[0.0; 7]; 6]
//!     }
//!
//!     fn flange_solve(
//!         _target: &Pose,
//!         _max_solutions: Option<usize>,
//!         _q_seed: Option<&[f64; 7]>,
//!         _respect_limits: bool,
//!         _allow_refinement: bool,
//!         _policy: TolerancePolicy,
//!         _refinement_max_iters: usize,
//!     ) -> [Option<[f64; 7]>; 1] {
//!         [None]
//!     }
//! }
//!
//! let q = [0.0; 7];
//! let mut tcp = Pose::identity();
//! tcp[(2, 3)] = 0.12; // 12 cm tool along the flange z axis.
//!
//! let tcp_pose = DemoArm::fk(&q, &tcp);        // base -> TCP forward kinematics
//! let j = DemoArm::jacobian(&q, &tcp);         // geometric Jacobian at the TCP
//! let payload = Payload::new(2.5, [0.0, 0.0, 0.06]);
//! let g = DemoArm::gravity(&q, &payload, &[0.0, 0.0, -9.81]);
//!
//! assert_eq!(tcp_pose[(2, 3)], 0.12);
//! assert_eq!(j[0][0], 0.0);
//! assert_eq!(g[0], 0.0);
//! ```
//!
//! Each arm is a zero-sized marker type whose [`ArmModel`] implementation lives
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
pub fn flange_offset_jacobian<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    offset: &[f64; 3],
) -> [[f64; 3]; DOF] {
    // r = R_flange · offset (point offset from flange origin, in base frame).
    let mut r = [0.0f64; 3];
    for i in 0..3 {
        let mut s = 0.0;
        for k in 0..3 {
            s += flange_pose[(i, k)] * offset[k];
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

    /// Gravity torque of the flange (DOF), if the driver exposes dynamics.
    ///
    /// Defaults to zero so kinematics-only OEM packages do not need to
    /// implement unused dynamics hooks.
    #[inline]
    fn flange_gravity(_q: &[f64; DOF], _gravity: &[f64; 3]) -> [f64; DOF] {
        [0.0; DOF]
    }

    /// Mass matrix of the flange (DOF × DOF), if the driver exposes dynamics.
    #[inline]
    fn flange_mass_matrix(_q: &[f64; DOF]) -> [[f64; DOF]; DOF] {
        [[0.0; DOF]; DOF]
    }

    /// Coriolis vector of the flange (DOF), if the driver exposes dynamics.
    #[inline]
    fn flange_coriolis(_q: &[f64; DOF], _qd: &[f64; DOF]) -> [f64; DOF] {
        [0.0; DOF]
    }

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
        let lin = flange_offset_jacobian(&flange, &jflange, &p);
        
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
        let jvp = flange_offset_jacobian(&Self::flange_fk(q), &Self::flange_jacobian(q), &payload.com);
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
        let jvp = flange_offset_jacobian(&Self::flange_fk(q), &Self::flange_jacobian(q), &payload.com);
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
        let jvp = flange_offset_jacobian(
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
        let jvp_p = flange_offset_jacobian(
            &Self::flange_fk(&qp),
            &Self::flange_jacobian(&qp),
            &payload.com,
        );
        let jvp_m = flange_offset_jacobian(
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

