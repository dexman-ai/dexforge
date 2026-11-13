//! Generic, arm-parameterised kinematics/dynamics facade with runtime TCP and
//! grasped-payload support.
//!
//! This module wraps the per-arm drivers behind a single [`ArmModel`] trait so the
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
//! methods ([`ArmModel::fk`], [`ArmModel::ik`], [`ArmModel::jacobian`],
//! [`ArmModel::inverse_jacobian`], [`ArmModel::gravity`], [`ArmModel::mass_matrix`],
//! [`ArmModel::coriolis`]) are default trait methods built directly in this module,
//! so adding TCP/payload support requires **no** code
//! regeneration and changes no per-arm driver code.
//!
//! Like the rest of the crate, every method is `#![no_std]`, allocates nothing,
//! runs in bounded time (fixed `DOF`/`3`/`6` loops, no iterative solvers beyond
//! the underlying `solve`), and is deterministic.
//!
//! # Layers
//!
//! This module is the whole contract surface of the crate:
//!
//! 1. [`ArmModel`] — what a driver crate implements: the *nominal*, stateless,
//!    generated flange primitives plus static facts (`NAME`, `JOINT_LIMITS`,
//!    `HAS_DYNAMICS`).
//! 2. [`Arm`] — one physical arm: an [`ArmModel`] plus its runtime state (TCP,
//!    payload, gravity direction, per-robot calibration). Statically typed and
//!    `DOF`-sized; this is what control code calls for FK / IK / Jacobian /
//!    dynamics.
//! 3. [`ArmMath`] / [`ArmDriver`] — the `MAX_DOF`-padded, object-safe form of
//!    the same thing, together with [`ArmManifest`], [`CoreState`] and
//!    [`CoreCommand`]. It mirrors DexZero's driver contract so a kernel can
//!    hold any arm as `dyn ArmDriver`.
//!
//! Calibration ([`Arm::set_calibration`]) swaps the flange FK and Jacobian for a
//! runtime [`KinematicChain`] (a fixed `DOF` loop) and polishes the nominal
//! analytic IK onto it with at most [`CHAIN_IK_MAX_ITERS`] Newton steps per
//! solution, so the bounded-time, allocation-free guarantees are unchanged and
//! an uncalibrated arm pays nothing.

use core::marker::PhantomData;

use nalgebra::SMatrix;

pub use crate::math::chain::{
    apply_dh_deltas, pose_from_xyz_quat, pose_to_xyz_quat, ChainJoint, DhConvention, DhRow,
    KinematicChain, ModelError, CHAIN_IK_MAX_ITERS,
};
use crate::math::postprocess::JointLimit;
use crate::math::ssik_core::{add3, cross3, dot3, wrap_to_pi};
use crate::math::tolerances::TolerancePolicy;

/// Finite-difference step (rad) for the payload Coriolis directional derivative.
const CORIOLIS_FD_STEP: f64 = 1e-6;

/// Standard gravity (m/s²) used for the default gravity vector `[0, 0, -g]`.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// Largest Frobenius deviation `‖fk_calibrated(q) − fk_nominal(q)‖` accepted by
/// [`Arm::set_calibration`]. Real calibrations are well under a centimetre; a
/// larger value means the chain is expressed in different base/flange frames
/// than the driver, which would silently break the nominal-seeded IK.
pub const CALIBRATION_MAX_DEVIATION: f64 = 0.05;

/// Number of probe configurations checked by [`Arm::set_calibration`].
pub const CALIBRATION_PROBES: usize = 6;

/// Refinement ceiling handed to the native solver by [`Arm::ik_nearest`].
pub const IK_NEAREST_REFINEMENT_ITERS: usize = 15;

/// FK closure tolerance (Frobenius norm of `fk(q) − target`) every calibrated
/// IK solution must meet; the crate-wide `fk_atol`.
pub const CALIBRATED_FK_ATOL: f64 = 1e-5;

/// A 4×4 homogeneous SE(3) pose, matching the per-arm `fk` / `solve` type.
pub type Pose = SMatrix<f64, 4, 4>;

/// A grasped tool / payload rigidly attached to the flange (end-effector link),
/// modelled as a point mass.
///
/// A `mass` and a centre of mass `com` expressed in the flange frame: everything
/// needed for gravity compensation (the dominant term when holding a tool in
/// contact-rich tasks) and for the point-mass inertia contribution. Use
/// [`RigidPayload`] to add a rotational inertia tensor.
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

/// A [`Payload`] with a rotational inertia tensor, as carried by an [`Arm`].
///
/// The inertia only matters for large tools at high angular rates; a plain
/// [`Payload`] converts with zero inertia.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidPayload {
    /// Tool mass in kilograms.
    pub mass: f64,
    /// Tool centre of mass `[x, y, z]` in the flange (end-effector) frame.
    pub com: [f64; 3],
    /// Rotational inertia (kg·m²) about `com`, in flange-frame axes.
    pub inertia: [[f64; 3]; 3],
}

impl From<Payload> for RigidPayload {
    fn from(p: Payload) -> RigidPayload {
        RigidPayload::new(p.mass, p.com)
    }
}

impl RigidPayload {
    /// A massless payload (no dynamics contribution).
    pub const NONE: RigidPayload = RigidPayload::new(0.0, [0.0, 0.0, 0.0]);

    /// Construct a point-mass payload from a mass and flange-frame centre of mass.
    #[inline]
    pub const fn new(mass: f64, com: [f64; 3]) -> RigidPayload {
        RigidPayload {
            mass,
            com,
            inertia: [[0.0; 3]; 3],
        }
    }

    /// Add a rotational inertia tensor about the centre of mass.
    #[inline]
    pub const fn with_inertia(mut self, inertia: [[f64; 3]; 3]) -> RigidPayload {
        self.inertia = inertia;
        self
    }

    /// Build and validate a payload from the driver-contract argument shape
    /// (`inertia` row-major 3×3).
    pub fn from_contract(
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<RigidPayload, ModelError> {
        let payload = RigidPayload::new(mass_kg, com).with_inertia([
            [inertia[0], inertia[1], inertia[2]],
            [inertia[3], inertia[4], inertia[5]],
            [inertia[6], inertia[7], inertia[8]],
        ]);
        payload.validate()?;
        Ok(payload)
    }

    /// Reject non-finite values and negative mass.
    pub fn validate(&self) -> Result<(), ModelError> {
        let finite = self.mass.is_finite()
            && self.com.iter().all(|v| v.is_finite())
            && self.inertia.iter().flatten().all(|v| v.is_finite());
        if !finite {
            return Err(ModelError::NonFinite);
        }
        if self.mass < 0.0 {
            return Err(ModelError::InvalidPayload);
        }
        Ok(())
    }

    #[inline]
    fn has_inertia(&self) -> bool {
        self.inertia.iter().flatten().any(|v| *v != 0.0)
    }

    #[inline]
    fn is_none(&self) -> bool {
        self.mass == 0.0 && !self.has_inertia()
    }
}

/// Payload inertia tensor rotated into base-frame axes: `R I Rᵀ`.
fn base_inertia(flange_pose: &Pose, inertia: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut ri = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ri[i][j] += flange_pose[(i, k)] * inertia[k][j];
            }
        }
    }
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += ri[i][k] * flange_pose[(j, k)];
            }
        }
    }
    out
}

#[inline]
fn mat3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [dot3(&m[0], v), dot3(&m[1], v), dot3(&m[2], v)]
}

#[inline]
fn angular_column<const DOF: usize>(jacobian: &[[f64; DOF]; 6], j: usize) -> [f64; 3] {
    [jacobian[3][j], jacobian[4][j], jacobian[5][j]]
}

/// Holding torque contributed by a grasped `payload`:
/// `tau_j = -m (g · ∂p_com/∂q_j)`.
///
/// `flange_pose` and `jacobian` are the flange FK and geometric Jacobian at
/// the configuration of interest.
pub fn payload_gravity_torque<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    payload: &RigidPayload,
    gravity: &[f64; 3],
) -> [f64; DOF] {
    let mut tau = [0.0f64; DOF];
    if payload.mass == 0.0 {
        return tau;
    }
    let jvp = flange_offset_jacobian(flange_pose, jacobian, &payload.com);
    for j in 0..DOF {
        tau[j] = -payload.mass * dot3(gravity, &jvp[j]);
    }
    tau
}

/// Contribution of a grasped `payload` to the joint-space inertia `M(q)`:
/// `m Jvpᵀ Jvp + Jwᵀ (R I Rᵀ) Jw`.
pub fn payload_inertia_matrix<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    payload: &RigidPayload,
) -> [[f64; DOF]; DOF] {
    let mut mm = [[0.0f64; DOF]; DOF];
    if payload.mass != 0.0 {
        let jvp = flange_offset_jacobian(flange_pose, jacobian, &payload.com);
        for i in 0..DOF {
            for j in 0..DOF {
                mm[i][j] = payload.mass * dot3(&jvp[i], &jvp[j]);
            }
        }
    }
    if payload.has_inertia() {
        let ib = base_inertia(flange_pose, &payload.inertia);
        for j in 0..DOF {
            let iw = mat3_vec(&ib, &angular_column(jacobian, j));
            for i in 0..DOF {
                mm[i][j] += dot3(&angular_column(jacobian, i), &iw);
            }
        }
    }
    mm
}

/// Coriolis & centrifugal torque `C(q, qd) qd` contributed by a grasped
/// `payload`.
///
/// `fk_jacobian` evaluates the flange FK and geometric Jacobian. The bias
/// accelerations `J̇ qd` come from a central difference along `qd` (two extra
/// evaluations), so the routine is allocation-free and constant-time.
pub fn payload_coriolis<const DOF: usize, F>(
    fk_jacobian: F,
    q: &[f64; DOF],
    qd: &[f64; DOF],
    payload: &RigidPayload,
) -> [f64; DOF]
where
    F: Fn(&[f64; DOF]) -> (Pose, [[f64; DOF]; 6]),
{
    let mut tau = [0.0f64; DOF];
    if payload.is_none() {
        return tau;
    }
    let h = CORIOLIS_FD_STEP;
    let mut qp = *q;
    let mut qm = *q;
    for j in 0..DOF {
        qp[j] += h * qd[j];
        qm[j] -= h * qd[j];
    }
    let (pose, jac) = fk_jacobian(q);
    let (pose_p, jac_p) = fk_jacobian(&qp);
    let (pose_m, jac_m) = fk_jacobian(&qm);

    if payload.mass != 0.0 {
        let jvp = flange_offset_jacobian(&pose, &jac, &payload.com);
        let jvp_p = flange_offset_jacobian(&pose_p, &jac_p, &payload.com);
        let jvp_m = flange_offset_jacobian(&pose_m, &jac_m, &payload.com);
        // a_bias = (d/dt Jvp) qd.
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
    }
    if payload.has_inertia() {
        let ib = base_inertia(&pose, &payload.inertia);
        let mut omega = [0.0f64; 3];
        let mut alpha_bias = [0.0f64; 3];
        for j in 0..DOF {
            let w = qd[j] / (2.0 * h);
            for r in 0..3 {
                omega[r] += jac[3 + r][j] * qd[j];
                alpha_bias[r] += (jac_p[3 + r][j] - jac_m[3 + r][j]) * w;
            }
        }
        // Euler: I α_bias + ω × I ω.
        let moment = add3(
            &mat3_vec(&ib, &alpha_bias),
            &cross3(&omega, &mat3_vec(&ib, &omega)),
        );
        for i in 0..DOF {
            tau[i] += dot3(&angular_column(&jac, i), &moment);
        }
    }
    tau
}

/// Re-reference a flange geometric Jacobian to the TCP origin.
fn tcp_jacobian<const DOF: usize>(
    flange_pose: &Pose,
    jflange: &[[f64; DOF]; 6],
    tcp: &Pose,
) -> [[f64; DOF]; 6] {
    let p = [tcp[(0, 3)], tcp[(1, 3)], tcp[(2, 3)]];
    let lin = flange_offset_jacobian(flange_pose, jflange, &p);
    let mut j = *jflange;
    for c in 0..DOF {
        j[0][c] = lin[c][0];
        j[1][c] = lin[c][1];
        j[2][c] = lin[c][2];
    }
    j
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
    /// Human-readable model name reported in the [`ArmManifest`].
    const NAME: &'static str = "";

    /// Position limits per joint (`None` = unlimited / unknown).
    const JOINT_LIMITS: [JointLimit; DOF] = [None; DOF];

    /// Set to `true` by drivers that override the `flange_gravity`,
    /// `flange_mass_matrix` and `flange_coriolis` hooks. Without it the zero
    /// defaults are indistinguishable from a real model, so [`Arm::dynamics`]
    /// refuses to answer.
    const HAS_DYNAMICS: bool = false;

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
        tcp_jacobian(&Self::flange_fk(q), &Self::flange_jacobian(q), tcp)
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
        if payload.mass == 0.0 {
            return [0.0; DOF];
        }
        payload_gravity_torque(
            &Self::flange_fk(q),
            &Self::flange_jacobian(q),
            &(*payload).into(),
            gravity,
        )
    }

    /// Grasped `payload` (point-mass) contribution to the joint-space inertia
    /// matrix `M(q)`. See [`payload_inertia_matrix`].
    #[inline]
    fn mass_matrix(q: &[f64; DOF], payload: &Payload) -> [[f64; DOF]; DOF] {
        if payload.mass == 0.0 {
            return [[0.0; DOF]; DOF];
        }
        payload_inertia_matrix(
            &Self::flange_fk(q),
            &Self::flange_jacobian(q),
            &(*payload).into(),
        )
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
        payload_coriolis(
            |q| (Self::flange_fk(q), Self::flange_jacobian(q)),
            q,
            qd,
            &(*payload).into(),
        )
    }
}

/// Whether `q` lies inside [`ArmModel::JOINT_LIMITS`] (bounds inclusive).
fn within_model_limits<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(
    q: &[f64; DOF],
) -> bool {
    for i in 0..DOF {
        if let Some((lo, hi)) = A::JOINT_LIMITS[i] {
            if q[i] < lo || q[i] > hi {
                return false;
            }
        }
    }
    true
}

// ===========================================================================
// Arm: one physical arm = model + runtime configuration
// ===========================================================================

/// Joint-space rigid-body terms of an arm including its payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dynamics<const DOF: usize> {
    /// Mass / inertia matrix `M(q)`.
    pub mass_matrix: [[f64; DOF]; DOF],
    /// Coriolis and centrifugal torque `C(q, qd) qd`.
    pub coriolis: [f64; DOF],
    /// Gravity torque `G(q)`.
    pub gravity: [f64; DOF],
}

/// One physical arm: the generated model `A` plus the state that differs per
/// installation and changes at runtime — TCP, payload, gravity direction
/// (mounting orientation) and per-robot kinematic calibration.
///
/// All setters validate their input and leave the arm unchanged on error, so
/// they are safe to call between control cycles. Every query is
/// allocation-free and bounded.
///
/// ```
/// # use dexhal::math::arm::*;
/// # use dexhal::math::tolerances::TolerancePolicy;
/// # struct Demo;
/// # impl ArmModel<6, 1> for Demo {
/// #     fn flange_fk(_q: &[f64; 6]) -> Pose { Pose::identity() }
/// #     fn flange_jacobian(_q: &[f64; 6]) -> [[f64; 6]; 6] { [[0.0; 6]; 6] }
/// #     fn flange_solve(_t: &Pose, _m: Option<usize>, _s: Option<&[f64; 6]>, _l: bool, _r: bool,
/// #         _p: TolerancePolicy, _i: usize) -> [Option<[f64; 6]>; 1] { [None] }
/// # }
/// let mut arm = Arm::<Demo, 6, 1>::new();
/// let mut tcp = Pose::identity();
/// tcp[(2, 3)] = 0.12;
/// arm.set_tcp(&tcp).unwrap();
/// arm.set_payload(Payload::new(2.5, [0.0, 0.0, 0.06])).unwrap();
/// assert_eq!(arm.fk(&[0.0; 6])[(2, 3)], 0.12);
/// ```
pub struct Arm<A, const DOF: usize, const MAX: usize> {
    tcp: Pose,
    tcp_inv: Pose,
    payload: RigidPayload,
    gravity: [f64; 3],
    calibration: Option<KinematicChain<DOF>>,
    _model: PhantomData<fn() -> A>,
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> Default for Arm<A, DOF, MAX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> Arm<A, DOF, MAX> {
    /// Nominal arm: identity TCP, no payload, gravity `[0, 0, -g]`, no calibration.
    pub fn new() -> Self {
        Arm {
            tcp: Pose::identity(),
            tcp_inv: Pose::identity(),
            payload: RigidPayload::NONE,
            gravity: [0.0, 0.0, -STANDARD_GRAVITY],
            calibration: None,
            _model: PhantomData,
        }
    }

    // --- configuration -----------------------------------------------------

    /// Set the flange → TCP transform.
    pub fn set_tcp(&mut self, tcp: &Pose) -> Result<(), ModelError> {
        crate::math::chain::Rigid::from_pose(tcp)?;
        self.tcp = *tcp;
        self.tcp_inv = se3_inv(tcp);
        Ok(())
    }

    pub fn tcp(&self) -> &Pose {
        &self.tcp
    }

    /// Set the grasped payload ([`Payload::NONE`] to release).
    ///
    /// Accepts a point-mass [`Payload`] or a [`RigidPayload`].
    pub fn set_payload(&mut self, payload: impl Into<RigidPayload>) -> Result<(), ModelError> {
        let payload = payload.into();
        payload.validate()?;
        self.payload = payload;
        Ok(())
    }

    pub fn payload(&self) -> &RigidPayload {
        &self.payload
    }

    /// Set the gravity vector in the arm's base frame (wall / ceiling mounts).
    pub fn set_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError> {
        if !gravity.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFinite);
        }
        self.gravity = gravity;
        Ok(())
    }

    pub fn gravity(&self) -> &[f64; 3] {
        &self.gravity
    }

    /// Install a per-robot calibrated chain.
    ///
    /// The chain must describe the same base → flange frames as the driver's
    /// `flange_fk`; it is compared against the nominal model at
    /// [`CALIBRATION_PROBES`] fixed configurations and rejected with
    /// [`ModelError::ExcessiveDeviation`] if any differs by more than
    /// [`CALIBRATION_MAX_DEVIATION`].
    pub fn set_calibration(&mut self, chain: KinematicChain<DOF>) -> Result<(), ModelError> {
        for k in 0..CALIBRATION_PROBES {
            let mut q = [0.0f64; DOF];
            for (i, v) in q.iter_mut().enumerate() {
                let sign = if (i + k) % 2 == 0 { 1.0 } else { -1.0 };
                *v = 0.35 * k as f64 * sign;
            }
            let deviation = (chain.fk(&q) - A::flange_fk(&q)).norm();
            if !deviation.is_finite() {
                return Err(ModelError::NonFinite);
            }
            if deviation > CALIBRATION_MAX_DEVIATION {
                return Err(ModelError::ExcessiveDeviation);
            }
        }
        self.calibration = Some(chain);
        Ok(())
    }

    /// Return to the nominal generated kinematics.
    pub fn clear_calibration(&mut self) {
        self.calibration = None;
    }

    pub fn calibration(&self) -> Option<&KinematicChain<DOF>> {
        self.calibration.as_ref()
    }

    // --- kinematics --------------------------------------------------------

    /// Flange FK and geometric Jacobian (calibrated if a calibration is set).
    pub fn flange_fk_jacobian(&self, q: &[f64; DOF]) -> (Pose, [[f64; DOF]; 6]) {
        match &self.calibration {
            Some(chain) => chain.fk_jacobian(q),
            None => (A::flange_fk(q), A::flange_jacobian(q)),
        }
    }

    /// Flange FK (calibrated if a calibration is set).
    pub fn flange_fk(&self, q: &[f64; DOF]) -> Pose {
        match &self.calibration {
            Some(chain) => chain.fk(q),
            None => A::flange_fk(q),
        }
    }

    /// Forward kinematics base → TCP.
    pub fn fk(&self, q: &[f64; DOF]) -> Pose {
        self.flange_fk(q) * self.tcp
    }

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) referenced at the TCP origin.
    pub fn jacobian(&self, q: &[f64; DOF]) -> [[f64; DOF]; 6] {
        let (pose, jac) = self.flange_fk_jacobian(q);
        tcp_jacobian(&pose, &jac, &self.tcp)
    }

    /// Minimum-norm pseudo-inverse of the TCP Jacobian; `None` at a singularity.
    pub fn inverse_jacobian(&self, q: &[f64; DOF]) -> Option<[[f64; 6]; DOF]> {
        pseudo_inverse(&self.jacobian(q))
    }

    /// All inverse-kinematics solutions placing the TCP at `tcp_target`.
    ///
    /// Arguments are those of the native solver. With a calibration installed
    /// the nominal analytic solutions are each polished onto the calibrated
    /// chain ([`KinematicChain::refine_ik`]), re-verified against
    /// [`CALIBRATED_FK_ATOL`] and, if `respect_limits`, re-checked against
    /// [`ArmModel::JOINT_LIMITS`]; solutions failing either are dropped. A
    /// target reachable only by the calibrated geometry (within a millimetre
    /// of the workspace boundary) therefore yields no solution.
    #[allow(clippy::too_many_arguments)]
    pub fn ik(
        &self,
        tcp_target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; DOF]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; DOF]>; MAX] {
        let flange_target = tcp_target * self.tcp_inv;
        let nominal = A::flange_solve(
            &flange_target,
            max_solutions,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        );
        let chain = match &self.calibration {
            Some(chain) => chain,
            None => return nominal,
        };
        let mut out = [None; MAX];
        let mut n = 0;
        for seed in nominal.iter().flatten() {
            let mut q = *seed;
            if chain.refine_ik(&flange_target, &mut q) > CALIBRATED_FK_ATOL {
                continue;
            }
            if respect_limits && !within_model_limits::<A, DOF, MAX>(&q) {
                continue;
            }
            out[n] = Some(q);
            n += 1;
        }
        out
    }

    /// The single IK solution closest to `q_seed`, for servoing.
    ///
    /// Distance is measured modulo 2π and each joint is returned on the
    /// 2π-branch nearest the seed when its limits allow, so a continuous
    /// Cartesian path maps to a continuous joint path.
    pub fn ik_nearest(
        &self,
        tcp_target: &Pose,
        q_seed: &[f64; DOF],
        policy: TolerancePolicy,
    ) -> Option<[f64; DOF]> {
        let solutions = self.ik(
            tcp_target,
            None,
            Some(q_seed),
            true,
            true,
            policy,
            IK_NEAREST_REFINEMENT_ITERS,
        );
        let mut best: Option<([f64; DOF], f64)> = None;
        for q in solutions.iter().flatten() {
            let mut dist = 0.0;
            for i in 0..DOF {
                let d = wrap_to_pi(q[i] - q_seed[i]);
                dist += d * d;
            }
            if best.is_none_or(|(_, b)| dist < b) {
                best = Some((*q, dist));
            }
        }
        let (mut q, _) = best?;
        for i in 0..DOF {
            let near = q_seed[i] + wrap_to_pi(q[i] - q_seed[i]);
            let allowed = match A::JOINT_LIMITS[i] {
                Some((lo, hi)) => near >= lo && near <= hi,
                None => true,
            };
            if allowed {
                q[i] = near;
            }
        }
        Some(q)
    }

    // --- dynamics ----------------------------------------------------------

    /// Holding torque of the payload alone under the configured gravity.
    pub fn payload_gravity(&self, q: &[f64; DOF]) -> [f64; DOF] {
        let (pose, jac) = self.flange_fk_jacobian(q);
        payload_gravity_torque(&pose, &jac, &self.payload, &self.gravity)
    }

    /// Payload contribution to `M(q)`.
    pub fn payload_mass_matrix(&self, q: &[f64; DOF]) -> [[f64; DOF]; DOF] {
        let (pose, jac) = self.flange_fk_jacobian(q);
        payload_inertia_matrix(&pose, &jac, &self.payload)
    }

    /// Payload contribution to `C(q, qd) qd`.
    pub fn payload_coriolis(&self, q: &[f64; DOF], qd: &[f64; DOF]) -> [f64; DOF] {
        payload_coriolis(|q| self.flange_fk_jacobian(q), q, qd, &self.payload)
    }

    /// `M(q)`, `C(q, qd) qd` and `G(q)` of the arm *including* its payload.
    ///
    /// `None` if the model has no dynamics ([`ArmModel::HAS_DYNAMICS`]). The
    /// arm's own terms use the nominal link geometry: kinematic calibration
    /// shifts them by parts per ten thousand, far below the uncertainty of the
    /// inertial parameters.
    pub fn dynamics(&self, q: &[f64; DOF], qd: &[f64; DOF]) -> Option<Dynamics<DOF>> {
        if !A::HAS_DYNAMICS {
            return None;
        }
        let mut out = Dynamics {
            mass_matrix: A::flange_mass_matrix(q),
            coriolis: A::flange_coriolis(q, qd),
            gravity: A::flange_gravity(q, &self.gravity),
        };
        if self.payload.is_none() {
            return Some(out);
        }
        let (pose, jac) = self.flange_fk_jacobian(q);
        let m = payload_inertia_matrix(&pose, &jac, &self.payload);
        let g = payload_gravity_torque(&pose, &jac, &self.payload, &self.gravity);
        let c = self.payload_coriolis(q, qd);
        for i in 0..DOF {
            out.gravity[i] += g[i];
            out.coriolis[i] += c[i];
            for j in 0..DOF {
                out.mass_matrix[i][j] += m[i][j];
            }
        }
        Some(out)
    }
}

// ===========================================================================
// Driver contract: manifest, wire layouts and the object-safe control API
// ===========================================================================

/// Maximum degrees of freedom carried by the padded contract types.
pub const MAX_DOF: usize = 8;

/// Semantic version of the driver contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractVersion {
    /// Breaking changes (e.g. the [`CoreState`] layout).
    pub major: u32,
    /// Backwards-compatible additions.
    pub minor: u32,
    pub patch: u32,
}

/// Contract version implemented by this crate.
pub const CONTRACT_VERSION: ContractVersion = ContractVersion {
    major: 0,
    minor: 1,
    patch: 0,
};

/// Bit flags of the targets a hardware controller executes natively.
pub mod control_mode {
    /// `[f64; N]` joint angles.
    pub const JOINT_POS: u32 = 1 << 0;
    /// `[f64; N]` joint velocities.
    pub const JOINT_VEL: u32 = 1 << 1;
    /// `[f64; N]` motor torques.
    pub const JOINT_TORQUE: u32 = 1 << 2;
    /// `[x, y, z, qx, qy, qz, qw]` absolute pose.
    pub const CARTESIAN_POSE: u32 = 1 << 3;
    /// `[f64; 6]` spatial velocity (linear, angular).
    pub const CARTESIAN_TWIST: u32 = 1 << 4;
    /// `[f64; 6]` force / torque.
    pub const CARTESIAN_WRENCH: u32 = 1 << 5;
}

/// Errors of the control API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArmError {
    /// The hardware does not implement this natively; the host model applies.
    NotSupported,
    /// An argument was rejected by the model.
    Model(ModelError),
    /// A buffer is shorter than the manifest's layout.
    BufferTooSmall,
    /// Hardware or transport fault, with a static description.
    Fault(&'static str),
}

impl From<ModelError> for ArmError {
    fn from(e: ModelError) -> Self {
        ArmError::Model(e)
    }
}

/// Where a configuration change took effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applied {
    /// The robot controller applied it (the host model is kept in sync).
    Native,
    /// Only the host-side model applied it; the kernel must compensate.
    Host,
}

/// One field of the driver-specific tail following [`CoreState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxField {
    /// e.g. `"motor_1_temp"`.
    pub label: &'static str,
    /// Number of `f64` slots (1 for scalars).
    pub size: u32,
}

/// What the host-side model of an arm can compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Number of actuated joints.
    pub dof: u8,
    /// Capacity of the analytic IK solution set.
    pub max_ik_solutions: u16,
    /// `M`, `C`, `G` are available ([`ArmModel::HAS_DYNAMICS`]).
    pub dynamics: bool,
}

/// Static description of an arm, read once by the kernel at boot.
///
/// Field-for-field the DexZero `DriverManifest`, with the heap-backed aux
/// schema replaced by a `'static` slice and three additions: `name`, `model`
/// and the `native_payload` / `native_tcp_offset` flags that let the kernel
/// plan fallbacks up front instead of probing for errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmManifest {
    pub name: &'static str,
    pub version: ContractVersion,
    /// Rate (Hz) at which the hardware expects commands.
    pub control_freq_hz: u32,
    /// OR of [`control_mode`] flags executed natively.
    pub control_modes: u32,
    /// The driver needs a real-time (SCHED_FIFO) thread.
    pub is_realtime_required: bool,
    pub active_dof_count: u8,
    /// Layout of the state tail following [`CoreState`].
    pub aux: &'static [AuxField],
    /// Cycles between the hardware sampling its state and `read_state` returning.
    pub state_staleness_cycles: u32,
    /// Cycles between `write_command` returning and the hardware acting.
    pub command_dispatch_cycles: u32,
    pub has_torque_sensing: bool,
    pub has_wrench_sensing: bool,
    pub has_integrated_gripper: bool,
    pub native_cartesian_impedance: bool,
    pub native_cartesian_admittance: bool,
    pub native_joint_impedance: bool,
    pub native_joint_admittance: bool,
    pub native_contact_detection: bool,
    pub native_reflexes: bool,
    /// The controller accepts payload changes itself.
    pub native_payload: bool,
    /// The controller accepts TCP changes itself.
    pub native_tcp_offset: bool,
    pub joint_limits_min: [f64; MAX_DOF],
    pub joint_limits_max: [f64; MAX_DOF],
    pub velocity_limits: [f64; MAX_DOF],
    pub effort_limits: [f64; MAX_DOF],
    pub model: ModelCapabilities,
}

impl ArmManifest {
    /// Manifest of a bare model: name, DOF, joint limits and model
    /// capabilities from `A`; no native features, no latency, unbounded
    /// velocity / effort. Drivers overwrite the public fields they know.
    pub fn for_model<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(
        control_freq_hz: u32,
        control_modes: u32,
    ) -> ArmManifest {
        const { assert!(DOF <= MAX_DOF) };
        let mut lo = [f64::NEG_INFINITY; MAX_DOF];
        let mut hi = [f64::INFINITY; MAX_DOF];
        for i in 0..DOF {
            if let Some((a, b)) = A::JOINT_LIMITS[i] {
                lo[i] = a;
                hi[i] = b;
            }
        }
        ArmManifest {
            name: A::NAME,
            version: CONTRACT_VERSION,
            control_freq_hz,
            control_modes,
            is_realtime_required: false,
            active_dof_count: DOF as u8,
            aux: &[],
            state_staleness_cycles: 0,
            command_dispatch_cycles: 0,
            has_torque_sensing: false,
            has_wrench_sensing: false,
            has_integrated_gripper: false,
            native_cartesian_impedance: false,
            native_cartesian_admittance: false,
            native_joint_impedance: false,
            native_joint_admittance: false,
            native_contact_detection: false,
            native_reflexes: false,
            native_payload: false,
            native_tcp_offset: false,
            joint_limits_min: lo,
            joint_limits_max: hi,
            velocity_limits: [f64::INFINITY; MAX_DOF],
            effort_limits: [f64::INFINITY; MAX_DOF],
            model: ModelCapabilities {
                dof: DOF as u8,
                max_ik_solutions: MAX as u16,
                dynamics: A::HAS_DYNAMICS,
            },
        }
    }

    /// Whether every flag in `modes` is executed natively.
    pub fn supports(&self, modes: u32) -> bool {
        self.control_modes & modes == modes
    }

    /// Length in `f64` of the state buffer: [`CoreState`] plus the aux tail.
    pub fn state_len(&self) -> usize {
        let mut n = CoreState::LEN;
        for f in self.aux {
            n += f.size as usize;
        }
        n
    }

    /// Length in `f64` of the command buffer.
    pub fn command_len(&self) -> usize {
        CoreCommand::LEN
    }
}

struct Reader<'a>(&'a [f64]);

impl Reader<'_> {
    fn array<const N: usize>(&mut self) -> [f64; N] {
        let (head, tail) = self.0.split_at(N);
        self.0 = tail;
        let mut out = [0.0; N];
        out.copy_from_slice(head);
        out
    }

    fn scalar(&mut self) -> f64 {
        self.array::<1>()[0]
    }
}

struct Writer<'a>(&'a mut [f64]);

impl Writer<'_> {
    fn array(&mut self, v: &[f64]) {
        let (head, tail) = core::mem::take(&mut self.0).split_at_mut(v.len());
        head.copy_from_slice(v);
        self.0 = tail;
    }
}

/// Typed prefix of the driver → kernel state buffer. Fields the hardware does
/// not provide stay `NAN`; the kernel then falls back to the host model.
///
/// `#[repr(C)]` and all-`f64`, so the struct's memory layout *is* the wire
/// layout ([`CoreState::LEN`] slots). This crate forbids `unsafe`, so it offers
/// the copying [`read_from`](Self::read_from) / [`write_to`](Self::write_to);
/// a consumer that wants a zero-copy overlay may cast the buffer itself.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreState {
    pub q: [f64; MAX_DOF],
    pub qd: [f64; MAX_DOF],
    /// `[x, y, z, qx, qy, qz, qw]`, only if the hardware computes its own TCP.
    pub tcp_pose: [f64; 7],
    pub tcp_twist: [f64; 6],
    pub torques: [f64; MAX_DOF],
    pub wrench: [f64; 6],
    /// 1.0 = contact, 0.0 = free, NAN = unknown.
    pub in_contact: f64,
    /// Digital input bitmask carried as a float.
    pub io_digital_in: f64,
    pub io_analog_in: [f64; 4],
    pub gripper_width: f64,
    pub gripper_force: f64,
    pub gripper_has_object: f64,
}

impl CoreState {
    /// Number of `f64` slots.
    pub const LEN: usize = core::mem::size_of::<CoreState>() / core::mem::size_of::<f64>();

    /// Every field `NAN` ("not provided").
    pub const UNKNOWN: CoreState = CoreState {
        q: [f64::NAN; MAX_DOF],
        qd: [f64::NAN; MAX_DOF],
        tcp_pose: [f64::NAN; 7],
        tcp_twist: [f64::NAN; 6],
        torques: [f64::NAN; MAX_DOF],
        wrench: [f64::NAN; 6],
        in_contact: f64::NAN,
        io_digital_in: f64::NAN,
        io_analog_in: [f64::NAN; 4],
        gripper_width: f64::NAN,
        gripper_force: f64::NAN,
        gripper_has_object: f64::NAN,
    };

    /// Decode from the front of `buffer`.
    pub fn read_from(buffer: &[f64]) -> Result<CoreState, ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut r = Reader(buffer);
        Ok(CoreState {
            q: r.array(),
            qd: r.array(),
            tcp_pose: r.array(),
            tcp_twist: r.array(),
            torques: r.array(),
            wrench: r.array(),
            in_contact: r.scalar(),
            io_digital_in: r.scalar(),
            io_analog_in: r.array(),
            gripper_width: r.scalar(),
            gripper_force: r.scalar(),
            gripper_has_object: r.scalar(),
        })
    }

    /// Encode into the front of `buffer`.
    pub fn write_to(&self, buffer: &mut [f64]) -> Result<(), ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut w = Writer(buffer);
        w.array(&self.q);
        w.array(&self.qd);
        w.array(&self.tcp_pose);
        w.array(&self.tcp_twist);
        w.array(&self.torques);
        w.array(&self.wrench);
        w.array(&[self.in_contact, self.io_digital_in]);
        w.array(&self.io_analog_in);
        w.array(&[
            self.gripper_width,
            self.gripper_force,
            self.gripper_has_object,
        ]);
        Ok(())
    }
}

/// Typed prefix of the kernel → driver command buffer. Targets the kernel does
/// not drive are `NAN`. Layout rules as for [`CoreState`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreCommand {
    pub target_q: [f64; MAX_DOF],
    pub target_qd: [f64; MAX_DOF],
    pub target_torques: [f64; MAX_DOF],
    pub target_pose: [f64; 7],
    pub target_twist: [f64; 6],
    pub target_wrench: [f64; 6],
    pub task_stiffness_k: [f64; 6],
    pub task_damping_b: [f64; 6],
    pub task_mass_m: [f64; 6],
    pub joint_stiffness_k: [f64; MAX_DOF],
    pub joint_damping_b: [f64; MAX_DOF],
    pub reflex_threshold: [f64; 6],
    /// 0 = stop, 1 = yield / freedrive, 2 = retract.
    pub reflex_action_code: f64,
    pub io_digital_out_mask: f64,
    pub io_digital_out_state: f64,
    pub gripper_target_width: f64,
    pub gripper_target_speed: f64,
    pub gripper_max_force: f64,
}

impl CoreCommand {
    /// Number of `f64` slots.
    pub const LEN: usize = core::mem::size_of::<CoreCommand>() / core::mem::size_of::<f64>();

    /// Every field `NAN` ("not commanded").
    pub const IDLE: CoreCommand = CoreCommand {
        target_q: [f64::NAN; MAX_DOF],
        target_qd: [f64::NAN; MAX_DOF],
        target_torques: [f64::NAN; MAX_DOF],
        target_pose: [f64::NAN; 7],
        target_twist: [f64::NAN; 6],
        target_wrench: [f64::NAN; 6],
        task_stiffness_k: [f64::NAN; 6],
        task_damping_b: [f64::NAN; 6],
        task_mass_m: [f64::NAN; 6],
        joint_stiffness_k: [f64::NAN; MAX_DOF],
        joint_damping_b: [f64::NAN; MAX_DOF],
        reflex_threshold: [f64::NAN; 6],
        reflex_action_code: f64::NAN,
        io_digital_out_mask: f64::NAN,
        io_digital_out_state: f64::NAN,
        gripper_target_width: f64::NAN,
        gripper_target_speed: f64::NAN,
        gripper_max_force: f64::NAN,
    };

    /// Decode from the front of `buffer`.
    pub fn read_from(buffer: &[f64]) -> Result<CoreCommand, ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut r = Reader(buffer);
        Ok(CoreCommand {
            target_q: r.array(),
            target_qd: r.array(),
            target_torques: r.array(),
            target_pose: r.array(),
            target_twist: r.array(),
            target_wrench: r.array(),
            task_stiffness_k: r.array(),
            task_damping_b: r.array(),
            task_mass_m: r.array(),
            joint_stiffness_k: r.array(),
            joint_damping_b: r.array(),
            reflex_threshold: r.array(),
            reflex_action_code: r.scalar(),
            io_digital_out_mask: r.scalar(),
            io_digital_out_state: r.scalar(),
            gripper_target_width: r.scalar(),
            gripper_target_speed: r.scalar(),
            gripper_max_force: r.scalar(),
        })
    }

    /// Encode into the front of `buffer`.
    pub fn write_to(&self, buffer: &mut [f64]) -> Result<(), ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut w = Writer(buffer);
        w.array(&self.target_q);
        w.array(&self.target_qd);
        w.array(&self.target_torques);
        w.array(&self.target_pose);
        w.array(&self.target_twist);
        w.array(&self.target_wrench);
        w.array(&self.task_stiffness_k);
        w.array(&self.task_damping_b);
        w.array(&self.task_mass_m);
        w.array(&self.joint_stiffness_k);
        w.array(&self.joint_damping_b);
        w.array(&self.reflex_threshold);
        w.array(&[
            self.reflex_action_code,
            self.io_digital_out_mask,
            self.io_digital_out_state,
            self.gripper_target_width,
            self.gripper_target_speed,
            self.gripper_max_force,
        ]);
        Ok(())
    }
}

/// [`Dynamics`] padded to [`MAX_DOF`]; rows and columns beyond the active DOF
/// are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicsState {
    pub m_matrix: [[f64; MAX_DOF]; MAX_DOF],
    pub c_vector: [f64; MAX_DOF],
    pub g_vector: [f64; MAX_DOF],
}

/// The host-side mathematical model of an arm with the DOF erased
/// ([`MAX_DOF`]-padded arrays, `[x, y, z, qx, qy, qz, qw]` poses).
///
/// Implemented by every [`Arm`]; entries beyond [`dof`](Self::dof) are ignored
/// on input and zero on output. Queries return `None` for non-finite input,
/// singular / unreachable targets, or a capability the model lacks.
pub trait ArmMath {
    /// Number of actuated joints.
    fn dof(&self) -> usize;

    /// Whether a per-robot calibration is active.
    fn is_calibrated(&self) -> bool;

    /// Base → TCP pose.
    fn forward_kinematics(&self, q: &[f64; MAX_DOF]) -> Option<[f64; 7]>;

    /// Geometric Jacobian `[Jv; Jw]` at the TCP.
    fn jacobian(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]>;

    /// Joint vector reaching `target_pose` closest to `q_seed`.
    fn inverse_kinematics(
        &self,
        target_pose: &[f64; 7],
        q_seed: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]>;

    /// `M`, `C qd`, `G` including the payload.
    fn dynamics(&self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF]) -> Option<DynamicsState>;

    /// Set the payload (`inertia` row-major about the centre of mass).
    fn set_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<(), ModelError>;

    /// Set the flange → TCP transform.
    fn set_tcp_offset(&mut self, pose: [f64; 7]) -> Result<(), ModelError>;

    /// Set the gravity vector in the base frame.
    fn set_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError>;
}

fn unpad<const DOF: usize>(q: &[f64; MAX_DOF]) -> Option<[f64; DOF]> {
    const { assert!(DOF <= MAX_DOF) };
    let mut out = [0.0; DOF];
    out.copy_from_slice(&q[..DOF]);
    out.iter().all(|v| v.is_finite()).then_some(out)
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> ArmMath for Arm<A, DOF, MAX> {
    fn dof(&self) -> usize {
        DOF
    }

    fn is_calibrated(&self) -> bool {
        self.calibration.is_some()
    }

    fn forward_kinematics(&self, q: &[f64; MAX_DOF]) -> Option<[f64; 7]> {
        Some(pose_to_xyz_quat(&self.fk(&unpad(q)?)))
    }

    fn jacobian(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]> {
        let jac = Arm::jacobian(self, &unpad(q)?);
        let mut out = [[0.0; MAX_DOF]; 6];
        for r in 0..6 {
            out[r][..DOF].copy_from_slice(&jac[r]);
        }
        Some(out)
    }

    fn inverse_kinematics(
        &self,
        target_pose: &[f64; 7],
        q_seed: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        let target = pose_from_xyz_quat(target_pose).ok()?;
        let q = self.ik_nearest(&target, &unpad(q_seed)?, TolerancePolicy::DEFAULT)?;
        let mut out = [0.0; MAX_DOF];
        out[..DOF].copy_from_slice(&q);
        Some(out)
    }

    fn dynamics(&self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF]) -> Option<DynamicsState> {
        let d = Arm::dynamics(self, &unpad(q)?, &unpad(qd)?)?;
        let mut out = DynamicsState {
            m_matrix: [[0.0; MAX_DOF]; MAX_DOF],
            c_vector: [0.0; MAX_DOF],
            g_vector: [0.0; MAX_DOF],
        };
        for i in 0..DOF {
            out.m_matrix[i][..DOF].copy_from_slice(&d.mass_matrix[i]);
        }
        out.c_vector[..DOF].copy_from_slice(&d.coriolis);
        out.g_vector[..DOF].copy_from_slice(&d.gravity);
        Some(out)
    }

    fn set_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<(), ModelError> {
        Arm::set_payload(self, RigidPayload::from_contract(mass_kg, com, inertia)?)
    }

    fn set_tcp_offset(&mut self, pose: [f64; 7]) -> Result<(), ModelError> {
        self.set_tcp(&pose_from_xyz_quat(&pose)?)
    }

    fn set_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError> {
        Arm::set_gravity(self, gravity)
    }
}

/// Everything needed to control one arm: capabilities, lifecycle, the
/// real-time data plane, the configuration plane and the host-side math.
///
/// A driver implements the hardware half (`manifest`, `connect`, `disconnect`,
/// `read_state`, `write_command`) and hands out its [`Arm`] through
/// [`arm`](Self::arm) / [`arm_mut`](Self::arm_mut); the math and the
/// configuration fallbacks are then provided. The trait is object-safe.
///
/// `read_state`, `write_command` and the four math queries are the data plane
/// and must not allocate or block.
pub trait ArmDriver {
    /// Static capabilities, read once at boot.
    fn manifest(&self) -> &ArmManifest;

    fn connect(&mut self) -> Result<(), ArmError>;
    fn disconnect(&mut self) -> Result<(), ArmError>;

    /// Fill `state_buffer` ([`ArmManifest::state_len`] slots): a [`CoreState`]
    /// followed by the aux tail.
    fn read_state(&mut self, state_buffer: &mut [f64]) -> Result<(), ArmError>;

    /// Execute the [`CoreCommand`] at the front of `cmd_buffer`.
    fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError>;

    /// The host-side model of this arm.
    fn arm(&self) -> &dyn ArmMath;
    fn arm_mut(&mut self) -> &mut dyn ArmMath;

    // --- configuration plane ----------------------------------------------

    /// Forward a payload change to the robot controller, if it has one.
    fn set_payload_native(&mut self, _payload: &RigidPayload) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    /// Forward a TCP change to the robot controller, if it has one.
    fn set_tcp_offset_native(&mut self, _pose: &[f64; 7]) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    /// Change the payload. The arguments are validated, offered to the
    /// controller, then applied to the host model; the result says which side
    /// compensates. A controller fault leaves the host model untouched.
    fn set_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<Applied, ArmError> {
        let payload = RigidPayload::from_contract(mass_kg, com, inertia)?;
        let applied = match self.set_payload_native(&payload) {
            Ok(()) => Applied::Native,
            Err(ArmError::NotSupported) => Applied::Host,
            Err(e) => return Err(e),
        };
        self.arm_mut().set_payload(mass_kg, com, inertia)?;
        Ok(applied)
    }

    /// Change the TCP; same protocol as [`set_payload`](Self::set_payload).
    fn set_tcp_offset(&mut self, pose: [f64; 7]) -> Result<Applied, ArmError> {
        pose_from_xyz_quat(&pose)?;
        let applied = match self.set_tcp_offset_native(&pose) {
            Ok(()) => Applied::Native,
            Err(ArmError::NotSupported) => Applied::Host,
            Err(e) => return Err(e),
        };
        self.arm_mut().set_tcp_offset(pose)?;
        Ok(applied)
    }

    fn clear_faults(&mut self) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    /// Escape hatch for vendor features outside the contract.
    fn handle_vendor_command(&mut self, _cmd_name: &str, _payload: &[u8]) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    // --- kinematics & dynamics --------------------------------------------

    fn forward_kinematics(&self, q: &[f64; MAX_DOF]) -> Option<[f64; 7]> {
        self.arm().forward_kinematics(q)
    }

    fn jacobian(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]> {
        self.arm().jacobian(q)
    }

    fn inverse_kinematics(
        &self,
        target_pose: &[f64; 7],
        q_seed: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        self.arm().inverse_kinematics(target_pose, q_seed)
    }

    fn dynamics(&self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF]) -> Option<DynamicsState> {
        self.arm().dynamics(q, qd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::chain::tests::{sample_q, ur5_calibrated, ur5_nominal};
    use core::f64::consts::PI;

    const TWO_PI: f64 = 2.0 * PI;

    /// Stand-in for a generated solver: one solution, wrapped to (-π, π].
    fn solve_from_seed(target: &Pose, q_seed: Option<&[f64; 6]>) -> [Option<[f64; 6]>; 1] {
        let mut q = q_seed.copied().unwrap_or([0.0; 6]);
        if ur5_nominal().refine_ik(target, &mut q) > 1e-7 {
            return [None];
        }
        [Some(q.map(wrap_to_pi))]
    }

    /// Kinematics-only UR5 built from the nominal DH table.
    struct Ur5;

    impl ArmModel<6, 1> for Ur5 {
        const NAME: &'static str = "ur5-test";
        const JOINT_LIMITS: [JointLimit; 6] = [
            Some((-PI, PI)),
            Some((-TWO_PI, TWO_PI)),
            Some((-TWO_PI, TWO_PI)),
            Some((-TWO_PI, TWO_PI)),
            Some((-TWO_PI, TWO_PI)),
            None,
        ];

        fn flange_fk(q: &[f64; 6]) -> Pose {
            ur5_nominal().fk(q)
        }

        fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
            ur5_nominal().jacobian(q)
        }

        fn flange_solve(
            target: &Pose,
            _max_solutions: Option<usize>,
            q_seed: Option<&[f64; 6]>,
            _respect_limits: bool,
            _allow_refinement: bool,
            _policy: TolerancePolicy,
            _refinement_max_iters: usize,
        ) -> [Option<[f64; 6]>; 1] {
            solve_from_seed(target, q_seed)
        }
    }

    /// Same kinematics with recognisable placeholder dynamics.
    struct Ur5Dyn;

    impl ArmModel<6, 1> for Ur5Dyn {
        const HAS_DYNAMICS: bool = true;

        fn flange_fk(q: &[f64; 6]) -> Pose {
            ur5_nominal().fk(q)
        }

        fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
            ur5_nominal().jacobian(q)
        }

        fn flange_solve(
            target: &Pose,
            _max_solutions: Option<usize>,
            q_seed: Option<&[f64; 6]>,
            _respect_limits: bool,
            _allow_refinement: bool,
            _policy: TolerancePolicy,
            _refinement_max_iters: usize,
        ) -> [Option<[f64; 6]>; 1] {
            solve_from_seed(target, q_seed)
        }

        fn flange_gravity(_q: &[f64; 6], gravity: &[f64; 3]) -> [f64; 6] {
            [gravity[2]; 6]
        }

        fn flange_mass_matrix(_q: &[f64; 6]) -> [[f64; 6]; 6] {
            let mut m = [[0.0; 6]; 6];
            for i in 0..6 {
                m[i][i] = 2.0;
            }
            m
        }

        fn flange_coriolis(_q: &[f64; 6], qd: &[f64; 6]) -> [f64; 6] {
            *qd
        }
    }

    fn tool() -> Pose {
        pose_from_xyz_quat(&[0.03, -0.02, 0.15, 0.0, 0.3, 0.0, 0.95]).unwrap()
    }

    fn tool_payload() -> RigidPayload {
        RigidPayload::new(2.5, [0.04, -0.03, 0.08]).with_inertia([
            [0.020, 0.002, 0.000],
            [0.002, 0.030, 0.001],
            [0.000, 0.001, 0.015],
        ])
    }

    fn pad(q: &[f64; 6]) -> [f64; MAX_DOF] {
        let mut out = [f64::NAN; MAX_DOF];
        out[..6].copy_from_slice(q);
        out
    }

    #[test]
    fn calibrated_ik_closes_on_the_calibrated_tcp() {
        let mut arm = Arm::<Ur5, 6, 1>::new();
        arm.set_tcp(&tool()).unwrap();
        let nominal_fk0 = arm.fk(&[0.2; 6]);
        arm.set_calibration(ur5_calibrated()).unwrap();
        assert!((arm.fk(&[0.2; 6]) - nominal_fk0).norm() > 1e-4);

        let mut uncalibrated = Arm::<Ur5, 6, 1>::new();
        uncalibrated.set_tcp(&tool()).unwrap();

        for seed in 0..20 {
            let q_true = sample_q::<6>(seed, 2.5);
            let target = arm.fk(&q_true);
            let solve = |a: &Arm<Ur5, 6, 1>| {
                a.ik(
                    &target,
                    None,
                    Some(&q_true),
                    false,
                    true,
                    TolerancePolicy::DEFAULT,
                    15,
                )[0]
            };
            let q = solve(&arm).expect("calibrated solution");
            assert!((arm.fk(&q) - target).norm() < 1e-9, "seed {seed}");
            // The nominal solver alone misses the real TCP by more than fk_atol.
            let q_nominal = solve(&uncalibrated).expect("nominal solution");
            assert!(
                (arm.fk(&q_nominal) - target).norm() > CALIBRATED_FK_ATOL,
                "seed {seed}"
            );
        }

        arm.clear_calibration();
        assert_eq!(arm.fk(&[0.2; 6]), nominal_fk0);
    }

    #[test]
    fn model_limit_gate_is_inclusive_and_skips_unlimited_joints() {
        let mut q = [PI, -1.0, 1.2, -0.7, 0.9, 100.0];
        assert!(within_model_limits::<Ur5, 6, 1>(&q));
        q[0] = PI + 1e-9;
        assert!(!within_model_limits::<Ur5, 6, 1>(&q));
    }

    #[test]
    fn calibration_in_the_wrong_frame_is_rejected() {
        let mut arm = Arm::<Ur5, 6, 1>::new();
        // UR's DH base is rotated by π about z relative to the URDF base_link.
        let flipped = pose_from_xyz_quat(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let chain = ur5_calibrated().with_base(&flipped).unwrap();
        assert_eq!(
            arm.set_calibration(chain),
            Err(ModelError::ExcessiveDeviation)
        );
        assert!(arm.calibration().is_none());
    }

    #[test]
    fn tcp_jacobian_matches_finite_difference_of_tcp_fk() {
        let mut arm = Arm::<Ur5, 6, 1>::new();
        arm.set_tcp(&tool()).unwrap();
        arm.set_calibration(ur5_calibrated()).unwrap();
        let q = sample_q::<6>(7, 2.0);
        let jac = arm.jacobian(&q);
        let h = 1e-6;
        for c in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[c] += h;
            qm[c] -= h;
            let d = (arm.fk(&qp) - arm.fk(&qm)) / (2.0 * h);
            for r in 0..3 {
                assert!((d[(r, 3)] - jac[r][c]).abs() < 1e-7, "row {r} col {c}");
            }
        }
        let jinv = arm.inverse_jacobian(&q).unwrap();
        for r in 0..6 {
            for c in 0..6 {
                let mut s = 0.0;
                for k in 0..6 {
                    s += jac[r][k] * jinv[k][c];
                }
                assert!((s - if r == c { 1.0 } else { 0.0 }).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn payload_gravity_is_the_potential_gradient() {
        let mut arm = Arm::<Ur5, 6, 1>::new();
        arm.set_payload(tool_payload()).unwrap();
        arm.set_gravity([1.0, -2.0, -9.0]).unwrap();
        let potential = |q: &[f64; 6]| {
            let t = arm.flange_fk(q);
            let p = arm.payload();
            let mut u = 0.0;
            for r in 0..3 {
                let com =
                    t[(r, 3)] + t[(r, 0)] * p.com[0] + t[(r, 1)] * p.com[1] + t[(r, 2)] * p.com[2];
                u -= p.mass * arm.gravity()[r] * com;
            }
            u
        };
        let q = sample_q::<6>(11, 2.0);
        let tau = arm.payload_gravity(&q);
        let h = 1e-6;
        for j in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[j] += h;
            qm[j] -= h;
            let grad = (potential(&qp) - potential(&qm)) / (2.0 * h);
            assert!(
                (tau[j] - grad).abs() < 1e-7,
                "joint {j}: {} vs {grad}",
                tau[j]
            );
        }
    }

    /// `C(q, qd) qd = Ṁ qd − ½ ∂(qdᵀ M qd)/∂q`, for a payload with an offset
    /// centre of mass and a full inertia tensor.
    #[test]
    fn payload_coriolis_is_consistent_with_its_mass_matrix() {
        let mut arm = Arm::<Ur5, 6, 1>::new();
        arm.set_payload(tool_payload()).unwrap();
        let q = sample_q::<6>(5, 2.0);
        let qd = sample_q::<6>(6, 1.0);
        let m_qd = |q: &[f64; 6]| {
            let m = arm.payload_mass_matrix(q);
            let mut out = [0.0; 6];
            for i in 0..6 {
                for j in 0..6 {
                    out[i] += m[i][j] * qd[j];
                }
            }
            out
        };
        let kinetic = |q: &[f64; 6]| {
            let v = m_qd(q);
            (0..6).map(|i| 0.5 * qd[i] * v[i]).sum::<f64>()
        };
        let h = 1e-5;
        let (mut qp, mut qm) = (q, q);
        for j in 0..6 {
            qp[j] += h * qd[j];
            qm[j] -= h * qd[j];
        }
        let (mp, mm) = (m_qd(&qp), m_qd(&qm));
        let tau = arm.payload_coriolis(&q, &qd);
        let m = arm.payload_mass_matrix(&q);
        for i in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[i] += h;
            qm[i] -= h;
            let want = (mp[i] - mm[i]) / (2.0 * h) - (kinetic(&qp) - kinetic(&qm)) / (2.0 * h);
            assert!(
                (tau[i] - want).abs() < 1e-5,
                "joint {i}: {} vs {want}",
                tau[i]
            );
            for j in 0..6 {
                assert!((m[i][j] - m[j][i]).abs() < 1e-12);
            }
        }
        assert!(tau.iter().any(|t| t.abs() > 1e-2));
    }

    #[test]
    fn dynamics_requires_a_model_and_adds_the_payload() {
        let q = sample_q::<6>(2, 2.0);
        let qd = sample_q::<6>(3, 1.0);
        assert!(Arm::<Ur5, 6, 1>::new().dynamics(&q, &qd).is_none());

        let mut arm = Arm::<Ur5Dyn, 6, 1>::new();
        let bare = arm.dynamics(&q, &qd).unwrap();
        assert_eq!(bare.gravity, [-STANDARD_GRAVITY; 6]);
        assert_eq!(bare.coriolis, qd);

        arm.set_payload(tool_payload()).unwrap();
        let loaded = arm.dynamics(&q, &qd).unwrap();
        let (m, c, g) = (
            arm.payload_mass_matrix(&q),
            arm.payload_coriolis(&q, &qd),
            arm.payload_gravity(&q),
        );
        for i in 0..6 {
            assert_eq!(loaded.gravity[i], bare.gravity[i] + g[i]);
            assert_eq!(loaded.coriolis[i], bare.coriolis[i] + c[i]);
            for j in 0..6 {
                assert_eq!(loaded.mass_matrix[i][j], bare.mass_matrix[i][j] + m[i][j]);
            }
        }
        assert!(g.iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn setters_reject_bad_input_and_keep_state() {
        let mut arm = Arm::<Ur5, 6, 1>::new();
        arm.set_payload(Payload::new(1.0, [0.0; 3])).unwrap();
        assert_eq!(
            arm.set_payload(Payload::new(-1.0, [0.0; 3])),
            Err(ModelError::InvalidPayload)
        );
        assert_eq!(
            arm.set_payload(Payload::new(f64::NAN, [0.0; 3])),
            Err(ModelError::NonFinite)
        );
        assert_eq!(arm.payload().mass, 1.0);
        let mut scaled = Pose::identity();
        scaled[(0, 0)] = 2.0;
        assert_eq!(arm.set_tcp(&scaled), Err(ModelError::NotRigid));
        assert_eq!(arm.tcp(), &Pose::identity());
        assert_eq!(
            arm.set_gravity([0.0, f64::INFINITY, 0.0]),
            Err(ModelError::NonFinite)
        );
    }

    #[test]
    fn ik_nearest_follows_the_seed_branch_within_limits() {
        let arm = Arm::<Ur5, 6, 1>::new();
        let q_true = [0.5, -1.0, 1.2, -0.7, 0.9, -1.0];
        let target = arm.fk(&q_true);
        let mut seed = q_true;
        seed[0] += TWO_PI; // beyond joint 0's (-π, π) limit
        seed[4] -= TWO_PI; // inside joint 4's (-2π, 2π) limit
        seed[5] += 3.0 * TWO_PI; // joint 5 is unlimited
        let q = arm
            .ik_nearest(&target, &seed, TolerancePolicy::DEFAULT)
            .unwrap();
        let want = [q_true[0], q_true[1], q_true[2], q_true[3], seed[4], seed[5]];
        for i in 0..6 {
            assert!(
                (q[i] - want[i]).abs() < 1e-6,
                "joint {i}: {} vs {}",
                q[i],
                want[i]
            );
        }

        let mut far = target;
        far[(0, 3)] += 5.0;
        assert!(arm
            .ik_nearest(&far, &q_true, TolerancePolicy::DEFAULT)
            .is_none());
    }

    #[test]
    fn erased_math_round_trips_and_pads() {
        let mut arm = Arm::<Ur5Dyn, 6, 1>::new();
        arm.set_calibration(ur5_calibrated()).unwrap();
        let math: &mut dyn ArmMath = &mut arm;
        math.set_tcp_offset([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 2.0])
            .unwrap();
        math.set_payload(1.5, [0.0, 0.0, 0.05], [0.0; 9]).unwrap();
        assert_eq!((math.dof(), math.is_calibrated()), (6, true));

        let q = pad(&sample_q::<6>(9, 2.0));
        let pose = math.forward_kinematics(&q).unwrap();
        let back = math.inverse_kinematics(&pose, &q).unwrap();
        for i in 0..6 {
            assert!((back[i] - q[i]).abs() < 1e-7, "joint {i}");
        }
        assert_eq!(&back[6..], &[0.0, 0.0]);

        let jac = math.jacobian(&q).unwrap();
        assert!(jac.iter().all(|row| row[6] == 0.0 && row[7] == 0.0));
        let dynamics = math.dynamics(&q, &q).unwrap();
        assert_eq!(dynamics.m_matrix[7], [0.0; MAX_DOF]);
        assert!(dynamics.m_matrix[0][0] > 2.0);

        let mut bad = q;
        bad[3] = f64::NAN;
        assert!(math.forward_kinematics(&bad).is_none());
        assert_eq!(
            math.set_tcp_offset([0.0; 7]),
            Err(ModelError::DegenerateAxis)
        );
    }

    #[test]
    fn wire_layouts_round_trip() {
        assert_eq!(CoreState::LEN, 52);
        assert_eq!(CoreCommand::LEN, 89);

        let mut buf = [0.0f64; 96];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = i as f64;
        }
        let state = CoreState::read_from(&buf).unwrap();
        assert_eq!(
            (state.q[0], state.tcp_pose[0], state.in_contact),
            (0.0, 16.0, 43.0)
        );
        assert_eq!(
            (state.io_analog_in, state.gripper_has_object),
            ([45.0, 46.0, 47.0, 48.0], 51.0)
        );
        let mut out = [-1.0f64; 96];
        state.write_to(&mut out).unwrap();
        assert_eq!(out[..CoreState::LEN], buf[..CoreState::LEN]);
        assert_eq!(out[CoreState::LEN], -1.0);

        let cmd = CoreCommand::read_from(&buf).unwrap();
        assert_eq!(
            (
                cmd.target_pose[0],
                cmd.reflex_threshold[0],
                cmd.gripper_max_force
            ),
            (24.0, 77.0, 88.0)
        );
        let mut out = [-1.0f64; 96];
        cmd.write_to(&mut out).unwrap();
        assert_eq!(out[..CoreCommand::LEN], buf[..CoreCommand::LEN]);

        assert_eq!(
            CoreState::read_from(&buf[..51]),
            Err(ArmError::BufferTooSmall)
        );
        assert_eq!(
            CoreCommand::IDLE.write_to(&mut out[..88]),
            Err(ArmError::BufferTooSmall)
        );
    }

    #[test]
    fn manifest_reflects_the_model() {
        static AUX: [AuxField; 2] = [
            AuxField {
                label: "motor_temp",
                size: 6,
            },
            AuxField {
                label: "bus_voltage",
                size: 1,
            },
        ];
        let mut manifest = ArmManifest::for_model::<Ur5, 6, 1>(
            500,
            control_mode::JOINT_POS | control_mode::JOINT_VEL,
        );
        manifest.aux = &AUX;
        assert_eq!((manifest.name, manifest.active_dof_count), ("ur5-test", 6));
        assert_eq!(
            (manifest.joint_limits_min[0], manifest.joint_limits_max[0]),
            (-PI, PI)
        );
        assert_eq!(manifest.joint_limits_max[5], f64::INFINITY);
        assert_eq!(manifest.joint_limits_min[7], f64::NEG_INFINITY);
        assert!(!manifest.model.dynamics);
        assert!(manifest.supports(control_mode::JOINT_POS));
        assert!(!manifest.supports(control_mode::JOINT_POS | control_mode::JOINT_TORQUE));
        assert_eq!(manifest.state_len(), CoreState::LEN + 7);
        assert!(
            ArmManifest::for_model::<Ur5Dyn, 6, 1>(500, 0)
                .model
                .dynamics
        );
    }

    struct FakeDriver {
        manifest: ArmManifest,
        arm: Arm<Ur5, 6, 1>,
        native: Result<(), ArmError>,
        native_calls: usize,
    }

    impl ArmDriver for FakeDriver {
        fn manifest(&self) -> &ArmManifest {
            &self.manifest
        }
        fn connect(&mut self) -> Result<(), ArmError> {
            Ok(())
        }
        fn disconnect(&mut self) -> Result<(), ArmError> {
            Ok(())
        }
        fn read_state(&mut self, state_buffer: &mut [f64]) -> Result<(), ArmError> {
            CoreState::UNKNOWN.write_to(state_buffer)
        }
        fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError> {
            CoreCommand::read_from(cmd_buffer).map(|_| ())
        }
        fn arm(&self) -> &dyn ArmMath {
            &self.arm
        }
        fn arm_mut(&mut self) -> &mut dyn ArmMath {
            &mut self.arm
        }
        fn set_payload_native(&mut self, _payload: &RigidPayload) -> Result<(), ArmError> {
            self.native_calls += 1;
            self.native
        }
    }

    #[test]
    fn driver_configuration_falls_back_to_the_host_model() {
        let mut driver = FakeDriver {
            manifest: ArmManifest::for_model::<Ur5, 6, 1>(500, control_mode::JOINT_POS),
            arm: Arm::new(),
            native: Err(ArmError::NotSupported),
            native_calls: 0,
        };
        let q = pad(&[0.3; 6]);
        let before = driver.forward_kinematics(&q).unwrap();

        // Not native: the host model takes over.
        assert_eq!(
            driver.set_payload(2.0, [0.0, 0.0, 0.1], [0.0; 9]),
            Ok(Applied::Host)
        );
        assert_eq!(driver.arm.payload().mass, 2.0);
        assert_eq!(
            driver.set_tcp_offset([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
            Ok(Applied::Host)
        );
        let after = driver.forward_kinematics(&q).unwrap();
        let moved: f64 = (0..3).map(|i| (after[i] - before[i]).powi(2)).sum();
        assert!((moved.sqrt() - 0.2).abs() < 1e-12);

        // Native: both sides updated.
        driver.native = Ok(());
        assert_eq!(
            driver.set_payload(3.0, [0.0; 3], [0.0; 9]),
            Ok(Applied::Native)
        );
        assert_eq!(driver.arm.payload().mass, 3.0);

        // A controller fault must not desynchronise the host model.
        driver.native = Err(ArmError::Fault("protective stop"));
        assert_eq!(
            driver.set_payload(4.0, [0.0; 3], [0.0; 9]),
            Err(ArmError::Fault("protective stop"))
        );
        assert_eq!(driver.arm.payload().mass, 3.0);

        // Invalid input never reaches the controller.
        let calls = driver.native_calls;
        assert_eq!(
            driver.set_payload(-1.0, [0.0; 3], [0.0; 9]),
            Err(ArmError::Model(ModelError::InvalidPayload))
        );
        assert_eq!(driver.native_calls, calls);
        assert_eq!(driver.clear_faults(), Err(ArmError::NotSupported));

        let object: &mut dyn ArmDriver = &mut driver;
        let mut state = [0.0; CoreState::LEN];
        object.read_state(&mut state).unwrap();
        assert!(state[0].is_nan());
    }
}
