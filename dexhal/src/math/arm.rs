//! One physical arm: a driver's [`ArmModel`] plus its runtime state.
//!
//! The driver supplies the nominal flange kinematics, its inverse kinematics
//! and its dynamics. [`Arm`] holds what differs per installation and changes
//! at runtime — TCP, payload, gravity direction, per-robot calibration — and
//! applies it around the driver's functions with the arm-agnostic math in
//! [`spatial`](crate::math::spatial), [`payload`](crate::math::payload) and
//! [`chain`](crate::math::chain). It contains no solver and no arm-specific
//! code.
//!
//! Calibration ([`Arm::set_calibration`]) swaps the flange FK and Jacobian for
//! a runtime [`KinematicChain`] (a fixed `DOF` loop) and polishes the driver's
//! IK solutions onto it with at most
//! [`CHAIN_IK_MAX_ITERS`](crate::math::chain::CHAIN_IK_MAX_ITERS) Newton steps each,
//! so every query stays allocation-free and bounded, and an uncalibrated arm
//! pays nothing.

use core::f64::consts::FRAC_PI_2;
use core::marker::PhantomData;

use crate::math::chain::{
    pose_error, transform_from_pose, transform_of, transform_to_pose, KinematicChain, ModelError,
};
use crate::math::kinematics::{residual_columns, se3_inv};
// Re-exported: a driver's `arm.rs` names its model through this module.
pub use crate::math::model::{ArmModel, Pose};
use crate::math::payload::{
    payload_coriolis_torque, payload_gravity_torque, payload_inertia_matrix,
    payload_inverse_dynamics, Payload,
};
use crate::math::contract::WrenchFrame;
use crate::math::postprocess::{q_close, within_limits};
use crate::math::spatial::{
    jacobian_in_frame, pose_rotation, pseudo_inverse, tcp_jacobian, wrench_tool_to_base,
};
use crate::math::ssik_core::{add3, cross3, mat3_vec, norm3, wrap_to_pi};
use crate::math::tolerances::TolerancePolicy;
use crate::math::torque_observer::{LstmObserver, ObserverMode};

/// Largest per-component difference (m/s²) between the runtime gravity and
/// the gravity a [`ObserverMode::FullDynamics`] observer was trained under.
pub const OBSERVER_GRAVITY_TOLERANCE: f64 = 1e-3;

/// Standard gravity (m/s²) used for the default gravity vector `[0, 0, -g]`.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// Largest Frobenius deviation `‖fk_calibrated(q) − fk_nominal(q)‖` accepted by
/// [`Arm::set_calibration`]. Real calibrations are well under a centimetre; a
/// larger value means the chain is expressed in different base/flange frames
/// than the driver, which would break the nominal-seeded IK.
pub const CALIBRATION_MAX_DEVIATION: f64 = 0.05;

/// Number of probe configurations checked by [`Arm::set_calibration`].
pub const CALIBRATION_PROBES: usize = 6;

/// Refinement ceiling handed to the driver's solver by [`Arm::ik_nearest`].
pub const IK_NEAREST_REFINEMENT_ITERS: usize = 15;

/// Default largest per-joint distance (rad) between the seed and the solution
/// returned by [`Arm::ik_nearest`]. See [`Arm::set_ik_max_joint_step`].
pub const IK_DEFAULT_MAX_JOINT_STEP: f64 = FRAC_PI_2;

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

/// Distance between two poses of the same frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseResidual {
    /// Distance between the origins (m).
    pub position: f64,
    /// Angle of the relative rotation (rad).
    pub rotation: f64,
}

/// One physical arm: the driver's model `A` plus the state that differs per
/// installation and changes at runtime.
///
/// All setters validate their input and leave the arm unchanged on error, so
/// they are safe to call between control cycles. Every query is
/// allocation-free and bounded.
///
/// ```
/// # use dexhal::math::arm::Arm;
/// # use dexhal::math::model::{ArmModel, Pose};
/// # use dexhal::math::payload::Payload;
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
    payload: Payload,
    gravity: [f64; 3],
    calibration: Option<KinematicChain<DOF>>,
    ik_max_joint_step: f64,
    observer: Option<LstmObserver<DOF>>,
    _model: PhantomData<fn() -> A>,
}

fn gravity_matches(a: &[f64; 3], b: &[f64; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= OBSERVER_GRAVITY_TOLERANCE)
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> Default for Arm<A, DOF, MAX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> Arm<A, DOF, MAX> {
    /// Nominal arm: identity TCP, no payload, gravity `[0, 0, -g]`, no
    /// calibration, [`IK_DEFAULT_MAX_JOINT_STEP`].
    pub fn new() -> Self {
        Arm {
            tcp: Pose::identity(),
            tcp_inv: Pose::identity(),
            payload: Payload::NONE,
            gravity: [0.0, 0.0, -STANDARD_GRAVITY],
            calibration: None,
            ik_max_joint_step: IK_DEFAULT_MAX_JOINT_STEP,
            observer: None,
            _model: PhantomData,
        }
    }

    // --- configuration -----------------------------------------------------

    /// Set the flange → TCP transform.
    pub fn set_tcp(&mut self, tcp: &Pose) -> Result<(), ModelError> {
        let rigid = transform_from_pose(tcp)?;
        self.tcp = *tcp;
        self.tcp_inv = transform_to_pose(&se3_inv(&rigid));
        Ok(())
    }

    pub fn tcp(&self) -> &Pose {
        &self.tcp
    }

    /// Set the grasped payload ([`Payload::NONE`] to release).
    pub fn set_payload(&mut self, payload: Payload) -> Result<(), ModelError> {
        payload.validate()?;
        self.payload = payload;
        Ok(())
    }

    pub fn payload(&self) -> &Payload {
        &self.payload
    }

    /// Set the gravity vector in the arm's base frame (wall / ceiling mounts).
    ///
    /// Rejected with [`ModelError::InvalidObserver`] while a
    /// [`ObserverMode::FullDynamics`] torque observer trained under a
    /// different gravity is installed: its output would silently be wrong.
    pub fn set_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError> {
        if !gravity.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFinite);
        }
        if let Some(obs) = &self.observer {
            if obs.mode() == ObserverMode::FullDynamics
                && !gravity_matches(&gravity, obs.trained_gravity())
            {
                return Err(ModelError::InvalidObserver);
            }
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

    /// Return to the driver's nominal kinematics.
    pub fn clear_calibration(&mut self) {
        self.calibration = None;
    }

    pub fn calibration(&self) -> Option<&KinematicChain<DOF>> {
        self.calibration.as_ref()
    }

    /// Set the continuity bound of [`ik_nearest`](Self::ik_nearest): the
    /// largest distance (rad) any joint may lie from the seed. `f64::INFINITY`
    /// disables the check.
    pub fn set_ik_max_joint_step(&mut self, step: f64) -> Result<(), ModelError> {
        if step.is_nan() {
            return Err(ModelError::NonFinite);
        }
        if step <= 0.0 {
            return Err(ModelError::NonPositive);
        }
        self.ik_max_joint_step = step;
        Ok(())
    }

    pub fn ik_max_joint_step(&self) -> f64 {
        self.ik_max_joint_step
    }

    /// Install a learned torque observer from a `.dexnext` file (FACTR 2).
    ///
    /// The bytes are validated ([`LstmObserver::from_bytes`]) and decoded in
    /// place every tick, so they must outlive the arm: embed them with
    /// `include_bytes!` or leak a buffer read at start-up. Rejected with
    /// [`ModelError::InvalidObserver`] if the file is malformed or built for
    /// another `DOF`, if it is a [`ObserverMode::FrictionResidual`] model and
    /// the driver has no dynamics, or if it is a
    /// [`ObserverMode::FullDynamics`] model trained under a gravity other
    /// than the configured one. The arm is unchanged on error.
    pub fn set_torque_observer(&mut self, bytes: &'static [u8]) -> Result<(), ModelError> {
        let obs = LstmObserver::<DOF>::from_bytes(bytes)?;
        match obs.mode() {
            ObserverMode::FrictionResidual if !A::HAS_DYNAMICS => {
                return Err(ModelError::InvalidObserver)
            }
            ObserverMode::FullDynamics if !gravity_matches(&self.gravity, obs.trained_gravity()) => {
                return Err(ModelError::InvalidObserver)
            }
            _ => {}
        }
        self.observer = Some(obs);
        Ok(())
    }

    /// Remove the torque observer; [`step_free_torque`](Self::step_free_torque)
    /// falls back to the rigid-body model.
    pub fn clear_torque_observer(&mut self) {
        self.observer = None;
    }

    /// Clear the observer's recurrent state and window, e.g. after the arm
    /// was disabled. No-op without an observer.
    pub fn reset_torque_observer(&mut self) {
        if let Some(obs) = &mut self.observer {
            obs.reset();
        }
    }

    pub fn torque_observer(&self) -> Option<&LstmObserver<DOF>> {
        self.observer.as_ref()
    }

    pub fn has_torque_observer(&self) -> bool {
        self.observer.is_some()
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

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) at the TCP origin, base axes.
    pub fn jacobian(&self, q: &[f64; DOF]) -> [[f64; DOF]; 6] {
        let (pose, jac) = self.flange_fk_jacobian(q);
        tcp_jacobian(&pose, &jac, &self.tcp)
    }

    /// Jacobian of [`pose_residual12`](crate::math::kinematics::pose_residual12)
    /// for the TCP pose, as columns (see [`residual_columns`]).
    pub fn residual_jacobian(&self, q: &[f64; DOF]) -> [[f64; 12]; DOF] {
        let (pose, jac) = self.flange_fk_jacobian(q);
        let tcp_pose = pose * self.tcp;
        let rot: [[f64; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|j| tcp_pose[(i, j)]));
        residual_columns(&rot, &tcp_jacobian(&pose, &jac, &self.tcp))
    }

    /// Geometric Jacobian at the TCP origin expressed in TCP axes: maps joint
    /// rates to the tool-frame twist.
    pub fn jacobian_tool(&self, q: &[f64; DOF]) -> [[f64; DOF]; 6] {
        let (pose, jac) = self.flange_fk_jacobian(q);
        jacobian_in_frame(&(pose * self.tcp), &tcp_jacobian(&pose, &jac, &self.tcp))
    }

    /// Minimum-norm pseudo-inverse of the TCP Jacobian; `None` at a singularity.
    pub fn inverse_jacobian(&self, q: &[f64; DOF]) -> Option<[[f64; 6]; DOF]> {
        pseudo_inverse(&self.jacobian(q))
    }

    /// Distance between the host model's `fk(q)` and a TCP pose reported by
    /// the hardware for the same `q`.
    ///
    /// A residual well above the robot's repeatability means the host model
    /// and the controller disagree: a missing or wrong calibration, or a TCP
    /// that differs between the two.
    pub fn fk_residual(
        &self,
        q: &[f64; DOF],
        reported_tcp: &Pose,
    ) -> Result<PoseResidual, ModelError> {
        let reported = transform_from_pose(reported_tcp)?;
        let e = pose_error(&reported, &transform_of(&self.fk(q)));
        Ok(PoseResidual {
            position: norm3(&[e[0], e[1], e[2]]),
            rotation: norm3(&[e[3], e[4], e[5]]),
        })
    }

    /// All inverse-kinematics solutions placing the TCP at `tcp_target`.
    ///
    /// The arguments and the solve are the driver's
    /// ([`ArmModel::flange_solve`]); this method retargets the TCP to the
    /// flange. With a calibration installed the driver's full nominal set is
    /// polished onto the calibrated chain ([`KinematicChain::refine_ik`]),
    /// re-verified against `policy.subproblem_numerical`, re-checked against
    /// [`ArmModel::JOINT_LIMITS`] if `respect_limits`, de-duplicated with
    /// `policy.subproblem_dedup` and only then truncated to `max_solutions`.
    /// The driver's ordering is kept. A target reachable only by the
    /// calibrated geometry (within a millimetre of the workspace boundary)
    /// has no nominal solution to start from and yields none.
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
        let chain = match &self.calibration {
            Some(chain) => chain,
            None => {
                return A::flange_solve(
                    &flange_target,
                    max_solutions,
                    q_seed,
                    respect_limits,
                    allow_refinement,
                    policy,
                    refinement_max_iters,
                )
            }
        };
        let nominal = A::flange_solve(
            &flange_target,
            None,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        );
        let cap = max_solutions.map_or(MAX, |m| m.min(MAX));
        let mut out = [None; MAX];
        let mut n = 0;
        for seed in nominal.iter().flatten() {
            if n == cap {
                break;
            }
            let mut q = *seed;
            if chain.refine_ik(&flange_target, &mut q) > policy.subproblem_numerical {
                continue;
            }
            if respect_limits && !within_limits(&q, &A::JOINT_LIMITS) {
                continue;
            }
            let duplicate = out[..n]
                .iter()
                .flatten()
                .any(|kept| q_close(&q, kept, policy.subproblem_dedup));
            if !duplicate {
                out[n] = Some(q);
                n += 1;
            }
        }
        out
    }

    /// The IK solution on the seed's branch, for servoing.
    ///
    /// The driver picks the branch: its solve is asked for the single solution
    /// nearest `q_seed` inside the joint limits. With a calibration installed
    /// only that solution is polished onto the calibrated chain. Each joint is
    /// then moved to the 2π turn nearest the seed when
    /// [`ArmModel::JOINT_LIMITS`] allow, so a continuous Cartesian path maps
    /// to a continuous joint path.
    ///
    /// Returns `None` if the target is unreachable, or if any joint of the
    /// result lies further than [`ik_max_joint_step`](Self::ik_max_joint_step)
    /// from the seed: near a singularity or a joint limit the nearest
    /// remaining solution is on another branch, and following it would
    /// command a jump.
    pub fn ik_nearest(
        &self,
        tcp_target: &Pose,
        q_seed: &[f64; DOF],
        policy: TolerancePolicy,
    ) -> Option<[f64; DOF]> {
        let flange_target = tcp_target * self.tcp_inv;
        let nominal = A::flange_solve(
            &flange_target,
            Some(1),
            Some(q_seed),
            true,
            true,
            policy,
            IK_NEAREST_REFINEMENT_ITERS,
        );
        let mut q = nominal.into_iter().flatten().next()?;
        if let Some(chain) = &self.calibration {
            if chain.refine_ik(&flange_target, &mut q) > policy.subproblem_numerical
                || !within_limits(&q, &A::JOINT_LIMITS)
            {
                return None;
            }
        }
        for i in 0..DOF {
            let near = q_seed[i] + wrap_to_pi(q[i] - q_seed[i]);
            if A::JOINT_LIMITS[i].is_none_or(|(lo, hi)| near >= lo && near <= hi) {
                q[i] = near;
            }
            let step = (q[i] - q_seed[i]).abs();
            if step.is_nan() || step > self.ik_max_joint_step {
                return None;
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
        let (pose, jac) = self.flange_fk_jacobian(q);
        payload_coriolis_torque(&pose, &jac, qd, &self.payload)
    }

    /// `M(q)`, `C(q, qd) qd` and `G(q)`: the driver's terms for the arm's own
    /// links plus the payload's.
    ///
    /// `None` if the driver has no dynamics ([`ArmModel::HAS_DYNAMICS`]). The
    /// driver's terms use the nominal link geometry: kinematic calibration
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
        let c = payload_coriolis_torque(&pose, &jac, qd, &self.payload);
        let g = payload_gravity_torque(&pose, &jac, &self.payload, &self.gravity);
        for i in 0..DOF {
            out.gravity[i] += g[i];
            out.coriolis[i] += c[i];
            for j in 0..DOF {
                out.mass_matrix[i][j] += m[i][j];
            }
        }
        Some(out)
    }

    /// Joint torque for the motion `(q, qd, qdd)`: the driver's inverse
    /// dynamics ([`ArmModel::flange_inverse_dynamics`]) plus the payload's.
    ///
    /// `None` if the driver has no dynamics.
    pub fn inverse_dynamics(
        &self,
        q: &[f64; DOF],
        qd: &[f64; DOF],
        qdd: &[f64; DOF],
    ) -> Option<[f64; DOF]> {
        if !A::HAS_DYNAMICS {
            return None;
        }
        let mut tau = A::flange_inverse_dynamics(q, qd, qdd, &self.gravity);
        if !self.payload.is_none() {
            let (pose, jac) = self.flange_fk_jacobian(q);
            let extra =
                payload_inverse_dynamics(&pose, &jac, qd, qdd, &self.payload, &self.gravity);
            for i in 0..DOF {
                tau[i] += extra[i];
            }
        }
        Some(tau)
    }

    /// Inverse dynamics of the configured payload alone for `(q, qd, qdd)`
    /// under the configured gravity: what a host adds to a bare-arm torque.
    pub fn payload_torque(
        &self,
        q: &[f64; DOF],
        qd: &[f64; DOF],
        qdd: &[f64; DOF],
    ) -> [f64; DOF] {
        if self.payload.is_none() {
            return [0.0; DOF];
        }
        let (pose, jac) = self.flange_fk_jacobian(q);
        payload_inverse_dynamics(&pose, &jac, qd, qdd, &self.payload, &self.gravity)
    }

    /// Free-space joint torque `τ_free` for this control tick: the torque the
    /// arm needs for `(q, qd, qdd)` without contact, so
    /// `τ_measured − τ_free` is the external torque.
    ///
    /// With a torque observer installed it is advanced one tick (at
    /// `control_freq_hz`; it re-evaluates every
    /// [`LstmObserver::stride_ticks`] ticks and holds its output between)
    /// and its prediction is used: plus the driver's rigid-body inverse
    /// dynamics for a [`ObserverMode::FrictionResidual`] model, plus the
    /// difference between the inverse dynamics of the runtime payload and of
    /// the payload the observer was trained with. `q_cmd` feeds the
    /// command-error input feature (zero if `None`); `qdd` is the reference
    /// acceleration. Without an observer this is
    /// [`inverse_dynamics`](Self::inverse_dynamics).
    ///
    /// `None` without an observer if the driver has no dynamics, if any
    /// input is not finite (the observer is not advanced), or while the
    /// observer warms up after a reset (at most `stride − 1` ticks; see
    /// [`LstmObserver::step`]).
    pub fn step_free_torque(
        &mut self,
        q: &[f64; DOF],
        qd: &[f64; DOF],
        qdd: &[f64; DOF],
        q_cmd: Option<&[f64; DOF]>,
        control_freq_hz: u32,
    ) -> Option<[f64; DOF]> {
        let finite = |v: &[f64; DOF]| v.iter().all(|x| x.is_finite());
        if !finite(q) || !finite(qd) || !finite(qdd) || !q_cmd.is_none_or(finite) {
            return None;
        }
        let Some(obs) = self.observer.as_mut() else {
            return self.inverse_dynamics(q, qd, qdd);
        };
        let mut u = [*q, *qd, [0.0; DOF]];
        if let Some(cmd) = q_cmd {
            for i in 0..DOF {
                u[2][i] = cmd[i] - q[i];
            }
        }
        let stride = LstmObserver::<DOF>::stride_ticks(control_freq_hz);
        let mut tau = obs.step(&u, stride)?;
        let (mode, trained) = (obs.mode(), *obs.trained_payload());
        if trained != self.payload {
            let (pose, jac) = self.flange_fk_jacobian(q);
            let runtime =
                payload_inverse_dynamics(&pose, &jac, qd, qdd, &self.payload, &self.gravity);
            let train = payload_inverse_dynamics(&pose, &jac, qd, qdd, &trained, &self.gravity);
            for i in 0..DOF {
                tau[i] += runtime[i] - train[i];
            }
        }
        if mode == ObserverMode::FrictionResidual {
            let rigid = A::flange_inverse_dynamics(q, qd, qdd, &self.gravity);
            for i in 0..DOF {
                tau[i] += rigid[i];
            }
        }
        Some(tau)
    }

    // --- sensing -----------------------------------------------------------

    /// The external wrench on the tool at the TCP origin in base axes, from a
    /// [`CoreState::wrench`](crate::math::contract::CoreState::wrench)
    /// reported in `frame` at configuration `q`.
    ///
    /// Unless `gravity_compensated`, the payload's weight under the configured
    /// gravity is removed, so a still arm holding its declared payload in free
    /// space reads zero.
    pub fn external_wrench(
        &self,
        q: &[f64; DOF],
        reported: &[f64; 6],
        frame: WrenchFrame,
        gravity_compensated: bool,
    ) -> [f64; 6] {
        let flange = self.flange_fk(q);
        let tcp = flange * self.tcp;
        let p_tcp = [tcp[(0, 3)], tcp[(1, 3)], tcp[(2, 3)]];
        let mut w = match frame {
            WrenchFrame::Base => *reported,
            WrenchFrame::Tcp => wrench_tool_to_base(&tcp, reported),
            WrenchFrame::Flange => {
                let at_flange = wrench_tool_to_base(&flange, reported);
                let f = [at_flange[0], at_flange[1], at_flange[2]];
                let lever = [
                    flange[(0, 3)] - p_tcp[0],
                    flange[(1, 3)] - p_tcp[1],
                    flange[(2, 3)] - p_tcp[2],
                ];
                let m = add3(&[at_flange[3], at_flange[4], at_flange[5]], &cross3(&lever, &f));
                [f[0], f[1], f[2], m[0], m[1], m[2]]
            }
        };
        if !gravity_compensated && self.payload.mass != 0.0 {
            let r = pose_rotation(&flange);
            let com = add3(
                &[flange[(0, 3)], flange[(1, 3)], flange[(2, 3)]],
                &mat3_vec(&r, &self.payload.com),
            );
            let weight = self.gravity.map(|g| self.payload.mass * g);
            let lever = [com[0] - p_tcp[0], com[1] - p_tcp[1], com[2] - p_tcp[2]];
            let moment = cross3(&lever, &weight);
            for k in 0..3 {
                w[k] -= weight[k];
                w[3 + k] -= moment[k];
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::chain::pose_from_xyz_quat;
    use crate::math::testing::{calibrated_chain, sample_q, SixR};
    use core::f64::consts::PI;

    const TWO_PI: f64 = 2.0 * PI;

    fn tool() -> Pose {
        pose_from_xyz_quat(&[0.03, -0.02, 0.15, 0.0, 0.3, 0.0, 0.95]).unwrap()
    }

    fn tool_payload() -> Payload {
        Payload::new(2.5, [0.04, -0.03, 0.08]).with_inertia([
            [0.020, 0.002, 0.000],
            [0.002, 0.030, 0.001],
            [0.000, 0.001, 0.015],
        ])
    }

    #[test]
    fn calibrated_ik_closes_on_the_calibrated_tcp() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        arm.set_tcp(&tool()).unwrap();
        let nominal_fk0 = arm.fk(&[0.2; 6]);
        arm.set_calibration(calibrated_chain()).unwrap();
        assert!((arm.fk(&[0.2; 6]) - nominal_fk0).norm() > 1e-4);

        let mut uncalibrated = Arm::<SixR, 6, 1>::new();
        uncalibrated.set_tcp(&tool()).unwrap();

        let policy = TolerancePolicy::DEFAULT;
        let mut solved = 0;
        for seed in 0..20 {
            let q_true = sample_q::<6>(seed, 2.5);
            let target = arm.fk(&q_true);
            let solve = |a: &Arm<SixR, 6, 1>| {
                a.ik(&target, None, Some(&q_true), false, true, policy, 15)[0]
            };
            // Where the driver has no nominal solution there is nothing to polish.
            let Some(q_nominal) = solve(&uncalibrated) else {
                assert!(solve(&arm).is_none(), "seed {seed}");
                continue;
            };
            solved += 1;
            let q = solve(&arm).unwrap_or_else(|| panic!("no calibrated solution, seed {seed}"));
            assert!((arm.fk(&q) - target).norm() < 1e-9, "seed {seed}");
            // The driver's nominal solution alone misses the real TCP.
            assert!(
                (arm.fk(&q_nominal) - target).norm() > policy.subproblem_numerical,
                "seed {seed}"
            );
        }
        assert!(solved >= 18, "stand-in driver solved only {solved} of 20");

        arm.clear_calibration();
        assert_eq!(arm.fk(&[0.2; 6]), nominal_fk0);
    }

    /// A scripted "driver" whose solve returns, in order: a solution outside
    /// joint 0's limit, the in-range solution, and a near-duplicate of it.
    struct Scripted;

    impl ArmModel<6, 3> for Scripted {
        const JOINT_LIMITS: [crate::math::postprocess::JointLimit; 6] =
            [Some((-PI, PI)), None, None, None, None, None];

        fn flange_fk(q: &[f64; 6]) -> Pose {
            SixR::flange_fk(q)
        }

        fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
            SixR::flange_jacobian(q)
        }

        fn flange_solve(
            target: &Pose,
            max_solutions: Option<usize>,
            q_seed: Option<&[f64; 6]>,
            respect_limits: bool,
            allow_refinement: bool,
            policy: TolerancePolicy,
            refinement_max_iters: usize,
        ) -> [Option<[f64; 6]>; 3] {
            // The nominal solve is scripted: it ignores the caller's tolerances.
            let _ = policy;
            let q = SixR::flange_solve(
                target,
                None,
                q_seed,
                respect_limits,
                allow_refinement,
                TolerancePolicy::DEFAULT,
                refinement_max_iters,
            )[0];
            let (mut over, mut twin) = (q, q);
            if let (Some(over), Some(twin)) = (over.as_mut(), twin.as_mut()) {
                over[0] += TWO_PI;
                twin[3] += 2e-4;
            }
            let mut out = [over, q, twin];
            for slot in out.iter_mut().skip(max_solutions.unwrap_or(3)) {
                *slot = None;
            }
            out
        }
    }

    #[test]
    fn residual_jacobian_matches_central_differences() {
        use crate::math::kinematics::pose_residual12;

        let mut arm = Arm::<SixR, 6, 1>::new();
        let tcp = pose_from_xyz_quat(&[0.05, -0.02, 0.1, 0.0, 0.0, 0.0, 1.0]).unwrap();
        arm.set_tcp(&tcp).unwrap();
        let goal = transform_of(&arm.fk(&[-0.1, 0.4, -0.6, 0.8, 0.2, -0.3]));
        let q = [0.3, -0.5, 0.9, -0.2, 0.6, 1.1];
        let cols = arm.residual_jacobian(&q);
        let h = 1e-6;
        for j in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[j] += h;
            qm[j] -= h;
            let rp = pose_residual12(&transform_of(&arm.fk(&qp)), &goal);
            let rm = pose_residual12(&transform_of(&arm.fk(&qm)), &goal);
            for row in 0..12 {
                let fd = (rp[row] - rm[row]) / (2.0 * h);
                assert!(
                    (fd - cols[j][row]).abs() < 1e-6,
                    "joint {j} row {row}: fd={fd} analytic={}",
                    cols[j][row]
                );
            }
        }
    }

    #[test]
    fn calibrated_ik_polishes_the_full_set_then_dedups_and_truncates() {
        let mut arm = Arm::<Scripted, 6, 3>::new();
        arm.set_calibration(calibrated_chain()).unwrap();
        let q_true = [0.5, -1.0, 1.2, -0.7, 0.9, -1.0];
        let target = arm.fk(&q_true);
        let policy = TolerancePolicy::DEFAULT;

        // Without limits all three polish onto the same solution modulo 2π.
        let all = arm.ik(&target, None, Some(&q_true), false, true, policy, 15);
        assert_eq!(all.iter().flatten().count(), 1);
        assert!((all[0].unwrap()[0] - q_true[0] - TWO_PI).abs() < 1e-6);

        // With limits the first nominal solution is dropped after the polish;
        // asking for one solution must still find the second.
        let one = arm.ik(&target, Some(1), Some(&q_true), true, true, policy, 15);
        assert_eq!(one.iter().flatten().count(), 1);
        let q = one[0].unwrap();
        assert!((arm.fk(&q) - target).norm() < 1e-9);
        assert!(q_close(&q, &q_true, 1e-6) && q[0].abs() < PI);

        // A tighter caller tolerance than the polish can meet rejects everything.
        let strict = TolerancePolicy {
            subproblem_numerical: 0.0,
            ..policy
        };
        let none = arm.ik(&target, None, Some(&q_true), false, true, strict, 15);
        assert!(none.iter().all(|s| s.is_none()));
    }

    #[test]
    fn ik_all_is_seedless_bounded_and_contains_the_seeded_answer() {
        let arm = Arm::<Scripted, 6, 3>::new();
        let q_true = [0.5, -1.0, 1.2, -0.7, 0.9, -1.0];
        let target = arm.fk(&q_true);
        let policy = TolerancePolicy::DEFAULT;

        let all = arm.ik(&target, Some(3), None, true, true, policy, IK_NEAREST_REFINEMENT_ITERS);
        let found = all.iter().flatten().count();
        assert!((1..=3).contains(&found));
        // `Scripted` includes a deliberately inexact twin (2e-4 off in joint 3).
        for q in all.iter().flatten() {
            assert!((arm.fk(q) - target).norm() < 1e-3);
        }

        let nearest = arm.ik_nearest(&target, &q_true, policy).unwrap();
        assert!(all
            .iter()
            .flatten()
            .any(|q| (0..6).all(|i| (q[i] - nearest[i]).abs() < 1e-6)));

        assert_eq!(
            arm.ik(&target, Some(0), None, true, true, policy, IK_NEAREST_REFINEMENT_ITERS)
                .iter()
                .flatten()
                .count(),
            0
        );
    }

    #[test]
    fn ik_nearest_follows_the_seed_branch_within_limits() {
        let arm = Arm::<SixR, 6, 1>::new();
        let q_true = [0.5, -1.0, 1.2, -0.7, 0.9, -1.0];
        let target = arm.fk(&q_true);
        let mut seed = q_true;
        seed[4] -= TWO_PI; // inside joint 4's (-2π, 2π) limit
        seed[5] += 3.0 * TWO_PI; // joint 5 is unlimited
        let q = arm
            .ik_nearest(&target, &seed, TolerancePolicy::DEFAULT)
            .unwrap();
        for i in 0..6 {
            assert!(
                (q[i] - seed[i]).abs() < 1e-6,
                "joint {i}: {} vs {}",
                q[i],
                seed[i]
            );
        }

        let mut far = target;
        far[(0, 3)] += 5.0;
        assert!(arm
            .ik_nearest(&far, &q_true, TolerancePolicy::DEFAULT)
            .is_none());
    }

    #[test]
    fn ik_nearest_refuses_to_jump() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        arm.set_calibration(calibrated_chain()).unwrap();
        let q_true = [0.5, -1.0, 1.2, -0.7, 0.9, -1.0];
        let target = arm.fk(&q_true);
        let policy = TolerancePolicy::DEFAULT;

        // Joint 0 is limited to (-π, π): a seed one turn away cannot be
        // followed, so the only solution is a full turn from the seed.
        let mut seed = q_true;
        seed[0] += TWO_PI;
        assert!(arm.ik_nearest(&target, &seed, policy).is_none());

        // A small step is followed, onto the calibrated chain.
        let mut near = q_true;
        near[1] += 0.05;
        near[3] -= 0.04;
        let q = arm.ik_nearest(&target, &near, policy).unwrap();
        assert!((arm.fk(&q) - target).norm() < 1e-9);

        // The bound is configurable, and can be disabled.
        arm.set_ik_max_joint_step(0.01).unwrap();
        assert!(arm.ik_nearest(&target, &near, policy).is_none());
        arm.set_ik_max_joint_step(f64::INFINITY).unwrap();
        let q = arm.ik_nearest(&target, &seed, policy).unwrap();
        assert!((q[0] - q_true[0]).abs() < 1e-6);
        assert_eq!(arm.set_ik_max_joint_step(0.0), Err(ModelError::NonPositive));
        assert_eq!(
            arm.set_ik_max_joint_step(f64::NAN),
            Err(ModelError::NonFinite)
        );
        assert_eq!(arm.ik_max_joint_step(), f64::INFINITY);
    }

    #[test]
    fn calibration_in_the_wrong_frame_is_rejected() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        // e.g. a DH base rotated by π about z relative to the driver's base.
        let flipped = pose_from_xyz_quat(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let chain = calibrated_chain().with_base(&flipped).unwrap();
        assert_eq!(
            arm.set_calibration(chain),
            Err(ModelError::ExcessiveDeviation)
        );
        assert!(arm.calibration().is_none());
    }

    #[test]
    fn tcp_jacobians_match_finite_differences_of_tcp_fk() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        arm.set_tcp(&tool()).unwrap();
        arm.set_calibration(calibrated_chain()).unwrap();
        let q = sample_q::<6>(7, 2.0);
        let jac = arm.jacobian(&q);
        let jac_tool = arm.jacobian_tool(&q);
        let pose = arm.fk(&q);
        let h = 1e-6;
        for c in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[c] += h;
            qm[c] -= h;
            let d = (arm.fk(&qp) - arm.fk(&qm)) / (2.0 * h);
            for r in 0..3 {
                assert!((d[(r, 3)] - jac[r][c]).abs() < 1e-7, "row {r} col {c}");
                // Tool axes: Rᵀ · (base-frame column).
                let (mut lin, mut ang) = (0.0, 0.0);
                for k in 0..3 {
                    lin += pose[(k, r)] * jac[k][c];
                    ang += pose[(k, r)] * jac[3 + k][c];
                }
                assert!((jac_tool[r][c] - lin).abs() < 1e-12);
                assert!((jac_tool[3 + r][c] - ang).abs() < 1e-12);
            }
        }
        let jinv = arm.inverse_jacobian(&q).unwrap();
        for r in 0..6 {
            for c in 0..6 {
                let s: f64 = (0..6).map(|k| jac[r][k] * jinv[k][c]).sum();
                assert!((s - if r == c { 1.0 } else { 0.0 }).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn dynamics_come_from_the_driver_plus_the_payload() {
        /// A driver without dynamics.
        struct Bare;
        impl ArmModel<6, 1> for Bare {
            fn flange_fk(q: &[f64; 6]) -> Pose {
                SixR::flange_fk(q)
            }
            fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
                SixR::flange_jacobian(q)
            }
            fn flange_solve(
                _: &Pose,
                _: Option<usize>,
                _: Option<&[f64; 6]>,
                _: bool,
                _: bool,
                _: TolerancePolicy,
                _: usize,
            ) -> [Option<[f64; 6]>; 1] {
                [None]
            }
        }
        let q = sample_q::<6>(2, 2.0);
        let qd = sample_q::<6>(3, 1.0);
        let qdd = sample_q::<6>(4, 2.0);
        assert!(Arm::<Bare, 6, 1>::new().dynamics(&q, &qd).is_none());
        assert!(Arm::<Bare, 6, 1>::new()
            .inverse_dynamics(&q, &qd, &qdd)
            .is_none());

        let mut arm = Arm::<SixR, 6, 1>::new();
        let bare = arm.dynamics(&q, &qd).unwrap();
        assert_eq!(bare.gravity, [-STANDARD_GRAVITY; 6]);
        assert_eq!(bare.coriolis, qd);
        let bare_tau = arm.inverse_dynamics(&q, &qd, &qdd).unwrap();
        for i in 0..6 {
            // The stand-in driver: M = 2·I, C qd = qd, G = g_z.
            assert_eq!(bare_tau[i], 2.0 * qdd[i] + qd[i] - STANDARD_GRAVITY);
        }

        arm.set_payload(tool_payload()).unwrap();
        arm.set_gravity([1.0, -2.0, -9.0]).unwrap();
        let bare = Arm::<SixR, 6, 1>::new().dynamics(&q, &qd).unwrap();
        let loaded = arm.dynamics(&q, &qd).unwrap();
        let (m, c, g) = (
            arm.payload_mass_matrix(&q),
            arm.payload_coriolis(&q, &qd),
            arm.payload_gravity(&q),
        );
        let tau = arm.inverse_dynamics(&q, &qd, &qdd).unwrap();
        for i in 0..6 {
            assert_eq!(loaded.gravity[i], -9.0 + g[i]);
            assert_eq!(loaded.coriolis[i], bare.coriolis[i] + c[i]);
            let mut want = loaded.coriolis[i] + loaded.gravity[i];
            for j in 0..6 {
                assert_eq!(loaded.mass_matrix[i][j], bare.mass_matrix[i][j] + m[i][j]);
                want += loaded.mass_matrix[i][j] * qdd[j];
            }
            assert!(
                (tau[i] - want).abs() < 1e-10,
                "joint {i}: {} vs {want}",
                tau[i]
            );
        }
        assert!(g.iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn setters_reject_bad_input_and_keep_state() {
        let mut arm = Arm::<SixR, 6, 1>::new();
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
    fn fk_residual_measures_host_against_reported_pose() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        arm.set_tcp(&tool()).unwrap();
        let q = sample_q::<6>(12, 2.0);
        // A controller that knows the calibrated geometry reports this pose.
        let mut truth = Arm::<SixR, 6, 1>::new();
        truth.set_tcp(&tool()).unwrap();
        truth.set_calibration(calibrated_chain()).unwrap();
        let reported = truth.fk(&q);

        let before = arm.fk_residual(&q, &reported).unwrap();
        assert!(before.position > 1e-4 && before.rotation > 1e-5);
        arm.set_calibration(calibrated_chain()).unwrap();
        let after = arm.fk_residual(&q, &reported).unwrap();
        assert!(after.position < 1e-12 && after.rotation < 1e-12);

        let mut shifted = reported;
        shifted[(1, 3)] += 0.003;
        let r = arm.fk_residual(&q, &shifted).unwrap();
        assert!((r.position - 0.003).abs() < 1e-12 && r.rotation < 1e-12);
        assert_eq!(
            arm.fk_residual(&q, &Pose::zeros()),
            Err(ModelError::NotRigid)
        );
    }

    #[test]
    fn external_wrench_moves_to_the_tcp_and_removes_payload_weight() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        arm.set_tcp(&tool()).unwrap();
        arm.set_payload(Payload::new(2.0, [0.01, 0.02, 0.1])).unwrap();
        let q = sample_q::<6>(5, 1.5);
        let flange = arm.flange_fk(&q);
        let tcp = arm.fk(&q);
        let close = |a: &[f64; 6], b: &[f64; 6]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9);

        // A pure external force at the TCP, seen by each sensor frame.
        let f_base = [3.0, -1.0, 2.0, 0.1, 0.2, -0.3];
        let at_tcp_axes = crate::math::spatial::wrench_base_to_tool(&tcp, &f_base);
        assert!(close(
            &arm.external_wrench(&q, &at_tcp_axes, WrenchFrame::Tcp, true),
            &f_base
        ));
        // The same wrench about the flange origin, in flange axes.
        let lever = [
            tcp[(0, 3)] - flange[(0, 3)],
            tcp[(1, 3)] - flange[(1, 3)],
            tcp[(2, 3)] - flange[(2, 3)],
        ];
        let m_flange = add3(&[f_base[3], f_base[4], f_base[5]], &cross3(&lever, &[f_base[0], f_base[1], f_base[2]]));
        let at_flange = crate::math::spatial::wrench_base_to_tool(
            &flange,
            &[f_base[0], f_base[1], f_base[2], m_flange[0], m_flange[1], m_flange[2]],
        );
        assert!(close(
            &arm.external_wrench(&q, &at_flange, WrenchFrame::Flange, true),
            &f_base
        ));

        // A still arm with only its payload: the sensor reads the weight.
        let r = pose_rotation(&flange);
        let com = add3(&[flange[(0, 3)], flange[(1, 3)], flange[(2, 3)]], &mat3_vec(&r, &[0.01, 0.02, 0.1]));
        let weight = arm.gravity().map(|g| 2.0 * g);
        let lever = [com[0] - tcp[(0, 3)], com[1] - tcp[(1, 3)], com[2] - tcp[(2, 3)]];
        let m = cross3(&lever, &weight);
        let raw = [weight[0], weight[1], weight[2], m[0], m[1], m[2]];
        assert!(close(&arm.external_wrench(&q, &raw, WrenchFrame::Base, false), &[0.0; 6]));
        assert!(close(&arm.external_wrench(&q, &raw, WrenchFrame::Base, true), &raw));
    }

    mod observer {
        use super::*;
        use crate::math::torque_observer::test_support::spec;
        use crate::math::torque_observer::{LstmObserver, ObserverMode};

        fn motion(k: usize) -> ([f64; 6], [f64; 6], [f64; 6], [f64; 6]) {
            let q = sample_q::<6>(k as u64, 1.5);
            let qd = sample_q::<6>(k as u64 + 100, 0.8);
            let qdd = sample_q::<6>(k as u64 + 200, 2.0);
            let cmd = q.map(|v| v + 0.01);
            (q, qd, qdd, cmd)
        }

        fn raw(obs: &mut LstmObserver<6>, q: &[f64; 6], qd: &[f64; 6], cmd: Option<&[f64; 6]>, stride: usize) -> Option<[f64; 6]> {
            let mut err = [0.0; 6];
            if let Some(cmd) = cmd {
                for i in 0..6 {
                    err[i] = cmd[i] - q[i];
                }
            }
            obs.step(&[*q, *qd, err], stride)
        }

        #[test]
        fn without_an_observer_the_model_is_used() {
            let mut arm = Arm::<SixR, 6, 1>::new();
            arm.set_payload(tool_payload()).unwrap();
            let (q, qd, qdd, cmd) = motion(1);
            assert_eq!(
                arm.step_free_torque(&q, &qd, &qdd, Some(&cmd), 1000),
                arm.inverse_dynamics(&q, &qd, &qdd)
            );
            let mut bare = Arm::<Scripted, 6, 3>::new();
            assert_eq!(bare.step_free_torque(&q, &qd, &qdd, None, 1000), None);
        }

        #[test]
        fn full_dynamics_output_is_the_network_plus_the_payload_delta() {
            let bytes = spec().leak(3);
            let mut arm = Arm::<Scripted, 6, 3>::new();
            arm.set_torque_observer(bytes).unwrap();
            assert!(arm.has_torque_observer());
            let mut reference = LstmObserver::<6>::from_bytes(bytes).unwrap();
            for k in 0..25 {
                let (q, qd, qdd, cmd) = motion(k);
                // 100 Hz model at 1 kHz: warming up for nine ticks, then
                // re-evaluated every tenth tick.
                let expected = raw(&mut reference, &q, &qd, Some(&cmd), 10);
                assert_eq!(expected.is_none(), k < 9, "tick {k}");
                let tau = arm.step_free_torque(&q, &qd, &qdd, Some(&cmd), 1000);
                assert_eq!(tau, expected, "tick {k}");
            }

            // A runtime payload adds exactly its inverse dynamics (trained bare).
            let (q, qd, qdd, _) = motion(40);
            let mut plain = Arm::<Scripted, 6, 3>::new();
            plain.set_torque_observer(bytes).unwrap();
            let mut loaded = Arm::<Scripted, 6, 3>::new();
            loaded.set_torque_observer(bytes).unwrap();
            loaded.set_payload(tool_payload()).unwrap();
            let a = plain.step_free_torque(&q, &qd, &qdd, None, 100).unwrap();
            let b = loaded.step_free_torque(&q, &qd, &qdd, None, 100).unwrap();
            let p = loaded.payload_torque(&q, &qd, &qdd);
            assert!(p.iter().any(|v| v.abs() > 1e-3));
            for i in 0..6 {
                assert!((b[i] - a[i] - p[i]).abs() < 1e-12);
            }
        }

        #[test]
        fn a_trained_payload_is_not_counted_twice() {
            let mut s = spec();
            s.payload = tool_payload();
            let bytes = s.leak(5);
            let (q, qd, qdd, _) = motion(2);

            let mut same = Arm::<Scripted, 6, 3>::new();
            same.set_torque_observer(bytes).unwrap();
            same.set_payload(tool_payload()).unwrap();
            let mut none = Arm::<Scripted, 6, 3>::new();
            none.set_torque_observer(bytes).unwrap();
            let mut net = LstmObserver::<6>::from_bytes(bytes).unwrap();
            let raw = raw(&mut net, &q, &qd, None, 1).unwrap();

            assert_eq!(same.step_free_torque(&q, &qd, &qdd, None, 100).unwrap(), raw);
            // Releasing the trained payload subtracts it.
            let released = none.step_free_torque(&q, &qd, &qdd, None, 100).unwrap();
            let p = same.payload_torque(&q, &qd, &qdd);
            for i in 0..6 {
                assert!((released[i] - (raw[i] - p[i])).abs() < 1e-12);
            }
        }

        #[test]
        fn friction_residual_adds_the_rigid_body_model() {
            let mut s = spec();
            s.mode = ObserverMode::FrictionResidual;
            let bytes = s.leak(9);

            assert_eq!(
                Arm::<Scripted, 6, 3>::new().set_torque_observer(bytes),
                Err(ModelError::InvalidObserver)
            );
            let mut arm = Arm::<SixR, 6, 1>::new();
            arm.set_torque_observer(bytes).unwrap();
            // A residual model is gravity-independent.
            arm.set_gravity([0.0, 0.0, STANDARD_GRAVITY]).unwrap();
            let mut net = LstmObserver::<6>::from_bytes(bytes).unwrap();
            for k in 0..4 {
                let (q, qd, qdd, _) = motion(k);
                let raw = raw(&mut net, &q, &qd, None, 1).unwrap();
                let rigid = arm.inverse_dynamics(&q, &qd, &qdd).unwrap();
                let tau = arm.step_free_torque(&q, &qd, &qdd, None, 100).unwrap();
                for i in 0..6 {
                    assert!((tau[i] - (raw[i] + rigid[i])).abs() < 1e-12, "tick {k}");
                }
            }
        }

        #[test]
        fn full_dynamics_is_bound_to_its_training_gravity() {
            let bytes = spec().leak(1);
            let ceiling = [0.0, 0.0, STANDARD_GRAVITY];

            let mut arm = Arm::<Scripted, 6, 3>::new();
            arm.set_gravity(ceiling).unwrap();
            assert_eq!(arm.set_torque_observer(bytes), Err(ModelError::InvalidObserver));
            assert!(!arm.has_torque_observer());

            let mut arm = Arm::<Scripted, 6, 3>::new();
            arm.set_torque_observer(bytes).unwrap();
            assert_eq!(arm.set_gravity(ceiling), Err(ModelError::InvalidObserver));
            assert_eq!(arm.gravity(), &[0.0, 0.0, -STANDARD_GRAVITY]);
            // Within tolerance is fine.
            arm.set_gravity([0.0, 5e-4, -STANDARD_GRAVITY]).unwrap();
            arm.clear_torque_observer();
            arm.set_gravity(ceiling).unwrap();

            let mut s = spec();
            s.gravity = ceiling;
            arm.set_torque_observer(s.leak(1)).unwrap();
        }

        #[test]
        fn non_finite_input_does_not_advance_the_observer() {
            let bytes = spec().leak(2);
            let mut arm = Arm::<Scripted, 6, 3>::new();
            arm.set_torque_observer(bytes).unwrap();
            let mut fresh = Arm::<Scripted, 6, 3>::new();
            fresh.set_torque_observer(bytes).unwrap();
            let (q, qd, qdd, cmd) = motion(3);
            let mut bad = qd;
            bad[2] = f64::NAN;
            assert_eq!(arm.step_free_torque(&q, &bad, &qdd, None, 100), None);
            let mut bad_cmd = cmd;
            bad_cmd[0] = f64::INFINITY;
            assert_eq!(arm.step_free_torque(&q, &qd, &qdd, Some(&bad_cmd), 100), None);
            assert_eq!(
                arm.step_free_torque(&q, &qd, &qdd, Some(&cmd), 100),
                fresh.step_free_torque(&q, &qd, &qdd, Some(&cmd), 100)
            );

            // Reset forgets the window: the next tick matches a fresh observer.
            let (q2, qd2, qdd2, _) = motion(4);
            let continued = arm.step_free_torque(&q2, &qd2, &qdd2, None, 100).unwrap();
            arm.reset_torque_observer();
            let after_reset = arm.step_free_torque(&q2, &qd2, &qdd2, None, 100).unwrap();
            let mut again = Arm::<Scripted, 6, 3>::new();
            again.set_torque_observer(bytes).unwrap();
            let fresh_first = again.step_free_torque(&q2, &qd2, &qdd2, None, 100).unwrap();
            assert_eq!(after_reset, fresh_first);
            assert_ne!(continued, fresh_first);
        }
    }
}
