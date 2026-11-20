//! The driver contract: what the DexZero kernel sees of an arm.
//!
//! * [`ArmManifest`] — static capabilities, limits (including optional
//!   [`jerk_limits`](ArmManifest::jerk_limits),
//!   [`has_static_inertials`](ArmManifest::has_static_inertials),
//!   [`has_residual_inertials`](ArmManifest::has_residual_inertials) and
//!   [`dynamics_source`](ArmManifest::dynamics_source)), and latency
//!   cycles, read once at boot.
//! * [`CoreState`] / [`CoreCommand`] and their zero-copy views — the two
//!   buffers exchanged every control cycle.
//! * [`ArmDriver`] — the object-safe control API (`Send`). A driver implements
//!   the hardware half (`manifest`, `connect`, `disconnect`, `read_state`,
//!   `write_command`) and hands out its host-side model.
//! * [`ArmMath`] — that host-side model with the DOF erased
//!   ([`MAX_DOF`]-padded arrays, `Send`), implemented by every [`Arm`].
//!
//! DexZero defines no copy of this contract; its executor thread holds the
//! driver as `Box<dyn ArmDriver>` and its planner thread holds an independent
//! `Box<dyn ArmMath>`.

pub mod manifest;
pub mod wire;

pub use manifest::*;
pub use wire::*;

use crate::math::arm::{Arm, IK_NEAREST_REFINEMENT_ITERS, PoseResidual};
use crate::math::chain::{
    pose_from_xyz_quat, pose_to_xyz_quat, ChainJoint, KinematicChain, ModelError,
};
use crate::math::model::{ArmModel, Pose};
use crate::math::payload::Payload;
use crate::math::tolerances::TolerancePolicy;

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
    minor: 4,
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

/// [`Dynamics`](crate::math::arm::Dynamics) padded to [`MAX_DOF`]; rows and
/// columns beyond the active DOF are zero. `#[repr(C)]`, all `f64`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicsState {
    pub m_matrix: [[f64; MAX_DOF]; MAX_DOF],
    pub c_vector: [f64; MAX_DOF],
    pub g_vector: [f64; MAX_DOF],
}

/// A per-robot [`KinematicChain`] in flat, [`MAX_DOF`]-padded form, so a
/// calibration can cross the DOF-erased control plane.
///
/// `#[repr(C)]` and all-`f64` ([`CalibrationData::LEN`] slots, in field
/// order). Joints beyond the arm's DOF are `NAN` and ignored.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationData {
    /// Per joint: the fixed transform from the previous joint's moving frame
    /// (or the base), `[x, y, z, qx, qy, qz, qw]`.
    pub origin: [[f64; 7]; MAX_DOF],
    /// Per joint: the rotation axis in the joint's frame.
    pub axis: [[f64; 3]; MAX_DOF],
    /// Per joint: the zero offset (rad).
    pub offset: [f64; MAX_DOF],
    /// Last joint → flange, `[x, y, z, qx, qy, qz, qw]`.
    pub flange: [f64; 7],
}

impl CalibrationData {
    /// Number of `f64` slots.
    pub const LEN: usize = core::mem::size_of::<CalibrationData>() / core::mem::size_of::<f64>();

    /// Flatten a chain.
    pub fn from_chain<const DOF: usize>(chain: &KinematicChain<DOF>) -> CalibrationData {
        const { assert!(DOF <= MAX_DOF) };
        let mut out = CalibrationData {
            origin: [[f64::NAN; 7]; MAX_DOF],
            axis: [[f64::NAN; 3]; MAX_DOF],
            offset: [f64::NAN; MAX_DOF],
            flange: pose_to_xyz_quat(&chain.flange()),
        };
        for i in 0..DOF {
            let joint = chain.joint(i);
            out.origin[i] = pose_to_xyz_quat(&joint.origin);
            out.axis[i] = joint.axis;
            out.offset[i] = joint.offset;
        }
        out
    }

    /// Rebuild and validate the chain of a `DOF`-joint arm.
    pub fn to_chain<const DOF: usize>(&self) -> Result<KinematicChain<DOF>, ModelError> {
        const { assert!(DOF <= MAX_DOF) };
        let mut joints = [ChainJoint {
            origin: Pose::identity(),
            axis: [0.0, 0.0, 1.0],
            offset: 0.0,
        }; DOF];
        for i in 0..DOF {
            joints[i] = ChainJoint {
                origin: pose_from_xyz_quat(&self.origin[i])?,
                axis: self.axis[i],
                offset: self.offset[i],
            };
        }
        KinematicChain::new(&joints, &pose_from_xyz_quat(&self.flange)?)
    }

    /// Copy out of the front of `buffer`.
    pub fn read_from(buffer: &[f64]) -> Result<CalibrationData, ArmError> {
        let mut chunks = buffer
            .get(..Self::LEN)
            .ok_or(ArmError::BufferTooSmall)?
            .iter()
            .copied();
        let mut next = || chunks.next().unwrap_or(f64::NAN);
        let mut out = CalibrationData {
            origin: [[0.0; 7]; MAX_DOF],
            axis: [[0.0; 3]; MAX_DOF],
            offset: [0.0; MAX_DOF],
            flange: [0.0; 7],
        };
        out.origin.iter_mut().flatten().for_each(|v| *v = next());
        out.axis.iter_mut().flatten().for_each(|v| *v = next());
        out.offset.iter_mut().for_each(|v| *v = next());
        out.flange.iter_mut().for_each(|v| *v = next());
        Ok(out)
    }

    /// Copy into the front of `buffer`.
    pub fn write_to(&self, buffer: &mut [f64]) -> Result<(), ArmError> {
        let slots = buffer
            .get_mut(..Self::LEN)
            .ok_or(ArmError::BufferTooSmall)?;
        let values = self
            .origin
            .iter()
            .flatten()
            .chain(self.axis.iter().flatten())
            .chain(&self.offset)
            .chain(&self.flange);
        for (slot, v) in slots.iter_mut().zip(values) {
            *slot = *v;
        }
        Ok(())
    }
}

/// The host-side model of an arm with the DOF erased ([`MAX_DOF`]-padded
/// arrays, `[x, y, z, qx, qy, qz, qw]` poses).
///
/// Implemented by every [`Arm`]; a kernel reaches it through
/// [`ArmDriver`]. Entries beyond [`dof`](Self::dof) are ignored on input and
/// zero on output. Queries return `None` for non-finite input, singular /
/// unreachable targets, or a capability the model lacks.
///
/// No method shares a name with an inherent method of [`Arm`]: queries whose
/// typed twin has the same natural name end in `_padded`, and configuration
/// methods are `apply_*` / `remove_*`.
pub trait ArmMath: Send {
    /// Number of actuated joints.
    fn dof(&self) -> usize;

    /// Whether a per-robot calibration is active.
    fn is_calibrated(&self) -> bool;

    /// Base → TCP pose.
    fn forward_kinematics(&self, q: &[f64; MAX_DOF]) -> Option<[f64; 7]>;

    /// Geometric Jacobian `[Jv; Jw]` at the TCP, base axes.
    fn jacobian_padded(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]>;

    /// Geometric Jacobian at the TCP, TCP axes.
    fn tool_jacobian_padded(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]>;

    /// Pseudo-inverse of the TCP Jacobian (one row per joint); `None` at a
    /// singularity.
    fn jacobian_pseudo_inverse(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; 6]; MAX_DOF]>;

    /// Joint vector reaching `target_pose` on the branch of `q_seed`
    /// ([`Arm::ik_nearest`], including its continuity bound).
    fn inverse_kinematics(
        &self,
        target_pose: &[f64; 7],
        q_seed: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]>;

    /// Up to `out.len()` joint solutions reaching `target_pose`, with no seed
    /// ranking: the caller selects among them (for example by end-effector
    /// orientation). Writes them to the front of `out` and returns how many
    /// were written (0 if unreachable). Order is the driver's.
    fn inverse_kinematics_all(&self, target_pose: &[f64; 7], out: &mut [[f64; MAX_DOF]]) -> usize;

    /// `M`, `C qd`, `G` including the payload.
    fn dynamics_padded(&self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF]) -> Option<DynamicsState>;

    /// Joint torque for `(q, qd, qdd)` including the payload.
    fn inverse_dynamics_padded(
        &self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]>;

    /// Distance between the model's TCP pose at `q` and a pose reported by
    /// the hardware.
    fn pose_residual(&self, q: &[f64; MAX_DOF], reported_tcp: &[f64; 7]) -> Option<PoseResidual>;

    /// External wrench at the TCP origin in base axes from a reported
    /// [`CoreState::wrench`] ([`Arm::external_wrench`]).
    fn external_wrench_padded(
        &self,
        q: &[f64; MAX_DOF],
        reported: &[f64; 6],
        frame: WrenchFrame,
        gravity_compensated: bool,
    ) -> Option<[f64; 6]>;

    /// Set the payload (`inertia` row-major about the centre of mass).
    fn apply_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<(), ModelError>;

    /// Set the flange → TCP transform.
    fn apply_tcp_offset(&mut self, pose: [f64; 7]) -> Result<(), ModelError>;

    /// Set the gravity vector in the base frame.
    fn apply_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError>;

    /// Install a per-robot calibration.
    fn apply_calibration(&mut self, calibration: &CalibrationData) -> Result<(), ModelError>;

    /// Return to the nominal kinematics.
    fn remove_calibration(&mut self);

    /// Set the continuity bound (rad) of [`inverse_kinematics`](Self::inverse_kinematics).
    fn apply_ik_max_joint_step(&mut self, step: f64) -> Result<(), ModelError>;

    /// Free-space joint torque for this control tick, advancing the learned
    /// torque observer if one is installed ([`Arm::step_free_torque`]);
    /// `q_cmd` feeds its command-error feature.
    fn step_free_torque_padded(
        &mut self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
        q_cmd: Option<&[f64; MAX_DOF]>,
        control_freq_hz: u32,
    ) -> Option<[f64; MAX_DOF]>;

    /// Clear the learned torque observer's history ([`Arm::reset_torque_observer`]);
    /// no-op without one.
    fn reset_torque_observer_state(&mut self);

    /// Inverse dynamics of the configured payload alone ([`Arm::payload_torque`]).
    fn payload_torque_padded(
        &self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]>;
}

fn unpad<const DOF: usize>(q: &[f64; MAX_DOF]) -> Option<[f64; DOF]> {
    const { assert!(DOF <= MAX_DOF) };
    let mut out = [0.0; DOF];
    out.copy_from_slice(&q[..DOF]);
    out.iter().all(|v| v.is_finite()).then_some(out)
}

fn pad<const DOF: usize>(v: &[f64; DOF]) -> [f64; MAX_DOF] {
    let mut out = [0.0; MAX_DOF];
    out[..DOF].copy_from_slice(v);
    out
}

fn pad_rows<const DOF: usize>(jac: &[[f64; DOF]; 6]) -> [[f64; MAX_DOF]; 6] {
    jac.map(|row| pad(&row))
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> ArmMath for Arm<A, DOF, MAX> {
    fn dof(&self) -> usize {
        DOF
    }

    fn is_calibrated(&self) -> bool {
        self.calibration().is_some()
    }

    fn forward_kinematics(&self, q: &[f64; MAX_DOF]) -> Option<[f64; 7]> {
        Some(pose_to_xyz_quat(&self.fk(&unpad(q)?)))
    }

    fn jacobian_padded(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]> {
        Some(pad_rows(&self.jacobian(&unpad(q)?)))
    }

    fn tool_jacobian_padded(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]> {
        Some(pad_rows(&self.jacobian_tool(&unpad(q)?)))
    }

    fn jacobian_pseudo_inverse(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; 6]; MAX_DOF]> {
        let jinv = self.inverse_jacobian(&unpad(q)?)?;
        let mut out = [[0.0; 6]; MAX_DOF];
        out[..DOF].copy_from_slice(&jinv);
        Some(out)
    }

    fn inverse_kinematics(
        &self,
        target_pose: &[f64; 7],
        q_seed: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        let target = pose_from_xyz_quat(target_pose).ok()?;
        let q = self.ik_nearest(&target, &unpad(q_seed)?, TolerancePolicy::DEFAULT)?;
        Some(pad(&q))
    }

    fn inverse_kinematics_all(&self, target_pose: &[f64; 7], out: &mut [[f64; MAX_DOF]]) -> usize {
        let Ok(target) = pose_from_xyz_quat(target_pose) else {
            return 0;
        };
        let solutions = self.ik(
            &target,
            Some(out.len().min(MAX)),
            None,
            true,
            true,
            TolerancePolicy::DEFAULT,
            IK_NEAREST_REFINEMENT_ITERS,
        );
        let mut n = 0;
        for q in solutions.iter().flatten().take(out.len()) {
            out[n] = pad(q);
            n += 1;
        }
        n
    }

    fn dynamics_padded(&self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF]) -> Option<DynamicsState> {
        let d = self.dynamics(&unpad(q)?, &unpad(qd)?)?;
        let mut out = DynamicsState {
            m_matrix: [[0.0; MAX_DOF]; MAX_DOF],
            c_vector: pad(&d.coriolis),
            g_vector: pad(&d.gravity),
        };
        for i in 0..DOF {
            out.m_matrix[i] = pad(&d.mass_matrix[i]);
        }
        Some(out)
    }

    fn inverse_dynamics_padded(
        &self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        let tau = self.inverse_dynamics(&unpad(q)?, &unpad(qd)?, &unpad(qdd)?)?;
        Some(pad(&tau))
    }

    fn pose_residual(&self, q: &[f64; MAX_DOF], reported_tcp: &[f64; 7]) -> Option<PoseResidual> {
        let reported = pose_from_xyz_quat(reported_tcp).ok()?;
        self.fk_residual(&unpad(q)?, &reported).ok()
    }

    fn external_wrench_padded(
        &self,
        q: &[f64; MAX_DOF],
        reported: &[f64; 6],
        frame: WrenchFrame,
        gravity_compensated: bool,
    ) -> Option<[f64; 6]> {
        if !reported.iter().all(|v| v.is_finite()) {
            return None;
        }
        Some(self.external_wrench(&unpad(q)?, reported, frame, gravity_compensated))
    }

    fn apply_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<(), ModelError> {
        self.set_payload(Payload::from_contract(mass_kg, com, inertia)?)
    }

    fn apply_tcp_offset(&mut self, pose: [f64; 7]) -> Result<(), ModelError> {
        self.set_tcp(&pose_from_xyz_quat(&pose)?)
    }

    fn apply_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError> {
        self.set_gravity(gravity)
    }

    fn apply_calibration(&mut self, calibration: &CalibrationData) -> Result<(), ModelError> {
        self.set_calibration(calibration.to_chain()?)
    }

    fn remove_calibration(&mut self) {
        self.clear_calibration();
    }

    fn apply_ik_max_joint_step(&mut self, step: f64) -> Result<(), ModelError> {
        self.set_ik_max_joint_step(step)
    }

    fn step_free_torque_padded(
        &mut self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
        q_cmd: Option<&[f64; MAX_DOF]>,
        control_freq_hz: u32,
    ) -> Option<[f64; MAX_DOF]> {
        let cmd = match q_cmd {
            Some(c) => Some(unpad::<DOF>(c)?),
            None => None,
        };
        let tau = self.step_free_torque(
            &unpad(q)?,
            &unpad(qd)?,
            &unpad(qdd)?,
            cmd.as_ref(),
            control_freq_hz,
        )?;
        Some(pad(&tau))
    }

    fn reset_torque_observer_state(&mut self) {
        self.reset_torque_observer();
    }

    fn payload_torque_padded(
        &self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        Some(pad(&self.payload_torque(&unpad(q)?, &unpad(qd)?, &unpad(qdd)?)))
    }
}

/// Offer a change to the controller, then say which side applied it.
fn applied(native: Result<(), ArmError>) -> Result<Applied, ArmError> {
    match native {
        Ok(()) => Ok(Applied::Native),
        Err(ArmError::NotSupported) => Ok(Applied::Host),
        Err(e) => Err(e),
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
/// `read_state`, `write_command` and the math queries are the data plane and
/// must not allocate or block.
pub trait ArmDriver: Send {
    /// Static capabilities, read once at boot.
    fn manifest(&self) -> &ArmManifest;

    /// Bring the arm to a commandable state: open the transport, power on,
    /// release the brakes, arm the hardware command-stream watchdog
    /// ([`ArmManifest::hardware_watchdog_ms`]) and enable. On success the arm
    /// is enabled and holding position, ready for
    /// [`write_command`](Self::write_command). There are no separate enable /
    /// brake hooks.
    fn connect(&mut self) -> Result<(), ArmError>;

    /// Stop motion, disable (brakes engaged) and close the transport.
    fn disconnect(&mut self) -> Result<(), ArmError>;

    /// Fill `state_buffer` ([`ArmManifest::state_len`] slots): a [`CoreState`]
    /// followed by the aux tail. See [`CoreStateViewMut`].
    fn read_state(&mut self, state_buffer: &mut [f64]) -> Result<(), ArmError>;

    /// Execute the [`CoreCommand`] at the front of `cmd_buffer`. See
    /// [`CoreCommandView`].
    ///
    /// The driver dispatches on [`CoreCommandView::active_mode`] and never
    /// infers a mode from which fields are finite. [`ActiveMode::Idle`] means
    /// hold: forward the controller's native hold / stop. A mode the manifest
    /// does not [support](ArmManifest::supports_mode) is
    /// [`ArmError::NotSupported`]. The driver runs no software watchdog; if
    /// calls stop, the hardware watchdog armed in `connect` acts.
    fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError>;

    /// The host-side model of this arm.
    fn arm(&self) -> &dyn ArmMath;
    fn arm_mut(&mut self) -> &mut dyn ArmMath;

    // --- configuration plane ----------------------------------------------

    /// Forward a payload change to the robot controller, if it has one.
    fn set_payload_native(&mut self, _payload: &Payload) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    /// Forward a TCP change to the robot controller, if it has one.
    fn set_tcp_offset_native(&mut self, _pose: &[f64; 7]) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    /// Forward a gravity-direction (mounting) change to the robot controller,
    /// if it has one.
    fn set_gravity_native(&mut self, _gravity: &[f64; 3]) -> Result<(), ArmError> {
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
        let payload = Payload::from_contract(mass_kg, com, inertia)?;
        let side = applied(self.set_payload_native(&payload))?;
        self.arm_mut().apply_payload(mass_kg, com, inertia)?;
        Ok(side)
    }

    /// Change the TCP; same protocol as [`set_payload`](Self::set_payload).
    fn set_tcp_offset(&mut self, pose: [f64; 7]) -> Result<Applied, ArmError> {
        pose_from_xyz_quat(&pose)?;
        let side = applied(self.set_tcp_offset_native(&pose))?;
        self.arm_mut().apply_tcp_offset(pose)?;
        Ok(side)
    }

    /// Change the gravity vector (base frame); same protocol as
    /// [`set_payload`](Self::set_payload).
    fn set_gravity(&mut self, gravity: [f64; 3]) -> Result<Applied, ArmError> {
        if !gravity.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFinite.into());
        }
        let side = applied(self.set_gravity_native(&gravity))?;
        self.arm_mut().apply_gravity(gravity)?;
        Ok(side)
    }

    /// Install a per-robot calibration on the host model. The robot
    /// controller keeps its own; compare the two with
    /// [`calibration_residual`](Self::calibration_residual).
    fn set_calibration(&mut self, calibration: &CalibrationData) -> Result<(), ArmError> {
        Ok(self.arm_mut().apply_calibration(calibration)?)
    }

    /// Return the host model to the nominal kinematics.
    fn clear_calibration(&mut self) {
        self.arm_mut().remove_calibration();
    }

    /// Set the largest per-joint distance (rad) between the seed and the
    /// result of [`inverse_kinematics`](Self::inverse_kinematics).
    fn set_ik_max_joint_step(&mut self, step: f64) -> Result<(), ArmError> {
        Ok(self.arm_mut().apply_ik_max_joint_step(step)?)
    }

    /// Re-enable after a protective stop or fault (the arm ends enabled, as
    /// after [`connect`](Self::connect)).
    fn clear_faults(&mut self) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    /// Escape hatch for vendor features outside the contract (e.g. manual
    /// brake release during commissioning).
    ///
    /// Any reply is written to the front of `response`; the result is its
    /// length in bytes (`0` for none). [`ArmError::BufferTooSmall`] if the
    /// reply does not fit.
    fn handle_vendor_command(
        &mut self,
        _cmd_name: &str,
        _payload: &[u8],
        _response: &mut [u8],
    ) -> Result<usize, ArmError> {
        Err(ArmError::NotSupported)
    }

    // --- kinematics & dynamics --------------------------------------------

    fn forward_kinematics(&self, q: &[f64; MAX_DOF]) -> Option<[f64; 7]> {
        self.arm().forward_kinematics(q)
    }

    fn jacobian(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]> {
        self.arm().jacobian_padded(q)
    }

    /// The Jacobian in TCP axes.
    fn tool_jacobian(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; MAX_DOF]; 6]> {
        self.arm().tool_jacobian_padded(q)
    }

    fn jacobian_pseudo_inverse(&self, q: &[f64; MAX_DOF]) -> Option<[[f64; 6]; MAX_DOF]> {
        self.arm().jacobian_pseudo_inverse(q)
    }

    fn inverse_kinematics(
        &self,
        target_pose: &[f64; 7],
        q_seed: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        self.arm().inverse_kinematics(target_pose, q_seed)
    }

    fn dynamics(&self, q: &[f64; MAX_DOF], qd: &[f64; MAX_DOF]) -> Option<DynamicsState> {
        self.arm().dynamics_padded(q, qd)
    }

    fn inverse_dynamics(
        &self,
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd: &[f64; MAX_DOF],
    ) -> Option<[f64; MAX_DOF]> {
        self.arm().inverse_dynamics_padded(q, qd, qdd)
    }

    /// Free-space joint torque `τ_free` for this control tick, for
    /// torque-residual wrench estimation (Route 6): call every cycle so a
    /// learned observer's window stays warm. `qdd_ref` is the reference
    /// acceleration and `q_cmd` the commanded position (the observer's
    /// command-error feature; zero if `None`).
    ///
    /// With [`DynamicsSource::Vendor`] and no learned observer
    /// ([`ArmManifest::has_torque_observer`]) it is `M qdd + C qd + G` from
    /// the `aux` tail of `state_buffer`, plus the host payload's inverse
    /// dynamics unless the controller compensates it
    /// ([`ArmManifest::native_payload`]). Otherwise it is
    /// [`ArmMath::step_free_torque_padded`] at the manifest's control rate.
    fn step_free_torque(
        &mut self,
        state_buffer: &[f64],
        q: &[f64; MAX_DOF],
        qd: &[f64; MAX_DOF],
        qdd_ref: &[f64; MAX_DOF],
        q_cmd: Option<&[f64; MAX_DOF]>,
    ) -> Option<[f64; MAX_DOF]> {
        let manifest = self.manifest();
        let control_freq_hz = manifest.control_freq_hz;
        if manifest.dynamics_source == DynamicsSource::Vendor && !manifest.has_torque_observer {
            let n = usize::from(manifest.active_dof_count).min(MAX_DOF);
            let native_payload = manifest.native_payload;
            let d = manifest.aux_dynamics(state_buffer)?;
            if !qdd_ref[..n].iter().all(|v| v.is_finite()) {
                return None;
            }
            let mut tau = [0.0; MAX_DOF];
            for i in 0..n {
                let inertial: f64 = (0..n).map(|j| d.m_matrix[i][j] * qdd_ref[j]).sum();
                tau[i] = inertial + d.c_vector[i] + d.g_vector[i];
            }
            if !native_payload {
                let p = self.arm().payload_torque_padded(q, qd, qdd_ref)?;
                for i in 0..n {
                    tau[i] += p[i];
                }
            }
            return Some(tau);
        }
        self.arm_mut()
            .step_free_torque_padded(q, qd, qdd_ref, q_cmd, control_freq_hz)
    }

    /// Clear the learned torque observer's history: call on connect, fault
    /// clear, freedrive toggles and reference snaps. No-op without one.
    fn reset_torque_observer(&mut self) {
        self.arm_mut().reset_torque_observer_state();
    }

    /// Calibration self-check: the distance between the host model's TCP pose
    /// and the one the hardware reports in the same state buffer.
    ///
    /// `None` if the buffer is too short or the hardware does not report
    /// `q` / `tcp_pose`. A residual above the robot's repeatability means the
    /// host model and the controller disagree on calibration or TCP.
    fn calibration_residual(&self, state_buffer: &[f64]) -> Option<PoseResidual> {
        let state = CoreStateView::new(state_buffer).ok()?;
        self.arm().pose_residual(state.q(), state.tcp_pose())
    }

    /// The external wrench on the tool (TCP origin, base axes, payload weight
    /// removed) from the state buffer, conditioned per the manifest's
    /// `sensing.wrench_frame` / `wrench_is_gravity_compensated`.
    ///
    /// `None` if the buffer is too short or the arm has no wrench sensing.
    fn external_wrench(&self, state_buffer: &[f64]) -> Option<[f64; 6]> {
        let manifest = self.manifest();
        if !manifest.has_wrench_sensing {
            return None;
        }
        let state = CoreStateView::new(state_buffer).ok()?;
        self.arm().external_wrench_padded(
            state.q(),
            state.wrench(),
            manifest.wrench_frame,
            manifest.wrench_is_gravity_compensated,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testing::{calibrated_chain, sample_q, SixR};

    fn padded(q: &[f64; 6]) -> [f64; MAX_DOF] {
        let mut out = [f64::NAN; MAX_DOF];
        out[..6].copy_from_slice(q);
        out
    }

    #[test]
    fn calibration_data_round_trips() {
        let chain = calibrated_chain();
        let flat = CalibrationData::from_chain(&chain);
        assert_eq!(CalibrationData::LEN, 95);
        assert!(flat.offset[6].is_nan() && flat.origin[7][0].is_nan());

        let mut wire = [0.0f64; CalibrationData::LEN + 1];
        flat.write_to(&mut wire).unwrap();
        // Slot order is field order.
        assert_eq!(wire[0], flat.origin[0][0]);
        assert_eq!(wire[56], flat.axis[0][0]);
        assert_eq!(wire[80], flat.offset[0]);
        assert_eq!(wire[94], flat.flange[6]);
        let back = CalibrationData::read_from(&wire).unwrap();
        assert_eq!(back.offset[..6], flat.offset[..6]);
        assert_eq!(back.flange, flat.flange);
        assert_eq!(
            CalibrationData::read_from(&wire[..94]),
            Err(ArmError::BufferTooSmall)
        );

        let rebuilt: KinematicChain<6> = back.to_chain().unwrap();
        for seed in 0..10 {
            let q = sample_q::<6>(seed, 3.0);
            assert!((rebuilt.fk(&q) - chain.fk(&q)).norm() < 1e-13);
        }

        let mut bad = flat;
        bad.axis[2] = [0.0; 3];
        assert_eq!(bad.to_chain::<6>().unwrap_err(), ModelError::DegenerateAxis);
        // A 7-joint arm needs the seventh joint, which is NAN here.
        assert_eq!(flat.to_chain::<7>().unwrap_err(), ModelError::NonFinite);
    }

    #[test]
    fn erased_math_matches_the_typed_arm_and_pads() {
        let mut arm = Arm::<SixR, 6, 1>::new();
        let q6 = sample_q::<6>(9, 2.0);
        let q = padded(&q6);
        {
            let math: &mut dyn ArmMath = &mut arm;
            math.apply_calibration(&CalibrationData::from_chain(&calibrated_chain()))
                .unwrap();
            math.apply_tcp_offset([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 2.0])
                .unwrap();
            math.apply_payload(1.5, [0.0, 0.0, 0.05], [0.0; 9]).unwrap();
            math.apply_gravity([0.0, -9.0, -3.0]).unwrap();
            assert_eq!((math.dof(), math.is_calibrated()), (6, true));

            let pose = math.forward_kinematics(&q).unwrap();
            let back = math.inverse_kinematics(&pose, &q).unwrap();
            for i in 0..6 {
                assert!((back[i] - q[i]).abs() < 1e-7, "joint {i}");
            }
            assert_eq!(&back[6..], &[0.0, 0.0]);
            assert_eq!(
                math.pose_residual(&q, &pose).map(|r| r.position < 1e-12),
                Some(true)
            );

            let mut bad = q;
            bad[3] = f64::NAN;
            assert!(math.forward_kinematics(&bad).is_none());
            assert!(math.jacobian_pseudo_inverse(&bad).is_none());
            assert_eq!(
                math.apply_tcp_offset([0.0; 7]),
                Err(ModelError::DegenerateAxis)
            );
            assert_eq!(
                math.apply_ik_max_joint_step(-1.0),
                Err(ModelError::NonPositive)
            );
        }

        // Typed and erased forms agree; both are callable without disambiguation.
        let math: &dyn ArmMath = &arm;
        let (jac, jac_tool) = (arm.jacobian(&q6), arm.jacobian_tool(&q6));
        let (jac_p, jac_tool_p) = (
            math.jacobian_padded(&q).unwrap(),
            math.tool_jacobian_padded(&q).unwrap(),
        );
        let pinv = arm.inverse_jacobian(&q6).unwrap();
        let pinv_p = math.jacobian_pseudo_inverse(&q).unwrap();
        for r in 0..6 {
            assert_eq!(jac_p[r][..6], jac[r]);
            assert_eq!(jac_tool_p[r][..6], jac_tool[r]);
            assert_eq!(jac_p[r][6..], [0.0, 0.0]);
            assert_eq!(pinv_p[r], pinv[r]);
        }
        assert_eq!(pinv_p[6..], [[0.0; 6]; 2]);

        let d = arm.dynamics(&q6, &q6).unwrap();
        let d_p = math.dynamics_padded(&q, &q).unwrap();
        assert_eq!(d_p.m_matrix[2][..6], d.mass_matrix[2]);
        assert_eq!(d_p.m_matrix[7], [0.0; MAX_DOF]);
        assert_eq!(d_p.g_vector[..6], d.gravity);
        assert!(d_p.m_matrix[0][0] > 2.0);
        let tau = arm.inverse_dynamics(&q6, &q6, &q6).unwrap();
        let tau_p = math.inverse_dynamics_padded(&q, &q, &q).unwrap();
        assert_eq!(tau_p[..6], tau);
        assert_eq!(tau_p[6..], [0.0, 0.0]);

        let math: &mut dyn ArmMath = &mut arm;
        math.remove_calibration();
        assert!(!math.is_calibrated());
    }

    struct FakeDriver {
        manifest: ArmManifest,
        arm: Arm<SixR, 6, 1>,
        /// What the "controller" answers to native configuration calls.
        native: Result<(), ArmError>,
        native_calls: usize,
        /// The geometry the "controller" reports TCP poses with.
        controller: Arm<SixR, 6, 1>,
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
            let mut state = CoreStateViewMut::new(state_buffer)?;
            state.clear();
            let q = [0.3, -0.8, 1.1, -0.4, 0.7, 0.2];
            state.q()[..6].copy_from_slice(&q);
            *state.tcp_pose() = pose_to_xyz_quat(&self.controller.fk(&q));
            Ok(())
        }
        fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError> {
            CoreCommandView::new(cmd_buffer).map(|_| ())
        }
        fn arm(&self) -> &dyn ArmMath {
            &self.arm
        }
        fn arm_mut(&mut self) -> &mut dyn ArmMath {
            &mut self.arm
        }
        fn set_payload_native(&mut self, _payload: &Payload) -> Result<(), ArmError> {
            self.native_calls += 1;
            self.native
        }
        fn set_gravity_native(&mut self, _gravity: &[f64; 3]) -> Result<(), ArmError> {
            self.native
        }
    }

    fn fake_driver() -> FakeDriver {
        let ratings = JointRatings {
            velocity: [3.0; 6],
            acceleration: [15.0; 6],
            effort: [150.0; 6],
        };
        let mut controller = Arm::new();
        controller.set_calibration(calibrated_chain()).unwrap();
        FakeDriver {
            manifest: ArmManifest::for_model::<SixR, 6, 1>(500, control_mode::JOINT_POS, &ratings)
                .unwrap(),
            arm: Arm::new(),
            native: Err(ArmError::NotSupported),
            native_calls: 0,
            controller,
        }
    }

    #[test]
    fn driver_configuration_falls_back_to_the_host_model() {
        let mut driver = fake_driver();
        let q = padded(&[0.3; 6]);
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
        assert_eq!(driver.set_gravity([0.0, 9.8, 0.0]), Ok(Applied::Host));
        assert_eq!(driver.arm.gravity(), &[0.0, 9.8, 0.0]);

        // Native: both sides updated.
        driver.native = Ok(());
        assert_eq!(
            driver.set_payload(3.0, [0.0; 3], [0.0; 9]),
            Ok(Applied::Native)
        );
        assert_eq!(driver.arm.payload().mass, 3.0);
        assert_eq!(driver.set_gravity([0.0, 0.0, -9.8]), Ok(Applied::Native));

        // A controller fault must not desynchronise the host model.
        driver.native = Err(ArmError::Fault("protective stop"));
        assert_eq!(
            driver.set_payload(4.0, [0.0; 3], [0.0; 9]),
            Err(ArmError::Fault("protective stop"))
        );
        assert_eq!(driver.arm.payload().mass, 3.0);
        assert_eq!(
            driver.set_gravity([1.0, 0.0, 0.0]),
            Err(ArmError::Fault("protective stop"))
        );
        assert_eq!(driver.arm.gravity(), &[0.0, 0.0, -9.8]);

        // Invalid input never reaches the controller.
        let calls = driver.native_calls;
        assert_eq!(
            driver.set_payload(-1.0, [0.0; 3], [0.0; 9]),
            Err(ArmError::Model(ModelError::InvalidPayload))
        );
        assert_eq!(driver.native_calls, calls);
        assert_eq!(
            driver.set_gravity([f64::NAN, 0.0, 0.0]),
            Err(ArmError::Model(ModelError::NonFinite))
        );
        assert_eq!(driver.clear_faults(), Err(ArmError::NotSupported));
        assert_eq!(
            driver.set_ik_max_joint_step(0.0),
            Err(ArmError::Model(ModelError::NonPositive))
        );
    }

    #[test]
    fn kernel_checks_and_installs_a_calibration_through_the_erased_driver() {
        let mut driver = fake_driver();
        let driver: &mut dyn ArmDriver = &mut driver;
        let mut state = [0.0; CoreState::LEN];
        driver.read_state(&mut state).unwrap();
        assert!(state[6].is_nan());

        // The host model is nominal, the controller is calibrated.
        let before = driver.calibration_residual(&state).unwrap();
        assert!(before.position > 1e-4, "{before:?}");

        let calibration = CalibrationData::from_chain(&calibrated_chain());
        driver.set_calibration(&calibration).unwrap();
        assert!(driver.arm().is_calibrated());
        let after = driver.calibration_residual(&state).unwrap();
        assert!(after.position < 1e-9 && after.rotation < 1e-9, "{after:?}");

        driver.clear_calibration();
        assert!(!driver.arm().is_calibrated());

        // Hardware that reports no TCP pose cannot be checked.
        CoreStateViewMut::new(&mut state)
            .unwrap()
            .tcp_pose()
            .fill(f64::NAN);
        assert!(driver.calibration_residual(&state).is_none());
        assert!(driver.calibration_residual(&state[..10]).is_none());

        let mut wrong = calibration;
        wrong.origin[0][0] += 1.0;
        assert_eq!(
            driver.set_calibration(&wrong),
            Err(ArmError::Model(ModelError::ExcessiveDeviation))
        );

        let q = padded(&[0.3, -0.8, 1.1, -0.4, 0.7, 0.2]);
        assert!(driver.tool_jacobian(&q).is_some());
        assert!(driver.jacobian_pseudo_inverse(&q).is_some());
        assert!(driver.inverse_dynamics(&q, &q, &q).is_some());
        assert!(driver.dynamics(&q, &q).is_some());
        assert!(driver.jacobian(&q).is_some());
    }

    #[test]
    fn step_free_torque_follows_the_dynamics_source() {
        use crate::math::torque_observer::test_support::spec;

        let q6 = sample_q::<6>(4, 1.5);
        let qd6 = sample_q::<6>(5, 0.7);
        let qdd6 = sample_q::<6>(6, 2.0);
        let (q, qd, qdd) = (padded(&q6), padded(&qd6), padded(&qdd6));

        // Model source, no observer: the rigid-body inverse dynamics.
        let mut driver = fake_driver();
        driver.arm.set_payload(Payload::new(1.0, [0.0, 0.0, 0.05])).unwrap();
        let tau = driver.step_free_torque(&[], &q, &qd, &qdd, None).unwrap();
        assert_eq!(tau, driver.inverse_dynamics(&q, &qd, &qdd).unwrap());
        driver.reset_torque_observer();

        // Vendor source: M qdd + C qd + G from aux, plus the host payload.
        let mut vendor = fake_driver();
        vendor.arm.set_payload(Payload::new(1.0, [0.0, 0.0, 0.05])).unwrap();
        vendor.manifest.dynamics_source = DynamicsSource::Vendor;
        vendor.manifest.aux = AuxSchema::from_slice(&[
            AuxField { label: aux_dynamics::GRAVITY_TORQUE, size: 6 },
            AuxField { label: aux_dynamics::CORIOLIS_TORQUE, size: 6 },
            AuxField { label: aux_dynamics::MASS_MATRIX, size: 36 },
        ])
        .unwrap();
        let mut state = [0.0; CoreState::LEN + 48];
        {
            let tail = &mut state[CoreState::LEN..];
            for i in 0..6 {
                tail[i] = 1.0 + i as f64;
                tail[6 + i] = 0.5;
                tail[12 + 7 * i] = 3.0;
            }
        }
        let p = vendor.arm.payload_torque(&q6, &qd6, &qdd6);
        let tau = vendor.step_free_torque(&state, &q, &qd, &qdd, None).unwrap();
        for i in 0..6 {
            let expected = 3.0 * qdd6[i] + 0.5 + 1.0 + i as f64 + p[i];
            assert!((tau[i] - expected).abs() < 1e-12, "joint {i}");
        }
        assert_eq!(tau[6..], [0.0, 0.0]);
        vendor.manifest.native_payload = true;
        let native = vendor.step_free_torque(&state, &q, &qd, &qdd, None).unwrap();
        assert!((native[0] - (3.0 * qdd6[0] + 1.5)).abs() < 1e-12);
        assert!(vendor.step_free_torque(&state[..10], &q, &qd, &qdd, None).is_none());

        // A learned observer overrides the vendor terms.
        let bytes = spec().leak(11);
        vendor.arm.set_payload(Payload::NONE).unwrap();
        vendor.arm.set_torque_observer(bytes).unwrap();
        vendor.manifest.has_torque_observer = true;
        let mut reference = Arm::<SixR, 6, 1>::new();
        reference.set_torque_observer(bytes).unwrap();
        let cmd = padded(&q6.map(|v| v + 0.02));
        let first6 = |t: [f64; MAX_DOF]| -> [f64; 6] { t[..6].try_into().unwrap() };
        // 500 Hz: the observer warms up for four ticks.
        for k in 0..12 {
            let tau = vendor.step_free_torque(&state, &q, &qd, &qdd, Some(&cmd)).map(first6);
            let expected =
                reference.step_free_torque(&q6, &qd6, &qdd6, Some(&q6.map(|v| v + 0.02)), 500);
            assert_eq!(tau.is_none(), k < 4, "tick {k}");
            assert_eq!(tau, expected, "tick {k}");
        }
        vendor.reset_torque_observer();
        reference.reset_torque_observer();
        assert_eq!(vendor.step_free_torque(&state, &q, &qd, &qdd, None), None);
        assert_eq!(reference.step_free_torque(&q6, &qd6, &qdd6, None, 500), None);
        for _ in 0..4 {
            assert_eq!(
                vendor.step_free_torque(&state, &q, &qd, &qdd, None).map(first6),
                reference.step_free_torque(&q6, &qd6, &qdd6, None, 500)
            );
        }
    }
}
