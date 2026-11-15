//! Mock arm driver (vendor `"mock"`) for integration tests of dexhal clients.
//!
//! `connect` and `disconnect` are no-ops. `write_command` pushes the command
//! into a fixed-depth queue whose depth is the manifest's round-trip latency.
//! `read_state` computes the state from the oldest queued command, so a
//! command becomes visible after `command_dispatch_cycles +
//! state_staleness_cycles` further writes.
//!
//! The state echoes the commanded values: joint velocity and torque, TCP
//! twist, wrench, digital outputs (looped back to the inputs) and gripper
//! width. Fields the command leaves unset read as NaN.

#![forbid(unsafe_code)]

use std::collections::VecDeque;

use dexhal::math::{
    pose_to_xyz_quat, Arm, ArmDriver, ArmError, ArmManifest, ArmMath, ArmModel, CoreCommand,
    CoreState, ManifestError, MAX_DOF,
};

mod model;

pub use model::MockArm;

/// Manifest of [`MockArm`], for clients that load it directly.
pub const MANIFEST: &str = include_str!("../manifest.yaml");

pub struct MockDriver<A, const DOF: usize, const MAX: usize> {
    manifest: ArmManifest,
    arm: Arm<A, DOF, MAX>,
    queue: VecDeque<CoreCommand>,
    held_q: [f64; MAX_DOF],
}

impl MockDriver<MockArm, 6, 1> {
    /// The driver for [`MockArm`] with the crate's [`MANIFEST`].
    pub fn default_arm() -> Result<Self, ManifestError> {
        Self::new(MANIFEST)
    }
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> MockDriver<A, DOF, MAX> {
    /// Build the driver from manifest text whose `driver:` field equals `A::NAME`.
    pub fn new(manifest_text: &'static str) -> Result<Self, ManifestError> {
        let manifest = ArmManifest::from_text::<A, DOF, MAX>(manifest_text)?;
        let depth = manifest.command_dispatch_cycles as usize
            + manifest.state_staleness_cycles as usize
            + 1;
        let queue = (0..depth).map(|_| CoreCommand::IDLE).collect();
        Ok(Self {
            manifest,
            arm: Arm::new(),
            queue,
            held_q: [0.0; MAX_DOF],
        })
    }
}

impl<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize> ArmDriver
    for MockDriver<A, DOF, MAX>
{
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
        let len = self.manifest.state_len();
        if state_buffer.len() < len {
            return Err(ArmError::BufferTooSmall);
        }
        let cmd = self
            .queue
            .front()
            .ok_or(ArmError::Fault("empty command queue"))?;

        for i in 0..DOF {
            if cmd.target_q[i].is_finite() {
                self.held_q[i] = cmd.target_q[i];
            }
        }

        let mut q = [0.0; DOF];
        q.copy_from_slice(&self.held_q[..DOF]);

        let mut state = CoreState::UNKNOWN;
        state.q = self.held_q;
        for i in 0..DOF {
            state.qd[i] = if cmd.target_qd[i].is_finite() {
                cmd.target_qd[i]
            } else {
                0.0
            };
            state.torques[i] = if cmd.target_torques[i].is_finite() {
                cmd.target_torques[i]
            } else {
                0.0
            };
        }
        state.tcp_pose = pose_to_xyz_quat(&self.arm.fk(&q));
        state.tcp_twist = cmd.target_twist;
        state.wrench = cmd.target_wrench;
        state.io_digital_in = cmd.io_digital_out_state;
        state.gripper_width = cmd.gripper_target_width;

        state.write_to(state_buffer)?;
        state_buffer[CoreState::LEN..len].fill(f64::NAN);
        Ok(())
    }

    fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError> {
        if cmd_buffer.len() < self.manifest.command_len() {
            return Err(ArmError::BufferTooSmall);
        }
        let cmd = CoreCommand::read_from(cmd_buffer)?;
        self.queue.pop_front();
        self.queue.push_back(cmd);
        Ok(())
    }

    fn arm(&self) -> &dyn ArmMath {
        &self.arm
    }

    fn arm_mut(&mut self) -> &mut dyn ArmMath {
        &mut self.arm
    }
}
