//! Mock arm driver (vendor `"mock"`) for integration tests of dexhal clients.
//!
//! `connect` and `disconnect` are no-ops. `write_command` pushes the command
//! into a fixed-depth queue whose depth is the manifest's round-trip latency.
//! `read_state` computes the state from the oldest queued command, so a
//! command becomes visible after `command_dispatch_cycles +
//! state_staleness_cycles` further writes.
//!
//! In a motion [`ActiveMode`] the state echoes the commanded values: joint
//! position, velocity and torque, TCP twist and wrench. [`ActiveMode::Idle`]
//! holds the last position with zero velocity and leaves the rest unset.
//! Digital and analog outputs (looped back to the inputs) and the gripper
//! width are echoed in every mode. Fields the command leaves unset read as
//! NaN.
//!
//! The vendor command `"echo"` copies its payload into the response.
//!
//! A learned torque observer (FACTR 2) can be installed with
//! [`MockDriver::with_torque_observer`], or through the plugin's `config` as
//! `observer=<path to .dexnext>`.
//!
//! The crate also builds as a `cdylib` exporting [`MockArm`]'s driver as a
//! `dexhal-plugin` (see [`dexhal_plugin_entry`]).

#![deny(unsafe_code)]

use std::collections::VecDeque;

use dexhal::math::{
    pose_to_xyz_quat, ActiveMode, Arm, ArmDriver, ArmError, ArmManifest, ArmMath, ArmModel,
    CoreCommand, CoreState, ManifestError, MAX_DOF,
};

mod model;
mod plugin;

pub use plugin::dexhal_plugin_entry;

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

    /// Install a learned torque observer (`.dexnext` bytes; see
    /// [`Arm::set_torque_observer`]) and advertise it in the manifest.
    pub fn with_torque_observer(mut self, bytes: &'static [u8]) -> Result<Self, ArmError> {
        self.arm.set_torque_observer(bytes)?;
        self.manifest.has_torque_observer = true;
        Ok(self)
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
        let moving = cmd.active_mode() != Some(ActiveMode::Idle);

        if moving {
            for i in 0..DOF {
                if cmd.target_q[i].is_finite() {
                    self.held_q[i] = cmd.target_q[i];
                }
            }
        }

        let mut q = [0.0; DOF];
        q.copy_from_slice(&self.held_q[..DOF]);

        let mut state = CoreState::UNKNOWN;
        state.q = self.held_q;
        let or_zero = |v: f64| if moving && v.is_finite() { v } else { 0.0 };
        for i in 0..DOF {
            state.qd[i] = or_zero(cmd.target_qd[i]);
            state.torques[i] = or_zero(cmd.target_torques[i]);
        }
        state.tcp_pose = pose_to_xyz_quat(&self.arm.fk(&q));
        if moving {
            state.tcp_twist = cmd.target_twist;
            state.wrench = cmd.target_wrench;
        }
        state.io_digital_in = cmd.io_digital_out_state;
        state.io_analog_in = cmd.io_analog_out;
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
        let mode = cmd
            .active_mode()
            .ok_or(ArmError::Fault("unknown control_mode"))?;
        if !self.manifest.supports_mode(mode) {
            return Err(ArmError::NotSupported);
        }
        self.queue.pop_front();
        self.queue.push_back(cmd);
        Ok(())
    }

    fn handle_vendor_command(
        &mut self,
        cmd_name: &str,
        payload: &[u8],
        response: &mut [u8],
    ) -> Result<usize, ArmError> {
        if cmd_name != "echo" {
            return Err(ArmError::NotSupported);
        }
        let out = response
            .get_mut(..payload.len())
            .ok_or(ArmError::BufferTooSmall)?;
        out.copy_from_slice(payload);
        Ok(payload.len())
    }

    fn arm(&self) -> &dyn ArmMath {
        &self.arm
    }

    fn arm_mut(&mut self) -> &mut dyn ArmMath {
        &mut self.arm
    }
}
