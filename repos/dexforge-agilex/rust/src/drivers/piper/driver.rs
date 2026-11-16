//! `ArmDriver` (hardware I/O) of the Piper.
//!
//! Scaffolded by `codegen/scaffold.py`; the coding agent replaces every
//! `TODO(agent)` with the OEM hardware API / protocol implementation (see
//! `dexhal/mock/rust` for a complete reference driver). Gate any OS or
//! vendor-SDK I/O behind the crate's `std` feature.

use dexhal::math::{Arm, ArmDriver, ArmError, ArmManifest, ArmMath, ManifestError};

use super::arm::{Piper, DOF, MAX_SOLUTIONS};

/// Driver manifest, validated by [`ArmManifest::from_text`].
pub const MANIFEST: &str = include_str!("../../../manifest.yaml");

/// Piper hardware driver.
pub struct PiperDriver {
    manifest: ArmManifest,
    arm: Arm<Piper, DOF, MAX_SOLUTIONS>,
}

impl PiperDriver {
    /// Build the driver from [`MANIFEST`].
    pub fn new() -> Result<Self, ManifestError> {
        Ok(Self {
            manifest: ArmManifest::from_text::<Piper, DOF, MAX_SOLUTIONS>(MANIFEST)?,
            arm: Arm::new(),
        })
    }
}

impl ArmDriver for PiperDriver {
    fn manifest(&self) -> &ArmManifest {
        &self.manifest
    }

    fn connect(&mut self) -> Result<(), ArmError> {
        // TODO(agent): open the OEM connection / session.
        Err(ArmError::NotSupported)
    }

    fn disconnect(&mut self) -> Result<(), ArmError> {
        // TODO(agent): close the OEM connection / session.
        Err(ArmError::NotSupported)
    }

    fn read_state(&mut self, state_buffer: &mut [f64]) -> Result<(), ArmError> {
        if state_buffer.len() < self.manifest.state_len() {
            return Err(ArmError::BufferTooSmall);
        }
        // TODO(agent): fill a `CoreState` from the OEM state and `write_to` the buffer.
        Err(ArmError::NotSupported)
    }

    fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError> {
        if cmd_buffer.len() < self.manifest.command_len() {
            return Err(ArmError::BufferTooSmall);
        }
        // TODO(agent): decode with `CoreCommand::read_from` and send to the OEM controller.
        Err(ArmError::NotSupported)
    }

    fn arm(&self) -> &dyn ArmMath {
        &self.arm
    }

    fn arm_mut(&mut self) -> &mut dyn ArmMath {
        &mut self.arm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dexhal::math::ArmModel;

    #[test]
    fn manifest_is_valid_for_the_model() {
        let driver = PiperDriver::new().expect("manifest.yaml parses");
        assert_eq!(driver.manifest().name, <Piper as ArmModel<DOF, MAX_SOLUTIONS>>::NAME);
        assert_eq!(driver.arm().dof(), DOF);
    }
}
