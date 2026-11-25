//! `ArmDriver` (hardware I/O) of the five-joint SO-ARM101.
//!
//! The model, manifest and buffer contract are complete; **hardware I/O is not implemented**.
//! The real arm is a chain of Feetech STS3215 servos on a serial bus, normally driven from
//! Python by LeRobot (`lerobot`'s `SO101Follower`). A Rust serial driver would be built from the
//! Feetech protocol documentation, which was not provided; until then `connect`, `disconnect`,
//! `read_state` and `write_command` return [`ArmError::NotSupported`]. The MuJoCo simulator
//! (`tools/sim_driver.py`) implements this same contract against a physics model instead.

use dexhal::math::{Arm, ArmDriver, ArmError, ArmManifest, ArmMath, ManifestError};

use super::arm::{So101, DOF, MAX_SOLUTIONS};

/// Driver manifest, validated by [`ArmManifest::from_text`].
pub const MANIFEST: &str = include_str!("../../manifest.yaml");

/// SO-ARM101 hardware driver (I/O not implemented, see the module docs).
pub struct So101Driver {
    manifest: ArmManifest,
    arm: Arm<So101, DOF, MAX_SOLUTIONS>,
}

impl So101Driver {
    /// Build the driver from [`MANIFEST`].
    pub fn new() -> Result<Self, ManifestError> {
        Ok(Self {
            manifest: ArmManifest::from_text::<So101, DOF, MAX_SOLUTIONS>(MANIFEST)?,
            arm: Arm::new(),
        })
    }
}

impl ArmDriver for So101Driver {
    fn manifest(&self) -> &ArmManifest {
        &self.manifest
    }

    fn connect(&mut self) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    fn disconnect(&mut self) -> Result<(), ArmError> {
        Err(ArmError::NotSupported)
    }

    fn read_state(&mut self, state_buffer: &mut [f64]) -> Result<(), ArmError> {
        if state_buffer.len() < self.manifest.state_len() {
            return Err(ArmError::BufferTooSmall);
        }
        Err(ArmError::NotSupported)
    }

    fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError> {
        if cmd_buffer.len() < self.manifest.command_len() {
            return Err(ArmError::BufferTooSmall);
        }
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
        let driver = So101Driver::new().expect("manifest.yaml parses");
        assert_eq!(
            driver.manifest().name,
            <So101 as ArmModel<DOF, MAX_SOLUTIONS>>::NAME
        );
        assert_eq!(driver.arm().dof(), DOF);
    }
}
