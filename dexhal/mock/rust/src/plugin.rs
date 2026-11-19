//! Export of [`MockDriver`]`<`[`MockArm`]`, 6, 1>` as a `dexhal-plugin`.

use dexhal::math::{Arm, ArmError, ArmMath};
use dexhal_plugin::{export_plugin, PluginDriver};

use crate::{MockArm, MockDriver};

impl PluginDriver for MockDriver<MockArm, 6, 1> {
    /// `config` is empty, or `observer=<path>` to load a `.dexnext` torque
    /// observer from a file.
    fn create(config: &[u8]) -> Result<Self, ArmError> {
        let driver =
            MockDriver::default_arm().map_err(|_| ArmError::Fault("mock manifest is invalid"))?;
        if config.is_empty() {
            return Ok(driver);
        }
        let path = std::str::from_utf8(config)
            .ok()
            .and_then(|c| c.trim().strip_prefix("observer="))
            .ok_or(ArmError::Fault("mock driver takes no configuration"))?;
        let bytes = std::fs::read(path).map_err(|_| ArmError::Fault("cannot read observer"))?;
        driver.with_torque_observer(Box::leak(bytes.into_boxed_slice()))
    }

    fn create_math() -> Box<dyn ArmMath> {
        Box::new(Arm::<MockArm, 6, 1>::new())
    }
}

export_plugin!(MockDriver<MockArm, 6, 1>);
