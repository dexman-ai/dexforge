//! Export of [`MockDriver`]`<`[`MockArm`]`, 6, 1>` as a `dexhal-plugin`.

use dexhal::math::{Arm, ArmError, ArmMath};
use dexhal_plugin::{export_plugin, PluginDriver};

use crate::{MockArm, MockDriver};

impl PluginDriver for MockDriver<MockArm, 6, 1> {
    /// The mock takes no configuration: `config` must be empty.
    fn create(config: &[u8]) -> Result<Self, ArmError> {
        if !config.is_empty() {
            return Err(ArmError::Fault("mock driver takes no configuration"));
        }
        MockDriver::default_arm().map_err(|_| ArmError::Fault("mock manifest is invalid"))
    }

    fn create_math() -> Box<dyn ArmMath> {
        Box::new(Arm::<MockArm, 6, 1>::new())
    }
}

export_plugin!(MockDriver<MockArm, 6, 1>);
