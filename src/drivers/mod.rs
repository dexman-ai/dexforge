//! Per-arm closed-form inverse kinematics drivers.

#[path = "franka/drivers/franka_panda/mod.rs"]
pub mod franka_panda;
#[path = "agilex/drivers/piper/mod.rs"]
pub mod piper;
#[path = "flexiv/drivers/rizon4/mod.rs"]
pub mod rizon4;
#[path = "flexiv/drivers/rizon10/mod.rs"]
pub mod rizon10;
#[path = "flexiv/drivers/rizon_common.rs"]
pub mod rizon_common;
#[path = "ur/drivers/ur5/mod.rs"]
pub mod ur5;
#[path = "ufactory/drivers/xarm7/mod.rs"]
pub mod xarm7;
