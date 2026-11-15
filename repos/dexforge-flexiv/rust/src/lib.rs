//! Flexiv arm drivers for dexhal.
//!
//! Each arm under [`drivers`] implements `dexhal::math::ArmModel` (generated
//! kinematics, IK and dynamics) and `dexhal::math::ArmDriver` (hardware I/O).

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

pub mod drivers;

// Per-arm re-exports (appended by codegen/scaffold.py).
pub use drivers::rizon_common;
pub use drivers::rizon4;
pub use drivers::rizon10;
pub use drivers::rizon4::{Rizon4, Rizon4Driver};
pub use drivers::rizon10::{Rizon10, Rizon10Driver};
