//! Ur arm drivers for dexhal.
//!
//! Each arm under [`drivers`] implements `dexhal::math::ArmModel` (generated
//! kinematics, IK and dynamics) and `dexhal::math::ArmDriver` (hardware I/O).

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

pub mod drivers;

// Per-arm re-exports (appended by codegen/scaffold.py).
pub use drivers::ur5;
pub use drivers::ur5::{Ur5, Ur5Driver};
pub use drivers::ur3e;
pub use drivers::ur3e::{Ur3e, Ur3eDriver};
