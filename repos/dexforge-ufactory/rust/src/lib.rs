//! Ufactory arm drivers for dexhal.
//!
//! Each arm under [`drivers`] implements `dexhal::math::ArmModel` (generated
//! kinematics, IK and dynamics) and `dexhal::math::ArmDriver` (hardware I/O).

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

pub mod drivers;

// Per-arm re-exports (appended by codegen/scaffold.py).
pub use drivers::xarm7;
pub use drivers::xarm7::{Xarm7, Xarm7Driver};
