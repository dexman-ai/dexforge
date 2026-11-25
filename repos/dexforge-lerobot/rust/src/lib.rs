//! Lerobot arm drivers for dexhal.
//!
//! Each arm under [`drivers`] implements `dexhal::math::ArmModel` (generated
//! kinematics, IK and dynamics) and `dexhal::math::ArmDriver` (hardware I/O).

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

pub mod drivers;
pub mod so101;

// Per-arm re-exports (appended by codegen/scaffold.py).
pub use so101::{So101, So101Driver};
pub use drivers::so101_6r::So1016r;
