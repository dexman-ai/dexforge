#![no_std]
#![forbid(unsafe_code)]

pub mod drivers;
pub mod math;

pub use drivers::xarm7;
pub use drivers::xarm7::ik_solver::{fk, solve, TolerancePolicy};
pub use drivers::xarm7::generated_kinematics;