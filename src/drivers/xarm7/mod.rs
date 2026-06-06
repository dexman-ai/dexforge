//! UFactory **xArm7** (7-DOF) driver.
//!
//! Port of the `ssik` prebuilt artifact `ssik/prebuilt/xarm7_ik.py`
//! (solver `jointlock.seven_r`, tier 1). See [`ik_solver`] for the public API.

pub mod generated_kinematics;
pub mod ik_solver;

mod consts;
