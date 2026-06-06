//! Flexiv **Rizon 4** (7-DOF) driver module.
//!
//! Port of the `ssik` prebuilt artifact `ssik/prebuilt/rizon4_ik.py`
//! (solver `jointlock.seven_r`, tier 1). See [`ik_solver`] for the public API.

mod consts;
pub mod generated_kinematics;
pub mod ik_solver;
mod pq;
