//! AgileX **Piper** 6-DOF arm driver.
//!
//! Port of the `ssik` prebuilt artifact `ssik/prebuilt/piper_ik.py`
//! (solver `ikgeo.general_6r`, tier 2). See [`ik_solver`] for the public API.

mod consts;
pub mod arm;
pub mod generated_kinematics;
pub mod ik_solver;
mod pq;
