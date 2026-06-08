//! `dexforge`: zero-allocation, `#![no_std]` analytical inverse kinematics.
//!
//! This crate is a faithful translation of the Python `ssik` prebuilt IK
//! artifacts (<https://github.com/personalrobotics/ssik>) into a deterministic,
//! heap-free Rust implementation suitable for hard real-time control loops.
//!
//! * [`math::ssik_core`] holds the generic, bounded-iteration numeric kernels
//!   (companion-matrix eigenvalues / quartic roots, small dense linear algebra)
//!   shared by every arm driver.
//! * [`drivers`] holds the per-arm closed-form solvers: the Universal Robots
//!   UR5 ([`drivers::ur5`]), the Franka Emika Panda ([`drivers::franka_panda`]),
//!   the AgileX Piper ([`drivers::piper`]), and the UFactory xArm7
//!   ([`drivers::xarm7`]).
//! * [`math::tcp`] adds runtime tool-center-point (TCP) retargeting and
//!   grasped-payload (mass + centre-of-mass) gravity/inertia terms, so the TCP
//!   and a held tool can change at runtime without regenerating any arm code.
//!
//! Every numeric loop in this crate carries a hardcoded iteration ceiling and
//! every data structure is a fixed-size, stack-allocated array, so the code
//! contains no heap allocation and no unbounded control flow.
#![no_std]
#![forbid(unsafe_code)]

pub mod drivers;
pub mod math;
