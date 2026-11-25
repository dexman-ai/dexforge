//! Shared math and the driver contract.
//!
//! * Common math, used by the drivers and by the contract layer:
//!   [`ssik_core`], [`kinematics`], [`postprocess`], [`tolerances`],
//!   [`spatial`], [`payload`], [`chain`], [`identify`], [`torque_observer`].
//! * The contract: [`model`] (what a driver implements), [`arm`] (one
//!   physical arm) and [`contract`] (what a kernel sees).

pub mod arm;
pub mod chain;
pub mod closed_form;
pub mod contract;
pub mod identify;
pub mod jointlock;
pub mod kinematics;
pub mod model;
pub mod payload;
pub mod postprocess;
pub mod raghavan_roth;
pub mod single;
pub mod spatial;
pub mod ssik_core;
pub mod tolerances;
pub mod torque_observer;

#[cfg(test)]
pub(crate) mod testing;

pub use arm::*;
pub use chain::*;
pub use contract::*;
pub use identify::*;
pub use kinematics::*;
pub use payload::*;
pub use postprocess::*;
pub use spatial::*;
pub use ssik_core::*;
pub use tolerances::*;
pub use torque_observer::*;
