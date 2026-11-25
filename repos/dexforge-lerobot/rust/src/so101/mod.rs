//! Five-joint SO-ARM101 (the physical arm; gripper jaw excluded).
//!
//! Built on the generated six-joint driver [`so101_6r`](crate::drivers::so101_6r) (the arm plus a virtual
//! wrist joint, which the ssik pipeline needs). `model.rs` and `arm.rs` are hand-written; see
//! `model.rs` for how the virtual joint is hidden. `dynamics_*` are generated from `so101.urdf`
//! by `codegen/generate_dynamics_test_data.py`; `reference_data.*` are generated from the
//! *unmodified* upstream URDF by `codegen/generate_reference_data.py`.

pub mod arm;
pub mod driver;
pub mod model;

/// The generated parity test (`dynamics_parity.rs`) imports `super::generated_dynamics`; this
/// exposes the five-joint functions under that name so it checks them against Pinocchio.
pub mod generated_dynamics {
    pub use super::model::{coriolis, fk, gravity, inverse_dynamics, jacobian, mass_matrix, DOF};
}

/// The entry points every arm module exposes (`fk`, `solve`, `TolerancePolicy`), so generic
/// consumers such as the simulator bindings treat this arm like the scaffolded ones. `solve`
/// ignores `allow_refinement`: the five real joints are always polished.
pub mod ik_solver {
    use dexhal::math::Pose;
    pub use dexhal::math::TolerancePolicy;

    pub use super::model::{fk, DOF, MAX_SOLUTIONS};

    /// See [`super::model::solve`].
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; DOF]>,
        respect_limits: bool,
        _allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; DOF]>; MAX_SOLUTIONS] {
        super::model::solve(
            target,
            max_solutions,
            q_seed,
            respect_limits,
            policy,
            refinement_max_iters,
        )
    }
}

pub use arm::{So101, DOF, MAX_SOLUTIONS};
pub use driver::{So101Driver, MANIFEST};

#[cfg(test)]
mod dynamics_parity;
