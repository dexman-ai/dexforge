//! `ArmModel` of the five-joint SO-ARM101.
//!
//! Hand-written (unlike the scaffolded `so101_6r/arm.rs`): `codegen/scaffold.py` derives an
//! `ArmModel` from a 6R/7R ssik artifact and cannot describe a five-joint arm. Every hook
//! delegates to [`super::model`], which evaluates the generated six-joint driver at the
//! virtual joint's zero. See that module for the embedding and its guarantees.
//!
//! `nominal_chain()` (the kinematic-calibration starting point) is not provided.

use dexhal::math::{ArmModel, JointLimit, Pose, TolerancePolicy};

use super::model;

/// Actuated joints, base to flange (the gripper jaw is not part of the arm).
pub const DOF: usize = model::DOF;

/// Capacity of the IK solution array.
pub const MAX_SOLUTIONS: usize = model::MAX_SOLUTIONS;

/// Five-joint SO-ARM101 model marker.
pub struct So101;

impl ArmModel<DOF, MAX_SOLUTIONS> for So101 {
    const NAME: &'static str = "so101";
    const JOINT_LIMITS: [JointLimit; DOF] = model::JOINT_LIMITS;
    const HAS_DYNAMICS: bool = true;

    #[inline]
    fn flange_fk(q: &[f64; DOF]) -> Pose {
        model::fk(q)
    }

    #[inline]
    fn flange_jacobian(q: &[f64; DOF]) -> [[f64; DOF]; 6] {
        model::jacobian(q)
    }

    /// `allow_refinement` is ignored: the five real joints are always polished onto the pose.
    #[inline]
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; DOF]>,
        respect_limits: bool,
        _allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; DOF]>; MAX_SOLUTIONS] {
        model::solve(
            target,
            max_solutions,
            q_seed,
            respect_limits,
            policy,
            refinement_max_iters,
        )
    }

    #[inline]
    fn flange_gravity(q: &[f64; DOF], gravity: &[f64; 3]) -> [f64; DOF] {
        model::gravity(q, gravity)
    }

    #[inline]
    fn flange_mass_matrix(q: &[f64; DOF]) -> [[f64; DOF]; DOF] {
        model::mass_matrix(q)
    }

    #[inline]
    fn flange_coriolis(q: &[f64; DOF], qd: &[f64; DOF]) -> [f64; DOF] {
        model::coriolis(q, qd)
    }

    #[inline]
    fn flange_inverse_dynamics(
        q: &[f64; DOF],
        qd: &[f64; DOF],
        qdd: &[f64; DOF],
        gravity: &[f64; 3],
    ) -> [f64; DOF] {
        model::inverse_dynamics(q, qd, qdd, gravity)
    }
}
