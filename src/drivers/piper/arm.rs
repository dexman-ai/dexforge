use crate::math::arm::{ArmModel, Pose};
use crate::math::tolerances::TolerancePolicy;

/// AgileX Piper (6-DOF) driver marker.
pub struct Piper;

impl ArmModel<6, 16> for Piper {
    #[inline]
    fn flange_fk(q: &[f64; 6]) -> Pose {
        crate::drivers::piper::ik_solver::fk(q)
    }

    #[inline]
    fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
        crate::drivers::piper::generated_kinematics::jacobian(q)
    }

    #[inline]
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; 6]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; 6]>; 16] {
        crate::drivers::piper::ik_solver::solve(
            target,
            max_solutions,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        )
    }
}