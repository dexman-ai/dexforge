use crate::math::arm::{ArmModel, Pose};
use crate::math::tolerances::TolerancePolicy;

/// UFactory xArm7 (7-DOF) driver marker.
pub struct Xarm7;

impl ArmModel<7, 128> for Xarm7 {
    #[inline]
    fn flange_fk(q: &[f64; 7]) -> Pose {
        crate::drivers::xarm7::ik_solver::fk(q)
    }

    #[inline]
    fn flange_jacobian(q: &[f64; 7]) -> [[f64; 7]; 6] {
        crate::drivers::xarm7::generated_kinematics::jacobian(q)
    }

    #[inline]
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; 7]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; 7]>; 128] {
        crate::drivers::xarm7::ik_solver::solve(
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