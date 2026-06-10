use crate::math::arm::{ArmModel, Pose};
use crate::math::tolerances::TolerancePolicy;

/// Flexiv Rizon 4 (7-DOF) driver marker.
pub struct Rizon4;

impl ArmModel<7, 256> for Rizon4 {
    #[inline]
    fn flange_fk(q: &[f64; 7]) -> Pose {
        crate::drivers::rizon4::ik_solver::fk(q)
    }

    #[inline]
    fn flange_jacobian(q: &[f64; 7]) -> [[f64; 7]; 6] {
        crate::drivers::rizon4::generated_kinematics::jacobian(q)
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
    ) -> [Option<[f64; 7]>; 256] {
        crate::drivers::rizon4::ik_solver::solve(
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