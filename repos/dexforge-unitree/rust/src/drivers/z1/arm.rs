use crate::math::arm::{ArmModel, Pose};
use crate::math::tolerances::TolerancePolicy;

/// Unitree Z1 (6-DOF) driver marker.
pub struct Z1;

impl ArmModel<6, 16> for Z1 {
    #[inline]
    fn flange_fk(q: &[f64; 6]) -> Pose {
        crate::drivers::z1::ik_solver::fk(q)
    }

    #[inline]
    fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
        crate::drivers::z1::generated_kinematics::jacobian(q)
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
        crate::drivers::z1::ik_solver::solve(
            target,
            max_solutions,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        )
    }

    #[inline]
    fn flange_gravity(q: &[f64; 6], gravity: &[f64; 3]) -> [f64; 6] {
        crate::drivers::z1::generated_dynamics::gravity(q, gravity)
    }

    #[inline]
    fn flange_mass_matrix(q: &[f64; 6]) -> [[f64; 6]; 6] {
        crate::drivers::z1::generated_dynamics::mass_matrix(q)
    }

    #[inline]
    fn flange_coriolis(q: &[f64; 6], qd: &[f64; 6]) -> [f64; 6] {
        crate::drivers::z1::generated_dynamics::coriolis(q, qd)
    }
}
