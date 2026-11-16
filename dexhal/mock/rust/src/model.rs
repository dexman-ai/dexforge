//! Default mock arm model: a nominal 6R with a standard-DH table.

use core::f64::consts::FRAC_PI_2;

use dexhal::math::{ArmModel, DhConvention, DhRow, KinematicChain, Pose, TolerancePolicy};

const DH: [DhRow; 6] = [
    DhRow::new(0.0, FRAC_PI_2, 0.12, 0.0),
    DhRow::new(-0.40, 0.0, 0.0, 0.0),
    DhRow::new(-0.35, 0.0, 0.0, 0.0),
    DhRow::new(0.0, FRAC_PI_2, 0.11, 0.0),
    DhRow::new(0.0, -FRAC_PI_2, 0.10, 0.0),
    DhRow::new(0.0, 0.0, 0.08, 0.0),
];

fn chain() -> KinematicChain<6> {
    KinematicChain::from_dh(DhConvention::Standard, &DH).expect("nominal DH table is valid")
}

/// Default model of [`MockDriver::default_arm`](crate::MockDriver::default_arm).
///
/// Kinematics come from the nominal DH table; there is no inverse kinematics,
/// so `flange_solve` returns no solutions.
pub struct MockArm;

impl ArmModel<6, 1> for MockArm {
    const NAME: &'static str = "mock-6r";

    fn flange_fk(q: &[f64; 6]) -> Pose {
        chain().fk(q)
    }

    fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
        chain().jacobian(q)
    }

    fn flange_solve(
        _target: &Pose,
        _max_solutions: Option<usize>,
        _q_seed: Option<&[f64; 6]>,
        _respect_limits: bool,
        _allow_refinement: bool,
        _policy: TolerancePolicy,
        _refinement_max_iters: usize,
    ) -> [Option<[f64; 6]>; 1] {
        [None]
    }
}
