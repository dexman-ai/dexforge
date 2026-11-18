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
/// Kinematics come from the nominal DH table. There is no closed-form inverse
/// kinematics: `flange_solve` polishes the seed (zero without one) onto the
/// target with the bounded damped-Newton [`KinematicChain::refine_ik`], so it
/// returns at most the one solution on the seed's branch.
pub struct MockArm;

impl ArmModel<6, 1> for MockArm {
    const NAME: &'static str = "mock-6r";

    fn flange_fk(q: &[f64; 6]) -> Pose {
        chain().fk(q)
    }

    fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
        chain().jacobian(q)
    }

    // `JOINT_LIMITS` is unconstrained, so `respect_limits` has nothing to drop.
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; 6]>,
        _respect_limits: bool,
        _allow_refinement: bool,
        policy: TolerancePolicy,
        _refinement_max_iters: usize,
    ) -> [Option<[f64; 6]>; 1] {
        if max_solutions == Some(0) {
            return [None];
        }
        let mut q = q_seed.copied().unwrap_or([0.0; 6]);
        let closes = chain().refine_ik(target, &mut q) <= policy.subproblem_numerical;
        [closes.then_some(q)]
    }
}
