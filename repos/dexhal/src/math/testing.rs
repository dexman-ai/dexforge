//! Test stand-in: a generic 6R arm that plays the role of a driver crate.
//!
//! Real arms are tested in their own driver repositories against their real
//! solvers. dexhal's tests only need *something* behind [`ArmModel`], so this
//! module describes a made-up arm with a DH table, and implements the trait
//! the way a driver's `arm.rs` does — except that its "generated" functions
//! are a [`KinematicChain`] and placeholder dynamics.

use core::f64::consts::{FRAC_PI_2, PI};

use crate::math::chain::{apply_dh_deltas, DhConvention, DhRow, KinematicChain};
use crate::math::model::{ArmModel, Pose};
use crate::math::postprocess::{within_limits, JointLimit};
use crate::math::ssik_core::wrap_to_pi;
use crate::math::tolerances::TolerancePolicy;

/// Nominal standard-DH table: a shoulder, three parallel axes and a wrist.
pub(crate) const SIX_R_DH: [DhRow; 6] = [
    DhRow::new(0.0, FRAC_PI_2, 0.12, 0.0),
    DhRow::new(-0.40, 0.0, 0.0, 0.0),
    DhRow::new(-0.35, 0.0, 0.0, 0.0),
    DhRow::new(0.0, FRAC_PI_2, 0.11, 0.0),
    DhRow::new(0.0, -FRAC_PI_2, 0.10, 0.0),
    DhRow::new(0.0, 0.0, 0.08, 0.0),
];

/// Per-robot deltas of the magnitude found in a vendor calibration file.
pub(crate) const SIX_R_DELTA: [DhRow; 6] = [
    DhRow::new(1.2e-4, -3.0e-4, 2.1e-4, 4.0e-4),
    DhRow::new(-2.5e-4, 5.0e-4, 0.0, -6.0e-4),
    DhRow::new(3.1e-4, -2.0e-4, 0.0, 3.0e-4),
    DhRow::new(-0.7e-4, 4.0e-4, -1.5e-4, -2.0e-4),
    DhRow::new(0.4e-4, -1.0e-4, 0.9e-4, 5.0e-4),
    DhRow::new(0.0, 0.0, -1.1e-4, -3.0e-4),
];

pub(crate) fn nominal_chain() -> KinematicChain<6> {
    KinematicChain::from_dh(DhConvention::Standard, &SIX_R_DH).unwrap()
}

pub(crate) fn calibrated_chain() -> KinematicChain<6> {
    KinematicChain::from_dh(
        DhConvention::Standard,
        &apply_dh_deltas(&SIX_R_DH, &SIX_R_DELTA),
    )
    .unwrap()
}

/// Deterministic pseudo-random joint vector in `[-amp, amp]`.
pub(crate) fn sample_q<const DOF: usize>(seed: u64, amp: f64) -> [f64; DOF] {
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03);
    let mut q = [0.0; DOF];
    for v in q.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0) * amp;
    }
    q
}

/// The stand-in driver.
pub(crate) struct SixR;

impl ArmModel<6, 1> for SixR {
    const NAME: &'static str = "six-r";
    const JOINT_LIMITS: [JointLimit; 6] = [
        Some((-PI, PI)),
        Some((-2.0 * PI, 2.0 * PI)),
        Some((-2.0 * PI, 2.0 * PI)),
        Some((-2.0 * PI, 2.0 * PI)),
        Some((-2.0 * PI, 2.0 * PI)),
        None,
    ];
    const HAS_DYNAMICS: bool = true;

    fn flange_fk(q: &[f64; 6]) -> Pose {
        nominal_chain().fk(q)
    }

    fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
        nominal_chain().jacobian(q)
    }

    /// Stand-in for a generated solver: the one solution reached from the
    /// seed, as principal values, honouring the limits and the tolerance.
    /// Being a bounded Newton polish it gives up close to a singularity,
    /// where a real closed-form solver would not.
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; 6]>,
        respect_limits: bool,
        _allow_refinement: bool,
        policy: TolerancePolicy,
        _refinement_max_iters: usize,
    ) -> [Option<[f64; 6]>; 1] {
        let mut q = q_seed.copied().unwrap_or([0.0; 6]);
        if max_solutions == Some(0)
            || nominal_chain().refine_ik(target, &mut q) > policy.subproblem_numerical
        {
            return [None];
        }
        let q = q.map(wrap_to_pi);
        if respect_limits && !within_limits(&q, &Self::JOINT_LIMITS) {
            return [None];
        }
        [Some(q)]
    }

    // Recognisable placeholder dynamics: G = g_z, M = 2·I, C qd = qd.

    fn flange_gravity(_q: &[f64; 6], gravity: &[f64; 3]) -> [f64; 6] {
        [gravity[2]; 6]
    }

    fn flange_mass_matrix(_q: &[f64; 6]) -> [[f64; 6]; 6] {
        let mut m = [[0.0; 6]; 6];
        for i in 0..6 {
            m[i][i] = 2.0;
        }
        m
    }

    fn flange_coriolis(_q: &[f64; 6], qd: &[f64; 6]) -> [f64; 6] {
        *qd
    }

    fn flange_inverse_dynamics(
        _q: &[f64; 6],
        qd: &[f64; 6],
        qdd: &[f64; 6],
        gravity: &[f64; 3],
    ) -> [f64; 6] {
        core::array::from_fn(|i| 2.0 * qdd[i] + qd[i] + gravity[2])
    }
}
