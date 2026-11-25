//! Shared seeded-path checks for 7-DOF joint-lock arms ([`ArmModel::flange_one`]).
//!
//! Not a crate on its own: each OEM crate includes this file from its
//! `tests/ik_path.rs` with `#[path]`, as it does with `perf/arm_timing.rs`.
//! Kept out of the generated `arm.rs` so regenerating an arm does not drop it.

use dexhal::math::arm::IK_DEFAULT_MAX_JOINT_STEP;
use dexhal::math::single::IK_LOCK_MAX_STEP;
use dexhal::math::{ArmModel, IkKind, DEFAULT_TOLERANCE_POLICY};

const STEPS: usize = 20;
const MAX_JUMP: f64 = 0.1;
const CLOSURE: f64 = 1e-5;

/// Track a short joint-space path from `start` with `flange_one(kind)`,
/// re-seeding from each solution. Every step must close within [`CLOSURE`] and
/// move less than [`MAX_JUMP`] from its seed. With `IkKind::Near` the joint
/// `locked` must also stay exactly at its seed value.
pub fn track_path<A: ArmModel<7, MAX>, const MAX: usize>(start: [f64; 7], locked: usize, kind: IkKind) {
    let mut q = start;
    let mut seed = q.map(|v| v + 0.02);
    for step in 0..STEPS {
        let target = A::flange_fk(&q);
        let sol = A::flange_one(&target, &seed, kind, DEFAULT_TOLERANCE_POLICY, 15)
            .unwrap_or_else(|| panic!("{} {kind:?} step {step}: no solution", A::NAME));
        if kind == IkKind::Near {
            assert_eq!(sol[locked], seed[locked], "{} step {step}: locked joint moved", A::NAME);
        }
        assert!((A::flange_fk(&sol) - target).norm() < CLOSURE, "{} {kind:?} step {step}: does not close", A::NAME);
        let jump = sol.iter().zip(seed.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        assert!(jump < MAX_JUMP, "{} {kind:?} step {step}: jumped {jump} to {sol:?}", A::NAME);
        seed = sol;
        q[0] += 0.01;
        q[3] += 0.005;
    }
}

/// `Fast` from a seed whose lock joint is offset from `start`'s by `off`
/// (the other joints match `start`). Any returned solution must close, keep
/// the lock joint within [`IK_LOCK_MAX_STEP`] of the seed and every other
/// joint within [`IK_DEFAULT_MAX_JOINT_STEP`]. Offsets inside the lock bound
/// must find a solution (`start` itself qualifies).
pub fn fast_lock_bound<A: ArmModel<7, MAX>, const MAX: usize>(start: [f64; 7], locked: usize) {
    let target = A::flange_fk(&start);
    for off in [0.05, -0.15, 0.3, -0.6, 1.2] {
        let mut seed = start;
        seed[locked] += off;
        let sol = A::flange_one(&target, &seed, IkKind::Fast, DEFAULT_TOLERANCE_POLICY, 15);
        let Some(sol) = sol else {
            assert!(off.abs() > IK_LOCK_MAX_STEP, "{} off {off}: no solution", A::NAME);
            continue;
        };
        assert!((A::flange_fk(&sol) - target).norm() < CLOSURE, "{} off {off}: does not close", A::NAME);
        for i in 0..7 {
            let bound = if i == locked { IK_LOCK_MAX_STEP } else { IK_DEFAULT_MAX_JOINT_STEP };
            assert!((sol[i] - seed[i]).abs() <= bound, "{} off {off}: joint {i} jumped to {sol:?}", A::NAME);
        }
    }
}
