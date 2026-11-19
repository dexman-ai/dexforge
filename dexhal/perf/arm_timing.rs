//! Shared release-mode wall-clock timing of IK and dynamics for any [`ArmModel`].
//!
//! Not a crate on its own: each OEM crate includes this file from its
//! `tests/perf.rs` with `#[path]` and calls [`measure`] once per arm.

use std::hint::black_box;
use std::time::Instant;

use dexhal::math::{Arm, ArmModel, TolerancePolicy};

const WARMUP: usize = 200;
const IK_CALLS: usize = 2_000;
const DYN_CALLS: usize = 50_000;
const REFINEMENT_ITERS: usize = 15;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn joints<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(&mut self) -> [f64; DOF] {
        let mut q = [0.0; DOF];
        for (i, v) in q.iter_mut().enumerate() {
            let (lo, hi) = A::JOINT_LIMITS[i].unwrap_or((-std::f64::consts::PI, std::f64::consts::PI));
            *v = lo + (hi - lo) * self.unit();
        }
        q
    }
}

/// Mean microseconds per call of `f`, after a warm-up of [`WARMUP`] calls.
fn mean_us(calls: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..WARMUP {
        f(i);
    }
    let t0 = Instant::now();
    for i in 0..calls {
        f(i);
    }
    t0.elapsed().as_secs_f64() * 1e6 / calls as f64
}

/// Times `ik`, `ik_nearest`, `dynamics` and `inverse_dynamics` for arm `A` and
/// prints one `PERF | name | DOF | ik_all | ik_nearest | dyn | inv_dyn | ik_hit% | nearest_hit% |` line (µs).
pub fn measure<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(name: &str) {
    let arm = Arm::<A, DOF, MAX>::new();
    let policy = TolerancePolicy::DEFAULT;
    let mut rng = Rng(0x5EED_0000_0000_0001);

    let n = WARMUP + IK_CALLS.max(DYN_CALLS);
    let qs: Vec<[f64; DOF]> = (0..n).map(|_| rng.joints::<A, DOF, MAX>()).collect();
    let qds: Vec<[f64; DOF]> = (0..n).map(|_| rng.joints::<A, DOF, MAX>()).collect();
    let qdds: Vec<[f64; DOF]> = (0..n).map(|_| rng.joints::<A, DOF, MAX>()).collect();
    let poses: Vec<_> = qs.iter().map(|q| arm.fk(q)).collect();

    // Solve rates over the timed targets. Misses are reported, not asserted:
    // random in-limit poses are not guaranteed reachable under the solver's limits.
    let mut ik_hits = 0;
    let mut nearest_hits = 0;
    for (q, pose) in qs.iter().zip(&poses).take(IK_CALLS) {
        let all = arm.ik(pose, None, None, true, true, policy, REFINEMENT_ITERS);
        ik_hits += all.iter().any(Option::is_some) as usize;
        nearest_hits += arm.ik_nearest(pose, q, policy).is_some() as usize;
    }
    for i in 0..DYN_CALLS {
        assert!(arm.dynamics(&qs[i], &qds[i]).is_some(), "{name}: no dynamics");
    }

    let ik_all = mean_us(IK_CALLS, |i| {
        black_box(arm.ik(&poses[i], None, None, true, true, policy, REFINEMENT_ITERS));
    });
    let ik_nearest = mean_us(IK_CALLS, |i| {
        black_box(arm.ik_nearest(&poses[i], &qs[i], policy));
    });
    let dynamics = mean_us(DYN_CALLS, |i| {
        black_box(arm.dynamics(&qs[i], &qds[i]));
    });
    let inverse_dynamics = mean_us(DYN_CALLS, |i| {
        black_box(arm.inverse_dynamics(&qs[i], &qds[i], &qdds[i]));
    });

    let pct = |hits: usize| 100.0 * hits as f64 / IK_CALLS as f64;
    println!(
        "PERF | {name} | {DOF} | {ik_all:.2} | {ik_nearest:.2} | {dynamics:.2} | {inverse_dynamics:.2} | {:.1} | {:.1} |",
        pct(ik_hits),
        pct(nearest_hits)
    );
}
