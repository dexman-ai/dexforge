//! Shared release-mode wall-clock timing of IK and dynamics for any [`ArmModel`].
//!
//! Not a crate on its own: each OEM crate includes this file from its
//! `tests/perf.rs` with `#[path]` and calls [`measure`] once per arm.

use std::hint::black_box;
use std::time::Instant;

use dexhal::math::ssik_core::wrap_to_pi;
use dexhal::math::{dls_ik, within_limits, Arm, ArmModel, IkKind, Pose, TolerancePolicy, Transform, DLS_FK_ATOL, DLS_MAX_ITERS};

const WARMUP: usize = 200;
const IK_CALLS: usize = 2_000;
const DYN_CALLS: usize = 50_000;
const REFINEMENT_ITERS: usize = 15;
const SEED_JITTER: f64 = 0.05;
/// Slowest targets re-timed for the worst-case column (see [`worst_seeds_us`]).
const WORST_SEEDS: usize = 10;
/// Back-to-back repeats of each re-timed target.
const WORST_RERUNS: usize = 100;

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

/// Worst-case targets of `f` over `calls` calls. One timed pass ranks every call
/// by its single-shot time; the [`WORST_SEEDS`] slowest, plus any `forced`
/// indices, are each re-timed as the mean of [`WORST_RERUNS`] back-to-back calls.
/// Returns `(index, single_pass_us, rerun_mean_us)` for those, sorted by
/// `rerun_mean_us` descending.
fn worst_seeds_us(calls: usize, forced: &[usize], f: impl Fn(usize)) -> Vec<(usize, f64, f64)> {
    for i in 0..WARMUP {
        f(i);
    }
    let mut ranked: Vec<(f64, usize)> = (0..calls)
        .map(|i| {
            let t0 = Instant::now();
            f(i);
            (t0.elapsed().as_secs_f64() * 1e6, i)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut picked: Vec<usize> = ranked.iter().take(WORST_SEEDS).map(|r| r.1).collect();
    for &i in forced {
        assert!(i < calls, "forced seed {i} outside {calls} targets");
        if !picked.contains(&i) {
            picked.push(i);
        }
    }
    let mut worst: Vec<(usize, f64, f64)> = picked
        .into_iter()
        .map(|i| {
            let pass = ranked.iter().find(|r| r.1 == i).map_or(0.0, |r| r.0);
            (i, pass, mean_us(WORST_RERUNS, |_| f(i)))
        })
        .collect();
    worst.sort_by(|a, b| b.2.total_cmp(&a.2));
    worst
}

/// Worst-case microseconds per call of `f`, as reported by [`worst_seeds_us`].
fn worst_avg_us(calls: usize, f: impl Fn(usize)) -> f64 {
    worst_seeds_us(calls, &[], f).first().map_or(0.0, |w| w.2)
}

/// Homogeneous pose → [`Transform`] without validation (poses here come from FK).
fn to_transform(p: &Pose) -> Transform {
    let mut t = Transform {
        r: [[0.0; 3]; 3],
        t: [0.0; 3],
    };
    for i in 0..3 {
        for j in 0..3 {
            t.r[i][j] = p[(i, j)];
        }
        t.t[i] = p[(i, 3)];
    }
    t
}

struct Timings {
    ik_all: f64,
    ik_all_worst: f64,
    ik_one: f64,
    ik_one_worst: f64,
    dls: f64,
    dls_worst: f64,
    dynamics: f64,
    inverse_dynamics: f64,
    ik_hits: usize,
    one_hits: usize,
    dls_hits: usize,
}

/// Times `ik_all`, `ik_one` (`IkKind::Fast`), `dls_ik`, `dynamics` and `inverse_dynamics` for
/// arm `A` and prints one
/// `PERF | name | DOF | ik_all | ik_one | dls_ik | dyn | inv_dyn | ik_hit% | one_hit% | dls_hit% |` line (µs).
/// `ik_all`, `ik_one` and `dls_ik` are seeded from the same jittered tracking
/// configuration and use the production acceptance checks. The solver cells
/// read `mean (max)`, where max is the [`worst_avg_us`] average.
#[allow(dead_code)] // called by the OEM perf wrappers
pub fn measure<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(name: &str) {
    let t = run::<A, DOF, MAX>(name);
    let pct = |hits: usize| 100.0 * hits as f64 / IK_CALLS as f64;
    println!(
        "PERF | {name} | {DOF} | {:.2} ({:.2}) | {:.2} ({:.2}) | {:.2} ({:.2}) | {:.2} | {:.2} | {:.1} | {:.1} | {:.1} |",
        t.ik_all,
        t.ik_all_worst,
        t.ik_one,
        t.ik_one_worst,
        t.dls,
        t.dls_worst,
        t.dynamics,
        t.inverse_dynamics,
        pct(t.ik_hits),
        pct(t.one_hits),
        pct(t.dls_hits)
    );
}

struct Targets<const DOF: usize> {
    qs: Vec<[f64; DOF]>,
    qds: Vec<[f64; DOF]>,
    qdds: Vec<[f64; DOF]>,
    seeds: Vec<[f64; DOF]>,
    poses: Vec<Pose>,
}

/// Fixed-seed targets (FK of random joints) and tracking seeds shared by every
/// timer. The first `k` entries are the same for any `k` up to the sample count.
fn targets<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(arm: &Arm<A, DOF, MAX>) -> Targets<DOF> {
    let mut rng = Rng(0x5EED_0000_0000_0001);
    let n = WARMUP + IK_CALLS.max(DYN_CALLS);
    let qs: Vec<[f64; DOF]> = (0..n).map(|_| rng.joints::<A, DOF, MAX>()).collect();
    let qds: Vec<[f64; DOF]> = (0..n).map(|_| rng.joints::<A, DOF, MAX>()).collect();
    let qdds: Vec<[f64; DOF]> = (0..n).map(|_| rng.joints::<A, DOF, MAX>()).collect();
    // Tracking seeds: each joint perturbed by at most ±SEED_JITTER from the
    // configuration that produced the target, then clamped into its joint
    // limits (a robot never reports an out-of-limit configuration). Seeding at
    // the exact solution would let DLS converge in zero iterations and time
    // only one FK check.
    let seeds: Vec<[f64; DOF]> = qs
        .iter()
        .map(|q| {
            std::array::from_fn(|j| {
                let s = q[j] + SEED_JITTER * (2.0 * rng.unit() - 1.0);
                A::JOINT_LIMITS[j].map_or(s, |(lo, hi)| s.clamp(lo, hi))
            })
        })
        .collect();
    let poses: Vec<Pose> = qs.iter().map(|q| arm.fk(q)).collect();
    Targets {
        qs,
        qds,
        qdds,
        seeds,
        poses,
    }
}

fn run<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(name: &str) -> Timings {
    let arm = Arm::<A, DOF, MAX>::new();
    let policy = TolerancePolicy::DEFAULT;
    let Targets {
        qs,
        qds,
        qdds,
        seeds,
        poses,
    } = targets::<A, DOF, MAX>(&arm);

    // Solve rates over the timed targets. Misses are reported, not asserted:
    // random in-limit poses are not guaranteed reachable under the solver's limits.
    let mut ik_hits = 0;
    let mut one_hits = 0;
    let mut dls_hits = 0;
    for (seed, pose) in seeds.iter().zip(&poses).take(IK_CALLS) {
        let all = arm.ik_all(pose, Some(seed), true, true, policy, REFINEMENT_ITERS);
        ik_hits += all.iter().any(Option::is_some) as usize;
        one_hits += arm.ik_one(pose, seed, IkKind::Fast, policy).is_some() as usize;
        dls_hits += dls_nearest(&arm, pose, seed).is_some() as usize;
    }
    for i in 0..DYN_CALLS {
        assert!(arm.dynamics(&qs[i], &qds[i]).is_some(), "{name}: no dynamics");
    }

    let ik_all = mean_us(IK_CALLS, |i| {
        black_box(arm.ik_all(&poses[i], Some(&seeds[i]), true, true, policy, REFINEMENT_ITERS));
    });
    let ik_all_worst = worst_avg_us(IK_CALLS, |i| {
        black_box(arm.ik_all(&poses[i], Some(&seeds[i]), true, true, policy, REFINEMENT_ITERS));
    });
    let ik_one = mean_us(IK_CALLS, |i| {
        black_box(arm.ik_one(&poses[i], &seeds[i], IkKind::Fast, policy));
    });
    let ik_one_worst = worst_avg_us(IK_CALLS, |i| {
        black_box(arm.ik_one(&poses[i], &seeds[i], IkKind::Fast, policy));
    });
    let dls = mean_us(IK_CALLS, |i| {
        black_box(dls_nearest(&arm, &poses[i], &seeds[i]));
    });
    let dls_worst = worst_avg_us(IK_CALLS, |i| {
        black_box(dls_nearest(&arm, &poses[i], &seeds[i]));
    });
    let dynamics = mean_us(DYN_CALLS, |i| {
        black_box(arm.dynamics(&qs[i], &qds[i]));
    });
    let inverse_dynamics = mean_us(DYN_CALLS, |i| {
        black_box(arm.inverse_dynamics(&qs[i], &qds[i], &qdds[i]));
    });

    Timings {
        ik_all,
        ik_all_worst,
        ik_one,
        ik_one_worst,
        dls,
        dls_worst,
        dynamics,
        inverse_dynamics,
        ik_hits,
        one_hits,
        dls_hits,
    }
}


/// [`dls_ik`] refinement of the arm's TCP pose, starting from `seed`, accepted
/// with the same checks as production `Arm::ik_one`: the residual gate
/// (`DLS_FK_ATOL`, equal to the policy's numerical tolerance), the seed-relative
/// 2π wrap, the per-joint step limit and the joint limits.
fn dls_nearest<A: ArmModel<DOF, MAX>, const DOF: usize, const MAX: usize>(
    arm: &Arm<A, DOF, MAX>,
    pose: &Pose,
    seed: &[f64; DOF],
) -> Option<[f64; DOF]> {
    let mut q = dls_ik(
        &to_transform(pose),
        seed,
        |q| to_transform(&arm.fk(q)),
        |q| arm.residual_jacobian(q),
        DLS_MAX_ITERS,
        DLS_FK_ATOL,
    )?;
    for i in 0..DOF {
        let near = seed[i] + wrap_to_pi(q[i] - seed[i]);
        if A::JOINT_LIMITS[i].is_none_or(|(lo, hi)| near >= lo && near <= hi) {
            q[i] = near;
        }
        let step = (q[i] - seed[i]).abs();
        if step.is_nan() || step > arm.ik_max_joint_step() {
            return None;
        }
    }
    within_limits(&q, &A::JOINT_LIMITS).then_some(q)
}
