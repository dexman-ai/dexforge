//! Single-solution inverse kinematics shared by every driver.
//!
//! [`six_dof_one`] and [`seven_dof_one`] are the bodies of
//! [`ArmModel::flange_one`]. They use only the nominal flange primitives
//! (`flange_fk`, `flange_jacobian`) and the driver's own candidate source, so
//! no per-arm finite differences are needed.
//!
//! * 6-DOF `Fast`: seeded analytic LM first; enumeration only if that fails.
//!   `Near`: enumeration only, candidates refined nearest-to-seed first.
//! * 7-DOF `Fast`: DLS with all joints free (accepted as is), then masked LM
//!   with the lock joint held at the seed value, then the two nearest lock
//!   samples with the lock joint pinned to the sample value.
//!   `Near`: masked LM with the lock joint at the seed value (short budget),
//!   then the two nearest samples with the lock joint set to the seed value,
//!   then masked DLS from the seed with the lock joint at the seed value (full
//!   [`DLS_MAX_ITERS`] budget). The lock joint never leaves the seed value.
//!   Both kinds return the first accepted solution (samples and candidates are
//!   visited nearest-first), so `Near` is nearest only approximately, by stage
//!   order.
//!
//! Every proposed solution passes `accept` after polishing: each joint is
//! unwrapped to the 2π turn nearest the seed (when in limits) and the candidate
//! is discarded if any joint is more than [`IK_DEFAULT_MAX_JOINT_STEP`] (π/2)
//! from the seed, the 7-DOF lock joint is more than [`IK_LOCK_MAX_STEP`] (π/8)
//! from the seed, or it is outside `JOINT_LIMITS` or off target. The lock bound
//! only bites in `Fast`, where the lock joint is free (DLS) or pinned to a
//! sample; `Near` holds it at the seed value.

use crate::math::arm::IK_DEFAULT_MAX_JOINT_STEP;
use crate::math::chain::transform_of;
use crate::math::closed_form::{pose_residual, refine6, rotation_of};
use crate::math::kinematics::{
    dls_ik, lm_refine_masked_analytic, mat4_frob_diff, residual_columns, Transform, DLS_MAX_ITERS,
};
use crate::math::model::{ArmModel, IkKind, Pose};
use crate::math::postprocess::{within_limits, wrap_distance, wrap_to_limits};
use crate::math::ssik_core::wrap_to_pi;
use crate::math::tolerances::TolerancePolicy;

/// Capacity of the candidate buffer a 6-DOF driver hands to [`six_dof_one`].
pub const CANDIDATES: usize = 16;

/// Largest distance (rad) the 7-DOF lock (redundant) joint of a proposed
/// solution may lie from the seed. Every other joint is bounded by
/// [`IK_DEFAULT_MAX_JOINT_STEP`].
pub const IK_LOCK_MAX_STEP: f64 = core::f64::consts::FRAC_PI_8;

/// `true` if every joint of `q` lies within [`IK_DEFAULT_MAX_JOINT_STEP`] of
/// `seed` modulo 2π. Driver tests use it to predict which `flange_all`
/// solution `flange_one` returns. The acceptance gate itself additionally
/// requires the 2π turn nearest the seed to be inside the joint limits.
pub fn within_max_joint_step<const N: usize>(q: &[f64; N], seed: &[f64; N]) -> bool {
    q.iter()
        .zip(seed.iter())
        .all(|(a, b)| wrap_to_pi(a - b).abs() <= IK_DEFAULT_MAX_JOINT_STEP)
}

/// Acceptance gate applied to every proposed solution, after any polish.
///
/// `q` must be finite. Each joint is wrapped by `±2π` into the joint limits,
/// then moved to the 2π turn nearest `seed` when the limits allow. The result
/// must lie inside the limits, have every joint within [`IK_DEFAULT_MAX_JOINT_STEP`] of
/// `seed` (joint `lock`, if any, within [`IK_LOCK_MAX_STEP`]), and reach
/// `target` within `policy.subproblem_numerical`. Returns the unwrapped
/// configuration.
fn accept<A: ArmModel<N, MAX>, const N: usize, const MAX: usize>(
    q: &[f64; N],
    seed: &[f64; N],
    lock: Option<usize>,
    target: &Pose,
    policy: TolerancePolicy,
) -> Option<[f64; N]> {
    if !q.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut wrapped = [*q];
    wrap_to_limits(&mut wrapped, 1, &A::JOINT_LIMITS);
    let mut q = wrapped[0];
    for i in 0..N {
        let near = seed[i] + wrap_to_pi(q[i] - seed[i]);
        if A::JOINT_LIMITS[i].is_none_or(|(lo, hi)| near >= lo && near <= hi) {
            q[i] = near;
        }
        let step = if lock == Some(i) { IK_LOCK_MAX_STEP } else { IK_DEFAULT_MAX_JOINT_STEP };
        if (q[i] - seed[i]).abs() > step {
            return None;
        }
    }
    if !within_limits(&q, &A::JOINT_LIMITS) {
        return None;
    }
    (mat4_frob_diff(&A::flange_fk(&q), target) <= policy.subproblem_numerical).then_some(q)
}

// ---------------------------------------------------------------------------
// 6-DOF
// ---------------------------------------------------------------------------

/// Nearest accepted candidate to `seed`. Candidates are ranked by their raw
/// wrap-to-π distance (earlier index wins ties) and tried in that order. A
/// candidate that does not already close is refined first.
fn nearest_accepted<A: ArmModel<6, MAX>, const MAX: usize>(
    cands: &[[f64; 6]],
    seed: &[f64; 6],
    target: &Pose,
    policy: TolerancePolicy,
    max_iters: usize,
) -> Option<[f64; 6]> {
    let mut used = [false; CANDIDATES];
    for _ in 0..cands.len() {
        let mut best: Option<(f64, usize)> = None;
        for (i, q) in cands.iter().enumerate() {
            if used[i] || !q.iter().all(|v| v.is_finite()) {
                continue;
            }
            let d = wrap_distance(q, seed);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, i));
            }
        }
        let Some((_, i)) = best else { break };
        used[i] = true;
        let q = &cands[i];
        let closed = mat4_frob_diff(&A::flange_fk(q), target) <= policy.subproblem_numerical;
        let refined = if closed {
            Some(*q)
        } else {
            refine6::<A, MAX>(q, target, max_iters, policy.subproblem_numerical)
        };
        if let Some(q) = refined.and_then(|r| accept::<A, 6, MAX>(&r, seed, None, target, policy)) {
            return Some(q);
        }
    }
    None
}

/// One 6-DOF solution (see the module docs for `kind`). `enumerate` supplies
/// the driver's raw closed-form candidates and is called only when needed.
/// Every proposed solution passes the acceptance gate after its polish; a
/// rejected one moves on to the next.
pub fn six_dof_one<A: ArmModel<6, MAX>, const MAX: usize>(
    target: &Pose,
    q_seed: &[f64; 6],
    kind: IkKind,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
    enumerate: impl FnOnce() -> ([[f64; 6]; CANDIDATES], usize),
) -> Option<[f64; 6]> {
    if kind == IkKind::Fast {
        if let Some(q) = refine6::<A, MAX>(
            q_seed,
            target,
            refinement_max_iters,
            policy.subproblem_numerical,
        )
        .and_then(|q| accept::<A, 6, MAX>(&q, q_seed, None, target, policy))
        {
            return Some(q);
        }
    }
    let (cands, n) = enumerate();
    nearest_accepted::<A, MAX>(&cands[..n], q_seed, target, policy, refinement_max_iters)
}

// ---------------------------------------------------------------------------
// 7-DOF (one joint locked, the rest solved by the driver's 6R sample solver)
// ---------------------------------------------------------------------------

/// Joint indices other than `lock`, in order.
fn free_joints(lock: usize) -> [usize; 6] {
    core::array::from_fn(|i| if i < lock { i } else { i + 1 })
}

/// DLS on all seven joints from `start` (the `Fast` first stage).
fn dls_free<A: ArmModel<7, MAX>, const MAX: usize>(
    goal: &Transform,
    start: &[f64; 7],
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 7]> {
    dls_ik(
        goal,
        start,
        |q| transform_of(&A::flange_fk(q)),
        |q| residual_columns(&rotation_of(&A::flange_fk(q)), &A::flange_jacobian(q)),
        max_iters,
        fk_atol,
    )
}

/// Analytic masked LM from `start` with `lock` pinned to `lock_value`, then
/// the acceptance gate against `seed`.
#[allow(clippy::too_many_arguments)]
fn lock_polish<A: ArmModel<7, MAX>, const MAX: usize>(
    start: &[f64; 7],
    lock: usize,
    lock_value: f64,
    seed: &[f64; 7],
    target: &Pose,
    policy: TolerancePolicy,
    max_iters: usize,
) -> Option<[f64; 7]> {
    let mut q = *start;
    q[lock] = lock_value;
    let free = free_joints(lock);
    let q = lm_refine_masked_analytic::<7, 6, _, _>(
        &q,
        &free,
        |q| pose_residual(&A::flange_fk(q), target),
        |q| residual_columns(&rotation_of(&A::flange_fk(q)), &A::flange_jacobian(q)),
        max_iters,
    );
    accept::<A, 7, MAX>(&q, seed, Some(lock), target, policy)
}

/// The two lock samples nearest (wrap-to-π) to `to`.
fn two_nearest(samples: &[f64], to: f64) -> [Option<f64>; 2] {
    let d = |v: f64| wrap_to_pi(v - to).abs();
    let mut out: [Option<f64>; 2] = [None; 2];
    for &v in samples {
        if out[0].is_none_or(|b| d(v) < d(b)) {
            out[1] = out[0];
            out[0] = Some(v);
        } else if out[1].is_none_or(|b| d(v) < d(b)) {
            out[1] = Some(v);
        }
    }
    out
}

/// Driver candidates at the two nearest samples (nearest sample first). Finite
/// candidates are visited nearest-to-seed first (after the lock joint is set to
/// `pin(sample)`), polished on the free joints, and the first one that passes
/// the acceptance gate after its polish is returned.
#[allow(clippy::too_many_arguments)]
fn sample_stage<A: ArmModel<7, MAX>, const MAX: usize>(
    sample: &impl Fn(f64) -> [Option<[f64; 7]>; MAX],
    nearest: [Option<f64>; 2],
    lock: usize,
    pin: impl Fn(f64) -> f64,
    seed: &[f64; 7],
    target: &Pose,
    policy: TolerancePolicy,
    max_iters: usize,
) -> Option<[f64; 7]> {
    for v in nearest.into_iter().flatten() {
        let mut cands = sample(v);
        let mut dist = [f64::INFINITY; MAX];
        for (c, d) in cands.iter_mut().zip(dist.iter_mut()) {
            let Some(q) = c else { continue };
            if !q.iter().all(|x| x.is_finite()) {
                *c = None;
                continue;
            }
            q[lock] = pin(v);
            *d = wrap_distance(q, seed);
        }
        loop {
            let mut best: Option<usize> = None;
            for (i, c) in cands.iter().enumerate() {
                if c.is_some() && best.is_none_or(|b| dist[i] < dist[b]) {
                    best = Some(i);
                }
            }
            let Some(i) = best else { break };
            let Some(c) = cands[i].take() else { break };
            if let Some(q) =
                lock_polish::<A, MAX>(&c, lock, pin(v), seed, target, policy, max_iters)
            {
                return Some(q);
            }
        }
    }
    None
}

/// Iterations of the seed-start stages (full-joint DLS and the seed-lock
/// polish). The sample stage uses the caller's `refinement_max_iters`.
pub const SEED_STAGE_ITERS: usize = 20;

/// One 7-DOF solution (see the module docs for `kind`).
///
/// `lock` is the joint held during the seeded stages, `lock_samples` the
/// driver's fixed sample values for it, and `sample(v)` the driver's 6R solve
/// with that joint at `v`. Every stage returns its first solution that passes
/// the acceptance gate after its polish.
#[allow(clippy::too_many_arguments)]
pub fn seven_dof_one<A: ArmModel<7, MAX>, const MAX: usize>(
    target: &Pose,
    q_seed: &[f64; 7],
    kind: IkKind,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
    lock: usize,
    lock_samples: &[f64],
    sample: impl Fn(f64) -> [Option<[f64; 7]>; MAX],
) -> Option<[f64; 7]> {
    let seed_lock = q_seed[lock];
    let nearest = two_nearest(lock_samples, seed_lock);
    let seed_polish = || {
        lock_polish::<A, MAX>(
            q_seed,
            lock,
            seed_lock,
            q_seed,
            target,
            policy,
            SEED_STAGE_ITERS,
        )
    };

    match kind {
        IkKind::Fast => {
            let goal = transform_of(target);
            dls_free::<A, MAX>(&goal, q_seed, SEED_STAGE_ITERS, policy.subproblem_numerical)
                .and_then(|q| accept::<A, 7, MAX>(&q, q_seed, Some(lock), target, policy))
                .or_else(seed_polish)
                .or_else(|| {
                    sample_stage::<A, MAX>(
                        &sample,
                        nearest,
                        lock,
                        |v| v,
                        q_seed,
                        target,
                        policy,
                        refinement_max_iters,
                    )
                })
        }
        // Stage 3 is the damped least-squares solve of `dls_ik` restricted to
        // the free joints. It shares stage 1's damped core and start, so it
        // differs by budget: it only helps seeds that need more than
        // `SEED_STAGE_ITERS` iterations.
        IkKind::Near => seed_polish()
            .or_else(|| {
                sample_stage::<A, MAX>(
                    &sample,
                    nearest,
                    lock,
                    |_| seed_lock,
                    q_seed,
                    target,
                    policy,
                    refinement_max_iters,
                )
            })
            .or_else(|| {
                lock_polish::<A, MAX>(
                    q_seed,
                    lock,
                    seed_lock,
                    q_seed,
                    target,
                    policy,
                    DLS_MAX_ITERS,
                )
            }),
    }
}
