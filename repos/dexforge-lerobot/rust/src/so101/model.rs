//! Five-joint **SO-ARM101** kinematics, dynamics and inverse kinematics.
//!
//! The SO-ARM101 arm has five actuated joints (shoulder pan, shoulder lift,
//! elbow flex, wrist flex, wrist roll); the gripper jaw is a separate actuator.
//! The ssik / dexforge pipeline only handles full 6R and 7R chains, so the
//! generated driver [`so101_6r`](crate::drivers::so101_6r) models the arm plus
//! **one zero-offset virtual revolute joint** (`virtual_wrist`, axis
//! perpendicular to the wrist-roll axis) appended after the wrist roll. At
//! `q6 = 0` that 6R chain is *exactly* the physical 5R arm.
//!
//! This module hides the virtual joint:
//!
//! * [`fk`], [`jacobian`] and the dynamics are the 6R functions evaluated at
//!   `q6 = 0`, restricted to the five real joints. Because the virtual joint is
//!   held stationary (`q6 = qd6 = qdd6 = 0`), taking the leading 5x5 block of
//!   `M(q)` and the first five entries of `g`, `C qd` and `tau` is exact.
//! * [`solve`] runs the 6R closed form, pins each candidate's virtual joint to
//!   zero and polishes the five real joints with a bounded masked
//!   Levenberg-Marquardt step (analytic Jacobian), then keeps a candidate only
//!   if the FK residual closes within `policy.subproblem_numerical`. That
//!   closure *is* the reachability test: the arm only reaches poses that a 6R
//!   solution reaches with `q6 = 0`. Like every other arm, the strictness is the
//!   tolerance policy's: the default (`1e-5`) accepts exact poses, and a caller
//!   that can live with millimetres passes a looser `subproblem_numerical`.
//! * [`solve_nearest`] is the nearest-reachable counterpart. The 5R arm cannot
//!   reach arbitrary SE(3) poses (a target from a camera or a teleop device is
//!   almost never exactly reachable), so it returns the closest reachable
//!   configuration together with its position and orientation error, and the
//!   caller decides whether that is close enough. It is a separate function on
//!   purpose: `ArmModel::flange_solve` promises solutions that close, and the
//!   exact functions here keep that promise.
//!
//! Like the rest of the crate this is `#![no_std]`, allocation-free, and every
//! loop has a hard iteration ceiling.

use core::f64::consts::PI;

use dexhal::math::kinematics::{lm_refine_masked_analytic, residual_columns};
use dexhal::math::ssik_core::wrap_to_pi;
use dexhal::math::{JointLimit, Pose, TolerancePolicy};

use crate::drivers::so101_6r::{generated_dynamics as dynx, ik_solver as ik6};

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "reference_data.rs"]
mod reference_data;

/// Number of real (actuated, non-gripper) joints.
pub const DOF: usize = 5;

/// Capacity of the IK solution array (inherited from the 6R closed form).
pub const MAX_SOLUTIONS: usize = 16;

/// Base link of the kinematic chain.
pub const BASE_LINK: &str = ik6::BASE_LINK;

/// End-effector link (`gripper_frame_link`).
pub const EE_LINK: &str = ik6::EE_LINK;

/// URDF names of the actuated joints, base to flange.
pub const JOINT_NAMES: [&str; DOF] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
];

/// Default weight of orientation against position in [`solve_nearest`]'s
/// least-squares metric: the rotation-matrix entries of the pose error are
/// multiplied by it, so `0.1` makes 1 rad of orientation error cost about as
/// much as 14 cm of position error, the right order for a 40 cm arm.
pub const DEFAULT_ROT_WEIGHT: f64 = 0.1;

/// Iteration ceiling per starting point that [`solve_nearest`]'s callers use by default. A
/// least-squares fit to a target that is *not* reachable needs more steps than polishing an
/// exact solution.
pub const NEAREST_MAX_ITERS: usize = 60;

/// Joint limits `(lo, hi)` in radians of the five real joints, from the
/// SO-101 URDF (`so101.urdf`).
pub const JOINT_LIMITS: [JointLimit; DOF] = [
    Some((-1.91986, 1.91986)),
    Some((-1.74533, 1.74533)),
    Some((-1.69, 1.69)),
    Some((-1.65806, 1.65806)),
    Some((-2.74385, 2.84121)),
];

/// Indices of the real joints inside the six-joint vector.
const REAL: [usize; 5] = [0, 1, 2, 3, 4];

const TWO_PI: f64 = 2.0 * PI;

#[inline]
fn lift(q: &[f64; 5]) -> [f64; 6] {
    [q[0], q[1], q[2], q[3], q[4], 0.0]
}

#[inline]
fn lift_rate(qd: &[f64; 5]) -> [f64; 6] {
    [qd[0], qd[1], qd[2], qd[3], qd[4], 0.0]
}

// ---------------------------------------------------------------------------
// Kinematics
// ---------------------------------------------------------------------------

/// Forward kinematics: pose of `gripper_frame_link` in `base_link`.
pub fn fk(q: &[f64; 5]) -> Pose {
    ik6::fk(&lift(q))
}

/// Geometric Jacobian `[Jv; Jw]` (6 x 5) of the end-effector frame.
pub fn jacobian(q: &[f64; 5]) -> [[f64; 5]; 6] {
    let j = dynx::jacobian(&lift(q));
    let mut out = [[0.0; 5]; 6];
    for r in 0..6 {
        for c in 0..5 {
            out[r][c] = j[r][c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dynamics (virtual joint held at q6 = 0, qd6 = qdd6 = 0)
// ---------------------------------------------------------------------------

/// Gravity holding torque for a base-frame gravity vector `g`.
pub fn gravity(q: &[f64; 5], g: &[f64; 3]) -> [f64; 5] {
    let t = dynx::gravity(&lift(q), g);
    [t[0], t[1], t[2], t[3], t[4]]
}

/// Joint-space inertia matrix `M(q)` (5 x 5).
pub fn mass_matrix(q: &[f64; 5]) -> [[f64; 5]; 5] {
    let m = dynx::mass_matrix(&lift(q));
    let mut out = [[0.0; 5]; 5];
    for r in 0..5 {
        for c in 0..5 {
            out[r][c] = m[r][c];
        }
    }
    out
}

/// Velocity-product torque `C(q, qd) qd`.
pub fn coriolis(q: &[f64; 5], qd: &[f64; 5]) -> [f64; 5] {
    let t = dynx::coriolis(&lift(q), &lift_rate(qd));
    [t[0], t[1], t[2], t[3], t[4]]
}

/// Feed-forward torque `M qdd + C qd + g`.
pub fn inverse_dynamics(q: &[f64; 5], qd: &[f64; 5], qdd: &[f64; 5], g: &[f64; 3]) -> [f64; 5] {
    let t = dynx::inverse_dynamics(&lift(q), &lift_rate(qd), &lift_rate(qdd), g);
    [t[0], t[1], t[2], t[3], t[4]]
}

// ---------------------------------------------------------------------------
// Inverse kinematics
// ---------------------------------------------------------------------------

/// 12-vector pose residual `[vec(R(q) - R_t); t(q) - t_t]` of the six-joint chain.
fn pose_residual12(q: &[f64; 6], target: &Pose) -> [f64; 12] {
    let tf = ik6::fk(q);
    let mut r = [0.0f64; 12];
    let mut k = 0usize;
    for i in 0..3 {
        for j in 0..3 {
            r[k] = tf[(i, j)] - target[(i, j)];
            k += 1;
        }
    }
    for i in 0..3 {
        r[9 + i] = tf[(i, 3)] - target[(i, 3)];
    }
    r
}

/// Analytic residual columns `d residual / d q_j` of the six-joint chain.
fn residual_jacobian(q: &[f64; 6]) -> [[f64; 12]; 6] {
    let tf = ik6::fk(q);
    let rot = [
        [tf[(0, 0)], tf[(0, 1)], tf[(0, 2)]],
        [tf[(1, 0)], tf[(1, 1)], tf[(1, 2)]],
        [tf[(2, 0)], tf[(2, 1)], tf[(2, 2)]],
    ];
    residual_columns(&rot, &dynx::jacobian(q))
}

fn frob_residual(q: &[f64; 6], target: &Pose) -> f64 {
    let tf = ik6::fk(q);
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = tf[(i, j)] - target[(i, j)];
            s += d * d;
        }
    }
    libm::sqrt(s)
}

fn q_close_wrap(a: &[f64; 5], b: &[f64; 5], tol: f64) -> bool {
    for i in 0..5 {
        if libm::fabs(wrap_to_pi(a[i] - b[i])) > tol {
            return false;
        }
    }
    true
}

fn limits(i: usize) -> (f64, f64) {
    JOINT_LIMITS[i].unwrap_or((f64::NEG_INFINITY, f64::INFINITY))
}

fn wrap_to_limits(q: &[f64; 5]) -> [f64; 5] {
    let mut out = *q;
    for (i, v) in out.iter_mut().enumerate() {
        let (lo, hi) = limits(i);
        if *v < lo {
            let w = *v + TWO_PI;
            if w <= hi {
                *v = w;
            }
        } else if *v > hi {
            let w = *v - TWO_PI;
            if w >= lo {
                *v = w;
            }
        }
    }
    out
}

/// Each joint moved to the nearest value inside its limits.
fn clamp_to_limits(q: &[f64; 5]) -> [f64; 5] {
    let mut out = *q;
    for (i, v) in out.iter_mut().enumerate() {
        let (lo, hi) = limits(i);
        *v = v.clamp(lo, hi);
    }
    out
}

fn in_limits(q: &[f64; 5]) -> bool {
    for (i, v) in q.iter().enumerate() {
        let (lo, hi) = limits(i);
        if *v < lo || *v > hi {
            return false;
        }
    }
    true
}

fn seed_dist(q: &[f64; 5], seed: &[f64; 5]) -> f64 {
    let mut s = 0.0;
    for i in 0..5 {
        let d = wrap_to_pi(q[i] - seed[i]);
        s += d * d;
    }
    s
}

/// Inverse kinematics for the five real joints.
///
/// * `t` - 4x4 target pose of `gripper_frame_link` in `base_link`.
/// * `max_solutions` - cap on returned solutions (after dedup and limits).
/// * `q_seed` - when `Some`, solutions are sorted nearest-first (wrap-to-pi).
/// * `respect_limits` - drop solutions outside [`JOINT_LIMITS`] (after a
///   `+-2*pi` wrap-into-limits attempt).
/// * `policy` - FK closure (`subproblem_numerical`) and dedup tolerances.
/// * `refinement_max_iters` - cap on masked LM iterations per candidate.
///
/// Returns every distinct solution whose FK closes to `policy.subproblem_numerical`
/// (Frobenius norm of the 4x4 difference, so about metres of position error and
/// radians of orientation error); poses outside that tolerance of the arm's
/// reachable set return all-`None`. Use [`solve_nearest`] to get the closest
/// reachable configuration instead.
pub fn solve(
    t: &Pose,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 5]>,
    respect_limits: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 5]>; MAX_SOLUTIONS] {
    let fk_atol = policy.subproblem_numerical;
    let dedup_atol = policy.subproblem_dedup;

    // Raw 6R closed form. No limit filter: the virtual joint's range is irrelevant.
    let raw = ik6::solve(t, None, None, false, false, policy, refinement_max_iters);

    let mut out: [Option<[f64; 5]>; MAX_SOLUTIONS] = [None; MAX_SOLUTIONS];
    let mut count = 0usize;

    // A candidate whose virtual joint is far from zero is a branch of the 6R problem that is not
    // on the 5R arm's reachable set. The bound scales with the policy so that a looser
    // `subproblem_numerical` admits targets farther from that set (a target `d` off the set gives a
    // candidate with q6 of order `d`); the default policy keeps the 1e-3 rad of an exact solve.
    let q6_bound = f64::max(1.0e-3, 100.0 * fk_atol);
    let mut out_res = [f64::INFINITY; MAX_SOLUTIONS];

    for cand in raw.iter().flatten() {
        if libm::fabs(wrap_to_pi(cand[5])) > q6_bound {
            continue;
        }
        // Pin q6 = 0 and polish the five real joints onto the pose. The FK closure below, not
        // the bound above, decides whether it is a solution.
        let start = [cand[0], cand[1], cand[2], cand[3], cand[4], 0.0];
        let refined = lm_refine_masked_analytic::<6, 5, _, _>(
            &start,
            &REAL,
            |q| pose_residual12(q, t),
            residual_jacobian,
            refinement_max_iters,
        );
        if !refined.iter().all(|v| v.is_finite()) {
            continue;
        }
        let pinned = [
            refined[0], refined[1], refined[2], refined[3], refined[4], 0.0,
        ];
        let residual = frob_residual(&pinned, t);
        if residual > fk_atol {
            continue;
        }
        let q5 = [pinned[0], pinned[1], pinned[2], pinned[3], pinned[4]];

        // De-duplicate modulo 2*pi, keeping the lowest residual (as the ssik reference does).
        let mut dup = None;
        for (k, slot) in out.iter().take(count).enumerate() {
            if let Some(existing) = slot {
                if q_close_wrap(&q5, existing, dedup_atol) {
                    dup = Some(k);
                    break;
                }
            }
        }
        match dup {
            Some(k) => {
                if residual < out_res[k] {
                    out[k] = Some(q5);
                    out_res[k] = residual;
                }
            }
            None if count < MAX_SOLUTIONS => {
                out[count] = Some(q5);
                out_res[count] = residual;
                count += 1;
            }
            None => {}
        }
    }

    if respect_limits {
        let mut filtered: [Option<[f64; 5]>; MAX_SOLUTIONS] = [None; MAX_SOLUTIONS];
        let mut n = 0usize;
        for s in out.iter().take(count).flatten() {
            let q = wrap_to_limits(s);
            if in_limits(&q) && n < MAX_SOLUTIONS {
                filtered[n] = Some(q);
                n += 1;
            }
        }
        out = filtered;
        count = n;
    }

    if let Some(seed) = q_seed {
        for i in 1..count {
            let mut j = i;
            while j > 0 {
                let prev = out[j - 1].unwrap();
                let curr = out[j].unwrap();
                if seed_dist(&prev, seed) > seed_dist(&curr, seed) {
                    out.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }

    if let Some(max) = max_solutions {
        let cap = max.min(count);
        for slot in out.iter_mut().take(count).skip(cap) {
            *slot = None;
        }
    }

    out
}

/// Number of extra starting configurations [`solve_nearest`] tries when no exact solution was found.
pub const FALLBACK_STARTS: usize = 24;

/// A fit whose weighted error is at or below this counts as exact (no fallback starts are tried).
const EXACT_SCORE: f64 = 1.0e-6;

/// Roberts' R5 low-discrepancy sequence: `alpha_j = frac(phi^-(j+1))` with `phi` the positive
/// root of `x^6 = x + 1`, giving well-spread points in five dimensions for any count.
const R5_ALPHA: [f64; 5] = [
    0.8812714616335696,
    0.7766393890897683,
    0.6844301295853427,
    0.6031687406857283,
    0.5315553977157914,
];

/// The `k`-th fallback starting configuration: a deterministic point spread over the joint limits.
fn fallback_start(k: usize) -> [f64; 6] {
    let mut q = [0.0f64; 6];
    for (i, v) in q.iter_mut().take(5).enumerate() {
        let (lo, hi) = limits(i);
        let u = (0.5 + (k as f64 + 1.0) * R5_ALPHA[i]) % 1.0;
        *v = lo + u * (hi - lo);
    }
    q
}

/// The closest reachable configuration to a target, and how far it falls short.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    /// The five joint angles.
    pub q: [f64; 5],
    /// Distance between the reached and the target position, in metres.
    pub position_error: f64,
    /// Angle of the rotation between the reached and the target orientation, in radians.
    pub orientation_error: f64,
}

/// Pose residual with the rotation-matrix entries scaled by `w` (see [`DEFAULT_ROT_WEIGHT`]).
fn weighted_residual12(q: &[f64; 6], target: &Pose, w: f64) -> [f64; 12] {
    let mut r = pose_residual12(q, target);
    for v in r.iter_mut().take(9) {
        *v *= w;
    }
    r
}

/// Analytic columns of [`weighted_residual12`].
fn weighted_residual_jacobian(q: &[f64; 6], w: f64) -> [[f64; 12]; 6] {
    let mut cols = residual_jacobian(q);
    for col in cols.iter_mut() {
        for v in col.iter_mut().take(9) {
            *v *= w;
        }
    }
    cols
}

fn norm12(r: &[f64; 12]) -> f64 {
    let mut s = 0.0;
    for v in r {
        s += v * v;
    }
    libm::sqrt(s)
}

/// `(position error in m, orientation error in rad)` of `q` against `target`.
fn pose_errors(q: &[f64; 6], target: &Pose) -> (f64, f64) {
    let tf = ik6::fk(q);
    let mut dp = 0.0;
    for i in 0..3 {
        let d = tf[(i, 3)] - target[(i, 3)];
        dp += d * d;
    }
    // trace(R^T R_target) = 1 + 2 cos(angle)
    let mut tr = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            tr += tf[(i, j)] * target[(i, j)];
        }
    }
    let c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
    (libm::sqrt(dp), libm::acos(c))
}

/// The reachable configuration closest to `t`, for targets that are *not* exactly reachable.
///
/// A five-joint arm reaches a 5-D subset of SE(3), and a target from a camera, a teleop device
/// or a hand-typed pose is almost never in it. This minimises the weighted pose error
/// `|p(q) - p_t|^2 + rot_weight^2 |R(q) - R_t|_F^2` over the five joints and returns the
/// minimiser with its position and orientation error, so the caller decides whether it is close
/// enough. (`rot_weight` is in metres per unit of rotation-matrix error; see
/// [`DEFAULT_ROT_WEIGHT`].)
///
/// Every start is polished with the same bounded, analytic-Jacobian Levenberg-Marquardt step as
/// [`solve`]: the 6R closed-form candidates of `t` (one per branch, virtual joint pinned to
/// zero), `q_seed` when given, and the zero configuration if there is nothing else. Candidates
/// that end at the same optimum (within `1e-6` relative) are different *branches* of it; the one
/// nearest `q_seed` wins, with each joint moved to the `2*pi` turn nearest the seed when the
/// limits allow, so a continuous target path gives a continuous joint path. When the target
/// is exactly reachable the result is the exact solution on that branch, with zero error.
///
/// With `respect_limits`, an optimum outside the joint limits is moved into them (wrapped by
/// `2*pi` where that helps, then clamped), and the returned errors are those of the clamped
/// configuration, so they stay honest. The fit itself is unconstrained: it does not search along a
/// limit, so near a limit the result is a good configuration, not necessarily the constrained
/// optimum.
///
/// Returns `None` when no start produces a finite optimum (for example a NaN target).
/// A target nowhere near the arm still returns the closest configuration, with a large error.
pub fn solve_nearest(
    t: &Pose,
    q_seed: Option<&[f64; 5]>,
    respect_limits: bool,
    rot_weight: f64,
    policy: TolerancePolicy,
    max_iters: usize,
) -> Option<Nearest> {
    const CAP: usize = MAX_SOLUTIONS + 2 + FALLBACK_STARTS;
    let raw = ik6::solve(t, None, None, false, false, policy, max_iters);

    // Starting points: seed, then the closed-form candidates; the zero configuration if empty.
    let mut starts = [[0.0f64; 6]; CAP];
    let mut n_starts = 0usize;
    if let Some(seed) = q_seed {
        starts[n_starts] = lift(seed);
        n_starts += 1;
    }
    for cand in raw.iter().flatten() {
        if n_starts < CAP {
            starts[n_starts] = [cand[0], cand[1], cand[2], cand[3], cand[4], 0.0];
            n_starts += 1;
        }
    }
    if n_starts == 0 {
        n_starts = 1;
    }

    let mut cand_q = [[0.0f64; 5]; CAP];
    let mut cand_score = [0.0f64; CAP];
    let mut n_cand = 0usize;
    let mut best = f64::INFINITY;

    // Polish one start; record the optimum (limits applied) and its weighted score.
    let mut polish = |start: &[f64; 6], n_cand: &mut usize, best: &mut f64| {
        let refined = lm_refine_masked_analytic::<6, 5, _, _>(
            start,
            &REAL,
            |q| weighted_residual12(q, t, rot_weight),
            |q| weighted_residual_jacobian(q, rot_weight),
            max_iters,
        );
        let q5 = [refined[0], refined[1], refined[2], refined[3], refined[4]];
        if !q5.iter().all(|v| v.is_finite()) {
            return;
        }
        let q5 = if respect_limits {
            clamp_to_limits(&wrap_to_limits(&q5))
        } else {
            q5
        };
        let score = norm12(&weighted_residual12(&lift(&q5), t, rot_weight));
        if !score.is_finite() {
            return;
        }
        cand_q[*n_cand] = q5;
        cand_score[*n_cand] = score;
        *n_cand += 1;
        if score < *best {
            *best = score;
        }
    };

    for start in starts.iter().take(n_starts) {
        polish(start, &mut n_cand, &mut best);
    }
    // The closed form is ill-conditioned near the shoulder singularity (the target close to the base
    // axis), where it can miss the right branch, and a fit from a distant seed can then settle in a
    // poor local optimum. Unless an exact solution was found, also try a fixed, well-spread set of
    // starting configurations (a low-discrepancy sequence over the joint limits; deterministic).
    if best > EXACT_SCORE {
        for k in 0..FALLBACK_STARTS {
            polish(&fallback_start(k), &mut n_cand, &mut best);
        }
    }
    if n_cand == 0 {
        return None;
    }

    // Among the candidates at the best optimum, the one nearest the seed (the first without a seed).
    let threshold = best * (1.0 + 1e-6) + 1e-12;
    let mut pick = usize::MAX;
    for i in 0..n_cand {
        if cand_score[i] > threshold {
            continue;
        }
        let better = match q_seed {
            Some(seed) => {
                pick == usize::MAX || seed_dist(&cand_q[i], seed) < seed_dist(&cand_q[pick], seed)
            }
            None => pick == usize::MAX,
        };
        if better {
            pick = i;
        }
    }
    let mut q = cand_q[pick];
    if let Some(seed) = q_seed {
        for (i, v) in q.iter_mut().enumerate() {
            let near = seed[i] + wrap_to_pi(*v - seed[i]);
            let (lo, hi) = limits(i);
            if near >= lo && near <= hi {
                *v = near;
            }
        }
    }
    let (position_error, orientation_error) = pose_errors(&lift(&q), t);
    Some(Nearest {
        q,
        position_error,
        orientation_error,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dexhal::math::Arm;
    use nalgebra::SMatrix;
    use std::vec::Vec;

    use super::super::arm::So101;

    const G: [f64; 3] = [0.0, 0.0, -9.81];
    const POLICY: TolerancePolicy = TolerancePolicy::DEFAULT;

    /// xorshift64* - deterministic, no external dependencies.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let v = self.0.wrapping_mul(0x2545F4914F6CDD1D);
            (v >> 11) as f64 / (1u64 << 53) as f64
        }
        /// A configuration inside the joint limits, shrunk toward the middle by `shrink`.
        fn q5(&mut self, shrink: f64) -> [f64; 5] {
            let mut q = [0.0; 5];
            for i in 0..5 {
                let (lo, hi) = limits(i);
                let mid = 0.5 * (lo + hi);
                let half = 0.5 * (hi - lo) * shrink;
                q[i] = mid + half * (2.0 * self.next() - 1.0);
            }
            q
        }
    }

    fn q_eq_mod(a: &[f64; 5], b: &[f64; 5], tol: f64) -> bool {
        (0..5).all(|i| libm::fabs(libm::remainder(a[i] - b[i], TWO_PI)) <= tol)
    }

    use reference_data as rd;

    // ---- Pinocchio ground truth: original, unmodified upstream URDF, jaw locked at 0 ----

    #[test]
    fn fk_matches_pinocchio_on_the_original_urdf() {
        let mut worst = 0.0f64;
        for i in 0..rd::N_CASES {
            let t = fk(&rd::q(i));
            let tref = rd::t(i);
            for r in 0..4 {
                for c in 0..4 {
                    worst = worst.max(libm::fabs(t[(r, c)] - tref[r * 4 + c]));
                }
            }
        }
        std::println!("max |T_dexforge - T_pinocchio(original URDF)| = {worst:.3e}");
        // Snapping the URDF's rounding noise moves FK by ~3e-6 m and ~2e-5 in rotation entries.
        assert!(
            worst < 5e-5,
            "FK deviates from Pinocchio on the original URDF: {worst}"
        );
    }

    #[test]
    fn dynamics_match_pinocchio_on_the_original_urdf() {
        let (mut wg, mut wm, mut wc, mut wt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..rd::N_CASES {
            let (q, qd, qdd) = (rd::q(i), rd::qd(i), rd::qdd(i));
            let g = gravity(&q, &G);
            let m = mass_matrix(&q);
            let c = coriolis(&q, &qd);
            let tau = inverse_dynamics(&q, &qd, &qdd, &G);
            let (gr, mr, cr, tr) = (
                rd::gravity_torque(i),
                rd::mass(i),
                rd::coriolis_torque(i),
                rd::tau(i),
            );
            for a in 0..5 {
                wg = wg.max(libm::fabs(g[a] - gr[a]));
                wc = wc.max(libm::fabs(c[a] - cr[a]));
                wt = wt.max(libm::fabs(tau[a] - tr[a]));
                for b in 0..5 {
                    wm = wm.max(libm::fabs(m[a][b] - mr[a * 5 + b]));
                }
            }
        }
        std::println!(
            "max dynamics deviation vs Pinocchio (original URDF): gravity {wg:.2e}, M {wm:.2e}, coriolis {wc:.2e}, tau {wt:.2e}"
        );
        // The jaw is folded into gripper_link and joint origins are snapped (~3 um); both are tiny.
        assert!(wg < 1e-4 && wm < 1e-4 && wc < 1e-4 && wt < 1e-4);
    }

    #[test]
    fn jacobian_linear_part_matches_finite_difference_of_fk() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let h = 1e-6;
        for _ in 0..50 {
            let q = rng.q5(0.8);
            let j = jacobian(&q);
            let t0 = fk(&q);
            for k in 0..5 {
                let mut qp = q;
                qp[k] += h;
                let t1 = fk(&qp);
                for r in 0..3 {
                    let fd = (t1[(r, 3)] - t0[(r, 3)]) / h;
                    assert!(
                        libm::fabs(fd - j[r][k]) < 1e-4,
                        "Jv[{r}][{k}]: fd {fd} vs {}",
                        j[r][k]
                    );
                }
            }
        }
    }

    // ---- IK ----

    #[test]
    fn ik_recovers_every_reachable_configuration() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut total = 0usize;
        for case in 0..300 {
            let q = rng.q5(0.9);
            let t = fk(&q);
            let sols = solve(&t, None, None, false, POLICY, 15);
            let got: Vec<[f64; 5]> = sols.iter().flatten().copied().collect();
            assert!(
                !got.is_empty(),
                "case {case}: no solutions for reachable q = {q:?}"
            );
            assert!(
                got.iter().any(|s| q_eq_mod(s, &q, 1e-5)),
                "case {case}: q = {q:?} not recovered; got {got:?}"
            );
            for s in &got {
                let res = frob_residual(&lift(s), &t);
                assert!(
                    res <= POLICY.subproblem_numerical,
                    "case {case}: residual {res}"
                );
            }
            total += got.len();
        }
        std::println!("300 reachable cases, {total} solutions in total");
    }

    #[test]
    fn ik_returns_nothing_for_unreachable_poses() {
        // Rotate the end-effector about the base y axis while holding position: an orientation
        // change the 5R arm (pan + three pitch joints + roll) cannot absorb in general.
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        let n = 100;
        let mut empty = 0usize;
        for _ in 0..n {
            let q = rng.q5(0.7);
            let mut t = fk(&q);
            let a = 0.25; // rad, far above numerical slack
            let (s, c) = (libm::sin(a), libm::cos(a));
            let ry = SMatrix::<f64, 3, 3>::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
            let rn = ry * t.fixed_view::<3, 3>(0, 0).into_owned();
            for i in 0..3 {
                for j in 0..3 {
                    t[(i, j)] = rn[(i, j)];
                }
            }
            let sols = solve(&t, None, None, false, POLICY, 15);
            // Whatever is returned must genuinely close; generically nothing should.
            for s in sols.iter().flatten() {
                assert!(frob_residual(&lift(s), &t) <= POLICY.subproblem_numerical);
            }
            if sols.iter().flatten().count() == 0 {
                empty += 1;
            }
        }
        assert!(
            empty >= n - 2,
            "expected (almost) all perturbed poses unreachable, only {empty}/{n} empty"
        );
    }

    #[test]
    fn respect_limits_keeps_the_in_range_solution() {
        let mut rng = Rng(0x0F0F_1234_ABCD_9876);
        for _ in 0..100 {
            let q = rng.q5(0.9);
            let t = fk(&q);
            let sols = solve(&t, None, None, true, POLICY, 15);
            for s in sols.iter().flatten() {
                assert!(in_limits(s), "returned solution outside the limits: {s:?}");
            }
            assert!(sols.iter().flatten().any(|s| q_eq_mod(s, &q, 1e-5)));
        }
    }

    #[test]
    fn seed_sorts_nearest_first_and_cap_truncates() {
        let mut rng = Rng(0xABCD_EF01_2345_6789);
        let mut saw_multi = false;
        for _ in 0..100 {
            let q = rng.q5(0.9);
            let t = fk(&q);
            let v: Vec<[f64; 5]> = solve(&t, None, Some(&q), false, POLICY, 15)
                .iter()
                .flatten()
                .copied()
                .collect();
            if v.len() > 1 {
                saw_multi = true;
                let d0 = seed_dist(&v[0], &q);
                for s in &v[1..] {
                    assert!(d0 <= seed_dist(s, &q) + 1e-12);
                }
                let one = solve(&t, Some(1), Some(&q), false, POLICY, 15);
                assert_eq!(one.iter().flatten().count(), 1);
            }
        }
        assert!(saw_multi, "no multi-solution case was exercised");
    }

    #[test]
    fn arm_facade_round_trips_with_a_tcp_offset() {
        let mut arm = Arm::<So101, DOF, MAX_SOLUTIONS>::new();
        let mut tcp = Pose::identity();
        tcp[(2, 3)] = 0.05;
        arm.set_tcp(&tcp).expect("tcp");
        let mut rng = Rng(0x5555_AAAA_5555_AAAA);
        let q = rng.q5(0.8);
        let target = arm.fk(&q);
        let sols = arm.ik(&target, None, None, false, true, POLICY, 15);
        assert!(sols.iter().flatten().any(|s| q_eq_mod(s, &q, 1e-5)));
    }

    // ---- leniency: tolerance policy and nearest-reachable ----

    /// `t` moved by up to `pos` metres per axis and rotated by about `ang` radians about a random axis.
    fn perturbed(rng: &mut Rng, t: &Pose, pos: f64, ang: f64) -> Pose {
        use nalgebra::{Rotation3, Vector3};
        let mut out = *t;
        for i in 0..3 {
            out[(i, 3)] += pos * (2.0 * rng.next() - 1.0);
        }
        let axis = Vector3::new(
            2.0 * rng.next() - 1.0,
            2.0 * rng.next() - 1.0,
            2.0 * rng.next() - 1.0,
        );
        let dr = Rotation3::from_scaled_axis(axis * (ang / 1.8));
        let r = dr.matrix() * t.fixed_view::<3, 3>(0, 0).into_owned();
        for i in 0..3 {
            for j in 0..3 {
                out[(i, j)] = r[(i, j)];
            }
        }
        out
    }

    #[test]
    fn a_looser_policy_admits_near_reachable_targets_and_the_default_stays_strict() {
        let mut rng = Rng(0xC0FF_EE00_1234_5678);
        let loose = TolerancePolicy {
            subproblem_numerical: 5e-3,
            ..POLICY
        };
        let (mut with_loose, mut with_default, n) = (0usize, 0usize, 200usize);
        for _ in 0..n {
            let q = rng.q5(0.8);
            let t = perturbed(&mut rng, &fk(&q), 1e-3, 0.01); // ~1 mm and ~0.6 degrees off
            let sols = solve(&t, None, None, false, loose, 15);
            let got: Vec<[f64; 5]> = sols.iter().flatten().copied().collect();
            for s in &got {
                assert!(
                    frob_residual(&lift(s), &t) <= 5e-3 + 1e-12,
                    "solution does not close to the policy"
                );
            }
            with_loose += (!got.is_empty()) as usize;
            with_default += (solve(&t, None, None, false, POLICY, 15)
                .iter()
                .flatten()
                .count()
                > 0) as usize;
        }
        std::println!("near-reachable targets with a solution: loose policy {with_loose}/{n}, default policy {with_default}/{n}");
        assert!(
            with_loose >= n * 95 / 100,
            "loose policy found a solution for only {with_loose}/{n}"
        );
        assert!(
            with_default <= n / 20,
            "the default policy must stay strict, but {with_default}/{n} closed"
        );
    }

    #[test]
    fn nearest_is_exact_for_reachable_poses_and_keeps_the_seed_branch() {
        let mut rng = Rng(0x0BAD_CAFE_F00D_0001);
        for case in 0..200 {
            let q = rng.q5(0.9);
            let t = fk(&q);
            let n = solve_nearest(
                &t,
                Some(&q),
                true,
                DEFAULT_ROT_WEIGHT,
                POLICY,
                NEAREST_MAX_ITERS,
            )
            .unwrap();
            assert!(
                n.position_error < 1e-6 && n.orientation_error < 1e-5,
                "case {case}: {n:?}"
            );
            assert!(
                q_eq_mod(&n.q, &q, 1e-5),
                "case {case}: left the seed branch: {:?} vs {q:?}",
                n.q
            );
            // without a seed it is still an exact solution
            let m = solve_nearest(
                &t,
                None,
                true,
                DEFAULT_ROT_WEIGHT,
                POLICY,
                NEAREST_MAX_ITERS,
            )
            .unwrap();
            assert!(
                m.position_error < 1e-6 && m.orientation_error < 1e-5,
                "case {case} (no seed): {m:?}"
            );
        }
    }

    #[test]
    fn nearest_never_does_worse_than_the_seed_and_improves_an_unreachable_pose() {
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);
        let mut improved = 0usize;
        let n = 100;
        for _ in 0..n {
            let q = rng.q5(0.7);
            let t = perturbed(&mut rng, &fk(&q), 0.0, 0.3); // an orientation the arm cannot absorb
            let start = norm12(&weighted_residual12(&lift(&q), &t, DEFAULT_ROT_WEIGHT));
            let r = solve_nearest(
                &t,
                Some(&q),
                false,
                DEFAULT_ROT_WEIGHT,
                POLICY,
                NEAREST_MAX_ITERS,
            )
            .unwrap();
            let end = norm12(&weighted_residual12(&lift(&r.q), &t, DEFAULT_ROT_WEIGHT));
            assert!(
                end <= start + 1e-9,
                "least squares got worse: {start} -> {end}"
            );
            improved += (end < start - 1e-6) as usize;
            // the answer is a real configuration, so its errors are what FK says they are
            let (dp, da) = pose_errors(&lift(&r.q), &t);
            assert!(
                (dp - r.position_error).abs() < 1e-12 && (da - r.orientation_error).abs() < 1e-12
            );
        }
        assert!(
            improved >= n * 9 / 10,
            "expected the fit to improve on the seed, only {improved}/{n} did"
        );
    }

    #[test]
    fn nearest_tracks_a_noisy_target_to_within_the_noise() {
        let mut rng = Rng(0x2468_ACE0_1357_9BDF);
        for _ in 0..200 {
            let q = rng.q5(0.8);
            let t = perturbed(&mut rng, &fk(&q), 2e-3, 0.02);
            let r = solve_nearest(
                &t,
                Some(&q),
                true,
                DEFAULT_ROT_WEIGHT,
                POLICY,
                NEAREST_MAX_ITERS,
            )
            .unwrap();
            // the true configuration is within the noise, so the optimum cannot be farther
            assert!(
                r.position_error < 5e-3 && r.orientation_error < 0.04,
                "{r:?}"
            );
        }
    }

    #[test]
    fn nearest_reports_a_target_nowhere_near_the_arm() {
        let mut t = fk(&[0.0, -0.3, 0.5, -0.2, 0.0]);
        t[(0, 3)] += 5.0;
        let r = solve_nearest(
            &t,
            Some(&[0.0, -0.3, 0.5, -0.2, 0.0]),
            true,
            DEFAULT_ROT_WEIGHT,
            POLICY,
            NEAREST_MAX_ITERS,
        )
        .expect("the closest configuration is still returned");
        assert!(
            r.position_error > 4.0,
            "a target 5 m away cannot be within {} m",
            r.position_error
        );
    }

    #[test]
    fn nearest_respects_the_limits_and_rejects_a_nan_target() {
        let mut rng = Rng(0x7777_1111_9999_3333);
        for _ in 0..100 {
            let q = rng.q5(1.0);
            let t = perturbed(&mut rng, &fk(&q), 2e-3, 0.02);
            if let Some(r) = solve_nearest(
                &t,
                None,
                true,
                DEFAULT_ROT_WEIGHT,
                POLICY,
                NEAREST_MAX_ITERS,
            ) {
                assert!(in_limits(&r.q), "outside the limits: {:?}", r.q);
            }
        }
        let mut nan = fk(&[0.0; 5]);
        nan[(0, 3)] = f64::NAN;
        assert!(solve_nearest(
            &nan,
            None,
            true,
            DEFAULT_ROT_WEIGHT,
            POLICY,
            NEAREST_MAX_ITERS
        )
        .is_none());
    }

    /// Regression: near the shoulder singularity (target high and close to the base axis) the 6R closed
    /// form is ill-conditioned, and a fit started from a distant seed used to settle 10 to 55 cm off a
    /// target that is within a millimetre of reachable. The fixed fallback starts find the branch.
    #[test]
    fn nearest_finds_the_right_branch_near_the_shoulder_singularity_from_a_distant_seed() {
        use dexhal::math::pose_from_xyz_quat;
        let far_seed = [0.0, -0.3, 0.5, -0.2, 0.0];
        let cases: [([f64; 5], [f64; 7]); 2] = [
            (
                [
                    0.31318276845477616,
                    -0.49478767320101286,
                    -1.1508361482740317,
                    -0.9431281913741131,
                    1.5109157214393674,
                ],
                [
                    -0.10608015028932995,
                    0.03807046469115335,
                    0.4471396005557163,
                    -0.2641351632983188,
                    0.4093695179601097,
                    0.6777764884583094,
                    -0.5506979616519395,
                ],
            ),
            (
                [
                    -0.8433646172068047,
                    -0.6563568724215465,
                    -1.0685657735839245,
                    0.8861027771729375,
                    0.6274471378835987,
                ],
                [
                    0.08919002237283397,
                    0.04800752405528007,
                    0.47010814695054864,
                    -0.23602203254283477,
                    0.27139158282741194,
                    0.12304857109811318,
                    0.924932028894823,
                ],
            ),
        ];
        for (q_true, pose) in cases {
            let t = pose_from_xyz_quat(&pose).unwrap();
            // the configuration the target was made from is ~1 mm / 0.3 degrees away: the fit must beat that
            let own = {
                let (dp, da) = pose_errors(&lift(&q_true), &t);
                (dp, da)
            };
            assert!(
                own.0 < 2e-3 && own.1 < 0.02,
                "test case is not near-reachable: {own:?}"
            );
            let r = solve_nearest(
                &t,
                Some(&far_seed),
                true,
                DEFAULT_ROT_WEIGHT,
                POLICY,
                NEAREST_MAX_ITERS,
            )
            .unwrap();
            assert!(
                r.position_error < 5e-3 && r.orientation_error < 0.1,
                "from a distant seed the fit missed by {:.1} mm / {:.3} rad: {r:?}",
                r.position_error * 1e3,
                r.orientation_error
            );
        }
    }
}
