//! Analytical inverse kinematics for the **SO-ARM101 embedded as a 6R chain**
//! (arm name `so101_6r`).
//!
//! The physical SO-ARM101 arm has five actuated joints. This driver models it as
//! a six-joint chain: the five real joints plus one zero-offset *virtual* wrist
//! joint (`virtual_wrist`, axis perpendicular to the wrist roll) appended after
//! the wrist roll. At `q6 = 0` the chain is exactly the physical arm. The
//! five-joint arm that callers should use is [`crate::drivers::so101`], which
//! pins the virtual joint to zero.
//!
//! This is a line-by-line translation of the Python `ssik` artifact
//! `codegen/so101_6r_ik.py`, built from `so101_6r.urdf` (solver `ikgeo.three_parallel`,
//! tier 0), into a deterministic, `#![no_std]`, zero-allocation Rust
//! implementation.
//!
//! The closed form is: subproblem **SP6** for the coupled wrist/shoulder angle
//! pair `(q1, q5)`, followed by the inlined `SP1 + SP3 + SP1` chain for the
//! remaining joints. SP6 reduces to intersecting two conics, which produces a
//! quartic solved through the companion-matrix eigenvalue routine in
//! [`dexhal::math::ssik_core`].
//!
//! ## Equivalence to the Python reference
//!
//! [`solve`] mirrors `so101_6r_ik.solve` with the same argument list:
//! `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`, `policy`,
//! and `refinement_max_iters`. The algebraic solution set is FK-verified to
//! [`TolerancePolicy::subproblem_numerical`] and de-duplicated modulo `2*pi`.
//!
//! [`fk`] mirrors `so101_6r_ik.fk` (alias of `_fk`): product-of-exponentials forward
//! kinematics using the baked chain constants.

use nalgebra::SMatrix;

use dexhal::math::kinematics::{lm_refine_masked_analytic, mat4_frob_diff, residual_columns};
use dexhal::math::ssik_core::{sp6_solve, wrap_to_pi};
pub use dexhal::math::tolerances::TolerancePolicy;

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom (five SO-101 joints plus the virtual wrist joint).
pub const DOF: usize = 6;

/// Base link name in the URDF.
pub const BASE_LINK: &str = "base_link";

/// End-effector link name in the URDF.
pub const EE_LINK: &str = "gripper_frame_link";

// --- algebraic-block degeneracy constants from so101_6r_ik.py ---
const TWO_PI: f64 = 2.0 * PI;

/// Joint angle limits (lo, hi) in radians — baked from `_JOINT_LIMITS` in
/// the Python artifact.  Used by [`solve`] when `respect_limits = true`.
const JOINT_LIMITS: [(f64, f64); 6] = [
    (-1.91986, 1.91986),
    (-1.74533, 1.74533),
    (-1.69, 1.69),
    (-1.65806, 1.65806),
    (-2.74385, 2.84121),
    (-3.141592653589793, 3.141592653589793),
];

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Inverse kinematics — mirrors `so101_6r_ik.solve`.
///
/// # Arguments
///
/// * `t` — 4×4 SE(3) target pose of `ee_link` in `base_link`.
/// * `max_solutions` — cap on returned solutions (post-dedup, post-limits).
///   `None` returns the full algebraic set (at most 8 here).
/// * `q_seed` — when `Some`, solutions are sorted by wrap-to-pi distance
///   from the seed (closest first). Has no effect on *which* solutions are
///   returned.
/// * `respect_limits` — when `true`, solutions outside the baked URDF joint
///   limits are dropped (after a `±2π` wrap-into-limits attempt). Pass
///   `false` for the raw geometric set.
/// * `allow_refinement` — when `true`, algebraic candidates that exceed
///   `policy.subproblem_numerical` are polished with a bounded Levenberg-Marquardt step.
///   For this arm (tier-0 arm) this is rarely needed; default `false`.
/// * `policy` — FK closure and dedup tolerances; use
///   [`TolerancePolicy::DEFAULT`] for the Python defaults.
/// * `refinement_max_iters` — maximum LM iterations when
///   `allow_refinement = true`.
///
/// Returns up to 16 candidate joint vectors `[q1, .., q6]` in radians;
/// unused slots are [`None`].
pub fn solve(
    t: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 6]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 6]>; 16] {
    let fk_atol = policy.subproblem_numerical;
    let dedup_atol = policy.subproblem_dedup;

    let mut solutions: [Option<[f64; 6]>; 16] = [None; 16];
    let mut solution_count = 0usize;

    // Raw algebraic candidates (stack array, at most 8 here).
    let mut cands = [[0.0f64; 6]; 8];
    let cand_count = solve_algebraic(t, &mut cands, policy);

    for ci in 0..cand_count {
        let q = cands[ci];
        // Reject non-finite candidates (mirrors np.isfinite filter).
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        // FK closure check.
        let residual = fk_residual(&q, t);
        let accepted_q: [f64; 6];
        if residual <= fk_atol {
            accepted_q = q;
        } else if allow_refinement {
            // Newton polish for near-miss candidates, mirroring Python's
            // `allow_refinement` path (lm_refine via _spatial_jacobian).
            match refine6(&q, t, refinement_max_iters, fk_atol) {
                Some(refined) => {
                    accepted_q = refined;
                }
                None => continue,
            }
        } else {
            continue;
        }

        // Wrap-to-pi de-duplication against accepted solutions.
        let mut dup = false;
        for s in solutions.iter().take(solution_count) {
            if let Some(existing) = s {
                if q_close_wrap(&accepted_q, existing, dedup_atol) {
                    dup = true;
                    break;
                }
            }
        }
        if dup {
            continue;
        }
        if solution_count < 16 {
            solutions[solution_count] = Some(accepted_q);
            solution_count += 1;
        }
    }

    // Post-processing (mirrors Python's _ps_wrap_to_limits + _ps_respect_limits).
    if respect_limits {
        let mut filtered: [Option<[f64; 6]>; 16] = [None; 16];
        let mut count = 0usize;
        for sol in solutions.iter().take(solution_count).flatten() {
            let q = wrap_to_limits(sol);
            if in_limits(&q) && count < 16 {
                filtered[count] = Some(q);
                count += 1;
            }
        }
        solutions = filtered;
        solution_count = count;
    }

    // Sort by seed distance (mirrors Python's _ps_nearest_to_seed).
    if let Some(seed) = q_seed {
        // Insertion sort on the first `solution_count` slots.
        for i in 1..solution_count {
            let mut j = i;
            while j > 0 {
                let prev = solutions[j - 1].unwrap();
                let curr = solutions[j].unwrap();
                if seed_dist(&prev, seed) > seed_dist(&curr, seed) {
                    solutions.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }

    // Truncate to max_solutions (mirrors Python's `solutions[:max_solutions]`).
    if let Some(max) = max_solutions {
        let cap = max.min(solution_count);
        for i in cap..solution_count {
            solutions[i] = None;
        }
    }

    solutions
}

/// `true` if joint vectors `a` and `b` agree (mod `2*pi`) within `tol` per axis.
fn q_close_wrap(a: &[f64; 6], b: &[f64; 6], tol: f64) -> bool {
    for i in 0..6 {
        if libm::fabs(wrap_to_pi(a[i] - b[i])) > tol {
            return false;
        }
    }
    true
}

/// Try `q[i] ± 2π` per joint to bring the vector into `JOINT_LIMITS`.
/// Mirrors Python's `_ps_wrap_to_limits`.
fn wrap_to_limits(q: &[f64; 6]) -> [f64; 6] {
    let mut out = *q;
    for i in 0..6 {
        let (lo, hi) = JOINT_LIMITS[i];
        if out[i] < lo {
            let wrapped = out[i] + TWO_PI;
            if wrapped <= hi {
                out[i] = wrapped;
            }
        } else if out[i] > hi {
            let wrapped = out[i] - TWO_PI;
            if wrapped >= lo {
                out[i] = wrapped;
            }
        }
    }
    out
}

/// `true` if every joint is within its URDF limit.
fn in_limits(q: &[f64; 6]) -> bool {
    for i in 0..6 {
        let (lo, hi) = JOINT_LIMITS[i];
        if q[i] < lo || q[i] > hi {
            return false;
        }
    }
    true
}

/// Squared wrap-to-π distance between `q` and `seed`. Used for seed sorting.
fn seed_dist(q: &[f64; 6], seed: &[f64; 6]) -> f64 {
    let mut s = 0.0;
    for i in 0..6 {
        let d = wrap_to_pi(q[i] - seed[i]);
        s += d * d;
    }
    s
}

// ---------------------------------------------------------------------------
// Levenberg-Marquardt refinement (for `allow_refinement = true`)
// ---------------------------------------------------------------------------

/// 12-vector pose residual `[vec(R(q) − R_t); t(q) − t_t]` for this arm.
fn pose_residual12(q: &[f64; 6], target: &SMatrix<f64, 4, 4>) -> [f64; 12] {
    let tf = fk(q);
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

/// Levenberg-Marquardt polish of a 6-DOF configuration onto the exact
/// pose root. Mirrors the Python `_lm_refine` path used when
/// `allow_refinement = True`. Returns `Some(q_refined)` if the FK residual
/// converges within `fk_atol`, otherwise `None`.
fn refine6(
    q_in: &[f64; 6],
    target: &SMatrix<f64, 4, 4>,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 6]> {
    const ALL: [usize; 6] = [0, 1, 2, 3, 4, 5];
    let q = lm_refine_masked_analytic::<6, 6, _, _>(
        q_in,
        &ALL,
        |q| pose_residual12(q, target),
        |q| {
            let f = super::generated_kinematics::fk(q);
            let rot = [
                [f[0][0], f[0][1], f[0][2]],
                [f[1][0], f[1][1], f[1][2]],
                [f[2][0], f[2][1], f[2][2]],
            ];
            residual_columns(&rot, &super::generated_kinematics::jacobian(q))
        },
        max_iters,
    );
    if q.iter().all(|v| v.is_finite()) && fk_residual(&q, target) <= fk_atol {
        Some(q)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Algebraic candidate generation (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Fills `out` with the raw algebraic IK candidates and returns the count.
fn solve_algebraic(
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]; 8],
    policy: TolerancePolicy,
) -> usize {
    let r_00 = t[(0, 0)];
    let r_01 = t[(0, 1)];
    let r_02 = t[(0, 2)];
    let r_10 = t[(1, 0)];
    let r_11 = t[(1, 1)];
    let r_12 = t[(1, 2)];
    let r_20 = t[(2, 0)];
    let r_21 = t[(2, 1)];
    let r_22 = t[(2, 2)];
    let p_x = t[(0, 3)];
    let p_y = t[(1, 3)];
    let p_z = t[(2, 3)];

    let mut count = 0usize;

    // T_target-derived SP6 inputs.
    let p_16_x = p_x - 0.00790000000000001*r_00 + 0.000218121*r_01 - 0.0981274*r_02 - 0.0388353;
    let p_16_y = p_y - 0.00790000000000001*r_10 + 0.000218121*r_11 - 0.0981274*r_12;
    let p_16_z = p_z - 0.00790000000000001*r_20 + 0.000218121*r_21 - 0.0981274*r_22 - 0.0624;
    let r_06_axes5_x = -1.0*r_00 + 1.22464679914735e-16*r_02;
    let r_06_axes5_y = -1.0*r_10 + 1.22464679914735e-16*r_12;
    let r_06_axes5_z = -1.0*r_20 + 1.22464679914735e-16*r_22;

    let p_16 = [p_16_x, p_16_y, p_16_z];
    let r_06_axes5 = [r_06_axes5_x, r_06_axes5_y, r_06_axes5_z];

    const H_SP_0: [f64; 3] = [-6.123233995736766e-17, 1.0, 1.2246467991473532e-16];
    const K_SP_0: [f64; 3] = [1.4997597826618576e-32, -1.2246467991473532e-16, 1.0];
    const K_SP_1: [f64; 3] = [-1.0, -7.253661990818735e-20, -5.825275737285599e-17];
    const NEG_P_5: [f64; 3] = [-0.0, -0.0, -0.0];
    const NEG_AXES_5: [f64; 3] = [5.818022075294779e-17, 0.04866027636027259, -0.9988153870984077];

    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, -0.0001777999999999968, 0.0, policy.subproblem_numerical, policy.subproblem_degeneracy, &mut theta15);

    for si in 0..n15 {
        let (q1, q5) = theta15[si];

        // NOTE: the literal constants below are the machine-generated symbolic
        // coefficients emitted by `ssik build`, copied verbatim from `so101_6r_ik.py`
        // (mechanically translated by `codegen/transpile_ik_solver.py`; most small ones
        // are rounding residue of the URDF's rotations).
        let th14_x0 = libm::sin(q5);
        let th14_x1 = libm::cos(q5);
        let th14_x2 = -0.998815387098408*r_00 + 0.0486602763602726*r_01 + 1.80572364047775e-16*r_02;
        let th14_x3 = libm::sin(q1);
        let th14_x4 = 1.22464679914735e-16*th14_x3;
        let th14_x5 = libm::cos(q1);
        let th14_x6 = 1.49975978266186e-32*th14_x5;
        let th14_x7 = th14_x4 - th14_x6 + 1.49975978266186e-32;
        let th14_x8 = 1.80644900667683e-16*r_00 - 2.90704596460347e-18*r_01 + 1.0*r_02;
        let th14_x9 = 5.81802207529478e-17*th14_x8;
        let th14_x10 = 0.0486602763602726*r_00 + 0.998815387098408*r_01 - 5.88662854911516e-18*r_02;
        let th14_x11 = 0.0486602763602726*th14_x10;
        let th14_x12 = -0.998815387098408*r_10 + 0.0486602763602726*r_11 + 1.80572364047775e-16*r_12;
        let th14_x13 = 1.49975978266186e-32*th14_x3;
        let th14_x14 = th14_x13 + 1.22464679914735e-16*th14_x5 - 1.22464679914735e-16;
        let th14_x15 = 1.80644900667683e-16*r_10 - 2.90704596460347e-18*r_11 + 1.0*r_12;
        let th14_x16 = 5.81802207529478e-17*th14_x15;
        let th14_x17 = 0.0486602763602726*r_10 + 0.998815387098408*r_11 - 5.88662854911516e-18*r_12;
        let th14_x18 = 0.0486602763602726*th14_x17;
        let th14_x19 = -1.0*r_20 + 1.22464679914735e-16*r_22 - th14_x11*th14_x7 + 0.998815387098408*th14_x12*th14_x14 - th14_x14*th14_x16 - th14_x14*th14_x18 + 0.998815387098408*th14_x2*th14_x7 - th14_x7*th14_x9;
        let th14_x20 = 1.0*th14_x5;
        let th14_x21 = th14_x20 + 1.49975978266186e-32;
        let th14_x22 = 1.0*th14_x3;
        let th14_x23 = th14_x22 + 1.83667601732677e-48*th14_x5 - 1.83667601732677e-48;
        let th14_x24 = -0.998815387098408*r_20 + 0.0486602763602726*r_21 + 1.80572364047775e-16*r_22;
        let th14_x25 = -th14_x13 + 1.22464679914735e-16*th14_x5 - 1.22464679914735e-16;
        let th14_x26 = 1.80644900667683e-16*r_20 - 2.90704596460347e-18*r_21 + 1.0*r_22;
        let th14_x27 = 5.81802207529478e-17*th14_x26;
        let th14_x28 = 0.0486602763602726*r_20 + 0.998815387098408*r_21 - 5.88662854911516e-18*r_22;
        let th14_x29 = 0.0486602763602726*th14_x28;
        let th14_x30 = th14_x11*th14_x23 - 0.998815387098408*th14_x12*th14_x21 + th14_x16*th14_x21 + th14_x18*th14_x21 - 0.998815387098408*th14_x2*th14_x23 + th14_x23*th14_x9 - 0.998815387098408*th14_x24*th14_x25 + th14_x25*th14_x27 + th14_x25*th14_x29;
        let th14_x31 = -th14_x30;
        let th14_x32 = 0.998815387098408*th14_x1 + 1.3684555315672e-48;
        let th14_x33 = th14_x20 + 2.24927940568994e-64;
        let th14_x34 = -th14_x22 + 1.83667601732677e-48*th14_x5 - 1.83667601732677e-48;
        let th14_x35 = -th14_x4 - th14_x6 + 1.49975978266186e-32;
        let th14_x36 = -th14_x11*th14_x33 + 0.998815387098408*th14_x12*th14_x34 - th14_x16*th14_x34 - th14_x18*th14_x34 + 0.998815387098408*th14_x2*th14_x33 + 0.998815387098408*th14_x24*th14_x35 - th14_x27*th14_x35 - th14_x29*th14_x35 - th14_x33*th14_x9;
        let theta14 = libm::atan2(th14_x19*(-5.8252757372856e-17*th14_x0 + 6.11598033374595e-17*th14_x1 - 2.46519032881566e-32) + th14_x31*(2.97958258451167e-18*th14_x0 + 6.11598033374595e-17*th14_x1 + 3.01898744547311e-48) + th14_x36*(0.0486602763602725*th14_x0 + th14_x32), th14_x19*(0.0486602763602726*th14_x0 + th14_x32) + th14_x31*(0.998815387098408*th14_x0 - 0.0486602763602726*th14_x1 + 1.3363823550461e-51) + th14_x36*(-2.90704596460349e-18*th14_x0 - 5.81802207529478e-17*th14_x1 + 2.46519032881566e-32) - (0.998815387098408*th14_x0 - 0.0486602763602725*th14_x1 - 1.50815734038151e-48)*(-1.22464679914735e-16*r_20 + 1.49975978266186e-32*r_22 + 2.97958258451167e-18*th14_x10*th14_x33 - 5.95916516902335e-18*th14_x10*th14_x7 + 1.22319606674919e-16*th14_x12*th14_x14 - 6.11598033374595e-17*th14_x12*th14_x34 - 7.12502211187839e-33*th14_x14*th14_x15 - 5.95916516902335e-18*th14_x14*th14_x17 + 3.5625110559392e-33*th14_x15*th14_x34 + 2.97958258451167e-18*th14_x17*th14_x34 - 6.11598033374595e-17*th14_x2*th14_x33 + 1.22319606674919e-16*th14_x2*th14_x7 - 6.11598033374595e-17*th14_x24*th14_x35 + 3.5625110559392e-33*th14_x26*th14_x35 + 2.97958258451167e-18*th14_x28*th14_x35 - th14_x30 + 3.5625110559392e-33*th14_x33*th14_x8 - 7.12502211187839e-33*th14_x7*th14_x8));
        let q6_x0 = libm::sin(q5);
        let q6_x1 = libm::cos(q5);
        let q6_x2 = 0.0486602763602726*r_00 + 0.998815387098408*r_01 - 5.88662854911516e-18*r_02;
        let q6_x3 = libm::cos(q1);
        let q6_x4 = 1.0*q6_x3;
        let q6_x5 = q6_x4 + 2.24927940568994e-64;
        let q6_x6 = 6.12323399573677e-17*q6_x5;
        let q6_x7 = 0.0486602763602726*r_10 + 0.998815387098408*r_11 - 5.88662854911516e-18*r_12;
        let q6_x8 = q6_x4 + 1.49975978266186e-32;
        let q6_x9 = 1.0*q6_x8;
        let q6_x10 = libm::sin(q1);
        let q6_x11 = 1.22464679914735e-16*q6_x10;
        let q6_x12 = 1.49975978266186e-32*q6_x3;
        let q6_x13 = q6_x11 - q6_x12 + 1.49975978266186e-32;
        let q6_x14 = 1.22464679914735e-16*q6_x13;
        let q6_x15 = 1.0*q6_x10;
        let q6_x16 = q6_x15 + 1.83667601732677e-48*q6_x3 - 1.83667601732677e-48;
        let q6_x17 = 1.0*q6_x16;
        let q6_x18 = -q6_x15 + 1.83667601732677e-48*q6_x3 - 1.83667601732677e-48;
        let q6_x19 = 6.12323399573677e-17*q6_x18;
        let q6_x20 = 1.49975978266186e-32*q6_x10;
        let q6_x21 = q6_x20 + 1.22464679914735e-16*q6_x3 - 1.22464679914735e-16;
        let q6_x22 = 1.22464679914735e-16*q6_x21;
        let q6_x23 = 0.0486602763602726*r_20 + 0.998815387098408*r_21 - 5.88662854911516e-18*r_22;
        let q6_x24 = -q6_x11 - q6_x12 + 1.49975978266186e-32;
        let q6_x25 = 6.12323399573677e-17*q6_x24;
        let q6_x26 = -q6_x20 + 1.22464679914735e-16*q6_x3 - 1.22464679914735e-16;
        let q6_x27 = 1.0*q6_x26;
        let q6_x28 = q6_x14*q6_x2 + q6_x17*q6_x2 - q6_x19*q6_x7 - q6_x2*q6_x6 + q6_x22*q6_x7 - q6_x23*q6_x25 + q6_x23*q6_x27 + q6_x7*q6_x9 + 5.95916516902335e-18*r_20 + 1.22319606674919e-16*r_21 - 7.20904081044331e-34*r_22;
        let q6_x29 = -0.998815387098408*r_00 + 0.0486602763602726*r_01 + 1.80572364047775e-16*r_02;
        let q6_x30 = -0.998815387098408*r_10 + 0.0486602763602726*r_11 + 1.80572364047775e-16*r_12;
        let q6_x31 = -0.998815387098408*r_20 + 0.0486602763602726*r_21 + 1.80572364047775e-16*r_22;
        let q6_x32 = q6_x14*q6_x29 + q6_x17*q6_x29 - q6_x19*q6_x30 + q6_x22*q6_x30 - q6_x25*q6_x31 + q6_x27*q6_x31 - q6_x29*q6_x6 + q6_x30*q6_x9 - 1.22319606674919e-16*r_20 + 5.95916516902335e-18*r_21 + 2.21137367645578e-32*r_22;
        let q6_x33 = 1.80644900667683e-16*r_00 - 2.90704596460347e-18*r_01 + 1.0*r_02;
        let q6_x34 = 1.80644900667683e-16*r_10 - 2.90704596460347e-18*r_11 + 1.0*r_12;
        let q6_x35 = 1.80644900667683e-16*r_20 - 2.90704596460347e-18*r_21 + 1.0*r_22;
        let q6_x36 = q6_x14*q6_x33 + q6_x17*q6_x33 - q6_x19*q6_x34 + q6_x22*q6_x34 - q6_x25*q6_x35 + q6_x27*q6_x35 - q6_x33*q6_x6 + q6_x34*q6_x9 + 2.2122619938497e-32*r_20 - 3.56010453552587e-34*r_21 + 1.22464679914735e-16*r_22;
        let q6 = libm::atan2(q6_x28*(3.52965197099219e-21*q6_x0 + 7.24506920924062e-20*q6_x1 + 6.10873526453671e-17) + q6_x32*(2.83459527251107e-18*q6_x0 + 5.81837504049188e-17*q6_x1 + 2.9760529325407e-18) + q6_x36*(0.0486602763602725*q6_x0 + 0.998815387098408*q6_x1 - 1.77794359488104e-34), q6_x28*(-1.22464679914735e-16*q6_x0 + 1.0*q6_x1 - 4.43632540834879e-36) + q6_x32*(1.0*q6_x0 + 1.22464679914735e-16*q6_x1 - 3.56272718478861e-33) + q6_x36*(-5.8252757372856e-17*q6_x0 - 7.25366199081945e-20*q6_x1 - 6.11598033374595e-17) - (-0.998815387098408*q6_x0 + 0.0486602763602725*q6_x1 + 6.84227765783602e-49)*(5.95916516902335e-18*q6_x13*q6_x2 - 1.22319606674919e-16*q6_x13*q6_x29 + 7.12502211187839e-33*q6_x13*q6_x33 + 0.0486602763602726*q6_x16*q6_x2 - 0.998815387098408*q6_x16*q6_x29 + 5.81802207529478e-17*q6_x16*q6_x33 + 6.11598033374595e-17*q6_x18*q6_x30 - 3.5625110559392e-33*q6_x18*q6_x34 - 2.97958258451167e-18*q6_x18*q6_x7 - 2.97958258451167e-18*q6_x2*q6_x5 - 1.22319606674919e-16*q6_x21*q6_x30 + 7.12502211187839e-33*q6_x21*q6_x34 + 5.95916516902335e-18*q6_x21*q6_x7 - 2.97958258451167e-18*q6_x23*q6_x24 + 0.0486602763602726*q6_x23*q6_x26 + 6.11598033374595e-17*q6_x24*q6_x31 - 3.5625110559392e-33*q6_x24*q6_x35 - 0.998815387098408*q6_x26*q6_x31 + 5.81802207529478e-17*q6_x26*q6_x35 + 6.11598033374595e-17*q6_x29*q6_x5 - 0.998815387098408*q6_x30*q6_x8 - 3.5625110559392e-33*q6_x33*q6_x5 + 5.81802207529478e-17*q6_x34*q6_x8 + 0.0486602763602726*q6_x7*q6_x8 + 1.22464679914735e-16*r_20 - 7.70371977754894e-34*r_21 - 1.49975978266186e-32*r_22));
        let dinr_x0 = libm::sin(theta14);
        let dinr_x1 = libm::cos(theta14);
        let dinr_x2 = libm::cos(q1);
        let dinr_x3 = 1.0*dinr_x2;
        let dinr_x4 = p_x - 0.00790000000000001*r_00 + 0.000218121*r_01 - 0.0981274*r_02 - 0.0388353;
        let dinr_x5 = libm::sin(q1);
        let dinr_x6 = 1.0*dinr_x5;
        let dinr_x7 = p_y - 0.00790000000000001*r_10 + 0.000218121*r_11 - 0.0981274*r_12;
        let dinr_x8 = 1.22464679914735e-16*dinr_x5;
        let dinr_x9 = 1.49975978266186e-32*dinr_x2;
        let dinr_x10 = 0.00790000000000001*r_20;
        let dinr_x11 = 0.0981274*r_22;
        let dinr_x12 = -dinr_x10 - dinr_x11 + p_z + 0.000218121*r_21 - 0.0624;
        let dinr_x13 = 1.49975978266186e-32*dinr_x5;
        let d_inner_x = 2.21661070645671e-18*dinr_x0 - 0.0611*dinr_x1 + dinr_x12*(-dinr_x8 - dinr_x9 + 1.49975978266186e-32) + dinr_x4*(dinr_x3 + 2.24927940568994e-64) + dinr_x7*(1.83667601732677e-48*dinr_x2 - dinr_x6 - 1.83667601732677e-48) - 0.0303992;
        let d_inner_y = -7.48259194279033e-18*dinr_x0 - 3.74129597139516e-18*dinr_x1 + dinr_x12*(-dinr_x13 + 1.22464679914735e-16*dinr_x2 - 1.22464679914735e-16) + dinr_x4*(1.83667601732677e-48*dinr_x2 + dinr_x6 - 1.83667601732677e-48) + dinr_x7*(dinr_x3 + 1.49975978266186e-32) + 0.000177800000000002;
        let d_inner_z = 0.0611*dinr_x0 + 2.21661070645671e-18*dinr_x1 - dinr_x10 - dinr_x11 + dinr_x4*(dinr_x8 - dinr_x9 + 1.49975978266186e-32) + dinr_x7*(dinr_x13 + 1.22464679914735e-16*dinr_x2 - 1.22464679914735e-16) + 1.0*p_z + 0.000218121*r_21 - 0.1166;
        let d_elbow = libm::sqrt(d_inner_x*d_inner_x + d_inner_y*d_inner_y + d_inner_z*d_inner_z);

        // SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        let q3_r_sq = 0.000245236362102745_f64;
        let q3_rhs = 0.01584052745 - 0.5 * (d_elbow * d_elbow);
        let q3_phi = -1.28848143677266 + PI;
        let theta_q3_plus;
        let theta_q3_minus;
        if q3_r_sq < policy.axis_parallel * policy.axis_parallel {
            theta_q3_plus = 0.0;
            theta_q3_minus = 0.0;
        } else {
            let q3_r = libm::sqrt(q3_r_sq);
            if libm::fabs(q3_rhs) > q3_r + policy.subproblem_feasibility {
                let p = if q3_rhs > 0.0 { q3_phi } else { q3_phi + PI };
                theta_q3_plus = p;
                theta_q3_minus = p;
            } else {
                let clipped = (q3_rhs / q3_r).clamp(-1.0, 1.0);
                let delta = libm::acos(clipped);
                theta_q3_plus = q3_phi + delta;
                theta_q3_minus = q3_phi - delta;
            }
        }

        for q3 in [theta_q3_plus, theta_q3_minus] {
            let q2_x0 = libm::sin(q3);
            let q2_x1 = libm::cos(q3);
            let q2_x2 = -0.1349*q2_x0 + 0.00520000000000001*q2_x1 + 0.11257;
            let q2_x3 = 0.00520000000000001*q2_x0 + 0.1349*q2_x1 + 0.028;
            let q2 = libm::atan2(d_inner_x*q2_x2 + d_inner_y*(-7.62342632469227e-18*q2_x0 + 1.68388934882761e-17*q2_x1 + 1.03219355466135e-17) - d_inner_z*q2_x3, d_inner_x*q2_x3 + d_inner_y*(1.68388934882761e-17*q2_x0 + 7.62342632469227e-18*q2_x1 - 1.10928732766459e-17) + d_inner_z*q2_x2 - (9.78470222549572e-19 - 7.70371977754894e-34*q2_x1)*(-6.12323399573677e-17*d_inner_x + 1.0*d_inner_y + 1.22464679914735e-16*d_inner_z));
            let q4 = libm::remainder(theta14 - q2 - q3, TWO_PI);

            if count < 8 {
                out[count] = [q1, q2, q3, q4, q5, q6];
                count += 1;
            }
        }
    }

    count
}


// ---------------------------------------------------------------------------
// Forward kinematics (for FK closure verification)
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics — mirrors `so101_6r_ik.fk`.
///
/// `q` is a 6-element joint-angle vector in radians. Returns the 4x4 SE(3)
/// pose of `ee_link` in `base_link`.
///
/// Delegates to the SymPy-generated, branch-free symbolic FK in
/// [`generated_kinematics::fk`] (regenerate with `codegen/generate_dynamics.py`).
pub fn fk(q: &[f64; 6]) -> SMatrix<f64, 4, 4> {
    let acc = super::generated_kinematics::fk(q);
    let mut out = SMatrix::<f64, 4, 4>::zeros();
    for i in 0..4 {
        for j in 0..4 {
            out[(i, j)] = acc[i][j];
        }
    }
    out
}

/// Frobenius-norm residual between FK(q) and the target pose.
fn fk_residual(q: &[f64; 6], target: &SMatrix<f64, 4, 4>) -> f64 {
    mat4_frob_diff(&fk(q), target)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrix;
    use std::vec::Vec;

    fn tf_from_row_major(a: &[f64; 16]) -> SMatrix<f64, 4, 4> {
        let mut m = SMatrix::<f64, 4, 4>::zeros();
        for i in 0..4 {
            for j in 0..4 {
                m[(i, j)] = a[i * 4 + j];
            }
        }
        m
    }

    /// Joint vectors equal modulo 2*pi within `tol`.
    fn q_eq_mod(a: &[f64; 6], b: &[f64; 6], tol: f64) -> bool {
        for i in 0..6 {
            let d = libm::remainder(a[i] - b[i], 2.0 * core::f64::consts::PI);
            if libm::fabs(d) > tol {
                return false;
            }
        }
        true
    }

    /// A generic (non-singular) configuration inside every joint limit; round-trip tests use it.
    const REF_Q: [f64; 6] = [0.3, -0.8, 1.0, -0.4, 0.8, 0.2];

    #[test]
    fn fk_home_pose() {
        let zero = [0.0f64; 6];
        let t = fk(&zero);
        // T_HOME at q = zeros(6), baked from the ssik artifact (the URDF's home pose).
        let t_home = [
            1.806449006676831e-16, -2.9070459646034717e-18, 1.0, 0.39136190000000004,
            0.04866027636027259, 0.9988153870984077, -5.886628549115158e-18, -1.1246427803140535e-05,
            -0.9988153870984077, 0.04866027636027259, 1.8057236404777494e-16, 0.2264687446137826,
            0.0, 0.0, 0.0, 1.0,
        ];
        for i in 0..4 {
            for j in 0..4 {
                let expected = t_home[i * 4 + j];
                let got = t[(i, j)];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "fk home pose mismatch at ({i},{j}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn reference_pose_round_trips() {
        let tf = fk(&REF_Q);
        let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &REF_Q, 1e-6));
        assert!(found, "reference configuration not recovered");
    }

    #[test]
    fn matches_ssik_reference_dataset() {
        assert!(!test_data::ssik_version().is_empty());
        let tol = 1e-5;
        let mut total_expected = 0usize;
        let mut case_idx = 0usize;
        for (tf_arr, expected) in &test_data::cases() {
            let tf = tf_from_row_major(tf_arr);
            let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            let got: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();

            assert_eq!(
                got.len(),
                expected.len(),
                "case {case_idx}: solution count {} != expected {}",
                got.len(),
                expected.len()
            );

            for exp in expected.iter() {
                let matched = got.iter().any(|g| q_eq_mod(g, exp, tol));
                assert!(
                    matched,
                    "case {case_idx}: expected solution {:?} not reproduced",
                    exp
                );
                total_expected += 1;
            }
            case_idx += 1;
        }
        assert!(total_expected > 0, "no expected solutions in dataset");
    }

    #[test]
    fn respect_limits_drops_out_of_range() {
        // REF_Q is within all joint limits.
        let tf = fk(&REF_Q);
        // With limits on, REF_Q should still survive (it's within all limits).
        let sols_lim = solve(&tf, None, None, true, false, TolerancePolicy::DEFAULT, 15);
        let found = sols_lim.iter().flatten().any(|s| q_eq_mod(s, &REF_Q, 1e-6));
        assert!(found, "REF_Q should survive respect_limits=true");
    }

    #[test]
    fn max_solutions_cap() {
        let tf = fk(&REF_Q);
        let all_sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let n_all: usize = all_sols.iter().flatten().count();
        if n_all > 1 {
            let capped = solve(&tf, Some(1), None, false, false, TolerancePolicy::DEFAULT, 15);
            let n_capped: usize = capped.iter().flatten().count();
            assert_eq!(n_capped, 1, "max_solutions=1 should return exactly 1 solution");
        }
    }

    #[test]
    fn q_seed_sorts_nearest_first() {
        let tf = fk(&REF_Q);
        let zero = [0.0f64; 6];
        let sols = solve(&tf, None, Some(&zero), false, false, TolerancePolicy::DEFAULT, 15);
        let solutions: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
        if solutions.len() > 1 {
            // First solution must be the closest to seed.
            let d0 = seed_dist(&solutions[0], &zero);
            for s in &solutions[1..] {
                assert!(
                    d0 <= seed_dist(s, &zero) + 1e-12,
                    "q_seed sort: first solution is not the nearest to seed"
                );
            }
        }
    }
}
