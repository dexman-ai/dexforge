//! Analytical inverse kinematics for the Universal Robots **UR5e** (arm name `ur5`).
//!
//! This is a line-by-line translation of the Python `ssik` artifact
//! `codegen/ur5_ik.py`, built from `ur5.urdf` (the UR5e imported from MuJoCo
//! Menagerie; solver `ikgeo.three_parallel`, tier 0), into a deterministic,
//! `#![no_std]`, zero-allocation Rust implementation.
//!
//! The closed form is: subproblem **SP6** for the coupled wrist/shoulder angle
//! pair `(q1, q5)`, followed by the inlined `SP1 + SP3 + SP1` chain for the
//! remaining joints. SP6 reduces to intersecting two conics, which produces a
//! quartic solved through the companion-matrix eigenvalue routine in
//! [`crate::math::ssik_core`].
//!
//! ## Equivalence to the Python reference
//!
//! [`solve`] mirrors `ur5_ik.solve` with the same argument list:
//! `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`, `policy`,
//! and `refinement_max_iters`. The algebraic solution set is FK-verified to
//! [`TolerancePolicy::subproblem_numerical`] and de-duplicated modulo `2*pi`.
//!
//! [`fk`] mirrors `ur5_ik.fk` (alias of `_fk`): product-of-exponentials forward
//! kinematics using the baked chain constants.

use nalgebra::SMatrix;

use dexhal::math::kinematics::{lm_refine6, mat4_frob_diff};
use dexhal::math::ssik_core::{sp6_solve, wrap_to_pi};
pub use dexhal::math::tolerances::TolerancePolicy;

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the UR5.
pub const DOF: usize = 6;

/// Base link name in the URDF.
pub const BASE_LINK: &str = "base_link";

/// End-effector link name in the URDF.
pub const EE_LINK: &str = "ee_link";

// --- algebraic-block degeneracy constants from ur5_ik.py ---
const TWO_PI: f64 = 2.0 * PI;

/// Joint angle limits (lo, hi) in radians — baked from `_JOINT_LIMITS` in
/// the Python artifact.  Used by [`solve`] when `respect_limits = true`.
const JOINT_LIMITS: [(f64, f64); 6] = [
    (-6.283185307179586, 6.283185307179586),
    (-6.283185307179586, 6.283185307179586),
    (-3.141592653589793, 3.141592653589793),
    (-6.283185307179586, 6.283185307179586),
    (-6.283185307179586, 6.283185307179586),
    (-6.283185307179586, 6.283185307179586),
];

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Inverse kinematics — mirrors `ur5_ik.solve`.
///
/// # Arguments
///
/// * `t` — 4×4 SE(3) target pose of `ee_link` in `base_link`.
/// * `max_solutions` — cap on returned solutions (post-dedup, post-limits).
///   `None` returns the full algebraic set (at most 8 for the UR5).
/// * `q_seed` — when `Some`, solutions are sorted by wrap-to-pi distance
///   from the seed (closest first). Has no effect on *which* solutions are
///   returned.
/// * `respect_limits` — when `true`, solutions outside the baked URDF joint
///   limits are dropped (after a `±2π` wrap-into-limits attempt). Pass
///   `false` for the raw geometric set.
/// * `allow_refinement` — when `true`, algebraic candidates that exceed
///   `policy.subproblem_numerical` are polished with a bounded Levenberg-Marquardt step.
///   For the UR5 (tier-0 arm) this is rarely needed; default `false`.
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

    // Raw algebraic candidates (stack array, at most 8 for the UR5).
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

/// 12-vector pose residual `[vec(R(q) − R_t); t(q) − t_t]` for the UR5.
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

/// Levenberg-Marquardt polish of a 6-DOF UR5 configuration onto the exact
/// pose root. Mirrors the Python `_lm_refine` path used when
/// `allow_refinement = True`. Returns `Some(q_refined)` if the FK residual
/// converges within `fk_atol`, otherwise `None`.
fn refine6(
    q_in: &[f64; 6],
    target: &SMatrix<f64, 4, 4>,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 6]> {
    lm_refine6(q_in, |q| pose_residual12(q, target), max_iters, fk_atol)
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
    let p_16_x = p_x - 1.22464679914735e-17 * r_00 - 6.12323399573677e-18 * r_01 - 0.1 * r_02;
    let p_16_y = p_y - 1.22464679914735e-17 * r_10 - 6.12323399573677e-18 * r_11 - 0.1 * r_12;
    let p_16_z =
        p_z - 1.22464679914735e-17 * r_20 - 6.12323399573677e-18 * r_21 - 0.1 * r_22 - 0.163;
    let r_06_axes5_x = 6.12323399573677e-17 * r_01 + 1.0 * r_02;
    let r_06_axes5_y = 6.12323399573677e-17 * r_11 + 1.0 * r_12;
    let r_06_axes5_z = 6.12323399573677e-17 * r_21 + 1.0 * r_22;

    const H_SP_0: [f64; 3] = [-1.2246467991473532e-16, -1.0, 0.0];
    const K_SP_0: [f64; 3] = [-0.0, -0.0, -1.0];
    const K_SP_1: [f64; 3] = [-1.2246467991473532e-16, 1.4997597826618576e-32, -1.0];
    const NEG_P_5: [f64; 3] = [-0.0, -0.0, 0.1];
    const NEG_AXES_5: [f64; 3] = [1.2246467991473532e-16, 1.0, -0.0];

    let p_16 = [p_16_x, p_16_y, p_16_z];
    let r_06_axes5 = [r_06_axes5_x, r_06_axes5_y, r_06_axes5_z];
    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(
        &h_sp,
        &k_sp,
        &p_sp,
        0.134,
        0.0,
        policy.subproblem_numerical,
        policy.subproblem_degeneracy,
        &mut theta15,
    );

    for si in 0..n15 {
        let (q1, q5) = theta15[si];

        // NOTE: the literal constants below are the machine-generated symbolic
        // coefficients emitted by `ssik build`, copied verbatim from `ur5_ik.py`
        // (mechanically translated; most are rounding residue of cos(pi/2)).
        // SP1 for theta14 = q1+q2+q3+q4 (sum of parallel-axis rotations).
        let th14_x0 = 6.12323399573677e-17 * r_21 + 1.0 * r_22;
        let th14_x1 = libm::sin(q5);
        let th14_x2 = 1.0 * th14_x1;
        let th14_x3 = libm::sin(q1);
        let th14_x4 = -1.22464679914735e-16 * r_00 - 6.12323399573677e-17 * r_01 - 1.0 * r_02;
        let th14_x5 = 1.0 * th14_x4;
        let th14_x6 = th14_x3 * th14_x5;
        let th14_x7 = 1.0 * r_00 + 1.22464679914735e-16 * r_01 - 1.22464679914735e-16 * r_02;
        let th14_x8 = 1.22464679914735e-16 * th14_x7;
        let th14_x9 = th14_x3 * th14_x8;
        let th14_x10 = libm::cos(q1);
        let th14_x11 = -1.22464679914735e-16 * r_10 - 6.12323399573677e-17 * r_11 - 1.0 * r_12;
        let th14_x12 = 1.0 * th14_x11;
        let th14_x13 = th14_x10 * th14_x12;
        let th14_x14 = 1.0 * r_10 + 1.22464679914735e-16 * r_11 - 1.22464679914735e-16 * r_12;
        let th14_x15 = 1.22464679914735e-16 * th14_x14;
        let th14_x16 = th14_x10 * th14_x15;
        let th14_x17 = -th14_x13 - th14_x16 + th14_x6 + th14_x9;
        let th14_x18 =
            -th14_x10 * th14_x5 - th14_x10 * th14_x8 - th14_x12 * th14_x3 - th14_x15 * th14_x3;
        let th14_x19 = 1.22464679914735e-16 * th14_x1;
        let th14_x20 = libm::cos(q5);
        let th14_x21 = 1.0 * th14_x20;
        let theta14 = libm::atan2(
            -th14_x0 * th14_x2 + 1.49975978266186e-32 * th14_x1 * th14_x17 - th14_x18 * th14_x19,
            th14_x0 * th14_x19
                + th14_x17 * (th14_x19 - th14_x21)
                + th14_x18 * (-th14_x2 - 1.22464679914735e-16 * th14_x20)
                - th14_x21
                    * (1.22464679914735e-16 * th14_x10 * th14_x4
                        + 1.49975978266186e-32 * th14_x10 * th14_x7
                        + 1.22464679914735e-16 * th14_x11 * th14_x3
                        + th14_x13
                        + 1.49975978266186e-32 * th14_x14 * th14_x3
                        + th14_x16
                        - th14_x6
                        - th14_x9),
        );
        // SP1 for q6 (wrist roll-2): closed-form atan2.
        let q6_x0 = libm::sin(q5);
        let q6_x1 = libm::sin(q1);
        let q6_x2 = 1.22464679914735e-16 * r_00;
        let q6_x3 = -q6_x2 - 6.12323399573677e-17 * r_01 - 1.0 * r_02;
        let q6_x4 = libm::cos(q1);
        let q6_x5 = 1.22464679914735e-16 * q6_x4;
        let q6_x6 = 1.22464679914735e-16 * r_10;
        let q6_x7 = -q6_x6 - 6.12323399573677e-17 * r_11 - 1.0 * r_12;
        let q6_x8 = 1.22464679914735e-16 * q6_x1;
        let q6_x9 = 1.0 * q6_x4;
        let q6_x10 = -1.0 * q6_x1 * q6_x3 + q6_x3 * q6_x5 + q6_x7 * q6_x8 + q6_x7 * q6_x9;
        let q6_x11 = -q6_x10;
        let q6_x12 = -q6_x2 + 1.0 * r_01 - 6.12323399573677e-17 * r_02;
        let q6_x13 = -q6_x6 + 1.0 * r_11 - 6.12323399573677e-17 * r_12;
        let q6_x14 = 1.0 * q6_x1 * q6_x12 - q6_x12 * q6_x5 - q6_x13 * q6_x8 - q6_x13 * q6_x9;
        let q6_x15 = 1.0 * q6_x0;
        let q6_x16 = 1.0 * r_00 + 1.22464679914735e-16 * r_01 - 1.22464679914735e-16 * r_02;
        let q6_x17 = 1.0 * r_10 + 1.22464679914735e-16 * r_11 - 1.22464679914735e-16 * r_12;
        let q6_x18 = 1.0 * q6_x1 * q6_x16 - q6_x16 * q6_x5 - q6_x17 * q6_x8 - q6_x17 * q6_x9;
        let q6_x19 = 1.22464679914735e-16 * q6_x0;
        let q6_x20 = libm::cos(q5);
        let q6_x21 = 1.0 * q6_x20;
        let q6 = libm::atan2(
            1.49975978266186e-32 * q6_x0 * q6_x11 - q6_x14 * q6_x15 - q6_x18 * q6_x19,
            q6_x11 * (-q6_x19 - q6_x21) - q6_x14 * q6_x19
                + q6_x18 * (q6_x15 - 1.22464679914735e-16 * q6_x20)
                + q6_x21
                    * (1.22464679914735e-16 * q6_x1 * q6_x16
                        - 1.49975978266186e-32 * q6_x1 * q6_x17
                        - q6_x10
                        - 1.49975978266186e-32 * q6_x16 * q6_x4
                        - q6_x17 * q6_x5),
        );
        // d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
        let dinr_x0 = libm::sin(theta14);
        let dinr_x1 = libm::cos(theta14);
        let dinr_x2 = libm::cos(q1);
        let dinr_x3 = p_x - 1.22464679914735e-17 * r_00 - 6.12323399573677e-18 * r_01 - 0.1 * r_02;
        let dinr_x4 = p_y - 1.22464679914735e-17 * r_10 - 6.12323399573677e-18 * r_11 - 0.1 * r_12;
        let dinr_x5 = 1.0 * libm::sin(q1);
        let dinr_x6 = 1.22464679914735e-16 - 1.22464679914735e-16 * dinr_x1;
        let dinr_x7 = libm::sin(q5);
        let dinr_x8 = libm::cos(q5);
        let dinr_x9 =
            1.22464679914735e-16 * dinr_x7 + 1.49975978266186e-32 * dinr_x8 - 1.49975978266186e-32;
        let dinr_x10 =
            1.49975978266186e-32 * dinr_x7 - 1.22464679914735e-16 * dinr_x8 + 1.22464679914735e-16;
        let dinr_x11 = 0.1 * dinr_x10;
        let dinr_x12 = 1.22464679914735e-17 * dinr_x0;
        let d_inner_x = -0.1 * dinr_x0 - 1.55530143491714e-17 * dinr_x1
            + dinr_x11 * (1.0 * dinr_x1 + 1.49975978266186e-32)
            + dinr_x2 * dinr_x3
            + dinr_x4 * dinr_x5
            + 0.1 * dinr_x6 * dinr_x9
            + 3.24531401774049e-17;
        let d_inner_y = dinr_x11 * dinr_x6 + dinr_x12 + dinr_x2 * dinr_x4 - dinr_x3 * dinr_x5
            + 1.22464679914735e-17 * dinr_x7
            + 1.49975978266186e-33 * dinr_x8
            + 0.265;
        let d_inner_z = 0.1 * dinr_x0 * dinr_x10 - 1.55530143491714e-17 * dinr_x0 + 0.1 * dinr_x1
            - dinr_x12 * dinr_x9
            + 1.0 * p_z
            - 1.22464679914735e-17 * r_20
            - 6.12323399573677e-18 * r_21
            - 0.1 * r_22
            - 0.163;
        let d_elbow =
            libm::sqrt(d_inner_x * d_inner_x + d_inner_y * d_inner_y + d_inner_z * d_inner_z);
        // SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        let q3_r_sq = 0.0277555600000000_f64;
        let q3_rhs = 0.175725 - 0.5 * d_elbow * d_elbow;
        let q3_phi = 5.49780309313178e-18 - PI;
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
        // LS fallback: theta = phi (or phi + pi if rhs < 0)
        for q3 in [theta_q3_plus, theta_q3_minus] {
            // SP1 for q2 (shoulder pitch): closed-form atan2.
            let q2_x0 = libm::sin(q3);
            let q2_x1 = libm::cos(q3);
            let q2_x2 = -0.392 * q2_x0 + 2.77555756156289e-17 * q2_x1 + 2.77555756156289e-17;
            let q2_x3 = -2.77555756156289e-17 * q2_x0 - 0.392 * q2_x1 - 0.425;
            let q2 = libm::atan2(
                -d_inner_x * q2_x2
                    + d_inner_y
                        * (-4.80061545265762e-17 * q2_x0
                            + 3.39907768361723e-33 * q2_x1
                            + 3.39907768361723e-33)
                    + d_inner_z * q2_x3,
                d_inner_x * q2_x3 - 1.60428730688303e-17 * d_inner_x
                    + d_inner_y
                        * (3.39907768361723e-33 * q2_x0 + 4.80061545265762e-17 * q2_x1 + 0.131)
                    - 0.131 * d_inner_y
                    + d_inner_z * q2_x2,
            );
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

/// Product-of-exponentials forward kinematics — mirrors `ur5_ik.fk`.
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

    /// A generic (non-singular) configuration. q = 0 is a wrist singularity for
    /// the UR5e (q5 = 0), where the solver returns only one shoulder branch (see
    /// `wrist_singularity_still_solves`), so round-trip tests use this instead.
    const REF_Q: [f64; 6] = [0.3, -1.0, 1.2, -0.5, 0.8, 0.2];

    #[test]
    fn fk_home_pose() {
        let zero = [0.0f64; 6];
        let t = fk(&zero);
        // Python T_HOME at q = zeros(6)
        let t_home = [
            1.0, 0.0, 0.0, -0.817,
            0.0, 0.0, -1.0, -0.234,
            0.0, 1.0, 0.0, 0.063,
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

    /// Wrist singularity (q5 = 0 or pi): joints 4 and 6 share an axis, so only their
    /// combination is defined and the original q4/q6 cannot be recovered.
    ///
    /// At (and within ~1e-6 of) the singularity the solver returns 4 solutions on a single
    /// shoulder branch instead of 8, and that branch need not be the one the arm is in. The
    /// Python reference artifact behaves identically (checked), so this is solver behaviour,
    /// not a porting error. What must hold: solutions exist, are finite, and reproduce the
    /// pose. Just off the singularity (q5 = 1e-3) all 8 are back and the original q1..q3, q5
    /// are among them.
    #[test]
    fn wrist_singularity_still_solves() {
        use core::f64::consts::PI;
        for q5 in [0.0, 1e-9, 1e-6, 1e-3, PI] {
            let q = [0.3, -1.0, 1.2, -0.5, q5, 0.2];
            let tf = fk(&q);
            let sols = solve(&tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            let all: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
            assert!(!all.is_empty(), "q5={q5}: no solution at/near the singularity");

            for s in &all {
                assert!(s.iter().all(|x| x.is_finite()), "q5={q5}: non-finite solution {s:?}");
                let t2 = fk(s);
                for i in 0..3 {
                    for j in 0..4 {
                        assert!(
                            (t2[(i, j)] - tf[(i, j)]).abs() < 1e-8,
                            "q5={q5}: solution {s:?} does not reproduce the pose at ({i},{j})"
                        );
                    }
                }
            }
        }

        // Just off the singularity the full solution set (and the original arm joints) returns.
        let q = [0.3, -1.0, 1.2, -0.5, 1e-3, 0.2];
        let sols = solve(&fk(&q), None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let all: Vec<[f64; 6]> = sols.iter().flatten().copied().collect();
        assert_eq!(all.len(), 8, "expected the full 8 solutions just off the singularity");
        let ok = all.iter().any(|s| {
            q_eq_mod(&[s[0], s[1], s[2], 0.0, s[4], 0.0], &[q[0], q[1], q[2], 0.0, q[4], 0.0], 1e-6)
        });
        assert!(ok, "original q1..q3, q5 not among {all:?}");
    }
}
