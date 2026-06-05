//! Shared deterministic IK solver for the Flexiv **Rizon** arm family
//! (`jointlock.seven_r` outer sweep + Raghavan-Roth 6R inner solver).
//!
//! Both Rizon variants (Rizon 4 / Rizon 10) are 7R, non-SRS arms with no
//! closed-form 6R structure, so the `ssik` solver `jointlock.seven_r` collapses
//! them to a series of 6R problems by **locking joint 2** (no chain reversal) to
//! each of 16 fixed sample values. Unlike the xArm7 / Franka members of this
//! class — whose locked sub-chains are spherical — the Rizon sub-chains are
//! generic 6R, so each one is solved with the deterministic **Raghavan-Roth**
//! kernel that already backs the Piper driver
//! ([`crate::math::ssik_core`]'s `rr_*` functions).
//!
//! The two arms differ only in their baked geometry, so this module factors the
//! whole pipeline behind a [`Model`] of `&'static` per-sample constants and
//! per-sample [`PqBuilder`] function pointers. Each arm's `ik_solver` supplies a
//! `Model` built from its generated `consts` / `pq` modules.
//!
//! ## Algorithm
//!
//! For every lock sample `s`:
//!
//! 1. Bridge the target into the sub-chain DH frame:
//!    `T_dh = T_PRE_INV[s] · T · T_POST_INV[s]`.
//! 2. Evaluate the per-sample `(P_sin, P_cos, P_one, Q)` matrices with
//!    `build_pq[s]`, eliminate `q0, q1` through the left null space of `Q`, run
//!    the Weierstrass substitution, and extract the real `tan(q/2)` roots and
//!    kernels via the bounded `24×24` companion eigenvalue solve
//!    ([`rr_real_roots`]).
//! 3. Back-substitute each root to the six sub-chain joints, map DH→POE, and
//!    pad the locked angle back in at index 2
//!    (`full = [sub0, sub1, q_lock, sub2, sub3, sub4, sub5]`).
//!
//! The collected candidates are FK-verified against the full 7R chain
//! ([`fk`]'s closure `< subproblem_numerical`), de-duplicated modulo `2π`, and — when
//! `allow_refinement` is set — polished by a bounded Levenberg-Marquardt step
//! that holds the locked joint fixed ([`refine7`]).
//!
//! ## Deterministic-execution note
//!
//! As in the Piper port, the reference's randomised Möbius fallback for
//! ill-conditioned sub-systems is intentionally **not** ported: every numeric
//! loop carries a hard iteration ceiling and every buffer is a fixed-size,
//! stack-allocated array. The deterministic kernel reproduces every reference
//! solution and may additionally surface a few extra *genuine* roots (each
//! verified to close the pose to `< subproblem_numerical`).

use nalgebra::{SMatrix, SVector};

use crate::math::kinematics::{mat4_mul, rot_homogeneous};
pub use crate::math::tolerances::TolerancePolicy;
use crate::math::ssik_core::{rr_left_null_space, rr_pseudo_inverse, rr_real_roots, wrap_to_pi};

use core::f64::consts::PI;

/// Degrees of freedom of every Rizon arm.
pub const DOF: usize = 7;

/// Number of locked-joint samples in the sweep.
pub const N_SAMPLES: usize = 16;

/// Maximum inner Raghavan-Roth candidates kept per lock sample.
const INNER_MAX: usize = 16;

/// Upper bound on raw 7R candidates collected across the sweep
/// (`N_SAMPLES * INNER_MAX`) before de-duplication.
const CAND_CAP: usize = N_SAMPLES * INNER_MAX;

/// Upper bound on the number of distinct IK solutions returned. The Rizon
/// exhaustive sweep can surface more roots than the wrist-decoupled 7R arms
/// (the deterministic RR kernel adds genuine extra roots at some samples), so
/// the output array is sized to [`CAND_CAP`].
pub const MAX_SOLUTIONS: usize = CAND_CAP;

const TWO_PI: f64 = 2.0 * PI;

/// Bounded LM iterations used to sharpen an already-accepted root onto exact FK
/// closure when `allow_refinement` is set.
const REFINE_EPS: f64 = 1e-7;
const LM_INITIAL_LAMBDA: f64 = 1e-6;
const LM_CONVERGENCE_THRESHOLD: f64 = 1e-13;
const LM_LAMBDA_GROWTH: f64 = 8.0;
const LM_LAMBDA_DECAY: f64 = 0.5;
const LM_LAMBDA_MIN: f64 = 1e-12;
const LM_LAMBDA_MAX: f64 = 1e8;
const LM_BACKTRACK_MAX: usize = 12;

/// Signature of a per-sample Raghavan-Roth `(P_sin, P_cos, P_one, Q)` builder.
pub type PqBuilder = fn(
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) -> ([[f64; 9]; 14], [[f64; 9]; 14], [[f64; 9]; 14], [[f64; 8]; 14]);

/// Per-arm baked geometry and per-sample Raghavan-Roth derivation data.
///
/// All fields are `&'static` references into the arm's generated `consts` / `pq`
/// modules, so a `Model` is a thin, copyable bundle of pointers with no owned
/// data.
#[derive(Clone, Copy)]
pub struct Model {
    /// Locked joint index (pinned across the sweep; `2` for both Rizon arms).
    pub lock_idx: usize,
    /// Locked-joint sample schedule.
    pub lock_samples: &'static [f64; N_SAMPLES],
    /// Per-sample 6R sub-chain DH `alpha`.
    pub dh_alpha: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample 6R sub-chain DH `a`.
    pub dh_a: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample 6R sub-chain DH `d`.
    pub dh_d: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample DH→POE `theta` offsets.
    pub dh_theta_offset: &'static [[f64; 6]; N_SAMPLES],
    /// Per-sample `T_PRE_INV` bridge (row-major 4×4).
    pub t_pre_inv: &'static [[[f64; 4]; 4]; N_SAMPLES],
    /// Per-sample `T_POST_INV` bridge (row-major 4×4).
    pub t_post_inv: &'static [[[f64; 4]; 4]; N_SAMPLES],
    /// Per-sample RR linearity joint.
    pub linearity_joint: &'static [usize; N_SAMPLES],
    /// Per-sample RR left-bilinear joint pair.
    pub left_bilinear: &'static [(usize, usize); N_SAMPLES],
    /// Per-sample RR right-bilinear joint pair.
    pub right_bilinear: &'static [(usize, usize); N_SAMPLES],
    /// Per-sample RR dropped joint.
    pub drop_joint: &'static [usize; N_SAMPLES],
    /// Per-sample PQ builder table.
    pub build_pq: &'static [PqBuilder; N_SAMPLES],
    /// Full 7R joint axes (base frame at `q = 0`).
    pub axes7: &'static [[f64; 3]; 7],
    /// Full 7R per-joint `T_left` (row-major 4×4).
    pub t_lefts7: &'static [[[f64; 4]; 4]; 7],
    /// Full 7R per-joint `T_right` (row-major 4×4).
    pub t_rights7: &'static [[[f64; 4]; 4]; 7],
    /// Joint angle limits `(lo, hi)` in radians.
    pub joint_limits: &'static [(f64, f64); 7],
}

// ---------------------------------------------------------------------------
// Ratio candidates (Manocha-Canny IV-C robust selection) — arm-independent.
// ---------------------------------------------------------------------------

/// Ratio candidates for `x_left[0]`.
const X0_RATIOS: [(usize, usize); 7] = [(5, 8), (2, 5), (11, 2), (4, 7), (10, 1), (3, 6), (9, 0)];
/// Ratio candidates for `x_left[1]`.
const X1_RATIOS: [(usize, usize); 5] = [(7, 8), (6, 7), (1, 2), (4, 5), (10, 11)];

/// Weierstrass trig→`x` basis transform (arm-independent).
const W_TRIG_TO_X: [[f64; 9]; 9] = [
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
];

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Inverse kinematics for a Rizon arm described by `model`.
///
/// Mirrors `rizon{4,10}_ik.solve` with the same argument list. Returns up to
/// [`MAX_SOLUTIONS`] candidate joint vectors `[q1, .., q7]` in radians; unused
/// slots are [`None`].
#[allow(clippy::too_many_arguments)]
pub fn solve(
    model: &Model,
    t: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 7]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    let fk_atol = policy.subproblem_numerical;
    let dedup_atol = policy.subproblem_dedup;

    let mut solutions: [Option<[f64; 7]>; MAX_SOLUTIONS] = [None; MAX_SOLUTIONS];

    // Determine lock-sample traversal order. With a seed, visit samples whose
    // locked angle is nearest the seed's locked joint first (mirrors the Python
    // `q_seed` reordering of the sample schedule).
    let mut sample_order: [usize; N_SAMPLES] = [0; N_SAMPLES];
    for (i, slot) in sample_order.iter_mut().enumerate() {
        *slot = i;
    }
    if let Some(seed) = q_seed {
        let sl = seed[model.lock_idx];
        for i in 1..N_SAMPLES {
            let mut j = i;
            while j > 0 {
                let da = libm::fabs(model.lock_samples[sample_order[j - 1]] - sl);
                let db = libm::fabs(model.lock_samples[sample_order[j]] - sl);
                if da > db {
                    sample_order.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    }

    // Joint-lock sweep: collect 7R candidates with their full-chain FK residual.
    let mut cand_q = [[0.0f64; 7]; CAND_CAP];
    let mut cand_res = [0.0f64; CAND_CAP];
    let mut n_cand = 0usize;

    let mut inner = [[0.0f64; 6]; INNER_MAX];

    for &sample in sample_order.iter() {
        let q_lock = model.lock_samples[sample];
        let n_inner = solve_sample(model, sample, t, &mut inner);
        for sub in inner.iter().take(n_inner) {
            // Pad the locked angle back in at `lock_idx` (no chain reversal):
            // full = [sub0, sub1, q_lock, sub2, sub3, sub4, sub5] for lock_idx 2.
            let full = pad_locked(model.lock_idx, sub, q_lock);
            if !full.iter().all(|v| v.is_finite()) {
                continue;
            }
            let residual = fk_residual(model, &full, t);
            if residual > fk_atol {
                continue;
            }
            if n_cand < CAND_CAP {
                cand_q[n_cand] = full;
                cand_res[n_cand] = residual;
                n_cand += 1;
            }
        }
    }

    // De-duplicate modulo 2π, keeping the lower-residual representative.
    let mut d_q = [[0.0f64; 7]; CAND_CAP];
    let mut d_res = [0.0f64; CAND_CAP];
    let n_d = dedup7(&cand_q, &cand_res, n_cand, &mut d_q, &mut d_res, dedup_atol);

    let mut sol_count = 0usize;
    for i in 0..n_d {
        if sol_count >= MAX_SOLUTIONS {
            break;
        }
        let q = if allow_refinement {
            refine7(model, &d_q[i], t, refinement_max_iters)
        } else {
            d_q[i]
        };
        solutions[sol_count] = Some(q);
        sol_count += 1;
    }

    // Post-processing (mirrors Python's _ps_wrap_to_limits + _ps_respect_limits).
    if respect_limits {
        let mut filtered: [Option<[f64; 7]>; MAX_SOLUTIONS] = [None; MAX_SOLUTIONS];
        let mut count = 0usize;
        for sol in solutions.iter().take(sol_count).flatten() {
            let q = wrap_to_limits7(model, sol);
            if in_limits7(model, &q) && count < MAX_SOLUTIONS {
                filtered[count] = Some(q);
                count += 1;
            }
        }
        solutions = filtered;
        sol_count = count;
    }

    // Sort by seed distance (mirrors Python's _ps_nearest_to_seed).
    if let Some(seed) = q_seed {
        for i in 1..sol_count {
            let mut j = i;
            while j > 0 {
                if let (Some(prev), Some(curr)) = (solutions[j - 1], solutions[j]) {
                    if seed_dist7(&prev, seed) > seed_dist7(&curr, seed) {
                        solutions.swap(j - 1, j);
                        j -= 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    // Truncate to max_solutions (mirrors Python's `solutions[:max_solutions]`).
    if let Some(max) = max_solutions {
        let cap = max.min(sol_count);
        solutions[cap..sol_count].fill(None);
    }

    solutions
}

/// Pad the six sub-chain joints with the locked angle at `lock_idx`.
///
/// For the Rizon arms (`lock_idx = 2`, no chain reversal) this is
/// `[sub0, sub1, q_lock, sub2, sub3, sub4, sub5]`.
fn pad_locked(lock_idx: usize, sub: &[f64; 6], q_lock: f64) -> [f64; 7] {
    let mut full = [0.0f64; 7];
    let mut k = 0usize;
    for (i, slot) in full.iter_mut().enumerate() {
        if i == lock_idx {
            *slot = q_lock;
        } else {
            *slot = sub[k];
            k += 1;
        }
    }
    full
}

// ---------------------------------------------------------------------------
// Per-sample Raghavan-Roth inner solver (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Solve the locked 6R sub-chain for lock sample `sample`, writing up to
/// [`INNER_MAX`] sub-chain joint vectors to `out`. Returns the count.
fn solve_sample(
    model: &Model,
    sample: usize,
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]; INNER_MAX],
) -> usize {
    let t_pre_inv = mat4(&model.t_pre_inv[sample]);
    let t_post_inv = mat4(&model.t_post_inv[sample]);
    let t_dh = t_pre_inv * t * t_post_inv;

    let (p_sin_a, p_cos_a, p_one_a, q_a) = (model.build_pq[sample])(
        t_dh[(0, 0)],
        t_dh[(0, 1)],
        t_dh[(0, 2)],
        t_dh[(0, 3)],
        t_dh[(1, 0)],
        t_dh[(1, 1)],
        t_dh[(1, 2)],
        t_dh[(1, 3)],
        t_dh[(2, 0)],
        t_dh[(2, 1)],
        t_dh[(2, 2)],
        t_dh[(2, 3)],
    );

    let p_sin = mat14x9(&p_sin_a);
    let p_cos = mat14x9(&p_cos_a);
    let p_one = mat14x9(&p_one_a);
    let q_mat = mat14x8(&q_a);

    // Eliminate q_0, q_1 through the left null space of Q.
    let nt = rr_left_null_space(&q_mat).transpose(); // 6x14
    let e_sin = nt * p_sin;
    let e_cos = nt * p_cos;
    let e_one = nt * p_one;

    // Weierstrass half-angle substitution into the polynomial basis.
    let w = mat9x9(&W_TRIG_TO_X);
    let e_quad = (e_one - e_cos) * w;
    let e_lin = (e_sin * 2.0) * w;
    let e_const = (e_one + e_cos) * w;

    // Doubled 12x12 matrix polynomial.
    let m_quad = embed_e(&e_quad);
    let m_lin = embed_e(&e_lin);
    let m_const = embed_e(&e_const);

    // Real tan(q/2) roots + kernels via the 24x24 companion eigenvalue route.
    let mut roots = [0.0f64; 24];
    let mut vecs = [[0.0f64; 12]; 24];
    let mut cond = 0.0f64;
    let n_roots = rr_real_roots(&m_quad, &m_lin, &m_const, &mut roots, &mut vecs, &mut cond);

    let q_pinv = rr_pseudo_inverse(&q_mat);

    let mut count = 0usize;
    for r in 0..n_roots {
        if count >= INNER_MAX {
            break;
        }
        if let Some(q_dh) = back_substitute(
            model, sample, roots[r], &vecs[r], &p_sin, &p_cos, &p_one, &q_pinv, &t_dh,
        ) {
            let mut q = [0.0f64; 6];
            for i in 0..6 {
                q[i] = q_dh[i] - model.dh_theta_offset[sample][i];
            }
            out[count] = q;
            count += 1;
        }
    }
    count
}

/// Embed a 6x9 `E` into the 12x12 doubled block structure for `M`.
fn embed_e(e: &SMatrix<f64, 6, 9>) -> SMatrix<f64, 12, 12> {
    let mut m = SMatrix::<f64, 12, 12>::zeros();
    m.fixed_view_mut::<6, 9>(0, 0).copy_from(e);
    let shift = e.fixed_view::<6, 6>(0, 3).into_owned();
    m.fixed_view_mut::<6, 6>(6, 0).copy_from(&shift);
    let wrap = e.fixed_view::<6, 3>(0, 0).into_owned();
    m.fixed_view_mut::<6, 3>(6, 9).copy_from(&wrap);
    m
}

/// Pick `v_12[num] / v_12[den]` using the pair whose denominator has the
/// largest magnitude. Returns `None` if the chosen denominator is ~0.
fn ratio_from(v12: &[f64; 12], candidates: &[(usize, usize)]) -> Option<f64> {
    let mut best = candidates[0];
    let mut best_den = libm::fabs(v12[best.1]);
    for &(num, den) in &candidates[1..] {
        let mag = libm::fabs(v12[den]);
        if mag > best_den {
            best_den = mag;
            best = (num, den);
        }
    }
    if libm::fabs(v12[best.1]) < 1e-12 {
        return None;
    }
    Some(v12[best.0] / v12[best.1])
}

/// Branch-level back-substitution — port of
/// `_raghavan_roth._back_substitute_inner` with the per-sample RR metadata.
#[allow(clippy::too_many_arguments)]
fn back_substitute(
    model: &Model,
    sample: usize,
    x_lin: f64,
    v12: &[f64; 12],
    p_sin: &SMatrix<f64, 14, 9>,
    p_cos: &SMatrix<f64, 14, 9>,
    p_one: &SMatrix<f64, 14, 9>,
    q_pinv: &SMatrix<f64, 8, 14>,
    t_dh: &SMatrix<f64, 4, 4>,
) -> Option<[f64; 6]> {
    let q_lin = 2.0 * libm::atan(x_lin);

    let x_l0 = ratio_from(v12, &X0_RATIOS)?;
    let x_l1 = ratio_from(v12, &X1_RATIOS)?;

    let q_l0 = 2.0 * libm::atan(x_l0);
    let q_l1 = 2.0 * libm::atan(x_l1);

    let (s_lin, c_lin) = (libm::sin(q_lin), libm::cos(q_lin));
    let (s_l0, c_l0) = (libm::sin(q_l0), libm::cos(q_l0));
    let (s_l1, c_l1) = (libm::sin(q_l1), libm::cos(q_l1));

    let v_left = SVector::<f64, 9>::from_column_slice(&[
        s_l0 * s_l1,
        s_l0 * c_l1,
        c_l0 * s_l1,
        c_l0 * c_l1,
        s_l0,
        c_l0,
        s_l1,
        c_l1,
        1.0,
    ]);
    let p_eff = p_sin * s_lin + p_cos * c_lin + p_one; // 14x9
    let rhs = p_eff * v_left; // 14
    let v_right = q_pinv * rhs; // 8

    let q_r0 = libm::atan2(v_right[4], v_right[5]);
    let q_r1 = libm::atan2(v_right[6], v_right[7]);

    let linearity = model.linearity_joint[sample];
    let (lb0, lb1) = model.left_bilinear[sample];
    let (rb0, rb1) = model.right_bilinear[sample];
    let drop = model.drop_joint[sample];

    let mut q = [0.0f64; 6];
    q[linearity] = q_lin;
    q[lb0] = q_l0;
    q[lb1] = q_l1;
    q[rb0] = q_r0;
    q[rb1] = q_r1;

    // Recover the dropped joint from the FK residual.
    let alpha = &model.dh_alpha[sample];
    let a = &model.dh_a[sample];
    let d = &model.dh_d[sample];
    let mut chain_before = SMatrix::<f64, 4, 4>::identity();
    for i in 0..drop {
        chain_before *= dh_matrix(q[i], alpha[i], a[i], d[i]);
    }
    let mut chain_after = SMatrix::<f64, 4, 4>::identity();
    for i in (drop + 1)..6 {
        chain_after *= dh_matrix(q[i], alpha[i], a[i], d[i]);
    }
    let before_inv = chain_before.try_inverse()?;
    let after_inv = chain_after.try_inverse()?;
    let a_drop_residual = before_inv * t_dh * after_inv;
    q[drop] = libm::atan2(a_drop_residual[(1, 0)], a_drop_residual[(0, 0)]);

    if q.iter().all(|v| v.is_finite()) {
        Some(q)
    } else {
        None
    }
}

/// Numeric DH transform matrix — mirrors `_raghavan_roth._dh_matrix_num`.
fn dh_matrix(theta: f64, alpha: f64, a: f64, d: f64) -> SMatrix<f64, 4, 4> {
    let (st, ct) = (libm::sin(theta), libm::cos(theta));
    let (sa, ca) = (libm::sin(alpha), libm::cos(alpha));
    SMatrix::<f64, 4, 4>::from_row_slice(&[
        ct, -st * ca, st * sa, a * ct, st, ct * ca, -ct * sa, a * st, 0.0, sa, ca, d, 0.0, 0.0,
        0.0, 1.0,
    ])
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// Product-of-exponentials forward kinematics of the full 7R chain — mirrors
/// `rizon{4,10}_ik.fk`. Returns the 4×4 SE(3) pose of the flange in `base_link`.
pub fn fk(model: &Model, q: &[f64; 7]) -> SMatrix<f64, 4, 4> {
    let mut acc = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for i in 0..7 {
        let rot = rot_homogeneous(&model.axes7[i], q[i]);
        let m = mat4_mul(&model.t_lefts7[i], &rot);
        let n = mat4_mul(&m, &model.t_rights7[i]);
        acc = mat4_mul(&acc, &n);
    }
    SMatrix::<f64, 4, 4>::from_fn(|i, j| acc[i][j])
}

/// Frobenius-norm residual between FK(q) and the target pose.
fn fk_residual(model: &Model, q: &[f64; 7], target: &SMatrix<f64, 4, 4>) -> f64 {
    let t = fk(model, q);
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = t[(i, j)] - target[(i, j)];
            s += d * d;
        }
    }
    libm::sqrt(s)
}

// ---------------------------------------------------------------------------
// Levenberg-Marquardt refinement (holds the locked joint fixed)
// ---------------------------------------------------------------------------

/// 12-vector pose residual `[vec(R(q) − R_t); t(q) − t_t]` of the full 7R FK.
fn residual12(model: &Model, q: &[f64; 7], target: &SMatrix<f64, 4, 4>) -> [f64; 12] {
    let tf = fk(model, q);
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

fn norm12(r: &[f64; 12]) -> f64 {
    let mut s = 0.0;
    for v in r.iter() {
        s += v * v;
    }
    libm::sqrt(s)
}

/// Levenberg-Marquardt polish of a 7R configuration, **holding the locked joint
/// fixed**. Refining only the six free joints targets the isolated 6R root and
/// keeps the redundant 7R from drifting along its self-motion manifold.
fn refine7(
    model: &Model,
    q_in: &[f64; 7],
    target: &SMatrix<f64, 4, 4>,
    max_iters: usize,
) -> [f64; 7] {
    // Free joints = all but the locked one.
    let mut free = [0usize; 6];
    let mut fi = 0usize;
    for i in 0..7 {
        if i != model.lock_idx {
            free[fi] = i;
            fi += 1;
        }
    }

    let mut q = *q_in;
    let mut lambda = LM_INITIAL_LAMBDA;
    let mut r = residual12(model, &q, target);
    let mut rn = norm12(&r);

    for _ in 0..max_iters {
        if rn < LM_CONVERGENCE_THRESHOLD {
            break;
        }
        let mut jac = SMatrix::<f64, 12, 6>::zeros();
        for (c, &jidx) in free.iter().enumerate() {
            let mut qp = q;
            qp[jidx] += REFINE_EPS;
            let rp = residual12(model, &qp, target);
            for row in 0..12 {
                jac[(row, c)] = (rp[row] - r[row]) / REFINE_EPS;
            }
        }
        let rv = SMatrix::<f64, 12, 1>::from_column_slice(&r);
        let jt = jac.transpose();
        let jtj = jt * jac; // 6x6
        let jtr = jt * rv; // 6x1

        let mut accepted = false;
        for _ in 0..LM_BACKTRACK_MAX {
            let mut damped = jtj;
            for d in 0..6 {
                damped[(d, d)] += lambda;
            }
            let inv = match damped.try_inverse() {
                Some(m) => m,
                None => {
                    lambda *= LM_LAMBDA_GROWTH;
                    continue;
                }
            };
            let dq = inv * jtr;
            let mut q_trial = q;
            for (c, &jidx) in free.iter().enumerate() {
                q_trial[jidx] -= dq[(c, 0)];
            }
            let r_trial = residual12(model, &q_trial, target);
            let rn_trial = norm12(&r_trial);
            if rn_trial < rn {
                q = q_trial;
                r = r_trial;
                rn = rn_trial;
                lambda = (lambda * LM_LAMBDA_DECAY).max(LM_LAMBDA_MIN);
                accepted = true;
                break;
            }
            lambda *= LM_LAMBDA_GROWTH;
            if lambda > LM_LAMBDA_MAX {
                break;
            }
        }
        if !accepted {
            break;
        }
    }
    q
}

// ---------------------------------------------------------------------------
// De-duplication and post-processing helpers
// ---------------------------------------------------------------------------

/// `true` if joint vectors `a` and `b` agree (mod `2π`) within `tol` per axis.
fn q_close7(a: &[f64; 7], b: &[f64; 7], tol: f64) -> bool {
    for i in 0..7 {
        if libm::fabs(wrap_to_pi(a[i] - b[i])) > tol {
            return false;
        }
    }
    true
}

/// Dedup 7-vectors by wrap-to-pi closeness within `tol`, keeping the
/// lower-residual representative on collision.
fn dedup7(
    q: &[[f64; 7]; CAND_CAP],
    res: &[f64; CAND_CAP],
    n: usize,
    out_q: &mut [[f64; 7]; CAND_CAP],
    out_res: &mut [f64; CAND_CAP],
    tol: f64,
) -> usize {
    let mut m = 0usize;
    for i in 0..n {
        let mut dup = usize::MAX;
        for j in 0..m {
            if q_close7(&q[i], &out_q[j], tol) {
                dup = j;
                break;
            }
        }
        if dup == usize::MAX {
            out_q[m] = q[i];
            out_res[m] = res[i];
            m += 1;
        } else if res[i] < out_res[dup] {
            out_q[dup] = q[i];
            out_res[dup] = res[i];
        }
    }
    m
}

/// Try `q[i] ± 2π` per joint to bring the vector into the model's joint limits.
fn wrap_to_limits7(model: &Model, q: &[f64; 7]) -> [f64; 7] {
    let mut out = *q;
    for i in 0..7 {
        let (lo, hi) = model.joint_limits[i];
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
fn in_limits7(model: &Model, q: &[f64; 7]) -> bool {
    for i in 0..7 {
        let (lo, hi) = model.joint_limits[i];
        if q[i] < lo || q[i] > hi {
            return false;
        }
    }
    true
}

/// Squared wrap-to-π distance between `q` and `seed`. Used for seed sorting.
fn seed_dist7(q: &[f64; 7], seed: &[f64; 7]) -> f64 {
    let mut s = 0.0;
    for i in 0..7 {
        let d = wrap_to_pi(q[i] - seed[i]);
        s += d * d;
    }
    s
}

// ---------------------------------------------------------------------------
// Small fixed-size array → nalgebra conversions
// ---------------------------------------------------------------------------

fn mat4(a: &[[f64; 4]; 4]) -> SMatrix<f64, 4, 4> {
    SMatrix::<f64, 4, 4>::from_fn(|i, j| a[i][j])
}

fn mat9x9(a: &[[f64; 9]; 9]) -> SMatrix<f64, 9, 9> {
    SMatrix::<f64, 9, 9>::from_fn(|i, j| a[i][j])
}

fn mat14x9(a: &[[f64; 9]; 14]) -> SMatrix<f64, 14, 9> {
    SMatrix::<f64, 14, 9>::from_fn(|i, j| a[i][j])
}

fn mat14x8(a: &[[f64; 8]; 14]) -> SMatrix<f64, 14, 8> {
    SMatrix::<f64, 14, 8>::from_fn(|i, j| a[i][j])
}
