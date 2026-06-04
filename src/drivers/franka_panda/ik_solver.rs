//! Analytical inverse kinematics for the **Franka Emika Panda** (7-DOF).
//!
//! This is a deterministic, `#![no_std]`, zero-allocation translation of the
//! Python `ssik` prebuilt artifact `ssik/prebuilt/franka_panda_ik.py`
//! (solver `jointlock.seven_r`, tier 1) into Rust.
//!
//! ## Algorithm
//!
//! The Franka is a redundant 7R arm with no closed-form 6R structure, so the
//! `ssik` solver collapses it to a series of 6R problems by **locking joint 4**
//! to each of 16 fixed sample values (`np.linspace` over the joint-4 range,
//! endpoint excluded). For each sample the remaining six joints form a 6R
//! sub-chain; reversing that chain (the EAIK `REVERSED` decomposition) places
//! its spherical wrist at the canonical position, where an inner ikgeo solver
//! applies:
//!
//! * `spherical` — for 15 of the 16 samples (SP5 shoulder + SP4/SP1 wrist), and
//! * `spherical_two_parallel` — for sample index 8 (SP4/SP3/SP1).
//!
//! The locked/reversed sub-chain geometry is a pure function of the arm
//! topology, so it is baked once by `scripts/generate_franka_consts.py` into
//! [`super::consts`]. At runtime each inner solve is verified against the
//! reversed sub-chain's forward kinematics, mapped back to original-chain
//! ordering, padded with the locked angle, FK-verified against the full 7R
//! chain, and de-duplicated modulo `2*pi`.
//!
//! ## Equivalence to the Python reference
//!
//! [`compute_ik`] reproduces `franka_panda_ik.solve(T, respect_limits=False)`:
//! the raw geometric solution set, FK-verified to `1e-5` and de-duplicated to
//! `1e-3` per joint. The exhaustive 16-sample sweep yields up to **128**
//! candidates (`16 * 8`); after dedup the count is typically 16–128, so unlike
//! the 6R drivers the output array holds 128 slots rather than 16. Unused
//! slots are `None`.
//!
//! Each de-duplicated root is finally polished by [`refine7`], a bounded
//! Levenberg-Marquardt step that holds the locked joint 4 fixed and refines the
//! six free joints onto the exact pose. Distinct roots are far apart (`>1e-3`),
//! so polishing never merges or splits the solution set; it only removes the
//! tiny `libm`-vs-`numpy` drift that the inner algebraic solvers accumulate at
//! near-singular configurations, guaranteeing the strict `1e-5` joint-space
//! agreement asserted by the reference test. The reference dataset is generated
//! with the byte-identical Levenberg-Marquardt polish, so both sides converge
//! to the same isolated root.

use nalgebra::SMatrix;

use crate::math::ssik_core::{
    add3, dot3, mat3_mul, mat3_transpose, mat3_vec, mat3t_vec, neg3, norm3, rotate,
    rotation_matrix, sp1_solve, sp3_solve, sp4_solve, sp5_solve, sub3, wrap_to_pi,
};

use super::consts;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the Franka Panda.
pub const DOF: usize = 7;

/// Upper bound on the number of distinct IK solutions (16 lock samples x up to
/// 8 inner solutions each). The Franka's exhaustive sweep can saturate this.
pub const MAX_SOLUTIONS: usize = 128;

// --- tolerances (ssik DEFAULT_TOLERANCE_POLICY) ---
const NUM_TOL: f64 = 1e-5; // subproblem_numerical
const DEG_TOL: f64 = 1e-12; // subproblem_degeneracy
const FEAS_TOL: f64 = 1e-9; // subproblem_feasibility
const DEDUP_TOL: f64 = 1e-3; // subproblem_dedup
const FK_ATOL: f64 = 1e-5; // FK closure tolerance (== subproblem_numerical)

/// Maximum inner-solver candidates per lock sample. The `spherical` solver can
/// return up to [`crate::math::ssik_core`]'s 16 SP5 shoulder triples (4 quartic
/// roots x 4 sign branches) and SP4 contributes up to 2 wrist branches each, so
/// 32 bounds every reachable case; `spherical_two_parallel` stays at 8.
const INNER_MAX: usize = 32;

/// Upper bound on raw 7R candidates collected across the 16-sample lock sweep
/// (`16 * INNER_MAX`) before de-duplication.
const CAND_CAP: usize = 16 * INNER_MAX;

/// A rigid transform stored as a rotation block and translation.
#[derive(Clone, Copy)]
struct Transform {
    r: [[f64; 3]; 3],
    t: [f64; 3],
}

const IDENTITY: Transform = Transform {
    r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    t: [0.0, 0.0, 0.0],
};

#[inline]
fn compose(a: &Transform, b: &Transform) -> Transform {
    Transform {
        r: mat3_mul(&a.r, &b.r),
        t: add3(&mat3_vec(&a.r, &b.t), &a.t),
    }
}

/// SE(3) inverse `[R^T | -R^T t]`.
#[inline]
fn se3_inv(tf: &Transform) -> Transform {
    let rt = mat3_transpose(&tf.r);
    Transform {
        r: rt,
        t: neg3(&mat3_vec(&rt, &tf.t)),
    }
}

/// Frobenius norm of the 4x4 difference between two transforms (the bottom
/// row `[0, 0, 0, 1]` is identical, so it contributes nothing).
#[inline]
fn frob_diff(a: &Transform, b: &Transform) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = a.r[i][j] - b.r[i][j];
            s += d * d;
        }
        let d = a.t[i] - b.t[i];
        s += d * d;
    }
    libm::sqrt(s)
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// POE forward kinematics of the reversed 6R sub-chain `sample`.
fn fk_sub(sample: usize, q: &[f64; 6]) -> Transform {
    let axes = &consts::SUB_AXES[sample];
    let tl = &consts::SUB_TL[sample];
    let mut t = IDENTITY;
    for i in 0..6 {
        let l = Transform {
            r: IDENTITY.r,
            t: tl[i],
        };
        let rj = Transform {
            r: rotation_matrix(&axes[i], q[i]),
            t: [0.0, 0.0, 0.0],
        };
        t = compose(&t, &l);
        t = compose(&t, &rj);
        if i == 5 {
            let tr = Transform {
                r: consts::SUB_TRR[sample],
                t: consts::SUB_TRT[sample],
            };
            t = compose(&t, &tr);
        }
    }
    t
}

/// POE forward kinematics of the full 7R Franka chain.
fn fk7(q: &[f64; 7]) -> Transform {
    let mut t = IDENTITY;
    for i in 0..7 {
        let l = Transform {
            r: IDENTITY.r,
            t: consts::TL7[i],
        };
        let rj = Transform {
            r: rotation_matrix(&consts::AXES7[i], q[i]),
            t: [0.0, 0.0, 0.0],
        };
        let tr = Transform {
            r: consts::TR7_R[i],
            t: consts::TR7_T[i],
        };
        t = compose(&t, &l);
        t = compose(&t, &rj);
        t = compose(&t, &tr);
    }
    t
}

// ---------------------------------------------------------------------------
// Inner 6R solvers (operate on the reversed sub-chain `sample`, target `tt`)
// ---------------------------------------------------------------------------

/// Wrist back-substitution shared by both inner solvers: given `r36` and the
/// three shoulder angles, append every `(q4, q5, q6)` branch as a candidate.
fn wrist_branches(
    sample: usize,
    r06: &[[f64; 3]; 3],
    q1: f64,
    q2: f64,
    q3: f64,
    out: &mut [[f64; 6]; INNER_MAX],
    n: &mut usize,
) {
    let axes = &consts::SUB_AXES[sample];
    let r36 = mat3_mul(
        &rotation_matrix(&neg3(&axes[2]), q3),
        &mat3_mul(
            &rotation_matrix(&neg3(&axes[1]), q2),
            &mat3_mul(&rotation_matrix(&neg3(&axes[0]), q1), r06),
        ),
    );

    let d = dot3(&axes[3], &mat3_vec(&r36, &axes[5]));
    let (t5, n5) = sp4_solve(&axes[3], &axes[4], &axes[5], d, FEAS_TOL, DEG_TOL);
    for j in 0..n5 {
        let q5 = t5[j];
        let q4 = sp1_solve(
            &axes[3],
            &rotate(&axes[4], q5, &axes[5]),
            &mat3_vec(&r36, &axes[5]),
        );
        let q6 = sp1_solve(
            &neg3(&axes[5]),
            &rotate(&neg3(&axes[4]), q5, &axes[3]),
            &mat3t_vec(&r36, &axes[3]),
        );
        if *n < INNER_MAX {
            out[*n] = [q1, q2, q3, q4, q5, q6];
            *n += 1;
        }
    }
}

/// `ikgeo.spherical` on reversed sub-chain `sample` with target `tt`.
fn spherical_candidates(sample: usize, tt: &Transform, out: &mut [[f64; 6]; INNER_MAX]) -> usize {
    let axes = &consts::SUB_AXES[sample];
    let tl = &consts::SUB_TL[sample];
    let p0 = tl[0];
    let p1 = tl[1];
    let p2 = tl[2];
    let p3 = add3(&add3(&tl[3], &tl[4]), &tl[5]);
    let p6 = consts::SUB_TRT[sample];
    let r_home = consts::SUB_TRR[sample];

    let r06 = mat3_mul(&tt.r, &mat3_transpose(&r_home));
    let p0t = tt.t;

    // p_16 = p0t - p0 - r06 @ p6
    let p16 = sub3(&sub3(&p0t, &p0), &mat3_vec(&r06, &p6));

    let (triples, n_tri, _is_ls) = sp5_solve(
        &neg3(&p1),
        &p16,
        &p2,
        &p3,
        &neg3(&axes[0]),
        &axes[1],
        &axes[2],
        NUM_TOL,
        DEG_TOL,
        DEDUP_TOL,
    );

    let mut n = 0usize;
    for ti in 0..n_tri {
        let (q1, q2, q3) = (triples[ti][0], triples[ti][1], triples[ti][2]);
        wrist_branches(sample, &r06, q1, q2, q3, out, &mut n);
    }
    n
}

/// `ikgeo.spherical_two_parallel` on reversed sub-chain `sample`, target `tt`.
fn sph2par_candidates(sample: usize, tt: &Transform, out: &mut [[f64; 6]; INNER_MAX]) -> usize {
    let axes = &consts::SUB_AXES[sample];
    let tl = &consts::SUB_TL[sample];
    let p0 = tl[0];
    let p1 = tl[1];
    let p2 = tl[2];
    let p3 = add3(&add3(&tl[3], &tl[4]), &tl[5]);
    let p6 = consts::SUB_TRT[sample];
    let r_home = consts::SUB_TRR[sample];

    let r06 = mat3_mul(&tt.r, &mat3_transpose(&r_home));
    let p0t = tt.t;

    // SP4 for q1: axes[1] . Rot(-axes0, q1)(p0t - r06 p6 - p0) = axes[1].(p1+p2+p3)
    let p_arg = sub3(&sub3(&p0t, &mat3_vec(&r06, &p6)), &p0);
    let d1 = dot3(&axes[1], &add3(&add3(&p1, &p2), &p3));
    let (t1, n1) = sp4_solve(&axes[1], &neg3(&axes[0]), &p_arg, d1, FEAS_TOL, DEG_TOL);

    let mut n = 0usize;
    for i1 in 0..n1 {
        let q1 = t1[i1];
        // shoulder = Rot(-axes0, q1)(-p0t + r06 p6 + p0) + p1
        let inner = add3(&add3(&neg3(&p0t), &mat3_vec(&r06, &p6)), &p0);
        let shoulder = add3(&mat3_vec(&rotation_matrix(&neg3(&axes[0]), q1), &inner), &p1);

        let (t3, n3) = sp3_solve(&axes[2], &neg3(&p3), &p2, norm3(&shoulder), FEAS_TOL, DEG_TOL);
        for i3 in 0..n3 {
            let q3 = t3[i3];
            // SP1 for q2 from the shoulder plane.
            let pp = sub3(&neg3(&p2), &mat3_vec(&rotation_matrix(&axes[2], q3), &p3));
            let q2 = sp1_solve(&axes[1], &pp, &shoulder);
            wrist_branches(sample, &r06, q1, q2, q3, out, &mut n);
        }
    }
    n
}

/// FK-verify, keep closures `<= FK_ATOL`, and dedup-by-wrap the inner 6R
/// candidates for `sample` against target `tt`. Writes survivors to `out`
/// (with residuals) and returns the count.
fn verify_inner(
    sample: usize,
    tt: &Transform,
    cands: &[[f64; 6]; INNER_MAX],
    n_cands: usize,
    out: &mut [[f64; 6]; INNER_MAX],
    res: &mut [f64; INNER_MAX],
) -> usize {
    let mut n = 0usize;
    for ci in 0..n_cands {
        let q = cands[ci];
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        let residual = frob_diff(&fk_sub(sample, &q), tt);
        if residual > FK_ATOL {
            continue;
        }
        // Dedup vs already-kept survivors (wrap-to-pi within DEDUP_TOL).
        let mut dup = usize::MAX;
        for j in 0..n {
            if q_close6(&q, &out[j], DEDUP_TOL) {
                dup = j;
                break;
            }
        }
        if dup == usize::MAX {
            out[n] = q;
            res[n] = residual;
            n += 1;
        } else if residual < res[dup] {
            out[dup] = q;
            res[dup] = residual;
        }
    }
    n
}

#[inline]
fn q_close6(a: &[f64; 6], b: &[f64; 6], tol: f64) -> bool {
    for i in 0..6 {
        if libm::fabs(wrap_to_pi(a[i] - b[i])) > tol {
            return false;
        }
    }
    true
}

#[inline]
fn q_close7(a: &[f64; 7], b: &[f64; 7], tol: f64) -> bool {
    for i in 0..7 {
        if libm::fabs(wrap_to_pi(a[i] - b[i])) > tol {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Shared Newton refinement
// ---------------------------------------------------------------------------

/// Iteration ceiling for the Levenberg-Marquardt pose-refinement polish.
const REFINE_ITERS: usize = 60;
const REFINE_EPS: f64 = 1e-7;

/// 12-vector pose residual `[vec(R(q) - R_t); t(q) - t_t]` of the full 7R FK.
#[inline]
fn residual12(q: &[f64; 7], target: &Transform) -> [f64; 12] {
    let tf = fk7(q);
    let mut r = [0.0f64; 12];
    let mut k = 0usize;
    for i in 0..3 {
        for j in 0..3 {
            r[k] = tf.r[i][j] - target.r[i][j];
            k += 1;
        }
    }
    for i in 0..3 {
        r[9 + i] = tf.t[i] - target.t[i];
    }
    r
}

#[inline]
fn norm12(r: &[f64; 12]) -> f64 {
    let mut s = 0.0;
    for v in r.iter() {
        s += v * v;
    }
    libm::sqrt(s)
}

/// Levenberg-Marquardt polish of a 7R configuration onto the exact pose root,
/// **holding the locked joint 4 fixed**.
///
/// Joint 4 is pinned to its lock-sample value (the surrounding sweep solved a
/// 6R sub-chain with that joint locked); freeing it would let the redundant 7R
/// drift along its self-motion manifold. Refining only the six free joints
/// `{0,1,2,3,5,6}` targets the *isolated* 6R root. Adaptive LM damping keeps the
/// step bounded at the near-singular configurations (where the raw algebraic
/// joint values are ill-conditioned), so this routine and the Python data
/// generator (`scripts/generate_franka_test_data.py`) — which applies the
/// identical step — converge to the same configuration well within the `1e-5`
/// test tolerance. The iteration count is hard-bounded for the real-time loop.
fn refine7(q_in: &[f64; 7], target: &Transform) -> [f64; 7] {
    const FREE: [usize; 6] = [0, 1, 2, 3, 5, 6];
    let mut q = *q_in;
    let mut lambda = 1e-6_f64;
    let mut r = residual12(&q, target);
    let mut rn = norm12(&r);

    for _ in 0..REFINE_ITERS {
        if rn < 1e-13 {
            break;
        }
        // Finite-difference Jacobian J (12x6) over the free joints.
        let mut jac = SMatrix::<f64, 12, 6>::zeros();
        for (c, &jidx) in FREE.iter().enumerate() {
            let mut qp = q;
            qp[jidx] += REFINE_EPS;
            let rp = residual12(&qp, target);
            for row in 0..12 {
                jac[(row, c)] = (rp[row] - r[row]) / REFINE_EPS;
            }
        }
        let rv = SMatrix::<f64, 12, 1>::from_column_slice(&r);
        let jt = jac.transpose();
        let jtj = jt * jac; // 6x6
        let jtr = jt * rv; // 6x1

        // Levenberg-Marquardt damped step with one accept/reject backtrack
        // ladder; lambda grows until the step decreases the residual.
        let mut accepted = false;
        for _ in 0..12 {
            let mut damped = jtj;
            for d in 0..6 {
                damped[(d, d)] += lambda;
            }
            let inv = match damped.try_inverse() {
                Some(m) => m,
                None => {
                    lambda *= 8.0;
                    continue;
                }
            };
            let dq = inv * jtr;
            let mut q_trial = q;
            for (c, &jidx) in FREE.iter().enumerate() {
                q_trial[jidx] -= dq[(c, 0)];
            }
            let r_trial = residual12(&q_trial, target);
            let rn_trial = norm12(&r_trial);
            if rn_trial < rn {
                q = q_trial;
                r = r_trial;
                rn = rn_trial;
                lambda = (lambda * 0.5).max(1e-12);
                accepted = true;
                break;
            }
            lambda *= 8.0;
            if lambda > 1e8 {
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
// Public interface
// ---------------------------------------------------------------------------

/// Computes the complete analytical inverse kinematics for the Franka Panda.
///
/// `target_tf` is the SE(3) pose of `ee_link` in `base_link`. The returned
/// fixed-size array holds every distinct solution (up to [`MAX_SOLUTIONS`]);
/// unused slots are `None`. This reproduces
/// `franka_panda_ik.solve(T, respect_limits=False)`.
pub fn compute_ik(target_tf: &SMatrix<f64, 4, 4>) -> [Option<[f64; 7]>; MAX_SOLUTIONS] {
    let mut solutions: [Option<[f64; 7]>; MAX_SOLUTIONS] = [None; MAX_SOLUTIONS];

    // Extract the target as a rigid transform and its SE(3) inverse (the inner
    // solvers run on the reversed chain with target T^{-1}).
    let target = Transform {
        r: [
            [target_tf[(0, 0)], target_tf[(0, 1)], target_tf[(0, 2)]],
            [target_tf[(1, 0)], target_tf[(1, 1)], target_tf[(1, 2)]],
            [target_tf[(2, 0)], target_tf[(2, 1)], target_tf[(2, 2)]],
        ],
        t: [target_tf[(0, 3)], target_tf[(1, 3)], target_tf[(2, 3)]],
    };
    let target_rev = se3_inv(&target);

    // Joint-lock sweep: collect 7R candidates with their inner FK residual.
    let mut cand_q = [[0.0f64; 7]; CAND_CAP];
    let mut cand_res = [0.0f64; CAND_CAP];
    let mut n_cand = 0usize;

    let mut raw = [[0.0f64; 6]; INNER_MAX];
    let mut inner = [[0.0f64; 6]; INNER_MAX];
    let mut inner_res = [0.0f64; INNER_MAX];

    for sample in 0..16 {
        let n_raw = if consts::SOLVER_KIND[sample] == 0 {
            spherical_candidates(sample, &target_rev, &mut raw)
        } else {
            sph2par_candidates(sample, &target_rev, &mut raw)
        };
        if n_raw == 0 {
            continue;
        }
        let n_inner = verify_inner(
            sample,
            &target_rev,
            &raw,
            n_raw,
            &mut inner,
            &mut inner_res,
        );

        let q_lock = consts::LOCK_SAMPLES[sample];
        for si in 0..n_inner {
            // map_reversed_q: reverse-order permutation.
            let s = inner[si];
            let sub_q = [s[5], s[4], s[3], s[2], s[1], s[0]];
            // Insert the locked joint at index 4.
            let full = [
                sub_q[0], sub_q[1], sub_q[2], sub_q[3], q_lock, sub_q[4], sub_q[5],
            ];
            if n_cand < CAND_CAP {
                cand_q[n_cand] = full;
                cand_res[n_cand] = inner_res[si];
                n_cand += 1;
            }
        }
    }

    // First dedup (mirrors `seven_r.solve`'s `dedup_by_wrap_close`).
    let mut d1_q = [[0.0f64; 7]; CAND_CAP];
    let mut d1_res = [0.0f64; CAND_CAP];
    let n_d1 = dedup7(&cand_q, &cand_res, n_cand, &mut d1_q, &mut d1_res);

    // Outer 7R FK verification, then final dedup (mirrors the prebuilt
    // `solve` body).
    let mut v_q = [[0.0f64; 7]; CAND_CAP];
    let mut v_res = [0.0f64; CAND_CAP];
    let mut n_v = 0usize;
    for i in 0..n_d1 {
        let q = d1_q[i];
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        let residual = frob_diff(&fk7(&q), &target);
        if residual <= FK_ATOL {
            v_q[n_v] = q;
            v_res[n_v] = residual;
            n_v += 1;
        }
    }

    let mut f_q = [[0.0f64; 7]; CAND_CAP];
    let mut f_res = [0.0f64; CAND_CAP];
    let n_final = dedup7(&v_q, &v_res, n_v, &mut f_q, &mut f_res);

    for i in 0..n_final {
        if i >= MAX_SOLUTIONS {
            break;
        }
        solutions[i] = Some(refine7(&f_q[i], &target));
    }
    solutions
}

/// Dedup 7-vectors by wrap-to-pi closeness, keeping the lower-residual
/// representative on collision (first-match-wins ordering).
fn dedup7(
    q: &[[f64; 7]; CAND_CAP],
    res: &[f64; CAND_CAP],
    n: usize,
    out_q: &mut [[f64; 7]; CAND_CAP],
    out_res: &mut [f64; CAND_CAP],
) -> usize {
    let mut m = 0usize;
    for i in 0..n {
        let mut dup = usize::MAX;
        for j in 0..m {
            if q_close7(&q[i], &out_q[j], DEDUP_TOL) {
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

    fn q_eq_mod(a: &[f64; 7], b: &[f64; 7], tol: f64) -> bool {
        for i in 0..7 {
            let d = libm::remainder(a[i] - b[i], 2.0 * core::f64::consts::PI);
            if libm::fabs(d) > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn home_pose_round_trips() {
        // T_HOME from the artifact (FK at q = 0).
        let t_home = [
            -0.7071068058785943, -0.7071067564945002, 0.0, 0.088,
            -0.7071067564944993, 0.7071068058785934, -4.440892098500621e-16, -3.241851231905455e-17,
            3.1401848077128287e-16, -3.140185027022262e-16, -0.9999999999999987, 0.9259999999999998,
            0.0, 0.0, 0.0, 1.0,
        ];
        let tf = tf_from_row_major(&t_home);
        let sols = compute_ik(&tf);
        let zero = [0.0; 7];
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-5));
        assert!(found, "home configuration q=0 not recovered");
    }

    #[test]
    fn matches_ssik_reference_dataset() {
        let tol = 1e-5;
        let mut total_expected = 0usize;
        let mut case_idx = 0usize;
        for (tf_arr, expected) in test_data::CASES {
            let tf = tf_from_row_major(tf_arr);
            let sols = compute_ik(&tf);
            let got: Vec<[f64; 7]> = sols.iter().flatten().copied().collect();

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
}
