//! Shared `jointlock.seven_r` IK for non-SRS 7R arms whose locked 6R
//! sub-chains are **spherical-wrist** (xArm7, Franka Panda, ...).
//!
//! The `ssik` solver collapses such a redundant arm to a series of 6R problems
//! by locking one joint to each of 16 fixed sample values. For each sample the
//! remaining six joints form a 6R sub-chain; reversing that chain (the EAIK
//! `REVERSED` decomposition) places its spherical wrist at the canonical
//! position, where an inner ikgeo solver applies:
//!
//! * `spherical` (`solver_kind == 0`) — SP5 shoulder + SP4/SP1 wrist, and
//! * `spherical_two_parallel` (`solver_kind == 1`) — SP4/SP3/SP1.
//!
//! Each inner solve is verified against the reversed sub-chain's forward
//! kinematics, mapped back to original-chain ordering, padded with the locked
//! angle, FK-verified against the full 7R chain and de-duplicated modulo `2π`;
//! [`refine7`] optionally polishes each root with the locked joint held fixed.
//!
//! What is shared here is the whole algorithm. A driver supplies only its
//! baked geometry as a [`SphericalLockModel`] of `&'static` constants and its
//! [`ArmModel`] (generated Jacobian and joint limits); its `ik_solver` keeps
//! thin `solve` / `lock_sample_solutions` wrappers. The arithmetic follows the
//! Python reference operation-for-operation, so the drivers' reference
//! datasets match to `1e-5`.

use nalgebra::SMatrix;

use crate::math::chain::transform_of;
use crate::math::kinematics::{
    compose, frob_diff, lm_refine_masked_analytic, pose_residual12, residual_columns, se3_inv,
    Transform, IDENTITY,
};
use crate::math::model::ArmModel;
use crate::math::postprocess::{
    dedup_solutions, nearest_to_seed, q_close, respect_limits as filter_limits, wrap_to_limits,
};
use crate::math::ssik_core::{
    add3, dot3, mat3_mul, mat3_transpose, mat3_vec, mat3t_vec, neg3, norm3, rotate,
    rotation_matrix, sp1_solve, sp3_solve, sp4_solve, sp5_solve, sub3,
};
use crate::math::tolerances::TolerancePolicy;

/// Number of locked-joint samples in the sweep.
pub const N_SAMPLES: usize = 16;

/// Maximum inner-solver candidates per lock sample. `spherical` can return up
/// to 16 SP5 shoulder triples (4 quartic roots x 4 sign branches) with up to 2
/// SP4 wrist branches each; `spherical_two_parallel` stays at 8.
pub const INNER_MAX: usize = 32;

/// Upper bound on raw 7R candidates collected across the sweep before
/// de-duplication.
pub const CAND_CAP: usize = N_SAMPLES * INNER_MAX;

/// Baked geometry of one spherical joint-lock 7R arm (from the driver's
/// generated `consts`).
pub struct SphericalLockModel {
    /// Locked-joint sample values.
    pub lock_samples: &'static [f64; N_SAMPLES],
    /// Inner solver per sample: `0` = `spherical`, `1` = `spherical_two_parallel`.
    pub solver_kind: &'static [u8; N_SAMPLES],
    /// Joint axes of each reversed 6R sub-chain.
    pub sub_axes: &'static [[[f64; 3]; 6]; N_SAMPLES],
    /// Pre-joint translations of each reversed 6R sub-chain.
    pub sub_tl: &'static [[[f64; 3]; 6]; N_SAMPLES],
    /// Tool rotation of each reversed 6R sub-chain.
    pub sub_trr: &'static [[[f64; 3]; 3]; N_SAMPLES],
    /// Tool translation of each reversed 6R sub-chain.
    pub sub_trt: &'static [[f64; 3]; N_SAMPLES],
    /// Joint axes of the full 7R chain.
    pub axes7: &'static [[f64; 3]; 7],
    /// Pre-joint translations of the full 7R chain.
    pub tl7: &'static [[f64; 3]; 7],
    /// Post-joint rotations of the full 7R chain.
    pub tr7_r: &'static [[[f64; 3]; 3]; 7],
    /// Post-joint translations of the full 7R chain.
    pub tr7_t: &'static [[f64; 3]; 7],
    /// Index of the locked joint (`0..7`).
    pub locked_joint: usize,
}

impl SphericalLockModel {
    /// The six free joints in original-chain order.
    fn free_joints(&self) -> [usize; 6] {
        let mut free = [0usize; 6];
        let mut k = 0;
        for j in 0..7 {
            if j != self.locked_joint {
                free[k] = j;
                k += 1;
            }
        }
        free
    }
}

/// POE forward kinematics of the reversed 6R sub-chain `sample`.
fn fk_sub(m: &SphericalLockModel, sample: usize, q: &[f64; 6]) -> Transform {
    let axes = &m.sub_axes[sample];
    let tl = &m.sub_tl[sample];
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
                r: m.sub_trr[sample],
                t: m.sub_trt[sample],
            };
            t = compose(&t, &tr);
        }
    }
    t
}

/// POE forward kinematics of the full 7R chain (the solver's verification FK).
pub fn fk7(m: &SphericalLockModel, q: &[f64; 7]) -> Transform {
    let mut t = IDENTITY;
    for i in 0..7 {
        let l = Transform {
            r: IDENTITY.r,
            t: m.tl7[i],
        };
        let rj = Transform {
            r: rotation_matrix(&m.axes7[i], q[i]),
            t: [0.0, 0.0, 0.0],
        };
        let tr = Transform {
            r: m.tr7_r[i],
            t: m.tr7_t[i],
        };
        t = compose(&t, &l);
        t = compose(&t, &rj);
        t = compose(&t, &tr);
    }
    t
}

/// Wrist back-substitution shared by both inner solvers: given `r06` and the
/// three shoulder angles, append every `(q4, q5, q6)` branch as a candidate.
#[allow(clippy::too_many_arguments)]
fn wrist_branches(
    axes: &[[f64; 3]; 6],
    r06: &[[f64; 3]; 3],
    q1: f64,
    q2: f64,
    q3: f64,
    out: &mut [[f64; 6]; INNER_MAX],
    n: &mut usize,
    policy: TolerancePolicy,
) {
    let r36 = mat3_mul(
        &rotation_matrix(&neg3(&axes[2]), q3),
        &mat3_mul(
            &rotation_matrix(&neg3(&axes[1]), q2),
            &mat3_mul(&rotation_matrix(&neg3(&axes[0]), q1), r06),
        ),
    );

    let d = dot3(&axes[3], &mat3_vec(&r36, &axes[5]));
    let (t5, n5) = sp4_solve(
        &axes[3],
        &axes[4],
        &axes[5],
        d,
        policy.subproblem_feasibility,
        policy.subproblem_degeneracy,
    );
    for &q5 in t5.iter().take(n5) {
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
fn spherical_candidates(
    m: &SphericalLockModel,
    sample: usize,
    tt: &Transform,
    out: &mut [[f64; 6]; INNER_MAX],
    policy: TolerancePolicy,
) -> usize {
    let axes = &m.sub_axes[sample];
    let tl = &m.sub_tl[sample];
    let p0 = tl[0];
    let p1 = tl[1];
    let p2 = tl[2];
    let p3 = add3(&add3(&tl[3], &tl[4]), &tl[5]);
    let p6 = m.sub_trt[sample];
    let r_home = m.sub_trr[sample];

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
        policy.subproblem_numerical,
        policy.subproblem_degeneracy,
        policy.subproblem_dedup,
    );

    let mut n = 0usize;
    for tri in triples.iter().take(n_tri) {
        wrist_branches(axes, &r06, tri[0], tri[1], tri[2], out, &mut n, policy);
    }
    n
}

/// `ikgeo.spherical_two_parallel` on reversed sub-chain `sample`, target `tt`.
fn sph2par_candidates(
    m: &SphericalLockModel,
    sample: usize,
    tt: &Transform,
    out: &mut [[f64; 6]; INNER_MAX],
    policy: TolerancePolicy,
) -> usize {
    let axes = &m.sub_axes[sample];
    let tl = &m.sub_tl[sample];
    let p0 = tl[0];
    let p1 = tl[1];
    let p2 = tl[2];
    let p3 = add3(&add3(&tl[3], &tl[4]), &tl[5]);
    let p6 = m.sub_trt[sample];
    let r_home = m.sub_trr[sample];

    let r06 = mat3_mul(&tt.r, &mat3_transpose(&r_home));
    let p0t = tt.t;

    // SP4 for q1: axes[1] . Rot(-axes0, q1)(p0t - r06 p6 - p0) = axes[1].(p1+p2+p3)
    let p_arg = sub3(&sub3(&p0t, &mat3_vec(&r06, &p6)), &p0);
    let d1 = dot3(&axes[1], &add3(&add3(&p1, &p2), &p3));
    let (t1, n1) = sp4_solve(
        &axes[1],
        &neg3(&axes[0]),
        &p_arg,
        d1,
        policy.subproblem_feasibility,
        policy.subproblem_degeneracy,
    );

    let mut n = 0usize;
    for &q1 in t1.iter().take(n1) {
        // shoulder = Rot(-axes0, q1)(-p0t + r06 p6 + p0) + p1
        let inner = add3(&add3(&neg3(&p0t), &mat3_vec(&r06, &p6)), &p0);
        let shoulder = add3(
            &mat3_vec(&rotation_matrix(&neg3(&axes[0]), q1), &inner),
            &p1,
        );

        let (t3, n3) = sp3_solve(
            &axes[2],
            &neg3(&p3),
            &p2,
            norm3(&shoulder),
            policy.subproblem_feasibility,
            policy.subproblem_degeneracy,
        );
        for &q3 in t3.iter().take(n3) {
            // SP1 for q2 from the shoulder plane.
            let pp = sub3(&neg3(&p2), &mat3_vec(&rotation_matrix(&axes[2], q3), &p3));
            let q2 = sp1_solve(&axes[1], &pp, &shoulder);
            wrist_branches(axes, &r06, q1, q2, q3, out, &mut n, policy);
        }
    }
    n
}

/// FK-verify (closure `<= subproblem_numerical`) and dedup-by-wrap the inner
/// 6R candidates of `sample` against `tt`. Writes survivors and their
/// residuals to `out` / `res` and returns the count.
#[allow(clippy::too_many_arguments)]
fn verify_inner(
    m: &SphericalLockModel,
    sample: usize,
    tt: &Transform,
    cands: &[[f64; 6]; INNER_MAX],
    n_cands: usize,
    out: &mut [[f64; 6]; INNER_MAX],
    res: &mut [f64; INNER_MAX],
    policy: TolerancePolicy,
) -> usize {
    let mut n = 0usize;
    for q in cands.iter().take(n_cands) {
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        let residual = frob_diff(&fk_sub(m, sample, q), tt);
        if residual > policy.subproblem_numerical {
            continue;
        }
        let dup = (0..n).find(|&j| q_close::<6>(q, &out[j], policy.subproblem_dedup));
        match dup {
            None => {
                out[n] = *q;
                res[n] = residual;
                n += 1;
            }
            Some(j) if residual < res[j] => {
                out[j] = *q;
                res[j] = residual;
            }
            Some(_) => {}
        }
    }
    n
}

/// Verified 7R candidates of lock sample `sample` against the reversed target
/// `target_rev`, with their inner residuals. Returns `(q, residual, count)`.
fn sample_candidates(
    m: &SphericalLockModel,
    sample: usize,
    target_rev: &Transform,
    policy: TolerancePolicy,
) -> ([[f64; 7]; INNER_MAX], [f64; INNER_MAX], usize) {
    let mut raw = [[0.0f64; 6]; INNER_MAX];
    let n_raw = if m.solver_kind[sample] == 0 {
        spherical_candidates(m, sample, target_rev, &mut raw, policy)
    } else {
        sph2par_candidates(m, sample, target_rev, &mut raw, policy)
    };
    let mut inner = [[0.0f64; 6]; INNER_MAX];
    let mut inner_res = [0.0f64; INNER_MAX];
    let n_inner = verify_inner(
        m,
        sample,
        target_rev,
        &raw,
        n_raw,
        &mut inner,
        &mut inner_res,
        policy,
    );

    // map_reversed_q: the reversed sub-chain's joint k is free joint `free[5 - k]`.
    let free = m.free_joints();
    let q_lock = m.lock_samples[sample];
    let mut out = [[0.0f64; 7]; INNER_MAX];
    for (o, s) in out.iter_mut().zip(inner.iter()).take(n_inner) {
        o[m.locked_joint] = q_lock;
        for (k, &j) in free.iter().enumerate() {
            o[j] = s[5 - k];
        }
    }
    (out, inner_res, n_inner)
}

/// Analytic Levenberg-Marquardt polish of a 7R configuration onto the exact
/// pose root, **holding the locked joint fixed**.
///
/// Freeing the locked joint would let the redundant arm drift along its
/// self-motion manifold; refining only the six free joints targets the
/// isolated 6R root. The Jacobian is `A::flange_jacobian` (the driver's
/// generated geometric Jacobian), so no finite differences are taken.
pub fn refine7<A: ArmModel<7, MAX>, const MAX: usize>(
    m: &SphericalLockModel,
    q_in: &[f64; 7],
    target: &Transform,
    max_iters: usize,
) -> [f64; 7] {
    lm_refine_masked_analytic::<7, 6, _, _>(
        q_in,
        &m.free_joints(),
        |q| pose_residual12(&fk7(m, q), target),
        |q| residual_columns(&fk7(m, q).r, &A::flange_jacobian(q)),
        max_iters,
    )
}

/// Candidates with the locked joint pinned to `lock_value` (one of
/// `m.lock_samples`); unknown values give none. The per-sample solver of
/// [`crate::math::single::seven_dof_one`].
pub fn lock_sample_solutions<const MAX: usize>(
    m: &SphericalLockModel,
    target_tf: &SMatrix<f64, 4, 4>,
    lock_value: f64,
    policy: TolerancePolicy,
) -> [Option<[f64; 7]>; MAX] {
    let mut out = [None; MAX];
    let Some(sample) = m.lock_samples.iter().position(|&v| v == lock_value) else {
        return out;
    };
    let target_rev = se3_inv(&transform_of(target_tf));
    let (q, _, n) = sample_candidates(m, sample, &target_rev, policy);
    for (o, q) in out.iter_mut().zip(q.iter()).take(n) {
        *o = Some(*q);
    }
    out
}

/// Every IK solution of the full 16-sample sweep (no early exit).
///
/// Candidates are FK-verified against the full chain to
/// `policy.subproblem_numerical` and de-duplicated to `policy.subproblem_dedup`;
/// with `allow_refinement` each is polished by [`refine7`]. With
/// `respect_limits` solutions are wrapped by `±2π` into `A::JOINT_LIMITS`
/// and those still outside are dropped. With `q_seed` the result is sorted
/// nearest-first (wrap-to-π distance). `max_solutions` truncates the output.
#[allow(clippy::too_many_arguments)]
pub fn solve<A: ArmModel<7, MAX>, const MAX: usize>(
    m: &SphericalLockModel,
    target_tf: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 7]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 7]>; MAX] {
    let fk_atol = policy.subproblem_numerical;
    let dedup_atol = policy.subproblem_dedup;

    let target = transform_of(target_tf);
    let target_rev = se3_inv(&target);

    let mut cand_q = [[0.0f64; 7]; CAND_CAP];
    let mut cand_res = [0.0f64; CAND_CAP];
    let mut n_cand = 0usize;
    for sample in 0..N_SAMPLES {
        let (q, res, n) = sample_candidates(m, sample, &target_rev, policy);
        cand_q[n_cand..n_cand + n].copy_from_slice(&q[..n]);
        cand_res[n_cand..n_cand + n].copy_from_slice(&res[..n]);
        n_cand += n;
    }

    // First dedup (mirrors `seven_r.solve`'s `dedup_by_wrap_close`).
    let mut d1_q = [[0.0f64; 7]; CAND_CAP];
    let mut d1_res = [0.0f64; CAND_CAP];
    let n_d1 = dedup_solutions::<7, CAND_CAP>(
        &cand_q,
        &cand_res,
        n_cand,
        &mut d1_q,
        &mut d1_res,
        dedup_atol,
    );

    // Outer 7R FK verification, then final dedup.
    let mut v_q = [[0.0f64; 7]; CAND_CAP];
    let mut v_res = [0.0f64; CAND_CAP];
    let mut n_v = 0usize;
    for q in d1_q.iter().take(n_d1) {
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        let residual = frob_diff(&fk7(m, q), &target);
        if residual <= fk_atol {
            v_q[n_v] = *q;
            v_res[n_v] = residual;
            n_v += 1;
        }
    }

    let mut f_q = [[0.0f64; 7]; CAND_CAP];
    let mut f_res = [0.0f64; CAND_CAP];
    let n_final =
        dedup_solutions::<7, CAND_CAP>(&v_q, &v_res, n_v, &mut f_q, &mut f_res, dedup_atol);

    let mut sols = [[0.0f64; 7]; MAX];
    let mut count = n_final.min(MAX);
    for (s, q) in sols.iter_mut().zip(f_q.iter()).take(count) {
        *s = if allow_refinement {
            refine7::<A, MAX>(m, q, &target, refinement_max_iters)
        } else {
            *q
        };
    }

    if respect_limits {
        wrap_to_limits(&mut sols, count, &A::JOINT_LIMITS);
        count = filter_limits(&mut sols, count, &A::JOINT_LIMITS);
    }
    if let Some(seed) = q_seed {
        nearest_to_seed(&mut sols, count, seed);
    }
    if let Some(max) = max_solutions {
        count = count.min(max);
    }

    let mut out = [None; MAX];
    for (o, q) in out.iter_mut().zip(sols.iter()).take(count) {
        *o = Some(*q);
    }
    out
}
