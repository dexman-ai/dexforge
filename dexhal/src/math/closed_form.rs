//! Post-processing shared by the 6-DOF closed-form drivers (UR, Z1, Piper).
//!
//! A driver's `solve` produces raw algebraic candidates and hands them to
//! [`finish6`], which runs the pipeline from `codegen/translate_ssik_artifact.md`
//! §4 item 5: drop non-finite candidates, accept on FK closure (or polish with
//! the analytic [`refine6`] when `allow_refinement`), de-duplicate modulo 2π,
//! wrap into and filter by the joint limits, sort nearest-first to the seed,
//! and truncate. It never early-exits; that is `single::six_dof_one`.
//!
//! FK, Jacobian and limits come from the driver's [`ArmModel`]
//! (`flange_fk`, `flange_jacobian`, `JOINT_LIMITS`), so no per-arm copy of
//! these helpers is needed.

use crate::math::kinematics::{lm_refine6, mat4_frob_diff, residual_columns};
use crate::math::model::{ArmModel, Pose};
use crate::math::postprocess::{nearest_to_seed, q_close, within_limits, wrap_to_limits};
use crate::math::tolerances::TolerancePolicy;

/// Pose residual `[vec(R − R_t); t − t_t]` (row-major rotation, then translation).
pub fn pose_residual(fk: &Pose, target: &Pose) -> [f64; 12] {
    let mut r = [0.0; 12];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = fk[(i, j)] - target[(i, j)];
        }
        r[9 + i] = fk[(i, 3)] - target[(i, 3)];
    }
    r
}

/// Rotation block of a pose as a row-major 3×3 array.
pub fn rotation_of(fk: &Pose) -> [[f64; 3]; 3] {
    core::array::from_fn(|i| core::array::from_fn(|j| fk[(i, j)]))
}

/// Analytic-Jacobian Levenberg-Marquardt polish of all six joints from `q`
/// onto `target`. `Some` when the FK residual converges within `fk_atol`.
pub fn refine6<A: ArmModel<6, MAX>, const MAX: usize>(
    q: &[f64; 6],
    target: &Pose,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 6]> {
    lm_refine6(
        q,
        |q| pose_residual(&A::flange_fk(q), target),
        |q| residual_columns(&rotation_of(&A::flange_fk(q)), &A::flange_jacobian(q)),
        max_iters,
        fk_atol,
    )
}

/// FK closure gate for one candidate: kept as is when it closes within
/// `fk_atol`, otherwise polished with [`refine6`] when `allow_refinement`.
pub fn close_or_refine6<A: ArmModel<6, MAX>, const MAX: usize>(
    q: &[f64; 6],
    target: &Pose,
    fk_atol: f64,
    allow_refinement: bool,
    max_iters: usize,
) -> Option<[f64; 6]> {
    if !q.iter().all(|v| v.is_finite()) {
        None
    } else if mat4_frob_diff(&A::flange_fk(q), target) <= fk_atol {
        Some(*q)
    } else if allow_refinement {
        refine6::<A, MAX>(q, target, max_iters, fk_atol)
    } else {
        None
    }
}

/// Full `ik_all` post-process of raw closed-form candidates (see the module
/// docs). Returns up to `MAX` solutions at the front; unused slots are `None`.
#[allow(clippy::too_many_arguments)]
pub fn finish6<A: ArmModel<6, MAX>, const MAX: usize>(
    cands: &[[f64; 6]],
    target: &Pose,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 6]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 6]>; MAX] {
    let mut sols = [[0.0f64; 6]; MAX];
    let mut n = 0usize;
    for q in cands {
        if n == MAX {
            break;
        }
        let Some(q) = close_or_refine6::<A, MAX>(
            q,
            target,
            policy.subproblem_numerical,
            allow_refinement,
            refinement_max_iters,
        ) else {
            continue;
        };
        if sols[..n]
            .iter()
            .any(|s| q_close(&q, s, policy.subproblem_dedup))
        {
            continue;
        }
        sols[n] = q;
        n += 1;
    }

    if respect_limits {
        wrap_to_limits(&mut sols, n, &A::JOINT_LIMITS);
        let mut kept = 0usize;
        for i in 0..n {
            if within_limits(&sols[i], &A::JOINT_LIMITS) {
                sols[kept] = sols[i];
                kept += 1;
            }
        }
        n = kept;
    }

    if let Some(seed) = q_seed {
        nearest_to_seed(&mut sols, n, seed);
    }
    if let Some(max) = max_solutions {
        n = n.min(max);
    }

    let mut out = [None; MAX];
    for (o, q) in out.iter_mut().zip(&sols[..n]) {
        *o = Some(*q);
    }
    out
}
