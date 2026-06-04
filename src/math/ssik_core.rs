//! Generic, allocation-free numeric kernels shared by the per-arm IK drivers.
//!
//! Everything in this module is `#![no_std]`-clean: there is no heap usage and
//! every iterative routine carries a hardcoded iteration ceiling so the worst
//! case execution time is bounded for a real-time control loop.
//!
//! The headline routine is [`quartic_roots`], which mirrors the behaviour of
//! NumPy's `numpy.roots` for the degree-`<= 4` polynomials produced by the
//! conic-intersection step of the IK-Geo subproblem solvers. As mandated by the
//! design constraints it does *not* use a dynamic eigenvalue solver: it forms
//! the polynomial *companion matrix* and extracts its eigenvalues with a
//! bounded-iteration real Schur (Francis double-shift) QR algorithm.

#[cfg(test)]
extern crate std;

use core::f64::consts::PI;
use libm::{atan2, cos, fabs, sin, sqrt};

// ---------------------------------------------------------------------------
// 3-vector primitives (deterministic, no BLAS) shared by the subproblem
// solvers. These mirror `ssik.kinematics._scalar3` and
// `ssik.subproblems._rotation`.
// ---------------------------------------------------------------------------

/// 3-vector dot product.
#[inline]
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3-vector cross product.
#[inline]
pub fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3-vector L2 norm.
#[inline]
pub fn norm3(a: &[f64; 3]) -> f64 {
    sqrt(dot3(a, a))
}

/// `a + b`.
#[inline]
pub fn add3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a - b`.
#[inline]
pub fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `s * a`.
#[inline]
pub fn scale3(a: &[f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Negate.
#[inline]
pub fn neg3(a: &[f64; 3]) -> [f64; 3] {
    [-a[0], -a[1], -a[2]]
}

/// Rodrigues rotation of `v` by `theta` about unit axis `k`.
///
/// `rotate(k, theta, v) = v cos + (k x v) sin + k (k.v)(1 - cos)`.
#[inline]
pub fn rotate(k: &[f64; 3], theta: f64, v: &[f64; 3]) -> [f64; 3] {
    let c = cos(theta);
    let s = sin(theta);
    let kv = dot3(k, v);
    let omc = 1.0 - c;
    let kxv = cross3(k, v);
    [
        v[0] * c + kxv[0] * s + k[0] * (kv * omc),
        v[1] * c + kxv[1] * s + k[1] * (kv * omc),
        v[2] * c + kxv[2] * s + k[2] * (kv * omc),
    ]
}

/// 3x3 Rodrigues rotation matrix about unit `axis` by `angle` (row-major).
#[inline]
pub fn rotation_matrix(axis: &[f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let c = cos(angle);
    let s = sin(angle);
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let oc = 1.0 - c;
    [
        [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
        [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
        [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
    ]
}

/// 3x3 matrix times 3-vector.
#[inline]
pub fn mat3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose of `m` times 3-vector (`m^T v`).
#[inline]
pub fn mat3t_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// 3x3 matrix product `a @ b`.
#[inline]
pub fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Transpose of a 3x3 matrix.
#[inline]
pub fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Wrap an angle to `(-pi, pi]` (mirrors `((a + pi) % 2pi) - pi`).
#[inline]
pub fn wrap_to_pi(a: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let mut r = (a + PI) % two_pi;
    if r < 0.0 {
        r += two_pi;
    }
    r - PI
}

// ---------------------------------------------------------------------------
// IK-Geo subproblem solvers (ports of ssik.subproblems.sp1/sp3/sp4/sp5).
//
// All tolerances are the ssik DEFAULT_TOLERANCE_POLICY values, passed in by
// the caller so the per-arm drivers stay the single source of truth.
// ---------------------------------------------------------------------------

/// Subproblem 1: angle `theta` (least-squares) with `Rot(k, theta) p ~= q`.
pub fn sp1_solve(k: &[f64; 3], p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let kxp = cross3(k, p);
    let kp = dot3(k, p);
    let kq = dot3(k, q);
    atan2(dot3(&kxp, q), dot3(p, q) - kp * kq)
}

/// Subproblem 4: up to two angles with `h . (Rot(k, theta) p) = d`.
///
/// Returns `(solutions, count)` with `count` in `1..=2`; degenerate and
/// least-squares regimes still yield a single representative angle, matching
/// the Python reference whose callers ignore the `is_ls` flag.
pub fn sp4_solve(
    h: &[f64; 3],
    k: &[f64; 3],
    p: &[f64; 3],
    d: f64,
    feas_tol: f64,
    deg_tol: f64,
) -> ([f64; 2], usize) {
    let hp = dot3(h, p);
    let kp = dot3(k, p);
    let hk = dot3(h, k);
    let coef_a = hp - kp * hk;
    let coef_b = dot3(h, &cross3(k, p));
    let coef_c = kp * hk;
    let r_sq = coef_a * coef_a + coef_b * coef_b;

    let deg_sq = deg_tol * deg_tol;
    if r_sq < deg_sq {
        return ([0.0, 0.0], 1);
    }

    let r = sqrt(r_sq);
    let rhs = d - coef_c;
    let phi = atan2(coef_b, coef_a);

    if fabs(rhs) - r > feas_tol {
        let theta = if rhs > 0.0 { phi } else { phi + PI };
        return ([theta, 0.0], 1);
    }

    let rhs_over_r = rhs / r;
    let rhs_clipped = if rhs_over_r < -1.0 {
        -1.0
    } else if rhs_over_r > 1.0 {
        1.0
    } else {
        rhs_over_r
    };
    let delta = libm::acos(rhs_clipped);
    if delta < feas_tol {
        return ([phi, 0.0], 1);
    }
    ([phi + delta, phi - delta], 2)
}

/// Subproblem 3: up to two angles with `|Rot(k, theta) p - q| = d`.
pub fn sp3_solve(
    k: &[f64; 3],
    p: &[f64; 3],
    q: &[f64; 3],
    d: f64,
    feas_tol: f64,
    deg_tol: f64,
) -> ([f64; 2], usize) {
    let target = 0.5 * (dot3(p, p) + dot3(q, q) - d * d);
    sp4_solve(q, k, p, target, feas_tol, deg_tol)
}

// --- SP5 polynomial helpers (ports of ssik.subproblems._aux) ---

#[inline]
fn vec_self_convolve_2(v: &[f64; 2]) -> [f64; 3] {
    let (a, b) = (v[0], v[1]);
    [a * a, 2.0 * a * b, b * b]
}

#[inline]
fn vec_self_convolve_3(v: &[f64; 3]) -> [f64; 5] {
    let (a, b, c) = (v[0], v[1], v[2]);
    [a * a, 2.0 * a * b, 2.0 * a * c + b * b, 2.0 * b * c, c * c]
}

#[inline]
fn vec_convolve_3(v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 5] {
    let (a, b, c) = (v1[0], v1[1], v1[2]);
    let (x, y, z) = (v2[0], v2[1], v2[2]);
    [
        a * x,
        b * x + a * y,
        a * z + b * y + c * x,
        b * z + c * y,
        c * z,
    ]
}

/// `(p, r)` cone-intersection polynomials used inside [`sp5_solve`].
fn cone_polynomials(
    p0_i: &[f64; 3],
    k_i: &[f64; 3],
    p_i: &[f64; 3],
    p_i_s: &[f64; 3],
    k2: &[f64; 3],
) -> ([f64; 2], [f64; 3]) {
    let ki_x_k2 = cross3(k_i, k2);
    let ki_x_ki_x_k2 = cross3(k_i, &ki_x_k2);
    let norm_ki_x_k2_sq = dot3(&ki_x_k2, &ki_x_k2);

    let ki_x_pi = cross3(k_i, p_i);
    let norm_ki_x_pi_sq = dot3(&ki_x_pi, &ki_x_pi);

    let alpha = dot3(p0_i, &ki_x_ki_x_k2) / norm_ki_x_k2_sq;
    let delta = dot3(k2, p_i_s);
    let beta = dot3(p0_i, &ki_x_k2) / norm_ki_x_k2_sq;

    let p_const = norm_ki_x_pi_sq + dot3(p_i_s, p_i_s) + 2.0 * alpha * delta;
    let p = [-2.0 * alpha, p_const];

    let two_beta = 2.0 * beta;
    let scale = two_beta * two_beta;
    let r = [
        scale * -1.0,
        scale * (2.0 * delta),
        scale * (-delta * delta + norm_ki_x_pi_sq * norm_ki_x_k2_sq),
    ];

    (p, r)
}

/// Maximum SP5 candidate triples returned (4 quartic roots x 4 sign branches).
const SP5_MAX: usize = 16;

/// Subproblem 5: up to four `(theta1, theta2, theta3)` triples with
/// `p0 + Rot(k1, t1) p1 = Rot(k2, t2) (p2 + Rot(k3, t3) p3)`.
///
/// Returns `(triples, count, is_ls)`. On exact feasibility `count` deduped
/// triples satisfy the equation within `num_tol`. On infeasibility a single
/// best least-squares triple is returned with `is_ls = true`; on degeneracy
/// `count = 0`.
#[allow(clippy::too_many_arguments)]
pub fn sp5_solve(
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    p3: &[f64; 3],
    k1: &[f64; 3],
    k2: &[f64; 3],
    k3: &[f64; 3],
    num_tol: f64,
    deg_tol: f64,
    dedup_tol: f64,
) -> ([[f64; 3]; SP5_MAX], usize, bool) {
    let empty = [[0.0f64; 3]; SP5_MAX];

    // Upfront degeneracy detection.
    let k1xk2 = cross3(k1, k2);
    let k3xk2 = cross3(k3, k2);
    if dot3(&k1xk2, &k1xk2) < deg_tol || dot3(&k3xk2, &k3xk2) < deg_tol {
        return (empty, 0, true);
    }
    let p1_perp = dot3(p1, p1) - dot3(k1, p1) * dot3(k1, p1);
    let p3_perp = dot3(p3, p3) - dot3(k3, p3) * dot3(k3, p3);
    if p1_perp < deg_tol || p3_perp < deg_tol {
        return (empty, 0, true);
    }

    let p1_s = add3(p0, &scale3(k1, dot3(k1, p1)));
    let p3_s = add3(p2, &scale3(k3, dot3(k3, p3)));

    let delta1 = dot3(k2, &p1_s);
    let delta3 = dot3(k2, &p3_s);

    let (p_1, r_1) = cone_polynomials(p0, k1, p1, &p1_s, k2);
    let (p_3, r_3) = cone_polynomials(p2, k3, p3, &p3_s, k2);

    let p_13 = [p_1[0] - p_3[0], p_1[1] - p_3[1]];
    let p_13_sq = vec_self_convolve_2(&p_13);

    let rhs = [
        r_3[0] - r_1[0] - p_13_sq[0],
        r_3[1] - r_1[1] - p_13_sq[1],
        r_3[2] - r_1[2] - p_13_sq[2],
    ];

    let lhs_poly = vec_self_convolve_3(&rhs);
    let conv = vec_convolve_3(&p_13_sq, &r_1);
    let mut eqn = [0.0f64; 5];
    for i in 0..5 {
        eqn[i] = lhs_poly[i] - 4.0 * conv[i];
    }

    let (roots, n_roots) = quartic_roots(&eqn);

    // Scale-aware real-root filter (matches the Python `_is_real_root`).
    let kxp1 = cross3(k1, p1);
    let kxp3 = cross3(k3, p3);
    // a_1 = [kxp1, -(k1 x kxp1)] as two columns; a_3 likewise.
    let a1_c0 = kxp1;
    let a1_c1 = neg3(&cross3(k1, &kxp1));
    let a3_c0 = kxp3;
    let a3_c1 = neg3(&cross3(k3, &kxp3));

    let signs_1 = [1.0, 1.0, -1.0, -1.0];
    let signs_3 = [1.0, -1.0, 1.0, -1.0];

    let mut cands = [[0.0f64; 3]; SP5_MAX];
    let mut n_cands = 0usize;

    for ri in 0..n_roots {
        let c = roots[ri];
        if fabs(c.im) >= max_f64(fabs(c.re), 1.0) * 1e-3 {
            continue;
        }
        let h = c.re;

        // a_i^T k2 (2-vectors).
        let a1k = [dot3(&a1_c0, k2), dot3(&a1_c1, k2)];
        let a3k = [dot3(&a3_c0, k2), dot3(&a3_c1, k2)];

        let hd1 = h - delta1;
        let hd3 = h - delta3;
        let const_1 = [a1k[0] * hd1, a1k[1] * hd1];
        let const_3 = [a3k[0] * hd3, a3k[1] * hd3];

        let a1k_ns = a1k[0] * a1k[0] + a1k[1] * a1k[1];
        let a3k_ns = a3k[0] * a3k[0] + a3k[1] * a3k[1];
        let sq1 = a1k_ns - hd1 * hd1;
        if sq1 < 0.0 {
            continue;
        }
        let sq3 = a3k_ns - hd3 * hd3;
        if sq3 < 0.0 {
            continue;
        }

        // J @ a_i_k * sqrt(sq) with J = [[0, 1], [-1, 0]] => [a_k[1], -a_k[0]].
        let pm_1 = [a1k[1] * sqrt(sq1), -a1k[0] * sqrt(sq1)];
        let pm_3 = [a3k[1] * sqrt(sq3), -a3k[0] * sqrt(sq3)];

        for si in 0..4 {
            let s1 = signs_1[si];
            let s3 = signs_3[si];
            let sc1 = [
                (const_1[0] + s1 * pm_1[0]) / a1k_ns,
                (const_1[1] + s1 * pm_1[1]) / a1k_ns,
            ];
            let sc3 = [
                (const_3[0] + s3 * pm_3[0]) / a3k_ns,
                (const_3[1] + s3 * pm_3[1]) / a3k_ns,
            ];

            // v1 = a_1 @ sc1 + p1_s ; v3 = a_3 @ sc3 + p3_s.
            let v1 = [
                a1_c0[0] * sc1[0] + a1_c1[0] * sc1[1] + p1_s[0],
                a1_c0[1] * sc1[0] + a1_c1[1] * sc1[1] + p1_s[1],
                a1_c0[2] * sc1[0] + a1_c1[2] * sc1[1] + p1_s[2],
            ];
            let v3 = [
                a3_c0[0] * sc3[0] + a3_c1[0] * sc3[1] + p3_s[0],
                a3_c0[1] * sc3[0] + a3_c1[1] * sc3[1] + p3_s[1],
                a3_c0[2] * sc3[0] + a3_c1[2] * sc3[1] + p3_s[2],
            ];

            let theta2 = sp1_solve(k2, &v3, &v1);
            cands[n_cands] = [atan2(sc1[0], sc1[1]), theta2, atan2(sc3[0], sc3[1])];
            n_cands += 1;
        }
    }

    // Gauss-Newton refine every candidate on the full SP5 equation.
    let mut refined = [[0.0f64; 3]; SP5_MAX];
    for i in 0..n_cands {
        refined[i] = refine_sp5(&cands[i], p0, p1, p2, p3, k1, k2, k3);
    }

    // Post-verify; keep exacts.
    let mut exact = [[0.0f64; 3]; SP5_MAX];
    let mut n_exact = 0usize;
    for i in 0..n_cands {
        if sp5_residual(&refined[i], p0, p1, p2, p3, k1, k2, k3) < num_tol {
            exact[n_exact] = refined[i];
            n_exact += 1;
        }
    }

    if n_exact > 0 {
        // Dedup triples (wrap-to-pi within dedup_tol on every element).
        let mut out = [[0.0f64; 3]; SP5_MAX];
        let mut n_out = 0usize;
        for i in 0..n_exact {
            let mut dup = false;
            for j in 0..n_out {
                if triple_close(&exact[i], &out[j], dedup_tol) {
                    dup = true;
                    break;
                }
            }
            if !dup {
                out[n_out] = exact[i];
                n_out += 1;
            }
        }
        return (out, n_out, false);
    }

    if n_cands == 0 {
        return (empty, 0, true);
    }

    // Best least-squares single.
    let mut best = 0usize;
    let mut best_res = sp5_residual(&refined[0], p0, p1, p2, p3, k1, k2, k3);
    for i in 1..n_cands {
        let res = sp5_residual(&refined[i], p0, p1, p2, p3, k1, k2, k3);
        if res < best_res {
            best_res = res;
            best = i;
        }
    }
    let mut out = [[0.0f64; 3]; SP5_MAX];
    out[0] = refined[best];
    (out, 1, true)
}

#[inline]
fn max_f64(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn triple_close(a: &[f64; 3], b: &[f64; 3], tol: f64) -> bool {
    fabs(wrap_to_pi(a[0] - b[0])) < tol
        && fabs(wrap_to_pi(a[1] - b[1])) < tol
        && fabs(wrap_to_pi(a[2] - b[2])) < tol
}

#[allow(clippy::too_many_arguments)]
fn sp5_residual(
    t: &[f64; 3],
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    p3: &[f64; 3],
    k1: &[f64; 3],
    k2: &[f64; 3],
    k3: &[f64; 3],
) -> f64 {
    let lhs = add3(p0, &rotate(k1, t[0], p1));
    let inner = add3(p2, &rotate(k3, t[2], p3));
    let rhs = rotate(k2, t[1], &inner);
    norm3(&sub3(&lhs, &rhs))
}

/// Gauss-Newton refinement of an SP5 triple (bounded at 20 iterations).
#[allow(clippy::too_many_arguments)]
fn refine_sp5(
    start: &[f64; 3],
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    p3: &[f64; 3],
    k1: &[f64; 3],
    k2: &[f64; 3],
    k3: &[f64; 3],
) -> [f64; 3] {
    let max_step = PI / 4.0;
    let step_tol = 1e-15;
    let mut t1 = start[0];
    let mut t2 = start[1];
    let mut t3 = start[2];

    for _ in 0..20 {
        let rotated_p1 = rotate(k1, t1, p1);
        let rotated_p3_inner = rotate(k3, t3, p3);
        let p2_plus = add3(p2, &rotated_p3_inner);
        let rotated_p2 = rotate(k2, t2, &p2_plus);

        let f = sub3(&add3(p0, &rotated_p1), &rotated_p2);

        let col1 = cross3(k1, &rotated_p1);
        let col2 = neg3(&cross3(k2, &rotated_p2));
        let col3 = neg3(&rotate(k2, t2, &cross3(k3, &rotated_p3_inner)));

        let (j00, j10, j20) = (col1[0], col1[1], col1[2]);
        let (j01, j11, j21) = (col2[0], col2[1], col2[2]);
        let (j02, j12, j22) = (col3[0], col3[1], col3[2]);

        let c00 = j11 * j22 - j12 * j21;
        let c01 = j12 * j20 - j10 * j22;
        let c02 = j10 * j21 - j11 * j20;
        let det = j00 * c00 + j01 * c01 + j02 * c02;
        if fabs(det) < 1e-15 {
            break;
        }
        let c10 = j02 * j21 - j01 * j22;
        let c11 = j00 * j22 - j02 * j20;
        let c12 = j01 * j20 - j00 * j21;
        let c20 = j01 * j12 - j02 * j11;
        let c21 = j02 * j10 - j00 * j12;
        let c22 = j00 * j11 - j01 * j10;
        let b0 = -f[0];
        let b1 = -f[1];
        let b2 = -f[2];
        let inv_det = 1.0 / det;
        let mut d0 = (c00 * b0 + c10 * b1 + c20 * b2) * inv_det;
        let mut d1 = (c01 * b0 + c11 * b1 + c21 * b2) * inv_det;
        let mut d2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det;

        let step_norm = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
        if step_norm > max_step {
            let sc = max_step / step_norm;
            d0 *= sc;
            d1 *= sc;
            d2 *= sc;
        }

        t1 = wrap_to_pi(t1 + d0);
        t2 = wrap_to_pi(t2 + d1);
        t3 = wrap_to_pi(t3 + d2);

        if step_norm < step_tol {
            break;
        }
    }

    [t1, t2, t3]
}

/// Minimal stack-only complex number used for polynomial roots / eigenvalues.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Construct a complex value.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// A purely real complex value.
    #[inline]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
}

/// Hard ceiling on the number of QR sweeps before the eigenvalue iteration
/// gives up. The loop is provably bounded by this constant regardless of input.
const QR_MAX_ITERATIONS: usize = 400;

/// Eigenvalues of a real upper-Hessenberg matrix via the bounded Francis
/// double-shift QR algorithm (real Schur decomposition).
///
/// `h` is consumed and overwritten with its (quasi-)triangular Schur form. The
/// returned array holds the `N` eigenvalues; complex eigenvalues appear as
/// conjugate pairs extracted from the 2x2 diagonal blocks of the Schur form.
///
/// The algorithm follows Golub & Van Loan, *Matrix Computations*, §7.5. It is a
/// fixed-size, stack-allocated port: there is no allocation and the outer loop
/// is capped at [`QR_MAX_ITERATIONS`].
pub fn hessenberg_eigenvalues<const N: usize>(mut h: [[f64; N]; N]) -> [Complex; N] {
    let mut eig = [Complex::new(0.0, 0.0); N];
    if N == 0 {
        return eig;
    }
    if N == 1 {
        eig[0] = Complex::real(h[0][0]);
        return eig;
    }

    // `n` is the size of the still-active leading block; trailing eigenvalues
    // are peeled off into `eig[n..]` as the block deflates.
    let mut n = N;
    let mut iterations = 0usize;

    while n > 0 {
        if iterations >= QR_MAX_ITERATIONS {
            // Bounded-time bail-out: fill the remaining diagonal as real
            // eigenvalues. In practice this branch is never taken for the
            // well-conditioned quartics produced by the IK pipeline.
            for i in 0..n {
                eig[i] = Complex::real(h[i][i]);
            }
            break;
        }

        if n == 1 {
            eig[0] = Complex::real(h[0][0]);
            break;
        }

        // Look for a negligible sub-diagonal entry that splits the block.
        let mut l = n;
        while l >= 2 {
            let s = fabs(h[l - 2][l - 2]) + fabs(h[l - 1][l - 1]);
            let s = if s == 0.0 { 1.0 } else { s };
            if fabs(h[l - 1][l - 2]) <= f64::EPSILON * s {
                h[l - 1][l - 2] = 0.0;
                break;
            }
            l -= 1;
        }
        let l = l - 1; // 0-based index of the first row of the active sub-block

        if l == n - 1 {
            // 1x1 block has converged.
            eig[n - 1] = Complex::real(h[n - 1][n - 1]);
            n -= 1;
            iterations = 0;
            continue;
        }

        if l == n - 2 {
            // 2x2 trailing block has converged: take its eigenvalues directly.
            let (e0, e1) = eig_2x2(
                h[n - 2][n - 2],
                h[n - 2][n - 1],
                h[n - 1][n - 2],
                h[n - 1][n - 1],
            );
            eig[n - 2] = e0;
            eig[n - 1] = e1;
            n -= 2;
            iterations = 0;
            continue;
        }

        // Otherwise perform one Francis double-shift QR sweep on rows `l..n`.
        francis_double_shift(&mut h, l, n);
        iterations += 1;
    }

    eig
}

/// Eigenvalues of the 2x2 real matrix `[[a, b], [c, d]]`.
#[inline]
fn eig_2x2(a: f64, b: f64, c: f64, d: f64) -> (Complex, Complex) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let s = sqrt(disc);
        (
            Complex::real(0.5 * (tr + s)),
            Complex::real(0.5 * (tr - s)),
        )
    } else {
        let s = sqrt(-disc);
        (
            Complex::new(0.5 * tr, 0.5 * s),
            Complex::new(0.5 * tr, -0.5 * s),
        )
    }
}

/// One Francis double-shift QR sweep applied to the active sub-block
/// `h[l..n][l..n]` of an upper-Hessenberg matrix, chasing the resulting bulge
/// down the diagonal with Householder reflectors of size 3 (then 2 at the
/// bottom). Operates in place on the full matrix to keep similarity.
fn francis_double_shift<const N: usize>(h: &mut [[f64; N]; N], l: usize, n: usize) {
    let m = n - 1; // last active row index
    // Wilkinson double shift from the trailing 2x2 block: s = trace, t = det.
    let s = h[m - 1][m - 1] + h[m][m];
    let t = h[m - 1][m - 1] * h[m][m] - h[m - 1][m] * h[m][m - 1];

    // First column of (H^2 - sH + tI) restricted to the active block.
    let a11 = h[l][l];
    let a12 = h[l][l + 1];
    let a21 = h[l + 1][l];
    let a22 = h[l + 1][l + 1];
    let a32 = h[l + 2][l + 1];

    let mut x = a11 * a11 + a12 * a21 - s * a11 + t;
    let mut y = a21 * (a11 + a22 - s);
    let mut z = a21 * a32;

    for k in l..m {
        // Build a Householder reflector zeroing (y, z) (and forming the bulge).
        let three = k + 2 < n; // do we still have a 3-element column?
        let (v0, v1, v2, beta) = householder3(x, y, if three { z } else { 0.0 });

        if beta != 0.0 {
            let q = if k > l { k - 1 } else { l };
            // Apply P from the left to rows k, k+1, (k+2): H <- P H.
            for j in q..N {
                let mut sum = v0 * h[k][j] + v1 * h[k + 1][j];
                if three {
                    sum += v2 * h[k + 2][j];
                }
                sum *= beta;
                h[k][j] -= sum * v0;
                h[k + 1][j] -= sum * v1;
                if three {
                    h[k + 2][j] -= sum * v2;
                }
            }
            // Apply P from the right to columns k, k+1, (k+2): H <- H P.
            let rmax = if k + 3 < n { k + 3 } else { n - 1 };
            for i in 0..=rmax {
                let mut sum = v0 * h[i][k] + v1 * h[i][k + 1];
                if three {
                    sum += v2 * h[i][k + 2];
                }
                sum *= beta;
                h[i][k] -= sum * v0;
                h[i][k + 1] -= sum * v1;
                if three {
                    h[i][k + 2] -= sum * v2;
                }
            }
        }

        // Update (x, y, z) for the next bulge-chase step.
        if k < m - 1 {
            x = h[k + 1][k];
            y = h[k + 2][k];
            z = if k + 3 < n { h[k + 3][k] } else { 0.0 };
        }
    }
}

/// Householder reflector mapping `(x, y, z)` onto the first axis. Returns the
/// reflector vector `(v0, v1, v2)` (normalised so `v0 == 1` implicitly via the
/// returned `beta`) together with `beta = 2 / (v . v)`.
#[inline]
fn householder3(x: f64, y: f64, z: f64) -> (f64, f64, f64, f64) {
    let alpha = sqrt(x * x + y * y + z * z);
    if alpha == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let alpha = if x >= 0.0 { -alpha } else { alpha };
    let v0 = x - alpha;
    let v1 = y;
    let v2 = z;
    let norm_sq = v0 * v0 + v1 * v1 + v2 * v2;
    if norm_sq == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    (v0, v1, v2, 2.0 / norm_sq)
}

/// Roots of the real polynomial
/// `coeffs[0] x^4 + coeffs[1] x^3 + coeffs[2] x^2 + coeffs[3] x + coeffs[4]`.
///
/// Mirrors `numpy.roots`: leading near-zero coefficients reduce the effective
/// degree and trailing zero coefficients contribute roots at the origin. The
/// remaining roots are the eigenvalues of the (monic) companion matrix, found
/// with the bounded [`hessenberg_eigenvalues`] solver.
///
/// Returns the fixed-size root buffer together with the number of valid roots
/// (the effective polynomial degree). Unused slots are left as zero.
pub fn quartic_roots(coeffs: &[f64; 5]) -> ([Complex; 4], usize) {
    let mut roots = [Complex::new(0.0, 0.0); 4];

    // Strip leading zero coefficients to find the true leading term. The
    // exact `== 0.0` test (rather than an epsilon) is deliberate: it mirrors
    // `numpy.roots`, which trims only exactly-zero leading/trailing entries so
    // this port stays bit-compatible with the Python reference's degree
    // reduction. Near-zero (but non-zero) leading terms are intentionally kept.
    let mut start = 0usize;
    while start < 5 && coeffs[start] == 0.0 {
        start += 1;
    }
    let active = &coeffs[start..]; // highest-degree first
    let len = active.len();
    if len <= 1 {
        // Constant (or all-zero) polynomial: no finite roots.
        return (roots, 0);
    }

    // Strip trailing zeros: each contributes a root at 0 (numpy-equivalent,
    // exact-zero test as above).
    let mut end = len;
    let mut zero_roots = 0usize;
    while end > 1 && active[end - 1] == 0.0 {
        end -= 1;
        zero_roots += 1;
    }
    let poly = &active[..end];
    let degree = poly.len() - 1; // degree of the non-trivial factor

    let mut count = 0usize;

    if degree >= 1 {
        // Monic coefficients c_{d-1} .. c_0 (companion matrix uses these).
        let lead = poly[0];
        // Build and solve via companion matrix for degree up to 4.
        match degree {
            1 => {
                roots[count] = Complex::real(-poly[1] / lead);
                count += 1;
            }
            2 => {
                let (e0, e1) = eig_2x2(
                    -poly[1] / lead,
                    -poly[2] / lead,
                    1.0,
                    0.0,
                );
                roots[count] = e0;
                count += 1;
                roots[count] = e1;
                count += 1;
            }
            3 => {
                let mut comp = [[0.0f64; 3]; 3];
                for j in 0..3 {
                    comp[0][j] = -poly[j + 1] / lead;
                }
                comp[1][0] = 1.0;
                comp[2][1] = 1.0;
                let e = hessenberg_eigenvalues(comp);
                for r in e {
                    roots[count] = r;
                    count += 1;
                }
            }
            4 => {
                let mut comp = [[0.0f64; 4]; 4];
                for j in 0..4 {
                    comp[0][j] = -poly[j + 1] / lead;
                }
                comp[1][0] = 1.0;
                comp[2][1] = 1.0;
                comp[3][2] = 1.0;
                let e = hessenberg_eigenvalues(comp);
                for r in e {
                    roots[count] = r;
                    count += 1;
                }
            }
            _ => unreachable!(),
        }
    }

    // Append the explicit zero roots from trailing zero coefficients.
    let mut zr = 0;
    while zr < zero_roots && count < 4 {
        roots[count] = Complex::real(0.0);
        count += 1;
        zr += 1;
    }

    (roots, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn check_roots(coeffs: [f64; 5], expected: &[(f64, f64)]) {
        let (roots, n) = quartic_roots(&coeffs);
        let mut got: Vec<(f64, f64)> = roots[..n].iter().map(|c| (c.re, c.im)).collect();
        // Match each expected root to a distinct computed root.
        for &(er, ei) in expected {
            let mut best = f64::INFINITY;
            let mut best_idx = usize::MAX;
            for (i, &(gr, gi)) in got.iter().enumerate() {
                let d = (gr - er) * (gr - er) + (gi - ei) * (gi - ei);
                if d < best {
                    best = d;
                    best_idx = i;
                }
            }
            assert!(best < 1e-8, "root ({er},{ei}) not matched, best dist {best}");
            got.remove(best_idx);
        }
    }

    #[test]
    fn quartic_real_roots() {
        // (x-1)(x-2)(x-3)(x-4) = x^4 -10x^3 +35x^2 -50x +24
        check_roots(
            [1.0, -10.0, 35.0, -50.0, 24.0],
            &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        );
    }

    #[test]
    fn quartic_complex_roots() {
        // (x^2+1)(x^2+4) = x^4 +5x^2 +4 -> +/- i, +/- 2i
        check_roots(
            [1.0, 0.0, 5.0, 0.0, 4.0],
            &[(0.0, 1.0), (0.0, -1.0), (0.0, 2.0), (0.0, -2.0)],
        );
    }

    #[test]
    fn cubic_reduced() {
        // leading zero -> cubic (x-1)(x-2)(x+3) = x^3 -7x +6 ... compute
        // (x-1)(x-2)=x^2-3x+2; *(x+3)= x^3 -7x +6
        check_roots(
            [0.0, 1.0, 0.0, -7.0, 6.0],
            &[(1.0, 0.0), (2.0, 0.0), (-3.0, 0.0)],
        );
    }

    #[test]
    fn quadratic_with_zero_root() {
        // x^2 - 3x = x(x-3): trailing zero -> root at 0 and 3
        check_roots([0.0, 0.0, 1.0, -3.0, 0.0], &[(0.0, 0.0), (3.0, 0.0)]);
    }
}
