//! Generic 6R **Raghavan-Roth** candidate generation (port of the `ssik`
//! `ikgeo.general_6r` `_solve_algebraic` route), shared by every driver whose
//! 6R (sub-)chain has no closed-form structure (Piper, Rizon 4 / 10).
//!
//! 1. Bridge the target into the DH frame: `T_dh = T_PRE_INV · T · T_POST_INV`.
//! 2. Evaluate the per-chain `(P_sin, P_cos, P_one, Q)` matrices with the
//!    generated [`PqBuilder`], eliminate `q_0, q_1` through the left null space
//!    of `Q`, and apply the Weierstrass half-angle substitution to get the
//!    `12×12` matrix polynomial `M(x) = M_quad x² + M_lin x + M_const`.
//! 3. Extract the real `tan(q/2)` roots and kernels with
//!    [`rr_real_roots`] (bounded `24×24` companion eigenvalue solve).
//! 4. Back-substitute each root to the six joints ([`back_substitute`]) and map
//!    DH → POE.
//!
//! Everything here is arm-independent; a driver supplies only an [`RrChain`]
//! of baked constants.

use nalgebra::{SMatrix, SVector};

use crate::math::ssik_core::{rr_left_null_space, rr_pseudo_inverse, rr_real_roots};

/// The generated `(P_sin, P_cos, P_one, Q)` matrices of one 6R chain.
pub type PqMatrices = (
    [[f64; 9]; 14],
    [[f64; 9]; 14],
    [[f64; 9]; 14],
    [[f64; 8]; 14],
);

/// Signature of a generated `(P_sin, P_cos, P_one, Q)` builder; the arguments
/// are the top three rows of `T_dh`, row-major.
pub type PqBuilder = fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> PqMatrices;

/// Weierstrass trig → `x` basis transform.
pub const W_TRIG_TO_X: [[f64; 9]; 9] = [
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

/// Ratio candidates for `x_left[0]` (Manocha-Canny IV-C robust selection).
pub const X0_RATIOS: [(usize, usize); 7] =
    [(5, 8), (2, 5), (11, 2), (4, 7), (10, 1), (3, 6), (9, 0)];
/// Ratio candidates for `x_left[1]`.
pub const X1_RATIOS: [(usize, usize); 5] = [(7, 8), (6, 7), (1, 2), (4, 5), (10, 11)];

/// Baked constants of one 6R chain (or one locked 6R sub-chain).
#[derive(Clone, Copy)]
pub struct RrChain<'a> {
    /// DH `alpha`.
    pub dh_alpha: &'a [f64; 6],
    /// DH `a`.
    pub dh_a: &'a [f64; 6],
    /// DH `d`.
    pub dh_d: &'a [f64; 6],
    /// DH → POE `theta` offsets (`q_poe = q_dh − offset`).
    pub dh_theta_offset: &'a [f64; 6],
    /// `T_PRE_INV` bridge (row-major 4×4).
    pub t_pre_inv: &'a [[f64; 4]; 4],
    /// `T_POST_INV` bridge (row-major 4×4).
    pub t_post_inv: &'a [[f64; 4]; 4],
    /// Linearity joint (the `tan(q/2)` eigenvalue joint).
    pub linearity_joint: usize,
    /// Left-bilinear joint pair.
    pub left_bilinear: (usize, usize),
    /// Right-bilinear joint pair.
    pub right_bilinear: (usize, usize),
    /// Joint dropped from the elimination and recovered from the FK residual.
    pub drop_joint: usize,
    /// Generated `(P_sin, P_cos, P_one, Q)` builder.
    pub build_pq: PqBuilder,
}

/// Raghavan-Roth candidates of `chain` for target `t`: writes up to
/// `out.len()` POE-frame joint vectors to `out` and returns the count (not
/// FK-verified). `cond_out` receives the 2-norm condition number of the
/// equilibrated leading matrix, so callers can detect ill-conditioned poses.
pub fn solve_6r(
    chain: &RrChain,
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]],
    cond_out: &mut f64,
) -> usize {
    let t_pre_inv = mat4(chain.t_pre_inv);
    let t_post_inv = mat4(chain.t_post_inv);
    let t_dh = t_pre_inv * t * t_post_inv;

    let (p_sin_a, p_cos_a, p_one_a, q_a) = (chain.build_pq)(
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
    let n_roots = rr_real_roots(&m_quad, &m_lin, &m_const, &mut roots, &mut vecs, cond_out);

    let q_pinv = rr_pseudo_inverse(&q_mat);

    let mut count = 0usize;
    for r in 0..n_roots {
        if count >= out.len() {
            break;
        }
        if let Some(q_dh) = back_substitute(
            chain, roots[r], &vecs[r], &p_sin, &p_cos, &p_one, &q_pinv, &t_dh,
        ) {
            // Map DH-frame q back to POE frame.
            let mut q = [0.0f64; 6];
            for i in 0..6 {
                q[i] = q_dh[i] - chain.dh_theta_offset[i];
            }
            out[count] = q;
            count += 1;
        }
    }
    count
}

/// Embed a 6x9 `E` into the 12x12 doubled block structure for `M`.
///
/// Top six rows hold `[E | 0]`; the bottom six rows are the `x_3`-shifted
/// version (cols 0-5 ← `E[:, 3:9]`, cols 9-11 ← `E[:, 0:3]`).
pub fn embed_e(e: &SMatrix<f64, 6, 9>) -> SMatrix<f64, 12, 12> {
    let mut m = SMatrix::<f64, 12, 12>::zeros();
    m.fixed_view_mut::<6, 9>(0, 0).copy_from(e);
    let shift = e.fixed_view::<6, 6>(0, 3).into_owned();
    m.fixed_view_mut::<6, 6>(6, 0).copy_from(&shift);
    let wrap = e.fixed_view::<6, 3>(0, 0).into_owned();
    m.fixed_view_mut::<6, 3>(6, 9).copy_from(&wrap);
    m
}

/// Pick `v_12[num] / v_12[den]` from `candidates`, using the pair whose
/// denominator has the largest magnitude (most-significant-entry rule).
/// Returns `None` if the chosen denominator is numerically zero.
pub fn ratio_from(v12: &[f64; 12], candidates: &[(usize, usize)]) -> Option<f64> {
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
/// `_raghavan_roth._back_substitute_inner`. Returns the DH-frame joints.
#[allow(clippy::too_many_arguments)]
pub fn back_substitute(
    chain: &RrChain,
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

    let mut q = [0.0f64; 6];
    q[chain.linearity_joint] = q_lin;
    q[chain.left_bilinear.0] = q_l0;
    q[chain.left_bilinear.1] = q_l1;
    q[chain.right_bilinear.0] = q_r0;
    q[chain.right_bilinear.1] = q_r1;

    // Recover the dropped joint from the FK residual.
    let drop = chain.drop_joint;
    let (alpha, a, d) = (chain.dh_alpha, chain.dh_a, chain.dh_d);
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

/// Numeric DH transform matrix at a specific joint angle — mirrors
/// `_raghavan_roth._dh_matrix_num`.
pub fn dh_matrix(theta: f64, alpha: f64, a: f64, d: f64) -> SMatrix<f64, 4, 4> {
    let (st, ct) = (libm::sin(theta), libm::cos(theta));
    let (sa, ca) = (libm::sin(alpha), libm::cos(alpha));
    SMatrix::<f64, 4, 4>::from_row_slice(&[
        ct,
        -st * ca,
        st * sa,
        a * ct,
        st,
        ct * ca,
        -ct * sa,
        a * st,
        0.0,
        sa,
        ca,
        d,
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}

/// Row-major `[[f64; 4]; 4]` → `SMatrix<4, 4>`.
pub fn mat4(a: &[[f64; 4]; 4]) -> SMatrix<f64, 4, 4> {
    SMatrix::<f64, 4, 4>::from_fn(|i, j| a[i][j])
}

/// Row-major `[[f64; 9]; 9]` → `SMatrix<9, 9>`.
pub fn mat9x9(a: &[[f64; 9]; 9]) -> SMatrix<f64, 9, 9> {
    SMatrix::<f64, 9, 9>::from_fn(|i, j| a[i][j])
}

/// Row-major `[[f64; 9]; 14]` → `SMatrix<14, 9>`.
pub fn mat14x9(a: &[[f64; 9]; 14]) -> SMatrix<f64, 14, 9> {
    SMatrix::<f64, 14, 9>::from_fn(|i, j| a[i][j])
}

/// Row-major `[[f64; 8]; 14]` → `SMatrix<14, 8>`.
pub fn mat14x8(a: &[[f64; 8]; 14]) -> SMatrix<f64, 14, 8> {
    SMatrix::<f64, 14, 8>::from_fn(|i, j| a[i][j])
}
