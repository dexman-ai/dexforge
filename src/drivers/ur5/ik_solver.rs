//! Analytical inverse kinematics for the Universal Robots **UR5**.
//!
//! This is a line-by-line translation of the Python `ssik` prebuilt artifact
//! `ssik/prebuilt/ur5_ik.py` (solver `ikgeo.three_parallel`, tier 0) into a
//! deterministic, `#![no_std]`, zero-allocation Rust implementation.
//!
//! The closed form is: subproblem **SP6** for the coupled wrist/shoulder angle
//! pair `(q1, q5)`, followed by the inlined `SP1 + SP3 + SP1` chain for the
//! remaining joints. SP6 reduces to intersecting two conics, which produces a
//! quartic solved through the companion-matrix eigenvalue routine in
//! [`crate::math::ssik_core`].
//!
//! ## Equivalence to the Python reference
//!
//! [`compute_ik`] reproduces `ur5_ik.solve(T, respect_limits=False)`: the raw
//! geometric solution set, FK-verified against the target pose and de-duplicated
//! modulo `2*pi`. Joint-limit clamping (`respect_limits=True`) is intentionally
//! *not* applied so the output is the pure algebraic set; callers can filter to
//! limits afterwards if desired.

use nalgebra::SMatrix;

use crate::math::ssik_core::quartic_roots;

use core::f64::consts::PI;

#[cfg(test)]
extern crate std;

#[cfg(test)]
#[path = "test_data.rs"]
mod test_data;

/// Degrees of freedom of the UR5.
pub const DOF: usize = 6;

// --- tolerances (ssik DEFAULT_TOLERANCE_POLICY) ---
const NUM_TOL: f64 = 1e-5; // subproblem_numerical
const DEG_TOL: f64 = 1e-12; // subproblem_degeneracy
const DEDUP_TOL: f64 = 1e-3; // subproblem_dedup
const FK_ATOL: f64 = 1e-5; // FK closure tolerance (== subproblem_numerical)

// --- algebraic-block degeneracy constants from ur5_ik.py ---
const DEG_SQ: f64 = 1e-16;
const FEAS_TOL: f64 = 1e-8;
const TWO_PI: f64 = 2.0 * PI;

// --- baked KinBody constants (for FK verification) ---
const JOINT_AXES: [[f64; 3]; 6] = [
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.0, 6.123233995736766e-17],
    [0.0, -1.2246467991473532e-16, -1.0],
    [0.0, -1.0, 6.123233995736766e-17],
];

const T_LEFTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.089159),
    eye_with(-0.425, 0.0, 0.0),
    eye_with(-0.39225000000000004, 0.0, 0.0),
    eye_with(0.0, -0.10915, 0.0),
    eye_with(0.0, -1.3877787807814457e-17, -0.09465),
];

const T_RIGHTS: [[[f64; 4]; 4]; 6] = [
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    eye_with(0.0, 0.0, 0.0),
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 6.123233995736766e-17, -1.0, -0.0823],
        [0.0, 1.0, 6.123233995736766e-17, 5.204170427930421e-18],
        [0.0, 0.0, 0.0, 1.0],
    ],
];

/// 4x4 homogeneous identity with the given translation column.
const fn eye_with(x: f64, y: f64, z: f64) -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// ---------------------------------------------------------------------------
// 3-vector primitives
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rodrigues rotation of `v` about unit axis `k` by `theta`.
#[inline]
fn rotate(k: &[f64; 3], theta: f64, v: &[f64; 3]) -> [f64; 3] {
    let c = libm::cos(theta);
    let s = libm::sin(theta);
    let kv = dot3(k, v);
    let omc = 1.0 - c;
    let kxv = cross3(k, v);
    [
        v[0] * c + kxv[0] * s + k[0] * (kv * omc),
        v[1] * c + kxv[1] * s + k[1] * (kv * omc),
        v[2] * c + kxv[2] * s + k[2] * (kv * omc),
    ]
}

#[inline]
fn wrap_to_pi(a: f64) -> f64 {
    libm::remainder(a, TWO_PI)
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Computes the complete analytical inverse kinematics for the UR5.
///
/// `target_tf` is the 4x4 SE(3) pose of `ee_link` expressed in `base_link`.
/// Returns up to 16 candidate joint vectors `[q1, .., q6]` (the UR5 yields at
/// most 8); empty slots are [`None`]. The set is FK-verified to within
/// [`FK_ATOL`] and de-duplicated modulo `2*pi`, matching
/// `ur5_ik.solve(T, respect_limits=False)`.
pub fn compute_ik(target_tf: &SMatrix<f64, 4, 4>) -> [Option<[f64; 6]>; 16] {
    let mut solutions: [Option<[f64; 6]>; 16] = [None; 16];
    let mut solution_count = 0usize;

    // Raw algebraic candidates (stack array, at most 8 for the UR5).
    let mut cands = [[0.0f64; 6]; 8];
    let cand_count = solve_algebraic(target_tf, &mut cands);

    for ci in 0..cand_count {
        let q = cands[ci];
        // Reject non-finite candidates (mirrors np.isfinite filter).
        if !q.iter().all(|v| v.is_finite()) {
            continue;
        }
        // FK closure check.
        let residual = fk_residual(&q, target_tf);
        if residual > FK_ATOL {
            continue;
        }
        // Wrap-to-pi de-duplication against accepted solutions.
        let mut dup = false;
        for s in solutions.iter().take(solution_count) {
            if let Some(existing) = s {
                if q_close_wrap(&q, existing, DEDUP_TOL) {
                    dup = true;
                    break;
                }
            }
        }
        if dup {
            continue;
        }
        if solution_count < 16 {
            solutions[solution_count] = Some(q);
            solution_count += 1;
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

// ---------------------------------------------------------------------------
// Algebraic candidate generation (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Fills `out` with the raw algebraic IK candidates and returns the count.
fn solve_algebraic(t: &SMatrix<f64, 4, 4>, out: &mut [[f64; 6]; 8]) -> usize {
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
    let p_16_x = p_x - 1.64748849439063e-19 * r_01 - 0.0823 * r_02;
    let p_16_y = p_y - 1.64748849439063e-19 * r_11 - 0.0823 * r_12;
    let p_16_z = p_z - 1.64748849439063e-19 * r_21 - 0.0823 * r_22;
    let r_06_axes5 = [1.0 * r_02, 1.0 * r_12, 1.0 * r_22];

    const H_SP_0: [f64; 3] = [0.0, -1.0, 6.123233995736766e-17];
    const K_SP_0: [f64; 3] = [-0.0, -0.0, -1.0];
    const K_SP_1: [f64; 3] = [0.0, -1.2246467991473532e-16, -1.0];
    const NEG_P_5: [f64; 3] = [-0.0, 1.3877787807814457e-17, 0.09465];
    const NEG_AXES_5: [f64; 3] = [-0.0, 1.0, -6.123233995736766e-17];

    let p_16 = [p_16_x, p_16_y, p_16_z];
    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, 0.10915, 0.0, &mut theta15);

    for si in 0..n15 {
        let (q1, q5) = theta15[si];

        // NOTE: the literal constants in the theta14 / q6 / d_inner blocks
        // below (e.g. `6.12323399573677e-17`, `7.49879891330929e-33`) are the
        // machine-generated symbolic coefficients emitted by `ssik build` and
        // are copied verbatim from `ur5_ik.py`. Most are rounding residue of
        // exact zeros (cos(pi/2)) preserved here so the floating-point
        // accumulation order matches the Python reference bit-for-bit.

        // SP1 for theta14 = q1+q2+q3+q4.
        let th14_x0 = libm::sin(q5);
        let th14_x1 = 1.0 * r_22;
        let th14_x2 = libm::sin(q1);
        let th14_x3 = 6.12323399573677e-17 * r_01 - 1.0 * r_02;
        let th14_x4 = th14_x2 * th14_x3;
        let th14_x5 = 1.0 * r_01 + 6.12323399573677e-17 * r_02;
        let th14_x6 = 6.12323399573677e-17 * th14_x2 * th14_x5;
        let th14_x7 = libm::cos(q1);
        let th14_x8 = 6.12323399573677e-17 * r_11 - 1.0 * r_12;
        let th14_x9 = th14_x7 * th14_x8;
        let th14_x10 = 1.0 * r_11 + 6.12323399573677e-17 * r_12;
        let th14_x11 = 6.12323399573677e-17 * th14_x10 * th14_x7;
        let th14_x12 = th14_x11 + th14_x4 - th14_x6 - th14_x9;
        let th14_x13 = libm::cos(q5);
        let th14_x14 = 6.12323399573677e-17 * th14_x10 * th14_x2 - th14_x2 * th14_x8
            - th14_x3 * th14_x7
            + 6.12323399573677e-17 * th14_x5 * th14_x7;
        let th14_x15 = 1.0 * th14_x13;
        let theta14 = libm::atan2(
            -th14_x0 * th14_x1 - 6.12323399573677e-17 * th14_x0 * th14_x12
                + th14_x14 * (6.12323399573677e-17 - 6.12323399573677e-17 * th14_x13),
            -1.0 * th14_x0 * th14_x14
                + th14_x1 * (1.22464679914735e-16 * th14_x13 - 6.12323399573677e-17)
                + th14_x12 * (-th14_x15 - 7.49879891330929e-33)
                - (th14_x15 + 3.74939945665464e-33)
                    * (6.12323399573677e-17 * r_22 - th14_x11 - th14_x4 + th14_x6 + th14_x9),
        );

        // SP1 for q6 (wrist roll-2).
        let q6_x0 = libm::cos(q5);
        let q6_x1 = libm::sin(q1);
        let q6_x2 = 1.0 * q6_x1;
        let q6_x3 = libm::cos(q1);
        let q6_x4 = 1.0 * q6_x3;
        let q6_x5 = q6_x2 * r_00 - q6_x4 * r_10 + 6.12323399573677e-17 * r_20;
        let q6_x6 = libm::sin(q5);
        let q6_x7 = 6.12323399573677e-17 * r_22;
        let q6_x8 = 6.12323399573677e-17 * r_01 - 1.0 * r_02;
        let q6_x9 = q6_x4 * (6.12323399573677e-17 * r_11 - 1.0 * r_12);
        let q6_x10 = q6_x2 * q6_x8 - q6_x7 - q6_x9 + 3.74939945665464e-33 * r_21;
        let q6_x11 = 1.0 * r_01 + 6.12323399573677e-17 * r_02;
        let q6_x12 = 1.0 * r_11 + 6.12323399573677e-17 * r_12;
        let q6_x13 = q6_x11 * q6_x2 - q6_x12 * q6_x4
            + 6.12323399573677e-17 * r_21
            + 3.74939945665464e-33 * r_22;
        let q6_x14 = 1.0 * q6_x6;
        let q6_x15 = 1.0 * q6_x0;
        let q6 = libm::atan2(
            -6.12323399573677e-17 * q6_x10 * q6_x6 - q6_x13 * q6_x14
                + q6_x5 * (6.12323399573677e-17 * q6_x0 - 6.12323399573677e-17),
            q6_x10 * (-q6_x15 - 7.49879891330929e-33)
                + q6_x13 * (1.22464679914735e-16 * q6_x0 - 6.12323399573677e-17)
                + q6_x14 * q6_x5
                - (-q6_x15 - 3.74939945665464e-33)
                    * (-6.12323399573677e-17 * q6_x1 * q6_x11
                        + 1.0 * q6_x1 * q6_x8
                        + 6.12323399573677e-17 * q6_x12 * q6_x3
                        - q6_x7
                        - q6_x9),
        );

        // d_inner vector.
        let dinr_x0 = libm::sin(theta14);
        let dinr_x1 = libm::sin(q5);
        let dinr_x2 = libm::cos(theta14);
        let dinr_x3 = libm::cos(q5);
        let dinr_x4 = 1.22464679914735e-16 - 1.22464679914735e-16 * dinr_x3;
        let dinr_x5 = 1.0 * dinr_x3 + 1.49975978266186e-32;
        let dinr_x6 = libm::sin(q1);
        let dinr_x7 = p_y - 1.64748849439063e-19 * r_11 - 0.0823 * r_12;
        let dinr_x8 = libm::cos(q1);
        let dinr_x9 = p_x - 1.64748849439063e-19 * r_01 - 0.0823 * r_02;
        let dinr_x10 = 6.12323399573677e-17 * dinr_x2 - 6.12323399573677e-17;
        let dinr_x11 = 1.38777878078145e-17 * dinr_x4;
        let d_inner_x = -1.96734287847793e-17 * dinr_x0 * dinr_x4
            - 8.49769420904307e-34 * dinr_x0 * dinr_x5
            - 0.09465 * dinr_x0
            + 2.28650585388476e-18 * dinr_x1 * dinr_x2
            + 1.0 * dinr_x6 * dinr_x7
            + dinr_x8 * dinr_x9;
        let d_inner_y = 1.40008103759583e-34 * dinr_x0 * dinr_x1
            + dinr_x10 * dinr_x11
            + 5.79564097696485e-18 * dinr_x2
            + 2.28650585388476e-18 * dinr_x3
            - 1.0 * dinr_x6 * dinr_x9
            + dinr_x7 * dinr_x8
            + 0.10915;
        let d_inner_z = 2.28650585388476e-18 * dinr_x0 * dinr_x1
            + 0.09465 * dinr_x10 * dinr_x4
            + 1.38777878078145e-17 * dinr_x10 * dinr_x5
            + dinr_x11 * (1.0 * dinr_x2 + 3.74939945665464e-33)
            + 0.09465 * dinr_x2
            + 1.0 * p_z
            - 1.64748849439063e-19 * r_21
            - 0.0823 * r_22
            - 0.089159;
        let d_elbow =
            libm::sqrt(d_inner_x * d_inner_x + d_inner_y * d_inner_y + d_inner_z * d_inner_z);

        // SP3 for q3 (elbow).
        let q3_r_sq = 0.0277909737890625_f64;
        let q3_rhs = 0.16724253125 - 0.5 * d_elbow * d_elbow;
        let q3_phi = PI;
        let theta_q3_plus;
        let theta_q3_minus;
        if q3_r_sq < DEG_SQ {
            theta_q3_plus = 0.0;
            theta_q3_minus = 0.0;
        } else {
            let q3_r = libm::sqrt(q3_r_sq);
            if libm::fabs(q3_rhs) > q3_r + FEAS_TOL {
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
            let q2_x1 = 0.39225 * q2_x0;
            let q2_x2 = libm::cos(q3);
            let q2_x3 = -0.39225 * q2_x2 - 0.425;
            let q2 = libm::atan2(
                d_inner_x * q2_x1
                    + d_inner_y * (-2.40183853482775e-17 * q2_x2 - 2.60237444818813e-17)
                    + d_inner_z * q2_x3,
                d_inner_x * q2_x3 - 2.40183853482775e-17 * d_inner_y * q2_x0 - d_inner_z * q2_x1,
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
// Subproblem 6 (port of ssik.subproblems.sp6 + _aux)
// ---------------------------------------------------------------------------

/// Solves SP6. Writes up to 4 `(theta1, theta2)` pairs into `out` and returns
/// the count. Matches `sp6.solve(...)` returning its exact-or-best-LS set.
fn sp6_solve(
    h: &[[f64; 3]; 4],
    k: &[[f64; 3]; 4],
    p: &[[f64; 3]; 4],
    d1: f64,
    d2: f64,
    out: &mut [(f64, f64); 4],
) -> usize {
    // Whole-system degeneracy: every p_i collinear with k_i.
    let mut all_collinear = true;
    for i in 0..4 {
        let kp = dot3(&k[i], &p[i]);
        let pp = dot3(&p[i], &p[i]) - kp * kp;
        if pp >= DEG_TOL {
            all_collinear = false;
            break;
        }
    }
    if all_collinear {
        return 0;
    }

    // Build A (2x4).
    let mut a_mat = [[0.0f64; 4]; 2]; // 2x4
    for i in 0..4 {
        let kxp = cross3(&k[i], &p[i]);
        let kxkxp = cross3(&k[i], &kxp);
        let col0 = kxp;
        let col1 = [-kxkxp[0], -kxkxp[1], -kxkxp[2]];
        let h_a0 = dot3(&h[i], &col0);
        let h_a1 = dot3(&h[i], &col1);
        let r = i / 2;
        let c = (i % 2) * 2;
        a_mat[r][c] = h_a0;
        a_mat[r][c + 1] = h_a1;
    }

    let b = [
        d1 - dot3(&h[0], &k[0]) * dot3(&k[0], &p[0]) - dot3(&h[1], &k[1]) * dot3(&k[1], &p[1]),
        d2 - dot3(&h[2], &k[2]) * dot3(&k[2], &p[2]) - dot3(&h[3], &k[3]) * dot3(&k[3], &p[3]),
    ];

    // QR (complete) of A^T (4x2).
    let mut m = [[0.0f64; 2]; 4]; // A^T
    for r in 0..2 {
        for c in 0..4 {
            m[c][r] = a_mat[r][c];
        }
    }
    let (q, rr) = qr_complete_4x2(&m);

    let r00 = rr[0][0];
    let r11 = rr[1][1];
    if libm::fabs(r00) < DEG_TOL || libm::fabs(r11) < DEG_TOL {
        return 0;
    }

    // x_min = q_range @ solve_lower_triangular(r_lower, b).
    let r01 = rr[0][1];
    let x0 = b[0] / r00;
    let x1 = (b[1] - x0 * r01) / r11;
    let mut x_min = [0.0f64; 4];
    for i in 0..4 {
        x_min[i] = q[i][0] * x0 + q[i][1] * x1;
    }

    // Null-space basis: columns 2 and 3 of Q.
    let null1 = [q[0][2], q[1][2], q[2][2], q[3][2]];
    let null2 = [q[0][3], q[1][3], q[2][3], q[3][3]];

    let xn1 = [[null1[0], null2[0]], [null1[1], null2[1]]];
    let xn2 = [[null1[2], null2[2]], [null1[3], null2[3]]];
    let xm1 = [x_min[0], x_min[1]];
    let xm2 = [x_min[2], x_min[3]];

    let mut xi = [(0.0f64, 0.0f64); 4];
    let nxi = solve_two_ellipse_numeric(&xm1, &xn1, &xm2, &xn2, &mut xi);

    // Build candidate (theta1, theta2) pairs satisfying the unit constraints.
    let mut cands = [(0.0f64, 0.0f64); 4];
    let mut ncand = 0usize;
    for j in 0..nxi {
        let (xi0, xi1) = xi[j];
        let mut x = [0.0f64; 4];
        for i in 0..4 {
            x[i] = x_min[i] + xi0 * null1[i] + xi1 * null2[i];
        }
        let n1 = libm::sqrt(x[0] * x[0] + x[1] * x[1]);
        let n2 = libm::sqrt(x[2] * x[2] + x[3] * x[3]);
        if libm::fabs(n1 - 1.0) > NUM_TOL || libm::fabs(n2 - 1.0) > NUM_TOL {
            continue;
        }
        let theta1 = libm::atan2(x[0], x[1]);
        let theta2 = libm::atan2(x[2], x[3]);
        cands[ncand] = (theta1, theta2);
        ncand += 1;
    }

    // Sort by pre-refinement residual (ascending).
    let mut resid = [0.0f64; 4];
    for j in 0..ncand {
        resid[j] = sp6_residual(cands[j].0, cands[j].1, h, k, p, d1, d2);
    }
    for i in 1..ncand {
        let cj = cands[i];
        let rj = resid[i];
        let mut jj = i;
        while jj > 0 && resid[jj - 1] > rj {
            resid[jj] = resid[jj - 1];
            cands[jj] = cands[jj - 1];
            jj -= 1;
        }
        resid[jj] = rj;
        cands[jj] = cj;
    }

    // Gauss-Newton refine each.
    let mut refined = [(0.0f64, 0.0f64); 4];
    for j in 0..ncand {
        refined[j] = sp6_refine(cands[j].0, cands[j].1, h, k, p, d1, d2);
    }

    // Keep exact solutions (residual < NUM_TOL); else best-LS fallback.
    let mut count = 0usize;
    for j in 0..ncand {
        if sp6_residual(refined[j].0, refined[j].1, h, k, p, d1, d2) < NUM_TOL {
            out[count] = refined[j];
            count += 1;
        }
    }
    if count > 0 {
        return count;
    }
    if ncand == 0 {
        return 0;
    }
    let mut best = 0usize;
    let mut best_r = f64::INFINITY;
    for j in 0..ncand {
        let rj = sp6_residual(refined[j].0, refined[j].1, h, k, p, d1, d2);
        if rj < best_r {
            best_r = rj;
            best = j;
        }
    }
    out[0] = refined[best];
    1
}

fn sp6_residual(
    theta1: f64,
    theta2: f64,
    h: &[[f64; 3]; 4],
    k: &[[f64; 3]; 4],
    p: &[[f64; 3]; 4],
    d1: f64,
    d2: f64,
) -> f64 {
    let r0 = rotate(&k[0], theta1, &p[0]);
    let r1 = rotate(&k[1], theta2, &p[1]);
    let r2 = rotate(&k[2], theta1, &p[2]);
    let r3 = rotate(&k[3], theta2, &p[3]);
    let lhs1 = dot3(&h[0], &r0) + dot3(&h[1], &r1);
    let lhs2 = dot3(&h[2], &r2) + dot3(&h[3], &r3);
    libm::fabs(lhs1 - d1).max(libm::fabs(lhs2 - d2))
}

/// Gauss-Newton refinement of an SP6 angle pair (bounded 20 iterations).
fn sp6_refine(
    mut t1: f64,
    mut t2: f64,
    h: &[[f64; 3]; 4],
    k: &[[f64; 3]; 4],
    p: &[[f64; 3]; 4],
    d1: f64,
    d2: f64,
) -> (f64, f64) {
    let max_step = PI / 4.0;
    for _ in 0..20 {
        let r0 = rotate(&k[0], t1, &p[0]);
        let r1 = rotate(&k[1], t2, &p[1]);
        let r2 = rotate(&k[2], t1, &p[2]);
        let r3 = rotate(&k[3], t2, &p[3]);

        let f0 = dot3(&h[0], &r0) + dot3(&h[1], &r1) - d1;
        let f1 = dot3(&h[2], &r2) + dot3(&h[3], &r3) - d2;

        let j00 = dot3(&h[0], &cross3(&k[0], &r0));
        let j01 = dot3(&h[1], &cross3(&k[1], &r1));
        let j10 = dot3(&h[2], &cross3(&k[2], &r2));
        let j11 = dot3(&h[3], &cross3(&k[3], &r3));

        let det = j00 * j11 - j01 * j10;
        if libm::fabs(det) < 1e-15 {
            break;
        }
        let inv_det = 1.0 / det;
        let mut delta0 = -(j11 * f0 - j01 * f1) * inv_det;
        let mut delta1 = -(-j10 * f0 + j00 * f1) * inv_det;

        let step_norm = libm::sqrt(delta0 * delta0 + delta1 * delta1);
        if step_norm > max_step {
            let scale = max_step / step_norm;
            delta0 *= scale;
            delta1 *= scale;
        }

        t1 = wrap_to_pi(t1 + delta0);
        t2 = wrap_to_pi(t2 + delta1);

        if step_norm < 1e-15 {
            break;
        }
    }
    (t1, t2)
}

/// Complete Householder QR of a 4x2 matrix `m`: returns `(Q, R)` with
/// `m = Q R`, `Q` orthogonal 4x4, `R` upper-triangular 4x2.
fn qr_complete_4x2(m: &[[f64; 2]; 4]) -> ([[f64; 4]; 4], [[f64; 2]; 4]) {
    let mut a = *m; // becomes R
    let mut q = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    for col in 0..2 {
        let mut norm = 0.0;
        for i in col..4 {
            norm += a[i][col] * a[i][col];
        }
        let norm = libm::sqrt(norm);
        // Exact-zero guard: skip the reflector only for a genuinely zero
        // column (it is already the identity there). A near-zero-but-nonzero
        // column still yields a valid Householder reflector; rank deficiency
        // is caught downstream by the `r00`/`r11` DEG_TOL check.
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[col][col] > 0.0 { -norm } else { norm };
        let mut v = [0.0f64; 4];
        for i in col..4 {
            v[i] = a[i][col];
        }
        v[col] -= alpha;
        let mut vnorm2 = 0.0;
        for i in col..4 {
            vnorm2 += v[i] * v[i];
        }
        if vnorm2 == 0.0 {
            continue;
        }
        let beta = 2.0 / vnorm2;

        // R <- H R
        for j in 0..2 {
            let mut s = 0.0;
            for i in col..4 {
                s += v[i] * a[i][j];
            }
            s *= beta;
            for i in col..4 {
                a[i][j] -= s * v[i];
            }
        }
        // Q <- Q H
        for i in 0..4 {
            let mut s = 0.0;
            for t in col..4 {
                s += q[i][t] * v[t];
            }
            s *= beta;
            for t in col..4 {
                q[i][t] -= s * v[t];
            }
        }
    }

    (q, a)
}

/// Real roots of `a x^2 + b x + c = 0`; linear or empty when degenerate.
fn quadratic_real_roots(a: f64, b: f64, c: f64, tol: f64, out: &mut [f64; 2]) -> usize {
    if libm::fabs(a) < tol {
        if libm::fabs(b) < tol {
            return 0;
        }
        out[0] = -c / b;
        return 1;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < -tol {
        return 0;
    }
    let root_disc = libm::sqrt(if disc > 0.0 { disc } else { 0.0 });
    out[0] = (-b + root_disc) / (2.0 * a);
    out[1] = (-b - root_disc) / (2.0 * a);
    2
}

/// Conic coefficient tuple `(a, b, c, d, e, f)`.
type Conic = (f64, f64, f64, f64, f64, f64);

/// Degenerate-conic intersection: `coef1` decouples to one variable.
fn solve_two_ellipse_degenerate(
    coef1: Conic,
    coef2: Conic,
    tol: f64,
    primary_y: bool,
    out: &mut [(f64, f64); 4],
) -> usize {
    let mut count = 0usize;
    if primary_y {
        let (_a, _b, c, _d, e, f) = coef1;
        let mut proots = [0.0f64; 2];
        let np = quadratic_real_roots(c, e, f, tol, &mut proots);
        let (a2, b2, c2, d2, e2, f2) = coef2;
        for pi in 0..np {
            let y = proots[pi];
            let mut oroots = [0.0f64; 2];
            let no =
                quadratic_real_roots(a2, b2 * y + d2, c2 * y * y + e2 * y + f2, tol, &mut oroots);
            for oi in 0..no {
                if count < 4 {
                    out[count] = (oroots[oi], y);
                    count += 1;
                }
            }
        }
    } else {
        let (a, _b, _c, d, _e, f) = coef1;
        let mut proots = [0.0f64; 2];
        let np = quadratic_real_roots(a, d, f, tol, &mut proots);
        let (a2, b2, c2, d2, e2, f2) = coef2;
        for pi in 0..np {
            let x = proots[pi];
            let mut oroots = [0.0f64; 2];
            let no =
                quadratic_real_roots(c2, b2 * x + e2, a2 * x * x + d2 * x + f2, tol, &mut oroots);
            for oi in 0..no {
                if count < 4 {
                    out[count] = (x, oroots[oi]);
                    count += 1;
                }
            }
        }
    }
    count
}

/// Real intersections of two conics parameterised as `|xm_i + xn_i @ [xi,eta]| = 1`.
fn solve_two_ellipse_numeric(
    xm1: &[f64; 2],
    xn1: &[[f64; 2]; 2],
    xm2: &[f64; 2],
    xn2: &[[f64; 2]; 2],
    out: &mut [(f64, f64); 4],
) -> usize {
    let epsilon = DEG_TOL;

    let a_1 = mat2_ata(xn1);
    let a = a_1[0][0];
    let b = 2.0 * a_1[1][0];
    let c = a_1[1][1];
    let b_1 = vec2_mat2(xm1, xn1);
    let d = 2.0 * b_1[0];
    let e = 2.0 * b_1[1];
    let f = (xm1[0] * xm1[0] + xm1[1] * xm1[1]) - 1.0;

    let a_2 = mat2_ata(xn2);
    let a1 = a_2[0][0];
    let b1 = 2.0 * a_2[1][0];
    let c1 = a_2[1][1];
    let b_2 = vec2_mat2(xm2, xn2);
    let d1c = 2.0 * b_2[0];
    let e1 = 2.0 * b_2[1];
    let fq = (xm2[0] * xm2[0] + xm2[1] * xm2[1]) - 1.0;

    let coef1: Conic = (a, b, c, d, e, f);
    let coef2: Conic = (a1, b1, c1, d1c, e1, fq);

    if libm::fabs(a) < epsilon && libm::fabs(b) < epsilon && libm::fabs(d) < epsilon {
        return solve_two_ellipse_degenerate(coef1, coef2, epsilon, true, out);
    }
    if libm::fabs(c) < epsilon && libm::fabs(b) < epsilon && libm::fabs(e) < epsilon {
        return solve_two_ellipse_degenerate(coef1, coef2, epsilon, false, out);
    }
    if libm::fabs(a1) < epsilon && libm::fabs(b1) < epsilon && libm::fabs(d1c) < epsilon {
        return solve_two_ellipse_degenerate(coef2, coef1, epsilon, true, out);
    }
    if libm::fabs(c1) < epsilon && libm::fabs(b1) < epsilon && libm::fabs(e1) < epsilon {
        return solve_two_ellipse_degenerate(coef2, coef1, epsilon, false, out);
    }

    let z0 = f * a * d1c * d1c + a * a * fq * fq - d * a * d1c * fq + a1 * a1 * f * f
        - 2.0 * a * fq * a1 * f
        - d * d1c * a1 * f
        + a1 * d * d * fq;

    let z1 = e1 * d * d * a1 - fq * d1c * a * b - 2.0 * a * fq * a1 * e - f * a1 * b1 * d
        + 2.0 * d1c * b1 * a * f
        + 2.0 * e1 * fq * a * a
        + d1c * d1c * a * e
        - e1 * d1c * a * d
        - 2.0 * a * e1 * a1 * f
        - f * a1 * d1c * b
        + 2.0 * f * e * a1 * a1
        - fq * b1 * a * d
        - e * a1 * d1c * d
        + 2.0 * fq * b * a1 * d;

    let z2 = e1 * e1 * a * a + 2.0 * c1 * fq * a * a - e * a1 * d1c * b + fq * a1 * b * b
        - e * a1 * b1 * d
        - fq * b1 * a * b
        - 2.0 * a * e1 * a1 * e
        + 2.0 * d1c * b1 * a * e
        - c1 * d1c * a * d
        - 2.0 * a * c1 * a1 * f
        + b1 * b1 * a * f
        + 2.0 * e1 * b * a1 * d
        + e * e * a1 * a1
        - c * a1 * d1c * d
        - e1 * b1 * a * d
        + 2.0 * f * c * a1 * a1
        - f * a1 * b1 * b
        + c1 * d * d * a1
        + d1c * d1c * a * c
        - e1 * d1c * a * b
        - 2.0 * a * fq * a1 * c;

    let z3 = -2.0 * a * a1 * c * e1 + e1 * a1 * b * b + 2.0 * c1 * b * a1 * d - c * a1 * b1 * d
        + b1 * b1 * a * e
        - e1 * b1 * a * b
        - 2.0 * a * c1 * a1 * e
        - e * a1 * b1 * b
        - c1 * b1 * a * d
        + 2.0 * e1 * c1 * a * a
        + 2.0 * e * c * a1 * a1
        - c * a1 * d1c * b
        + 2.0 * d1c * b1 * a * c
        - c1 * d1c * a * b;

    let z4 = a * a * c1 * c1 - 2.0 * a * c1 * a1 * c + a1 * a1 * c * c - b * a * b1 * c1
        - b * b1 * a1 * c
        + b * b * a1 * c1
        + c * a * b1 * b1;

    let (roots, nr) = quartic_roots(&[z4, z3, z2, z1, z0]);

    let mut count = 0usize;
    for i in 0..nr {
        if libm::fabs(roots[i].im) >= epsilon {
            continue;
        }
        let y = roots[i].re;
        let num = -((a * c1 * y * y + a * fq) - a1 * c * y * y + a * e1 * y - a1 * e * y - a1 * f);
        let den = (a * b1 * y + a * d1c) - a1 * b * y - a1 * d;
        if libm::fabs(den) < epsilon {
            continue;
        }
        let x = num / den;
        if count < 4 {
            out[count] = (x, y);
            count += 1;
        }
    }
    count
}

/// `M^T M` for a 2x2 matrix stored row-major.
#[inline]
fn mat2_ata(m: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let c00 = m[0][0] * m[0][0] + m[1][0] * m[1][0];
    let c01 = m[0][0] * m[0][1] + m[1][0] * m[1][1];
    let c11 = m[0][1] * m[0][1] + m[1][1] * m[1][1];
    [[c00, c01], [c01, c11]]
}

/// `v^T M` (1x2) for 2-vector `v` and 2x2 matrix `M`.
#[inline]
fn vec2_mat2(v: &[f64; 2], m: &[[f64; 2]; 2]) -> [f64; 2] {
    [
        v[0] * m[0][0] + v[1] * m[1][0],
        v[0] * m[0][1] + v[1] * m[1][1],
    ]
}

// ---------------------------------------------------------------------------
// Forward kinematics (for FK closure verification)
// ---------------------------------------------------------------------------

fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for kk in 0..4 {
                s += a[i][kk] * b[kk][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn rot_homogeneous(axis: &[f64; 3], angle: f64) -> [[f64; 4]; 4] {
    let c = libm::cos(angle);
    let s = libm::sin(angle);
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let oc = 1.0 - c;
    [
        [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s, 0.0],
        [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s, 0.0],
        [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Product-of-exponentials forward kinematics using the baked chain constants.
fn fk(q: &[f64; 6]) -> [[f64; 4]; 4] {
    let mut acc = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for i in 0..6 {
        let rot = rot_homogeneous(&JOINT_AXES[i], q[i]);
        let m = mat4_mul(&T_LEFTS[i], &rot);
        let n = mat4_mul(&m, &T_RIGHTS[i]);
        acc = mat4_mul(&acc, &n);
    }
    acc
}

/// Frobenius-norm residual between FK(q) and the target pose.
fn fk_residual(q: &[f64; 6], target: &SMatrix<f64, 4, 4>) -> f64 {
    let t = fk(q);
    let mut s = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = t[i][j] - target[(i, j)];
            s += d * d;
        }
    }
    libm::sqrt(s)
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

    #[test]
    fn home_pose_round_trips() {
        let t_home = [
            1.0, 0.0, 0.0, -0.81725,
            0.0, 6.123233995736766e-17, -1.0, -0.19145,
            0.0, 1.0, 6.123233995736766e-17, -0.005490999999999996,
            0.0, 0.0, 0.0, 1.0,
        ];
        let tf = tf_from_row_major(&t_home);
        let sols = compute_ik(&tf);
        let zero = [0.0; 6];
        let found = sols.iter().flatten().any(|s| q_eq_mod(s, &zero, 1e-6));
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
}
