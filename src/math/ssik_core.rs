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

use libm::{fabs, sqrt};

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

// ===========================================================================
// IK-Geo subproblem core (arm-agnostic)
// ===========================================================================
//
// The routines below are a 1-1 extraction of the non-arm-specific geometric
// subproblem machinery used by every `ikgeo` closed-form driver: the SP6
// solver (two coupled conic constraints) and the small dense linear-algebra
// helpers it depends on (complete QR of a 4x2, quadratic roots, and the
// two-conic intersection that bottoms out in [`quartic_roots`]). They were
// originally inlined in the UR5 driver and are shared verbatim by the other
// three-parallel / spherical `ikgeo` arms.

use core::f64::consts::PI;

/// `2*pi`, the period used by the angle-wrapping helpers.
pub const TWO_PI: f64 = 2.0 * PI;

/// SP6 numerical tolerance (`ssik` `subproblem_numerical`).
const SP_NUM_TOL: f64 = 1e-5;
/// SP6 degeneracy tolerance (`ssik` `subproblem_degeneracy`).
const SP_DEG_TOL: f64 = 1e-12;

// --- 3-vector primitives ---------------------------------------------------

/// Euclidean dot product of two 3-vectors.
#[inline]
pub fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
#[inline]
pub fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rodrigues rotation of `v` about unit axis `k` by `theta`.
#[inline]
pub fn rotate(k: &[f64; 3], theta: f64, v: &[f64; 3]) -> [f64; 3] {
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

/// Wrap an angle into `(-pi, pi]` (IEEE remainder by `2*pi`).
#[inline]
pub fn wrap_to_pi(a: f64) -> f64 {
    libm::remainder(a, TWO_PI)
}

// --- Subproblem 6 (port of `ssik.subproblems.sp6` + `_aux`) -----------------

/// Solves IK-Geo Subproblem 6.
///
/// Given the four unit axes `h`, four rotation axes `k`, four position vectors
/// `p` and the two scalar targets `d1`, `d2`, finds the angle pairs
/// `(theta1, theta2)` satisfying
///
/// ```text
/// h0 . R(k0, t1) p0 + h1 . R(k1, t2) p1 = d1
/// h2 . R(k2, t1) p2 + h3 . R(k3, t2) p3 = d2
/// ```
///
/// Writes up to four `(theta1, theta2)` pairs into `out` and returns the count.
/// Matches `sp6.solve(...)`: it returns the exact solution set (residual below
/// the subproblem tolerance) or, failing that, the single best least-squares
/// pair.
pub fn sp6_solve(
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
        if pp >= SP_DEG_TOL {
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
    if libm::fabs(r00) < SP_DEG_TOL || libm::fabs(r11) < SP_DEG_TOL {
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
        if libm::fabs(n1 - 1.0) > SP_NUM_TOL || libm::fabs(n2 - 1.0) > SP_NUM_TOL {
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

    // Keep exact solutions (residual < SP_NUM_TOL); else best-LS fallback.
    let mut count = 0usize;
    for j in 0..ncand {
        if sp6_residual(refined[j].0, refined[j].1, h, k, p, d1, d2) < SP_NUM_TOL {
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
        // is caught downstream by the `r00`/`r11` SP_DEG_TOL check.
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
    let epsilon = SP_DEG_TOL;

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
