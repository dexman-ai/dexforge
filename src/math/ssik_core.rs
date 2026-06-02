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

    // Strip leading (near-)zero coefficients to find the true leading term.
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

    // Strip trailing zeros: each contributes a root at 0.
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
