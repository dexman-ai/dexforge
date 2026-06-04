//! Arm-agnostic post-processing utilities for IK solutions.
//!
//! Port of `ssik.postprocess`. These are composable filters that operate on
//! the raw geometric solution sets returned by the analytical IK kernels.
//!
//! # What lives here
//!
//! * [`respect_limits`] — drop solutions outside joint-limit ranges
//! * [`wrap_to_limits`] — try `q ± 2π` integer wraps to bring into range
//! * [`nearest_to_seed`] — sort solutions by wrap-to-pi joint-space distance
//!
//! # `no_std`
//!
//! All functions operate on fixed-size arrays (no `Vec`, no heap). The
//! generic `N` parameter is the DOF (number of joints). Solutions are
//! represented as `[f64; N]` joint-angle arrays.

use core::f64::consts::PI;
use crate::math::ssik_core::wrap_to_pi;

/// Joint limit specification: `(lower, upper)` bounds in radians.
/// Use `None` for unconstrained (continuous) joints.
pub type JointLimit = Option<(f64, f64)>;

/// Check whether all joints in `q` satisfy the given limits.
///
/// Joints with `limits[i] = None` are unconstrained and always pass.
/// Values exactly on the boundary are accepted.
///
/// Mirrors `ssik.postprocess.respect_limits` (single-solution predicate).
#[inline]
pub fn within_limits<const N: usize>(q: &[f64; N], limits: &[JointLimit; N]) -> bool {
    for i in 0..N {
        if let Some((lo, hi)) = limits[i] {
            if q[i] < lo || q[i] > hi {
                return false;
            }
        }
    }
    true
}

/// Filter an array of solutions, keeping only those within joint limits.
///
/// `solutions` has `count` valid entries on input. Returns the new count of
/// valid entries (compacted to the front). Mirrors `ssik.postprocess.respect_limits`.
pub fn respect_limits<const N: usize>(
    solutions: &mut [[f64; N]],
    count: usize,
    limits: &[JointLimit; N],
) -> usize {
    let mut out = 0usize;
    for i in 0..count {
        if within_limits(&solutions[i], limits) {
            solutions[out] = solutions[i];
            out += 1;
        }
    }
    out
}

/// Try wrapping each joint by `±2π` integer multiples to bring it into range.
///
/// Searches `k ∈ {-2, -1, 0, +1, +2}` (smallest `|k|` first). Joints with
/// `limits[i] = None` are left unchanged.
///
/// Mirrors `ssik.postprocess.wrap_to_limits`.
pub fn wrap_to_limits<const N: usize>(
    solutions: &mut [[f64; N]],
    count: usize,
    limits: &[JointLimit; N],
) {
    let two_pi = 2.0 * PI;
    for idx in 0..count {
        for i in 0..N {
            if let Some((lo, hi)) = limits[i] {
                let q_i = solutions[idx][i];
                if q_i >= lo && q_i <= hi {
                    continue;
                }
                // Try wraps with smallest |k| first.
                let wraps: [i32; 4] = [1, -1, 2, -2];
                for &k in wraps.iter() {
                    let candidate = q_i + two_pi * (k as f64);
                    if candidate >= lo && candidate <= hi {
                        solutions[idx][i] = candidate;
                        break;
                    }
                }
            }
        }
    }
}

/// Compute wrap-to-pi L2 distance between two joint configurations.
///
/// Treats angle differences modulo 2π, appropriate for revolute joints.
#[inline]
pub fn wrap_distance<const N: usize>(q: &[f64; N], seed: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        let d = wrap_to_pi(q[i] - seed[i]);
        sum += d * d;
    }
    libm::sqrt(sum)
}

/// Sort solutions by ascending wrap-to-pi distance to a seed configuration.
///
/// Only the first `count` entries of `solutions` are considered. Uses
/// insertion sort (bounded, O(n²) for small n — typical IK sets have ≤16
/// solutions). Stable: ties preserve input order.
///
/// Mirrors `ssik.postprocess.nearest_to_seed`.
pub fn nearest_to_seed<const N: usize>(
    solutions: &mut [[f64; N]],
    count: usize,
    seed: &[f64; N],
) {
    // Insertion sort — bounded iterations for the small solution sets
    // produced by analytical IK (≤ 16 for most arms).
    for i in 1..count {
        let key = solutions[i];
        let key_dist = wrap_distance(&key, seed);
        let mut j = i;
        while j > 0 && wrap_distance(&solutions[j - 1], seed) > key_dist {
            solutions[j] = solutions[j - 1];
            j -= 1;
        }
        solutions[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn within_limits_all_pass() {
        let q = [0.0, 1.0, -0.5];
        let limits: [JointLimit; 3] = [Some((-PI, PI)), Some((0.0, 2.0)), Some((-1.0, 0.0))];
        assert!(within_limits(&q, &limits));
    }

    #[test]
    fn within_limits_fail() {
        let q = [0.0, 3.0, -0.5];
        let limits: [JointLimit; 3] = [Some((-PI, PI)), Some((0.0, 2.0)), Some((-1.0, 0.0))];
        assert!(!within_limits(&q, &limits));
    }

    #[test]
    fn within_limits_unconstrained() {
        let q = [100.0, -999.0];
        let limits: [JointLimit; 2] = [None, None];
        assert!(within_limits(&q, &limits));
    }

    #[test]
    fn respect_limits_filters() {
        let mut sols = [[0.5, 0.5], [5.0, 0.5], [0.1, 0.1], [0.0; 2]];
        let limits: [JointLimit; 2] = [Some((-1.0, 1.0)), Some((-1.0, 1.0))];
        let n = respect_limits(&mut sols, 3, &limits);
        assert_eq!(n, 2);
        assert_eq!(sols[0], [0.5, 0.5]);
        assert_eq!(sols[1], [0.1, 0.1]);
    }

    #[test]
    fn wrap_to_limits_wraps_correctly() {
        // Joint at 7.0 rad should wrap to 7.0 - 2*pi ≈ 0.717
        let mut sols = [[7.0_f64; 1]];
        let limits: [JointLimit; 1] = [Some((-PI, PI))];
        wrap_to_limits(&mut sols, 1, &limits);
        assert!(
            sols[0][0] >= -PI && sols[0][0] <= PI,
            "wrapped value {} not in [-pi, pi]",
            sols[0][0]
        );
        let expected = 7.0 - 2.0 * PI;
        assert!((sols[0][0] - expected).abs() < 1e-10);
    }

    #[test]
    fn wrap_distance_zero_for_same() {
        let q = [1.0, 2.0, 3.0];
        assert!(wrap_distance(&q, &q) < 1e-15);
    }

    #[test]
    fn wrap_distance_modular() {
        // q differs by 2*pi in each joint -> distance should be ~0
        let q1 = [0.0, 0.0];
        let q2 = [2.0 * PI, -2.0 * PI];
        assert!(wrap_distance(&q1, &q2) < 1e-10);
    }

    #[test]
    fn nearest_to_seed_sorts() {
        let seed = [0.0, 0.0];
        let mut sols = [[3.0, 3.0], [0.1, 0.1], [1.0, 1.0], [0.0; 2]];
        nearest_to_seed(&mut sols, 3, &seed);
        // After sort: [0.1,0.1] closest, then [1,1], then [3,3]
        assert_eq!(sols[0], [0.1, 0.1]);
        assert_eq!(sols[1], [1.0, 1.0]);
        assert_eq!(sols[2], [3.0, 3.0]);
    }
}
