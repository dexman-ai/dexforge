//! Numeric tolerances used across ssik (predicates + subproblem solvers).
//!
//! Port of `ssik.core.tolerances.TolerancePolicy`. This is the single source
//! for every user-tunable threshold in the library. Each field captures the
//! *reason* the tolerance exists so diagnostics can reference named fields.
//!
//! # `no_std`
//!
//! This module is entirely `#![no_std]`-clean: [`TolerancePolicy`] is `Copy`
//! and stack-allocated with no dynamic data.

/// Numeric tolerances for kinematic predicates and subproblem solvers.
///
/// Mirrors the Python `ssik.core.tolerances.TolerancePolicy` dataclass.
///
/// Structural-predicate fields (`axis_parallel`, `axis_intersect`) are used
/// by topology classification.  Subproblem-solver fields are consumed by
/// SP1–SP6 under [`crate::math::ssik_core`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TolerancePolicy {
    /// Cross-product magnitude below which two unit-vector axes are considered
    /// parallel (or anti-parallel). Default `1e-8`.
    pub axis_parallel: f64,
    /// Perpendicular distance below which two lines in 3D are considered to
    /// intersect. Default `1e-8`.
    pub axis_intersect: f64,
    /// Threshold on residuals that decide whether a subproblem input admits
    /// an exact solution or only a least-squares approximation. Default `1e-9`.
    pub subproblem_feasibility: f64,
    /// Threshold for filtering spurious candidates produced by quartic / ellipse
    /// root-finding. Unit-circle closure is checked to this tolerance. Default `1e-5`.
    pub subproblem_numerical: f64,
    /// Rank / collinearity threshold. A QR diagonal or sin-of-angle-between-axes
    /// below this marks the input as degenerate. Default `1e-12`.
    pub subproblem_degeneracy: f64,
    /// Angle-space tolerance for deduplicating SP5/SP6 solutions. Two solutions
    /// within this radian distance collapse to one. Default `1e-3`.
    pub subproblem_dedup: f64,
}

/// The default tolerance policy matching the Python `DEFAULT_TOLERANCE_POLICY`.
pub const DEFAULT_TOLERANCE_POLICY: TolerancePolicy = TolerancePolicy {
    axis_parallel: 1e-8,
    axis_intersect: 1e-8,
    subproblem_feasibility: 1e-9,
    subproblem_numerical: 1e-5,
    subproblem_degeneracy: 1e-12,
    subproblem_dedup: 1e-3,
};

impl TolerancePolicy {
    /// Default policy matching the Python `DEFAULT_TOLERANCE_POLICY`.
    pub const DEFAULT: Self = DEFAULT_TOLERANCE_POLICY;

    /// Create a new tolerance policy with all default values.
    #[inline]
    pub const fn new() -> Self {
        DEFAULT_TOLERANCE_POLICY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_matches_python() {
        let p = DEFAULT_TOLERANCE_POLICY;
        assert_eq!(p.axis_parallel, 1e-8);
        assert_eq!(p.axis_intersect, 1e-8);
        assert_eq!(p.subproblem_feasibility, 1e-9);
        assert_eq!(p.subproblem_numerical, 1e-5);
        assert_eq!(p.subproblem_degeneracy, 1e-12);
        assert_eq!(p.subproblem_dedup, 1e-3);
    }

    #[test]
    fn new_equals_default() {
        assert_eq!(TolerancePolicy::new(), DEFAULT_TOLERANCE_POLICY);
    }

    #[test]
    fn policy_is_copy() {
        let a = DEFAULT_TOLERANCE_POLICY;
        let b = a; // Copy
        assert_eq!(a, b);
    }
}
