//! Kinematic self-identification from tracked motion.
//!
//! Vendors such as Universal Robots ship per-robot DH deltas; for arms that do
//! not, the same information can be recovered by moving the arm through a set
//! of configurations while an external tracker (laser tracker, motion capture,
//! calibrated camera) measures a flange-mounted target. [`ChainIdentifier`]
//! fits a [`KinematicChain`] to those samples.
//!
//! An external reference is required: joint encoders alone cannot observe the
//! link geometry.
//!
//! # Method
//!
//! Gauss–Newton with Levenberg damping over a small rigid correction of every
//! fixed transform of the chain (`6 · (DOF + 1)` parameters). The normal
//! equations are accumulated one sample at a time, so the caller owns the
//! sample storage and nothing is allocated; one *pass* is
//!
//! ```text
//! for (q, measured) in samples { identifier.add_sample(q, measured); }
//! let report = identifier.step();
//! ```
//!
//! repeated a fixed number of times (three to six passes suffice for
//! millimetre-scale errors). Each call is bounded; the solve is a fixed-size
//! Cholesky. This is a commissioning tool, not a control-loop routine: the
//! identifier holds a [`IDENTIFY_MAX_PARAMS`]² matrix (≈ 23 kB) and `step`
//! needs as much stack again.
//!
//! # Weighting and outliers
//!
//! Position residuals are in metres and orientation residuals in radians.
//! Orientation rows are scaled by a *rotation weight* in metres per radian
//! ([`ChainIdentifier::with_rotation_weight`]): the ratio of the tracker's
//! position noise to its orientation noise, or equivalently the lever arm at
//! which an orientation error matters as much as a position error.
//!
//! Zero-mean measurement noise averages out with the number of samples. Gross
//! errors (occlusions, a mis-indexed sample) do not; set
//! [`ChainIdentifier::with_outlier_gate`] to skip samples whose residual
//! exceeds a bound. The gate must be wider than the error of the initial
//! chain, or valid samples are skipped too — [`IdentifyReport::rejected`]
//! tells.
//!
//! # Frames
//!
//! The chain being fitted predicts the measured target in the *measurement*
//! frame: start from the nominal chain wrapped with the best guess of
//! tracker → base ([`KinematicChain::with_base`]) and flange → target
//! ([`KinematicChain::with_flange`]).
//!
//! The fit cannot tell an error in those two guesses from an error in the
//! arm's first and last link, so the result is exact only as a tracker →
//! target map. Removing the guesses again gives a chain in the driver's
//! frames whose *internal* geometry is calibrated but whose base and flange
//! frames are displaced by the guess errors. To obtain a chain that is exact
//! in the driver's own frames, measure tracker → base and / or flange →
//! target independently and hold them with [`ChainIdentifier::fix_base`] /
//! [`ChainIdentifier::fix_flange`].
//!
//! # Observability
//!
//! The parameterisation is deliberately redundant: a correction along or
//! about a joint axis can be attributed to either neighbouring transform. The
//! damping resolves the choice and the predicted target motion is unaffected.
//! [`IdentifyReport::observable`] counts the directions the samples actually
//! determine — `4·DOF + 6` for full-pose samples of a well-excited arm, three
//! fewer for position-only samples, which leave the target's orientation
//! free. A lower count means the motion did not excite every joint.

use crate::math::chain::{rotation_about, rotation_error, transform_from_pose, KinematicChain};
use crate::math::kinematics::{compose, Transform, IDENTITY};
use crate::math::model::Pose;
use crate::math::ssik_core::{cross3, dot3, norm3, sub3};

/// Largest `DOF` supported by [`ChainIdentifier`].
pub const IDENTIFY_MAX_DOF: usize = 8;
/// Capacity of the normal equations: six parameters per fixed transform.
pub const IDENTIFY_MAX_PARAMS: usize = 6 * (IDENTIFY_MAX_DOF + 1);
/// Levenberg damping relative to the diagonal of the normal matrix.
pub const IDENTIFY_RELATIVE_DAMPING: f64 = 1e-6;
/// Absolute damping keeping unobservable parameters at zero.
pub const IDENTIFY_ABSOLUTE_DAMPING: f64 = 1e-9;
/// Default rotation weight (m/rad): 1 mrad counts as 0.1 mm.
pub const IDENTIFY_DEFAULT_ROTATION_WEIGHT: f64 = 0.1;
/// A parameter is counted as observable when the samples determine at least
/// this fraction of its normal-matrix diagonal independently of the
/// parameters before it.
pub const IDENTIFY_OBSERVABLE_RATIO: f64 = 1e-4;

const NP: usize = IDENTIFY_MAX_PARAMS;

/// One tracker observation of the chain's flange frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Measurement {
    /// Full 6-DOF pose of the target.
    Pose(Pose),
    /// Position of the target origin only.
    Position([f64; 3]),
}

/// Outcome of one [`ChainIdentifier::step`], describing the pass *before* the
/// update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentifyReport {
    /// RMS position residual per sample (m).
    pub rms_position: f64,
    /// RMS orientation residual per pose sample (rad); zero without any.
    pub rms_rotation: f64,
    /// Samples used in this pass.
    pub samples: usize,
    /// Samples skipped by the outlier gate.
    pub rejected: usize,
    /// Parameters being estimated (six per free transform).
    pub parameters: usize,
    /// How many of them the samples determine; the rest are held by the
    /// damping. See the [module docs](self#observability).
    pub observable: usize,
}

/// Streaming least-squares fit of a [`KinematicChain`]. See the [module docs](self).
pub struct ChainIdentifier<const DOF: usize> {
    chain: KinematicChain<DOF>,
    estimate_base: bool,
    estimate_flange: bool,
    rotation_weight: f64,
    outlier_gate: f64,
    h: [[f64; NP]; NP],
    g: [f64; NP],
    sum_sq_position: f64,
    sum_sq_rotation: f64,
    pose_samples: usize,
    samples: usize,
    rejected: usize,
}

impl<const DOF: usize> ChainIdentifier<DOF> {
    /// Start from an initial chain (normally the nominal geometry).
    pub fn new(initial: KinematicChain<DOF>) -> Self {
        const { assert!(DOF >= 1 && DOF <= IDENTIFY_MAX_DOF) };
        ChainIdentifier {
            chain: initial,
            estimate_base: true,
            estimate_flange: true,
            rotation_weight: IDENTIFY_DEFAULT_ROTATION_WEIGHT,
            outlier_gate: f64::INFINITY,
            h: [[0.0; NP]; NP],
            g: [0.0; NP],
            sum_sq_position: 0.0,
            sum_sq_rotation: 0.0,
            pose_samples: 0,
            samples: 0,
            rejected: 0,
        }
    }

    /// Hold the base transform (measurement frame → first joint) fixed.
    pub fn fix_base(mut self) -> Self {
        self.estimate_base = false;
        self
    }

    /// Hold the last-joint → target transform fixed.
    pub fn fix_flange(mut self) -> Self {
        self.estimate_flange = false;
        self
    }

    /// Weight of orientation residuals relative to position residuals, in
    /// metres per radian. Non-finite or negative values are ignored; zero
    /// makes pose samples count as position samples.
    pub fn with_rotation_weight(mut self, metres_per_radian: f64) -> Self {
        if metres_per_radian.is_finite() && metres_per_radian >= 0.0 {
            self.rotation_weight = metres_per_radian;
        }
        self
    }

    /// Skip samples whose weighted residual `√(|Δp|² + (w·|Δθ|)²)` exceeds
    /// `metres`. Disabled (infinite) by default.
    pub fn with_outlier_gate(mut self, metres: f64) -> Self {
        if metres > 0.0 {
            self.outlier_gate = metres;
        }
        self
    }

    /// Current estimate.
    pub fn chain(&self) -> &KinematicChain<DOF> {
        &self.chain
    }

    /// Accumulate one sample into the current pass. Returns `false` if the
    /// sample was not used: non-finite, or beyond the outlier gate.
    pub fn add_sample(&mut self, q: &[f64; DOF], measured: &Measurement) -> bool {
        // Frame in which each fixed transform is corrected: the frame it is
        // left-multiplied into.
        let mut prefix = [IDENTITY; IDENTIFY_MAX_DOF + 1];
        let mut t = IDENTITY;
        for i in 0..DOF {
            prefix[i] = t;
            t = compose(
                &compose(&t, &self.chain.origin[i]),
                &self.chain.joint_rotation(i, q[i]),
            );
        }
        prefix[DOF] = t;
        let tip = compose(&t, &self.chain.flange);

        // Residual in metres: orientation rows carry the rotation weight.
        let mut r = [0.0f64; 6];
        let (dp, dw, rows) = match measured {
            Measurement::Pose(pose) => {
                let target = match transform_from_pose(pose) {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                (
                    sub3(&target.t, &tip.t),
                    rotation_error(&target.r, &tip.r),
                    6,
                )
            }
            Measurement::Position(p) => (sub3(p, &tip.t), [0.0; 3], 3),
        };
        for k in 0..3 {
            r[k] = dp[k];
            r[k + 3] = self.rotation_weight * dw[k];
        }
        let mut norm_sq = 0.0;
        for v in r {
            norm_sq += v * v;
        }
        if !norm_sq.is_finite() {
            return false;
        }
        if libm::sqrt(norm_sq) > self.outlier_gate {
            self.rejected += 1;
            return false;
        }

        // A twist (v, ω) in frame P moves the tip by R v + (R ω) × (p_tip − p_P)
        // and rotates it by R ω.
        let n = 6 * (DOF + 1);
        let mut jac = [[0.0f64; NP]; 6];
        for b in 0..=DOF {
            let p = &prefix[b];
            let arm = sub3(&tip.t, &p.t);
            for k in 0..3 {
                let col = [p.r[0][k], p.r[1][k], p.r[2][k]];
                let lever = cross3(&col, &arm);
                for row in 0..3 {
                    jac[row][6 * b + k] = col[row];
                    jac[row][6 * b + 3 + k] = lever[row];
                    jac[row + 3][6 * b + 3 + k] = self.rotation_weight * col[row];
                }
            }
        }
        for row in 0..rows {
            let jr = &jac[row];
            for i in 0..n {
                self.g[i] += jr[i] * r[row];
                for j in i..n {
                    self.h[i][j] += jr[i] * jr[j];
                }
            }
        }
        self.sum_sq_position += dot3(&dp, &dp);
        if rows == 6 {
            self.sum_sq_rotation += dot3(&dw, &dw);
            self.pose_samples += 1;
        }
        self.samples += 1;
        true
    }

    /// Solve the accumulated pass, update the chain and start a new pass.
    ///
    /// Returns `None` (chain unchanged, pass discarded) if no samples were
    /// used or the normal equations are not positive definite.
    pub fn step(&mut self) -> Option<IdentifyReport> {
        let n = 6 * (DOF + 1);
        let mut a = self.h;
        let mut x = self.g;
        let rms = |sum_sq: f64, count: usize| {
            if count > 0 {
                libm::sqrt(sum_sq / count as f64)
            } else {
                0.0
            }
        };
        let fixed = |i: usize| {
            let block = i / 6;
            (block == 0 && !self.estimate_base) || (block == DOF && !self.estimate_flange)
        };
        let mut report = IdentifyReport {
            rms_position: rms(self.sum_sq_position, self.samples),
            rms_rotation: rms(self.sum_sq_rotation, self.pose_samples),
            samples: self.samples,
            rejected: self.rejected,
            parameters: (0..n).filter(|i| !fixed(*i)).count(),
            observable: 0,
        };
        let is_fixed: [bool; NP] = core::array::from_fn(|i| i < n && fixed(i));
        self.h = [[0.0; NP]; NP];
        self.g = [0.0; NP];
        self.sum_sq_position = 0.0;
        self.sum_sq_rotation = 0.0;
        self.pose_samples = 0;
        self.samples = 0;
        self.rejected = 0;
        if report.samples == 0 {
            return None;
        }

        let mut diagonal = [0.0f64; NP];
        for i in 0..n {
            diagonal[i] = a[i][i];
            if is_fixed[i] {
                for j in 0..n {
                    a[i.min(j)][i.max(j)] = 0.0;
                }
                a[i][i] = 1.0;
                x[i] = 0.0;
            } else {
                a[i][i] += IDENTIFY_RELATIVE_DAMPING * a[i][i] + IDENTIFY_ABSOLUTE_DAMPING;
            }
        }

        // In-place Cholesky A = Uᵀ U on the upper triangle. (Not nalgebra's:
        // the system is `n ≤ NP` at run time and lives in the caller's frame.)
        for i in 0..n {
            for j in i..n {
                let mut s = a[i][j];
                for k in 0..i {
                    s -= a[k][i] * a[k][j];
                }
                if i == j {
                    if !(s > 0.0) {
                        return None;
                    }
                    // What remains of the diagonal once the earlier parameters
                    // are accounted for is what the samples say about this one.
                    if !is_fixed[i] && s > IDENTIFY_OBSERVABLE_RATIO * diagonal[i] {
                        report.observable += 1;
                    }
                    a[i][i] = libm::sqrt(s);
                } else {
                    a[i][j] = s / a[i][i];
                }
            }
        }
        for i in 0..n {
            let mut s = x[i];
            for k in 0..i {
                s -= a[k][i] * x[k];
            }
            x[i] = s / a[i][i];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..n {
                s -= a[i][k] * x[k];
            }
            x[i] = s / a[i][i];
        }
        if !x[..n].iter().all(|v| v.is_finite()) {
            return None;
        }

        for b in 0..=DOF {
            let v = [x[6 * b], x[6 * b + 1], x[6 * b + 2]];
            let w = [x[6 * b + 3], x[6 * b + 4], x[6 * b + 5]];
            let angle = norm3(&w);
            let mut delta: Transform = if angle > 0.0 {
                rotation_about(&[w[0] / angle, w[1] / angle, w[2] / angle], angle)
            } else {
                IDENTITY
            };
            delta.t = v;
            if b < DOF {
                self.chain.origin[b] = compose(&delta, &self.chain.origin[b]);
            } else {
                self.chain.flange = compose(&delta, &self.chain.flange);
            }
        }
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::math::testing::{calibrated_chain, nominal_chain, sample_q};
    use std::boxed::Box;

    const SAMPLES: u64 = 60;
    const PASSES: usize = 6;

    fn tool() -> Pose {
        let mut t = Pose::identity();
        t[(0, 3)] = 0.05;
        t[(1, 3)] = -0.02;
        t[(2, 3)] = 0.11;
        t
    }

    fn position(chain: &KinematicChain<6>, q: &[f64; 6]) -> [f64; 3] {
        let t = chain.fk(q);
        [t[(0, 3)], t[(1, 3)], t[(2, 3)]]
    }

    /// Zero-mean pseudo-noise in `[-amp, amp]³`, independent of `sample_q`
    /// (SplitMix64 finaliser).
    fn noise(seed: u64, amp: f64) -> [f64; 3] {
        core::array::from_fn(|k| {
            let mut z = (3 * seed + k as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0) * amp
        })
    }

    /// Largest held-out position error of `chain` against `truth`.
    fn held_out_error(chain: &KinematicChain<6>, truth: &KinematicChain<6>) -> f64 {
        (1000..1040)
            .map(|s| {
                let q = sample_q::<6>(s, 3.0);
                norm3(&sub3(&position(chain, &q), &position(truth, &q)))
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn pose_tracking_recovers_calibrated_kinematics() {
        let truth = calibrated_chain();
        let mut ident = Box::new(ChainIdentifier::new(nominal_chain()));
        let mut first = None;
        let mut last = None;
        for _ in 0..PASSES {
            for s in 0..SAMPLES {
                let q = sample_q::<6>(s, 2.8);
                assert!(ident.add_sample(&q, &Measurement::Pose(truth.fk(&q))));
            }
            let report = ident.step().unwrap();
            assert_eq!((report.samples, report.rejected), (SAMPLES as usize, 0));
            // 4 parameters per revolute joint plus the two end frames.
            assert_eq!((report.parameters, report.observable), (42, 30));
            first = first.or(Some(report));
            last = Some(report);
        }
        let (first, last) = (first.unwrap(), last.unwrap());
        assert!(first.rms_position > 1e-4 && first.rms_rotation > 1e-4);
        assert!(last.rms_position < 1e-9 && last.rms_rotation < 1e-9);
        // Held-out configurations.
        for s in 1000..1040 {
            let q = sample_q::<6>(s, 3.0);
            let err = (ident.chain().fk(&q) - truth.fk(&q)).norm();
            assert!(err < 1e-8, "held-out error {err}");
        }
    }

    #[test]
    fn position_tracking_recovers_positioning_and_marker_offset() {
        let truth = calibrated_chain().with_flange(&tool()).unwrap();
        // The marker mount is only known to a few millimetres.
        let mut guess = tool();
        guess[(0, 3)] += 3e-3;
        guess[(2, 3)] -= 2e-3;
        let initial = nominal_chain().with_flange(&guess).unwrap();
        let mut ident = Box::new(ChainIdentifier::new(initial));
        for _ in 0..PASSES {
            for s in 0..SAMPLES {
                let q = sample_q::<6>(s, 2.8);
                ident.add_sample(&q, &Measurement::Position(position(&truth, &q)));
            }
            let report = ident.step().unwrap();
            assert_eq!(report.rms_rotation, 0.0);
            // The target's orientation is not observable from its position.
            assert_eq!(report.observable, 27);
        }
        assert!(held_out_error(ident.chain(), &truth) < 1e-7);
    }

    #[test]
    fn measurement_noise_averages_out() {
        let truth = calibrated_chain();
        let amp = 1e-4; // ±0.1 mm per axis
        let fit = |samples: u64| {
            let mut ident = Box::new(ChainIdentifier::new(nominal_chain()));
            let mut rms = 0.0;
            for _ in 0..PASSES {
                for s in 0..samples {
                    let q = sample_q::<6>(s, 2.8);
                    let (p, e) = (position(&truth, &q), noise(s, amp));
                    let measured = [p[0] + e[0], p[1] + e[1], p[2] + e[2]];
                    ident.add_sample(&q, &Measurement::Position(measured));
                }
                rms = ident.step().unwrap().rms_position;
            }
            (held_out_error(ident.chain(), &truth), rms)
        };
        let nominal_error = held_out_error(&nominal_chain(), &truth);
        let (few, _) = fit(40);
        let (many, rms) = fit(400);
        // The fit converges to the noise floor, not below it …
        assert!(rms > 0.5 * amp && rms < 2.0 * amp, "rms {rms}");
        // … and more samples buy accuracy well inside the noise amplitude.
        assert!(
            many < 0.5 * amp,
            "held-out {many} (few {few}, rms {rms}, nominal {nominal_error})"
        );
        assert!(many < few, "{many} vs {few}");
        assert!(many < 0.1 * nominal_error, "{many} vs {nominal_error}");
    }

    #[test]
    fn outlier_gate_rejects_gross_errors() {
        let truth = calibrated_chain();
        let fit = |gate: Option<f64>| {
            let mut ident = ChainIdentifier::new(nominal_chain()).with_rotation_weight(0.2);
            if let Some(gate) = gate {
                ident = ident.with_outlier_gate(gate);
            }
            let mut ident = Box::new(ident);
            let mut rejected = 0;
            for _ in 0..PASSES {
                for s in 0..SAMPLES {
                    let q = sample_q::<6>(s, 2.8);
                    let mut pose = truth.fk(&q);
                    if s % 10 == 3 {
                        pose[(1, 3)] += 0.03; // a 3 cm glitch on every tenth sample
                    }
                    ident.add_sample(&q, &Measurement::Pose(pose));
                }
                rejected = ident.step().unwrap().rejected;
            }
            (held_out_error(ident.chain(), &truth), rejected)
        };
        let (ungated, none) = fit(None);
        let (gated, rejected) = fit(Some(0.01));
        assert_eq!((none, rejected), (0, 6));
        assert!(ungated > 1e-3, "outliers should corrupt the fit: {ungated}");
        assert!(gated < 1e-8, "gated fit error {gated}");
    }

    #[test]
    fn rotation_weight_scales_orientation_rows() {
        let truth = calibrated_chain();
        let q = sample_q::<6>(1, 2.0);
        let pose = truth.fk(&q);
        // With zero weight a pose sample carries no orientation information.
        let mut by_pose = Box::new(ChainIdentifier::new(nominal_chain()).with_rotation_weight(0.0));
        let mut by_position = Box::new(ChainIdentifier::new(nominal_chain()));
        for s in 0..SAMPLES {
            let q = sample_q::<6>(s, 2.8);
            by_pose.add_sample(&q, &Measurement::Pose(truth.fk(&q)));
            by_position.add_sample(&q, &Measurement::Position(position(&truth, &q)));
        }
        let (a, b) = (by_pose.step().unwrap(), by_position.step().unwrap());
        assert_eq!(a.observable, b.observable);
        assert!(a.rms_rotation > 0.0 && b.rms_rotation == 0.0);
        assert_eq!(by_pose.chain(), by_position.chain());

        // Invalid weights are ignored; a non-rigid pose is not a sample.
        let mut ident = Box::new(
            ChainIdentifier::new(nominal_chain())
                .with_rotation_weight(f64::NAN)
                .with_rotation_weight(-1.0),
        );
        assert!(ident.add_sample(&q, &Measurement::Pose(pose)));
        assert!(!ident.add_sample(&q, &Measurement::Pose(Pose::zeros())));
    }

    #[test]
    fn fixed_blocks_are_not_modified_and_empty_pass_is_rejected() {
        let truth = calibrated_chain();
        let initial = nominal_chain();
        let mut ident = Box::new(ChainIdentifier::new(initial).fix_base().fix_flange());
        assert!(ident.step().is_none());
        for s in 0..SAMPLES {
            let q = sample_q::<6>(s, 2.8);
            ident.add_sample(&q, &Measurement::Pose(truth.fk(&q)));
        }
        assert!(!ident.add_sample(&[f64::NAN; 6], &Measurement::Position([0.0; 3])));
        let report = ident.step().unwrap();
        assert_eq!((report.samples, report.parameters), (SAMPLES as usize, 30));
        assert_eq!(ident.chain().joint(0).origin, initial.joint(0).origin);
        assert_eq!(ident.chain().flange(), initial.flange());
        assert_ne!(ident.chain().joint(2).origin, initial.joint(2).origin);
    }
}
