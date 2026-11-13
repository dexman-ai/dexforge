//! Kinematic self-identification from tracked motion.
//!
//! Vendors such as Universal Robots ship per-robot DH deltas; for arms that do
//! not, the same information can be recovered by moving the arm through a set
//! of configurations while an external tracker (laser tracker, motion capture,
//! calibrated camera) measures a flange-mounted target. [`ChainIdentifier`]
//! fits a [`KinematicChain`] to those samples and the result is installed with
//! [`crate::math::arm::Arm::set_calibration`] exactly like a vendor calibration.
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
//! # Frames and observability
//!
//! The chain predicts the measured target in the measurement frame, so start
//! from the nominal chain wrapped with the best guess of tracker → base
//! ([`KinematicChain::with_base`]) and flange → target
//! ([`KinematicChain::with_flange`]); errors in those guesses are estimated
//! along with the link geometry. Removing the same two guesses from the result
//! gives the calibrated chain in the driver's own frames.
//!
//! The parameterisation is deliberately redundant (a correction along a joint
//! axis can be attributed to either neighbouring transform); the damping
//! resolves the choice and the predicted flange motion is unaffected.
//! Position-only measurements leave the target's orientation undetermined, so
//! they calibrate positioning accuracy only.

use crate::math::arm::Pose;
use crate::math::chain::{rotation_error, KinematicChain, Rigid};
use crate::math::ssik_core::{cross3, dot3};

/// Largest `DOF` supported by [`ChainIdentifier`].
pub const IDENTIFY_MAX_DOF: usize = 8;
/// Capacity of the normal equations: six parameters per fixed transform.
pub const IDENTIFY_MAX_PARAMS: usize = 6 * (IDENTIFY_MAX_DOF + 1);
/// Levenberg damping relative to the diagonal of the normal matrix.
pub const IDENTIFY_RELATIVE_DAMPING: f64 = 1e-6;
/// Absolute damping keeping unobservable parameters at zero.
pub const IDENTIFY_ABSOLUTE_DAMPING: f64 = 1e-9;

const NP: usize = IDENTIFY_MAX_PARAMS;

/// One tracker observation of the chain's flange frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Measurement {
    /// Full 6-DOF pose of the target.
    Pose(Pose),
    /// Position of the target origin only.
    Position([f64; 3]),
}

/// Outcome of one [`ChainIdentifier::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdentifyReport {
    /// RMS residual of the samples of this pass, *before* the update
    /// (metres and radians mixed for pose measurements).
    pub rms: f64,
    /// Number of samples accumulated in this pass.
    pub samples: usize,
}

/// Streaming least-squares fit of a [`KinematicChain`]. See the [module docs](self).
pub struct ChainIdentifier<const DOF: usize> {
    chain: KinematicChain<DOF>,
    estimate_base: bool,
    estimate_flange: bool,
    h: [[f64; NP]; NP],
    g: [f64; NP],
    sum_sq: f64,
    rows: usize,
    samples: usize,
}

impl<const DOF: usize> ChainIdentifier<DOF> {
    /// Start from an initial chain (normally the nominal geometry).
    pub fn new(initial: KinematicChain<DOF>) -> Self {
        const { assert!(DOF >= 1 && DOF <= IDENTIFY_MAX_DOF) };
        ChainIdentifier {
            chain: initial,
            estimate_base: true,
            estimate_flange: true,
            h: [[0.0; NP]; NP],
            g: [0.0; NP],
            sum_sq: 0.0,
            rows: 0,
            samples: 0,
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

    /// Current estimate.
    pub fn chain(&self) -> &KinematicChain<DOF> {
        &self.chain
    }

    /// Accumulate one sample into the current pass. Non-finite samples are
    /// ignored and reported as `false`.
    pub fn add_sample(&mut self, q: &[f64; DOF], measured: &Measurement) -> bool {
        // Frame in which each fixed transform is corrected: the frame it is
        // left-multiplied into.
        let mut prefix = [Rigid::IDENTITY; IDENTIFY_MAX_DOF + 1];
        let mut t = Rigid::IDENTITY;
        for i in 0..DOF {
            prefix[i] = t;
            t = t.mul(&self.chain.origin[i]).mul(&Rigid::rotation(
                &self.chain.axis[i],
                q[i] + self.chain.offset[i],
            ));
        }
        prefix[DOF] = t;
        let tip = t.mul(&self.chain.flange);

        let mut r = [0.0f64; 6];
        let m = match measured {
            Measurement::Pose(pose) => {
                let target = match Rigid::from_pose(pose) {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                let w = rotation_error(&target.r, &tip.r);
                for k in 0..3 {
                    r[k] = target.p[k] - tip.p[k];
                    r[k + 3] = w[k];
                }
                6
            }
            Measurement::Position(p) => {
                for k in 0..3 {
                    r[k] = p[k] - tip.p[k];
                }
                3
            }
        };
        if !r.iter().all(|v| v.is_finite()) {
            return false;
        }

        // A twist (v, ω) in frame P moves the tip by R v + (R ω) × (p_tip − p_P)
        // and rotates it by R ω.
        let n = 6 * (DOF + 1);
        let mut jac = [[0.0f64; NP]; 6];
        for b in 0..=DOF {
            let p = &prefix[b];
            let arm = [tip.p[0] - p.p[0], tip.p[1] - p.p[1], tip.p[2] - p.p[2]];
            for k in 0..3 {
                let col = [p.r[0][k], p.r[1][k], p.r[2][k]];
                let lever = cross3(&col, &arm);
                for row in 0..3 {
                    jac[row][6 * b + k] = col[row];
                    jac[row][6 * b + 3 + k] = lever[row];
                    jac[row + 3][6 * b + 3 + k] = col[row];
                }
            }
        }
        for row in 0..m {
            let jr = &jac[row];
            for i in 0..n {
                self.g[i] += jr[i] * r[row];
                for j in i..n {
                    self.h[i][j] += jr[i] * jr[j];
                }
            }
            self.sum_sq += r[row] * r[row];
        }
        self.rows += m;
        self.samples += 1;
        true
    }

    /// Solve the accumulated pass, update the chain and start a new pass.
    ///
    /// Returns `None` (chain unchanged, pass discarded) if no samples were
    /// added or the normal equations are not positive definite.
    pub fn step(&mut self) -> Option<IdentifyReport> {
        let n = 6 * (DOF + 1);
        let mut a = self.h;
        let mut x = self.g;
        let report = IdentifyReport {
            rms: if self.rows > 0 {
                libm::sqrt(self.sum_sq / self.rows as f64)
            } else {
                0.0
            },
            samples: self.samples,
        };
        self.h = [[0.0; NP]; NP];
        self.g = [0.0; NP];
        self.sum_sq = 0.0;
        self.rows = 0;
        self.samples = 0;
        if report.samples == 0 {
            return None;
        }

        for i in 0..n {
            let block = i / 6;
            let fixed =
                (block == 0 && !self.estimate_base) || (block == DOF && !self.estimate_flange);
            if fixed {
                for j in 0..n {
                    a[i.min(j)][i.max(j)] = 0.0;
                }
                a[i][i] = 1.0;
                x[i] = 0.0;
            } else {
                a[i][i] += IDENTIFY_RELATIVE_DAMPING * a[i][i] + IDENTIFY_ABSOLUTE_DAMPING;
            }
        }

        // In-place Cholesky A = Uᵀ U on the upper triangle.
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
            let angle = libm::sqrt(dot3(&w, &w));
            let mut delta = if angle > 0.0 {
                Rigid::rotation(&[w[0] / angle, w[1] / angle, w[2] / angle], angle)
            } else {
                Rigid::IDENTITY
            };
            delta.p = v;
            if b < DOF {
                self.chain.origin[b] = delta.mul(&self.chain.origin[b]);
            } else {
                self.chain.flange = delta.mul(&self.chain.flange);
            }
        }
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::math::chain::tests::{sample_q, ur5_calibrated, ur5_nominal};
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

    #[test]
    fn pose_tracking_recovers_calibrated_kinematics() {
        let truth = ur5_calibrated();
        let mut ident = Box::new(ChainIdentifier::new(ur5_nominal()));
        let mut first_rms = 0.0;
        let mut last_rms = 0.0;
        for pass in 0..PASSES {
            for s in 0..SAMPLES {
                let q = sample_q::<6>(s, 2.8);
                assert!(ident.add_sample(&q, &Measurement::Pose(truth.fk(&q))));
            }
            let report = ident.step().unwrap();
            assert_eq!(report.samples, SAMPLES as usize);
            if pass == 0 {
                first_rms = report.rms;
            }
            last_rms = report.rms;
        }
        assert!(
            first_rms > 1e-4,
            "calibration error should be visible: {first_rms}"
        );
        assert!(last_rms < 1e-9, "did not converge: {last_rms}");
        // Held-out configurations.
        for s in 1000..1040 {
            let q = sample_q::<6>(s, 3.0);
            let err = (ident.chain().fk(&q) - truth.fk(&q)).norm();
            assert!(err < 1e-8, "held-out error {err}");
        }
    }

    #[test]
    fn position_tracking_recovers_positioning_and_marker_offset() {
        let truth = ur5_calibrated().with_flange(&tool()).unwrap();
        // The marker mount is only known to a few millimetres.
        let mut guess = tool();
        guess[(0, 3)] += 3e-3;
        guess[(2, 3)] -= 2e-3;
        let initial = ur5_nominal().with_flange(&guess).unwrap();
        let mut ident = Box::new(ChainIdentifier::new(initial));
        let position = |chain: &KinematicChain<6>, q: &[f64; 6]| {
            let t = chain.fk(q);
            [t[(0, 3)], t[(1, 3)], t[(2, 3)]]
        };
        for _ in 0..PASSES {
            for s in 0..SAMPLES {
                let q = sample_q::<6>(s, 2.8);
                ident.add_sample(&q, &Measurement::Position(position(&truth, &q)));
            }
            ident.step().unwrap();
        }
        for s in 1000..1040 {
            let q = sample_q::<6>(s, 3.0);
            let (a, b) = (position(ident.chain(), &q), position(&truth, &q));
            let err = (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
            assert!(err < 1e-7, "held-out position error {err}");
        }
    }

    #[test]
    fn fixed_blocks_are_not_modified_and_empty_pass_is_rejected() {
        let truth = ur5_calibrated();
        let initial = ur5_nominal();
        let mut ident = Box::new(ChainIdentifier::new(initial).fix_base().fix_flange());
        assert!(ident.step().is_none());
        for s in 0..SAMPLES {
            let q = sample_q::<6>(s, 2.8);
            ident.add_sample(&q, &Measurement::Pose(truth.fk(&q)));
        }
        assert!(!ident.add_sample(&[f64::NAN; 6], &Measurement::Position([0.0; 3])));
        assert_eq!(ident.step().unwrap().samples, SAMPLES as usize);
        assert_eq!(ident.chain().joint(0).origin, initial.joint(0).origin);
        assert_eq!(ident.chain().flange(), initial.flange());
        assert_ne!(ident.chain().joint(2).origin, initial.joint(2).origin);
    }
}
