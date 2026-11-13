//! Runtime (calibrated) kinematic chain.
//!
//! The per-arm drivers bake the *nominal* geometry into generated code. A
//! physical arm deviates from it by a few tenths of a millimetre / milliradian,
//! which vendors such as Universal Robots ship as per-robot DH deltas. A
//! [`KinematicChain`] carries that per-robot geometry as plain data so it can be
//! swapped at runtime without regenerating anything:
//!
//! ```text
//! base→flange = Π_i [ origin_i · Rot(axis_i, q_i + offset_i) ] · flange
//! ```
//!
//! This form is a superset of standard DH, modified (Craig) DH and URDF joint
//! origins, so one type covers vendor calibration files ([`KinematicChain::from_dh`]
//! and [`apply_dh_deltas`]) and chains identified from tracked motion
//! ([`crate::math::identify`]).
//!
//! Everything here is `no_std`, allocation-free and bounded: FK and the
//! geometric Jacobian are a single fixed `DOF` loop, and [`KinematicChain::refine_ik`]
//! is a Newton polish capped at [`CHAIN_IK_MAX_ITERS`].

use nalgebra::SMatrix;

use crate::math::arm::Pose;
use crate::math::ssik_core::{add3, cross3, dot3};

/// Iteration ceiling of [`KinematicChain::refine_ik`].
pub const CHAIN_IK_MAX_ITERS: usize = 12;
/// Twist-norm threshold below which [`KinematicChain::refine_ik`] stops early.
pub const CHAIN_IK_CONVERGENCE: f64 = 1e-11;
/// Tikhonov damping added to `J Jᵀ` in [`KinematicChain::refine_ik`].
pub const CHAIN_IK_DAMPING: f64 = 1e-10;
/// Largest joint step (rad, infinity norm) taken by one polish iteration.
pub const CHAIN_IK_MAX_STEP: f64 = 0.5;
/// Orthonormality tolerance for rotation blocks accepted as rigid transforms.
pub const RIGID_ATOL: f64 = 1e-6;
/// Axes shorter than this are rejected as degenerate.
pub const AXIS_MIN_NORM: f64 = 1e-9;
/// `sin(angle)` threshold below which the rotation log uses its small-angle form.
const ROT_LOG_SMALL: f64 = 1e-9;

/// Reasons a runtime model parameter is rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// A value is NaN or infinite.
    NonFinite,
    /// A joint axis or quaternion has (near) zero norm.
    DegenerateAxis,
    /// A transform is not a proper rigid motion `[R | t; 0 0 0 1]`.
    NotRigid,
    /// Payload mass is negative.
    InvalidPayload,
    /// A calibrated chain disagrees with the nominal model by more than the
    /// accepted deviation; usually a base/flange frame convention mismatch.
    ExcessiveDeviation,
}

/// Compact rigid transform used internally (27 + 9 multiplies to compose).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rigid {
    pub(crate) r: [[f64; 3]; 3],
    pub(crate) p: [f64; 3],
}

impl Rigid {
    pub(crate) const IDENTITY: Rigid = Rigid {
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        p: [0.0; 3],
    };

    pub(crate) fn from_pose(m: &Pose) -> Result<Rigid, ModelError> {
        let mut out = Rigid::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                out.r[i][j] = m[(i, j)];
            }
            out.p[i] = m[(i, 3)];
        }
        if !m.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFinite);
        }
        if m[(3, 0)] != 0.0 || m[(3, 1)] != 0.0 || m[(3, 2)] != 0.0 || m[(3, 3)] != 1.0 {
            return Err(ModelError::NotRigid);
        }
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += out.r[k][i] * out.r[k][j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                if (s - want).abs() > RIGID_ATOL {
                    return Err(ModelError::NotRigid);
                }
            }
        }
        if dot3(&out.r[0], &cross3(&out.r[1], &out.r[2])) < 0.0 {
            return Err(ModelError::NotRigid);
        }
        Ok(out)
    }

    pub(crate) fn to_pose(self) -> Pose {
        let mut m = Pose::identity();
        for i in 0..3 {
            for j in 0..3 {
                m[(i, j)] = self.r[i][j];
            }
            m[(i, 3)] = self.p[i];
        }
        m
    }

    #[inline]
    pub(crate) fn rotate(&self, v: &[f64; 3]) -> [f64; 3] {
        [
            dot3(&self.r[0], v),
            dot3(&self.r[1], v),
            dot3(&self.r[2], v),
        ]
    }

    pub(crate) fn mul(&self, b: &Rigid) -> Rigid {
        let mut out = Rigid::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                out.r[i][j] =
                    self.r[i][0] * b.r[0][j] + self.r[i][1] * b.r[1][j] + self.r[i][2] * b.r[2][j];
            }
        }
        out.p = add3(&self.rotate(&b.p), &self.p);
        out
    }

    /// Rotation by `angle` about the unit `axis` (Rodrigues), zero translation.
    pub(crate) fn rotation(axis: &[f64; 3], angle: f64) -> Rigid {
        let (s, c) = (libm::sin(angle), libm::cos(angle));
        let t = 1.0 - c;
        let [x, y, z] = *axis;
        Rigid {
            r: [
                [c + t * x * x, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, c + t * y * y, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, c + t * z * z],
            ],
            p: [0.0; 3],
        }
    }

    /// Frobenius norm of the difference of the two homogeneous matrices.
    pub(crate) fn frob_diff(&self, o: &Rigid) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let d = self.r[i][j] - o.r[i][j];
                s += d * d;
            }
            let d = self.p[i] - o.p[i];
            s += d * d;
        }
        libm::sqrt(s)
    }
}

/// Rotation vector of `a · bᵀ` (the base-frame rotation taking `b` to `a`).
pub(crate) fn rotation_error(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut e = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            e[i][j] = a[i][0] * b[j][0] + a[i][1] * b[j][1] + a[i][2] * b[j][2];
        }
    }
    let w = [
        0.5 * (e[2][1] - e[1][2]),
        0.5 * (e[0][2] - e[2][0]),
        0.5 * (e[1][0] - e[0][1]),
    ];
    let sin_t = libm::sqrt(dot3(&w, &w));
    if sin_t < ROT_LOG_SMALL {
        return w;
    }
    let cos_t = 0.5 * (e[0][0] + e[1][1] + e[2][2] - 1.0);
    let k = libm::atan2(sin_t, cos_t) / sin_t;
    [w[0] * k, w[1] * k, w[2] * k]
}

/// Pose error twist `[Δp; Δθ]` taking `current` to `target`, in the base frame.
pub(crate) fn pose_error(target: &Rigid, current: &Rigid) -> [f64; 6] {
    let w = rotation_error(&target.r, &current.r);
    [
        target.p[0] - current.p[0],
        target.p[1] - current.p[1],
        target.p[2] - current.p[2],
        w[0],
        w[1],
        w[2],
    ]
}

/// Build a pose from `[x, y, z, qx, qy, qz, qw]` (the quaternion is normalised).
pub fn pose_from_xyz_quat(p: &[f64; 7]) -> Result<Pose, ModelError> {
    if !p.iter().all(|v| v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    let n = libm::sqrt(p[3] * p[3] + p[4] * p[4] + p[5] * p[5] + p[6] * p[6]);
    if n < AXIS_MIN_NORM {
        return Err(ModelError::DegenerateAxis);
    }
    let (x, y, z, w) = (p[3] / n, p[4] / n, p[5] / n, p[6] / n);
    let mut m = Pose::identity();
    m[(0, 0)] = 1.0 - 2.0 * (y * y + z * z);
    m[(0, 1)] = 2.0 * (x * y - z * w);
    m[(0, 2)] = 2.0 * (x * z + y * w);
    m[(1, 0)] = 2.0 * (x * y + z * w);
    m[(1, 1)] = 1.0 - 2.0 * (x * x + z * z);
    m[(1, 2)] = 2.0 * (y * z - x * w);
    m[(2, 0)] = 2.0 * (x * z - y * w);
    m[(2, 1)] = 2.0 * (y * z + x * w);
    m[(2, 2)] = 1.0 - 2.0 * (x * x + y * y);
    m[(0, 3)] = p[0];
    m[(1, 3)] = p[1];
    m[(2, 3)] = p[2];
    Ok(m)
}

/// Convert a pose to `[x, y, z, qx, qy, qz, qw]` with `qw ≥ 0`.
pub fn pose_to_xyz_quat(m: &Pose) -> [f64; 7] {
    let r = |i: usize, j: usize| m[(i, j)];
    let tr = r(0, 0) + r(1, 1) + r(2, 2);
    // Shepperd: pivot on the largest of (trace, r00, r11, r22) to stay well-conditioned.
    let (x, y, z, w);
    if tr > 0.0 {
        let s = 2.0 * libm::sqrt(tr + 1.0);
        w = 0.25 * s;
        x = (r(2, 1) - r(1, 2)) / s;
        y = (r(0, 2) - r(2, 0)) / s;
        z = (r(1, 0) - r(0, 1)) / s;
    } else if r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2) {
        let s = 2.0 * libm::sqrt(1.0 + r(0, 0) - r(1, 1) - r(2, 2));
        w = (r(2, 1) - r(1, 2)) / s;
        x = 0.25 * s;
        y = (r(0, 1) + r(1, 0)) / s;
        z = (r(0, 2) + r(2, 0)) / s;
    } else if r(1, 1) > r(2, 2) {
        let s = 2.0 * libm::sqrt(1.0 + r(1, 1) - r(0, 0) - r(2, 2));
        w = (r(0, 2) - r(2, 0)) / s;
        x = (r(0, 1) + r(1, 0)) / s;
        y = 0.25 * s;
        z = (r(1, 2) + r(2, 1)) / s;
    } else {
        let s = 2.0 * libm::sqrt(1.0 + r(2, 2) - r(0, 0) - r(1, 1));
        w = (r(1, 0) - r(0, 1)) / s;
        x = (r(0, 2) + r(2, 0)) / s;
        y = (r(1, 2) + r(2, 1)) / s;
        z = 0.25 * s;
    }
    let sign = if w < 0.0 { -1.0 } else { 1.0 };
    [
        m[(0, 3)],
        m[(1, 3)],
        m[(2, 3)],
        sign * x,
        sign * y,
        sign * z,
        sign * w,
    ]
}

/// One row of a Denavit–Hartenberg table (metres / radians).
///
/// The same struct carries a *delta* row (e.g. UR `calibration.conf`:
/// `delta_a`, `delta_alpha`, `delta_d`, `delta_theta`), see [`apply_dh_deltas`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DhRow {
    pub a: f64,
    pub alpha: f64,
    pub d: f64,
    /// Joint zero offset: the joint angle is `q + theta`.
    pub theta: f64,
}

impl DhRow {
    pub const ZERO: DhRow = DhRow::new(0.0, 0.0, 0.0, 0.0);

    pub const fn new(a: f64, alpha: f64, d: f64, theta: f64) -> DhRow {
        DhRow { a, alpha, d, theta }
    }
}

/// DH table convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DhConvention {
    /// Distal: `A_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)` (UR, Kinova).
    Standard,
    /// Proximal / Craig: `A_i = Rx(α_{i-1}) · Tx(a_{i-1}) · Rz(θ_i) · Tz(d_i)`
    /// (Franka); row `i` stores `a_{i-1}`, `α_{i-1}`, `d_i`, `θ_i`.
    Modified,
}

/// Add per-robot calibration deltas to a nominal DH table.
pub fn apply_dh_deltas<const DOF: usize>(
    nominal: &[DhRow; DOF],
    delta: &[DhRow; DOF],
) -> [DhRow; DOF] {
    let mut out = *nominal;
    for i in 0..DOF {
        out[i].a += delta[i].a;
        out[i].alpha += delta[i].alpha;
        out[i].d += delta[i].d;
        out[i].theta += delta[i].theta;
    }
    out
}

/// A revolute joint of a [`KinematicChain`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainJoint {
    /// Fixed transform from the previous joint's moving frame (or the base) to
    /// this joint's frame.
    pub origin: Pose,
    /// Rotation axis in this joint's frame (normalised on construction).
    pub axis: [f64; 3],
    /// Joint zero offset (rad): the joint rotates by `q + offset`.
    pub offset: f64,
}

/// Per-robot serial chain of `DOF` revolute joints. See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicChain<const DOF: usize> {
    pub(crate) origin: [Rigid; DOF],
    pub(crate) axis: [[f64; 3]; DOF],
    pub(crate) offset: [f64; DOF],
    pub(crate) flange: Rigid,
}

impl<const DOF: usize> KinematicChain<DOF> {
    /// Build a chain from explicit joints and the last-joint → flange transform.
    pub fn new(joints: &[ChainJoint; DOF], flange: &Pose) -> Result<Self, ModelError> {
        let mut chain = KinematicChain {
            origin: [Rigid::IDENTITY; DOF],
            axis: [[0.0, 0.0, 1.0]; DOF],
            offset: [0.0; DOF],
            flange: Rigid::from_pose(flange)?,
        };
        for i in 0..DOF {
            let j = &joints[i];
            if !j.offset.is_finite() || !j.axis.iter().all(|v| v.is_finite()) {
                return Err(ModelError::NonFinite);
            }
            let n = libm::sqrt(dot3(&j.axis, &j.axis));
            if n < AXIS_MIN_NORM {
                return Err(ModelError::DegenerateAxis);
            }
            chain.origin[i] = Rigid::from_pose(&j.origin)?;
            chain.axis[i] = [j.axis[0] / n, j.axis[1] / n, j.axis[2] / n];
            chain.offset[i] = j.offset;
        }
        Ok(chain)
    }

    /// Build a chain from a DH table.
    ///
    /// The chain's base and flange frames are the DH ones. If the arm driver
    /// uses different frames (URDF `base_link` / `tool0`, …) adapt them with
    /// [`with_base`](Self::with_base) and [`with_flange`](Self::with_flange).
    pub fn from_dh(convention: DhConvention, rows: &[DhRow; DOF]) -> Result<Self, ModelError> {
        let mut chain = KinematicChain {
            origin: [Rigid::IDENTITY; DOF],
            axis: [[0.0, 0.0, 1.0]; DOF],
            offset: [0.0; DOF],
            flange: Rigid::IDENTITY,
        };
        // Rx(alpha) with translation (a, 0, d).
        let link = |a: f64, alpha: f64, d: f64| {
            let mut t = Rigid::rotation(&[1.0, 0.0, 0.0], alpha);
            t.p = [a, 0.0, d];
            t
        };
        for i in 0..DOF {
            let row = &rows[i];
            if !(row.a.is_finite()
                && row.alpha.is_finite()
                && row.d.is_finite()
                && row.theta.is_finite())
            {
                return Err(ModelError::NonFinite);
            }
            chain.offset[i] = row.theta;
            match convention {
                DhConvention::Standard => {
                    let fixed = link(row.a, row.alpha, row.d);
                    if i + 1 < DOF {
                        chain.origin[i + 1] = fixed;
                    } else {
                        chain.flange = fixed;
                    }
                }
                DhConvention::Modified => {
                    let d_prev = if i == 0 { 0.0 } else { rows[i - 1].d };
                    chain.origin[i] = link(row.a, row.alpha, d_prev);
                    if i + 1 == DOF {
                        chain.flange.p = [0.0, 0.0, row.d];
                    }
                }
            }
        }
        Ok(chain)
    }

    /// Re-express the chain in a different base frame: `base · chain`.
    pub fn with_base(mut self, base: &Pose) -> Result<Self, ModelError> {
        let base = Rigid::from_pose(base)?;
        if DOF == 0 {
            self.flange = base.mul(&self.flange);
        } else {
            self.origin[0] = base.mul(&self.origin[0]);
        }
        Ok(self)
    }

    /// Append a fixed transform after the flange: `chain · flange`.
    pub fn with_flange(mut self, flange: &Pose) -> Result<Self, ModelError> {
        self.flange = self.flange.mul(&Rigid::from_pose(flange)?);
        Ok(self)
    }

    /// Joint `i` as stored.
    pub fn joint(&self, i: usize) -> ChainJoint {
        ChainJoint {
            origin: self.origin[i].to_pose(),
            axis: self.axis[i],
            offset: self.offset[i],
        }
    }

    /// Last-joint → flange transform.
    pub fn flange(&self) -> Pose {
        self.flange.to_pose()
    }

    pub(crate) fn fk_jacobian_rigid(&self, q: &[f64; DOF]) -> (Rigid, [[f64; DOF]; 6]) {
        let mut t = Rigid::IDENTITY;
        let mut axes = [[0.0f64; 3]; DOF];
        let mut points = [[0.0f64; 3]; DOF];
        for i in 0..DOF {
            t = t.mul(&self.origin[i]);
            axes[i] = t.rotate(&self.axis[i]);
            points[i] = t.p;
            t = t.mul(&Rigid::rotation(&self.axis[i], q[i] + self.offset[i]));
        }
        t = t.mul(&self.flange);
        let mut jac = [[0.0f64; DOF]; 6];
        for i in 0..DOF {
            let arm = [
                t.p[0] - points[i][0],
                t.p[1] - points[i][1],
                t.p[2] - points[i][2],
            ];
            let lin = cross3(&axes[i], &arm);
            for r in 0..3 {
                jac[r][i] = lin[r];
                jac[r + 3][i] = axes[i][r];
            }
        }
        (t, jac)
    }

    /// Forward kinematics base → flange.
    pub fn fk(&self, q: &[f64; DOF]) -> Pose {
        let mut t = Rigid::IDENTITY;
        for i in 0..DOF {
            t = t
                .mul(&self.origin[i])
                .mul(&Rigid::rotation(&self.axis[i], q[i] + self.offset[i]));
        }
        t.mul(&self.flange).to_pose()
    }

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) of the flange origin, base frame.
    pub fn jacobian(&self, q: &[f64; DOF]) -> [[f64; DOF]; 6] {
        self.fk_jacobian_rigid(q).1
    }

    /// FK and Jacobian from a single pass over the chain.
    pub fn fk_jacobian(&self, q: &[f64; DOF]) -> (Pose, [[f64; DOF]; 6]) {
        let (t, j) = self.fk_jacobian_rigid(q);
        (t.to_pose(), j)
    }

    /// Polish `q` so that `fk(q)` matches `target`, starting from a nearby
    /// solution (typically the nominal analytic IK).
    ///
    /// Damped minimum-norm Newton steps, at most [`CHAIN_IK_MAX_ITERS`]. Returns
    /// the final Frobenius residual `‖fk(q) − target‖`; callers accept the
    /// result against their `fk_atol`. `q` is left untouched if the target is
    /// not a rigid transform (residual `f64::INFINITY`).
    pub fn refine_ik(&self, target: &Pose, q: &mut [f64; DOF]) -> f64 {
        let target = match Rigid::from_pose(target) {
            Ok(t) => t,
            Err(_) => return f64::INFINITY,
        };
        let mut cur = *q;
        let mut best = *q;
        let mut best_res = f64::INFINITY;
        for iter in 0..=CHAIN_IK_MAX_ITERS {
            let (tip, jac) = self.fk_jacobian_rigid(&cur);
            let res = tip.frob_diff(&target);
            if !res.is_finite() {
                break;
            }
            if res < best_res {
                best_res = res;
                best = cur;
            }
            let e = pose_error(&target, &tip);
            let mut e_norm = 0.0;
            for v in e {
                e_norm += v * v;
            }
            if iter == CHAIN_IK_MAX_ITERS || libm::sqrt(e_norm) < CHAIN_IK_CONVERGENCE {
                break;
            }
            let mut a = SMatrix::<f64, 6, 6>::zeros();
            for i in 0..6 {
                for k in 0..6 {
                    let mut s = 0.0;
                    for c in 0..DOF {
                        s += jac[i][c] * jac[k][c];
                    }
                    a[(i, k)] = s;
                }
                a[(i, i)] += CHAIN_IK_DAMPING;
            }
            let chol = match a.cholesky() {
                Some(c) => c,
                None => break,
            };
            let y = chol.solve(&SMatrix::<f64, 6, 1>::from_column_slice(&e));
            let mut dq = [0.0f64; DOF];
            let mut step = 0.0f64;
            for c in 0..DOF {
                for i in 0..6 {
                    dq[c] += jac[i][c] * y[i];
                }
                step = step.max(dq[c].abs());
            }
            let scale = if step > CHAIN_IK_MAX_STEP {
                CHAIN_IK_MAX_STEP / step
            } else {
                1.0
            };
            for c in 0..DOF {
                cur[c] += scale * dq[c];
            }
        }
        if best_res.is_finite() {
            *q = best;
        }
        best_res
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};

    /// Nominal UR5 DH table (standard convention), as published by Universal Robots.
    pub(crate) const UR5_DH: [DhRow; 6] = [
        DhRow::new(0.0, FRAC_PI_2, 0.089159, 0.0),
        DhRow::new(-0.425, 0.0, 0.0, 0.0),
        DhRow::new(-0.39225, 0.0, 0.0, 0.0),
        DhRow::new(0.0, FRAC_PI_2, 0.10915, 0.0),
        DhRow::new(0.0, -FRAC_PI_2, 0.09465, 0.0),
        DhRow::new(0.0, 0.0, 0.0823, 0.0),
    ];

    /// Deltas of the magnitude found in a UR `calibration.conf`.
    pub(crate) const UR5_DELTA: [DhRow; 6] = [
        DhRow::new(1.2e-4, -3.0e-4, 2.1e-4, 4.0e-4),
        DhRow::new(-2.5e-4, 5.0e-4, 0.0, -6.0e-4),
        DhRow::new(3.1e-4, -2.0e-4, 0.0, 3.0e-4),
        DhRow::new(-0.7e-4, 4.0e-4, -1.5e-4, -2.0e-4),
        DhRow::new(0.4e-4, -1.0e-4, 0.9e-4, 5.0e-4),
        DhRow::new(0.0, 0.0, -1.1e-4, -3.0e-4),
    ];

    pub(crate) fn ur5_nominal() -> KinematicChain<6> {
        KinematicChain::from_dh(DhConvention::Standard, &UR5_DH).unwrap()
    }

    pub(crate) fn ur5_calibrated() -> KinematicChain<6> {
        KinematicChain::from_dh(
            DhConvention::Standard,
            &apply_dh_deltas(&UR5_DH, &UR5_DELTA),
        )
        .unwrap()
    }

    /// Deterministic pseudo-random joint vector in `[-amp, amp]`.
    pub(crate) fn sample_q<const DOF: usize>(seed: u64, amp: f64) -> [f64; DOF] {
        let mut s = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0xD1B5_4A32_D192_ED03);
        let mut q = [0.0; DOF];
        for v in q.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0) * amp;
        }
        q
    }

    fn frob(a: &Pose, b: &Pose) -> f64 {
        (a - b).norm()
    }

    #[test]
    fn standard_dh_matches_ur5_home_pose() {
        let t = ur5_nominal().fk(&[0.0; 6]);
        let want = [-0.81725, -0.19145, -0.005491];
        for i in 0..3 {
            assert!((t[(i, 3)] - want[i]).abs() < 1e-12, "{t}");
        }
    }

    #[test]
    fn modified_dh_matches_explicit_product() {
        let rows = [
            DhRow::new(0.0, 0.0, 0.333, 0.1),
            DhRow::new(0.05, -FRAC_PI_2, 0.0, -0.2),
            DhRow::new(-0.08, FRAC_PI_2, 0.316, 0.3),
        ];
        let q = [0.4, -0.7, 1.1];
        let chain = KinematicChain::from_dh(DhConvention::Modified, &rows).unwrap();
        let mut want = Rigid::IDENTITY;
        for i in 0..3 {
            let mut tx = Rigid::IDENTITY;
            tx.p = [rows[i].a, 0.0, 0.0];
            let mut tz = Rigid::IDENTITY;
            tz.p = [0.0, 0.0, rows[i].d];
            want = want
                .mul(&Rigid::rotation(&[1.0, 0.0, 0.0], rows[i].alpha))
                .mul(&tx)
                .mul(&Rigid::rotation(&[0.0, 0.0, 1.0], q[i] + rows[i].theta))
                .mul(&tz);
        }
        assert!(frob(&chain.fk(&q), &want.to_pose()) < 1e-12);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let chain = ur5_calibrated();
        let q = sample_q::<6>(3, 2.0);
        let (pose, jac) = chain.fk_jacobian(&q);
        let base = Rigid::from_pose(&pose).unwrap();
        let h = 1e-6;
        for c in 0..6 {
            let mut qp = q;
            qp[c] += h;
            let moved = Rigid::from_pose(&chain.fk(&qp)).unwrap();
            let twist = pose_error(&moved, &base);
            for r in 0..6 {
                assert!((twist[r] / h - jac[r][c]).abs() < 1e-5, "row {r} col {c}");
            }
        }
    }

    #[test]
    fn refine_ik_recovers_calibrated_solution_from_nominal() {
        let nominal = ur5_nominal();
        let calibrated = ur5_calibrated();
        for seed in 0..20 {
            let q_true = sample_q::<6>(seed, 2.5);
            let target = calibrated.fk(&q_true);
            // Calibration moves the flange by far more than fk_atol …
            assert!(frob(&nominal.fk(&q_true), &target) > 1e-4);
            // … and a polish from a nominal-quality seed closes the gap.
            let mut q = q_true;
            for (i, v) in q.iter_mut().enumerate() {
                *v += 2e-3 * (i as f64 - 2.5);
            }
            let res = calibrated.refine_ik(&target, &mut q);
            assert!(res < 1e-9, "seed {seed}: residual {res}");
        }
    }

    #[test]
    fn refine_ik_reports_unreachable_target() {
        let chain = ur5_nominal();
        let mut target = chain.fk(&[0.3; 6]);
        target[(0, 3)] += 5.0;
        let mut q = [0.3; 6];
        assert!(chain.refine_ik(&target, &mut q) > 1.0);
        assert!(q.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_non_rigid_and_degenerate_input() {
        let mut skew = Pose::identity();
        skew[(0, 1)] = 0.1;
        let joint = ChainJoint {
            origin: Pose::identity(),
            axis: [0.0, 0.0, 1.0],
            offset: 0.0,
        };
        assert_eq!(
            KinematicChain::new(&[joint], &skew),
            Err(ModelError::NotRigid)
        );
        let zero_axis = ChainJoint {
            axis: [0.0; 3],
            ..joint
        };
        assert_eq!(
            KinematicChain::new(&[zero_axis], &Pose::identity()),
            Err(ModelError::DegenerateAxis)
        );
        let mut mirror = Pose::identity();
        mirror[(2, 2)] = -1.0;
        assert_eq!(ur5_nominal().with_base(&mirror), Err(ModelError::NotRigid));
    }

    #[test]
    fn quaternion_round_trip_covers_all_pivots() {
        for (axis, angle) in [
            ([0.0, 0.0, 1.0], 0.3),
            ([1.0, 0.0, 0.0], PI - 1e-3),
            ([0.0, 1.0, 0.0], PI - 1e-3),
            ([0.0, 0.0, 1.0], PI - 1e-3),
            ([0.6, 0.0, 0.8], 2.0),
        ] {
            let mut t = Rigid::rotation(&axis, angle);
            t.p = [0.1, -0.2, 0.3];
            let pose = t.to_pose();
            let back = pose_from_xyz_quat(&pose_to_xyz_quat(&pose)).unwrap();
            assert!(frob(&pose, &back) < 1e-12, "axis {axis:?}");
        }
        assert_eq!(
            pose_from_xyz_quat(&[0.0; 7]),
            Err(ModelError::DegenerateAxis)
        );
    }
}
