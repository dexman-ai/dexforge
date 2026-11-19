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

use crate::math::kinematics::{compose, frob_diff, Transform, IDENTITY};
use crate::math::model::Pose;
use crate::math::ssik_core::{cross3, dot3, mat3_mul, mat3_transpose, mat3_vec, rotation_matrix};

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
    /// A value that must be strictly positive is not.
    NonPositive,
    /// A calibrated chain disagrees with the nominal model by more than the
    /// accepted deviation; usually a base/flange frame convention mismatch.
    ExcessiveDeviation,
    /// A learned torque observer (`.dexnext`) is malformed, was trained for
    /// another arm, or does not fit the arm's model or mounting.
    InvalidObserver,
}

/// Validate a homogeneous pose and convert it to a [`Transform`].
pub fn transform_from_pose(m: &Pose) -> Result<Transform, ModelError> {
    if !m.iter().all(|v| v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    if m[(3, 0)] != 0.0 || m[(3, 1)] != 0.0 || m[(3, 2)] != 0.0 || m[(3, 3)] != 1.0 {
        return Err(ModelError::NotRigid);
    }
    let out = transform_of(m);
    let gram = mat3_mul(&mat3_transpose(&out.r), &out.r);
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            if (gram[i][j] - want).abs() > RIGID_ATOL {
                return Err(ModelError::NotRigid);
            }
        }
    }
    if dot3(&out.r[0], &cross3(&out.r[1], &out.r[2])) < 0.0 {
        return Err(ModelError::NotRigid);
    }
    Ok(out)
}

/// Upper 3×4 block of a homogeneous pose, without validation.
pub(crate) fn transform_of(m: &Pose) -> Transform {
    let mut out = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            out.r[i][j] = m[(i, j)];
        }
        out.t[i] = m[(i, 3)];
    }
    out
}

/// Homogeneous 4×4 form of a [`Transform`].
pub fn transform_to_pose(t: &Transform) -> Pose {
    let mut m = Pose::identity();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = t.r[i][j];
        }
        m[(i, 3)] = t.t[i];
    }
    m
}

/// Rotation by `angle` about the unit `axis`, zero translation.
#[inline]
pub(crate) fn rotation_about(axis: &[f64; 3], angle: f64) -> Transform {
    Transform {
        r: rotation_matrix(axis, angle),
        t: [0.0; 3],
    }
}

/// Exponential map `exp([w]×)` of a 3-D rotation vector `w` (axis × angle, rad).
pub fn rotation_exp(w: &[f64; 3]) -> [[f64; 3]; 3] {
    let angle = libm::sqrt(dot3(w, w));
    if angle < ROT_LOG_SMALL {
        let (wx, wy, wz) = (w[0], w[1], w[2]);
        let (xx, yy, zz) = (wx * wx, wy * wy, wz * wz);
        let (xy, xz, yz) = (wx * wy, wx * wz, wy * wz);
        return [
            [1.0 - 0.5 * (yy + zz), -wz + 0.5 * xy, wy + 0.5 * xz],
            [wz + 0.5 * xy, 1.0 - 0.5 * (xx + zz), -wx + 0.5 * yz],
            [-wy + 0.5 * xz, wx + 0.5 * yz, 1.0 - 0.5 * (xx + yy)],
        ];
    }
    let inv = 1.0 / angle;
    rotation_matrix(&[w[0] * inv, w[1] * inv, w[2] * inv], angle)
}

/// Rotation vector `log(a · bᵀ)` (the base-frame rotation taking `b` to `a`).
pub fn rotation_error(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [f64; 3] {
    let e = mat3_mul(a, &mat3_transpose(b));
    let w = [
        0.5 * (e[2][1] - e[1][2]),
        0.5 * (e[0][2] - e[2][0]),
        0.5 * (e[1][0] - e[0][1]),
    ];
    let sin_t = libm::sqrt(dot3(&w, &w));
    let cos_t = (0.5 * (e[0][0] + e[1][1] + e[2][2] - 1.0)).clamp(-1.0, 1.0);
    if cos_t < 0.0 {
        // Angle above π/2: `w / sin θ` loses precision towards π, so take the
        // axis from the symmetric part (e + eᵀ)/2 = cos θ I + (1 − cos θ) u uᵀ
        // and its sign from `w`.
        let denom = 1.0 - cos_t;
        let d0 = ((e[0][0] - cos_t) / denom).max(0.0);
        let d1 = ((e[1][1] - cos_t) / denom).max(0.0);
        let d2 = ((e[2][2] - cos_t) / denom).max(0.0);
        let mut u = if d0 >= d1 && d0 >= d2 {
            let ux = libm::sqrt(d0).max(ROT_LOG_SMALL);
            let s = 2.0 * denom * ux;
            [ux, (e[0][1] + e[1][0]) / s, (e[0][2] + e[2][0]) / s]
        } else if d1 >= d2 {
            let uy = libm::sqrt(d1).max(ROT_LOG_SMALL);
            let s = 2.0 * denom * uy;
            [(e[0][1] + e[1][0]) / s, uy, (e[1][2] + e[2][1]) / s]
        } else {
            let uz = libm::sqrt(d2).max(ROT_LOG_SMALL);
            let s = 2.0 * denom * uz;
            [(e[0][2] + e[2][0]) / s, (e[1][2] + e[2][1]) / s, uz]
        };
        let un = libm::sqrt(dot3(&u, &u));
        if un > ROT_LOG_SMALL {
            u = [u[0] / un, u[1] / un, u[2] / un];
        }
        if dot3(&u, &w) < 0.0 {
            u = [-u[0], -u[1], -u[2]];
        }
        let theta = libm::atan2(sin_t, cos_t);
        return [u[0] * theta, u[1] * theta, u[2] * theta];
    }
    if sin_t < ROT_LOG_SMALL {
        return w;
    }
    let k = libm::atan2(sin_t, cos_t) / sin_t;
    [w[0] * k, w[1] * k, w[2] * k]
}

/// Pose error twist `[Δp; Δθ]` taking `current` to `target`, in the base frame.
pub fn pose_error(target: &Transform, current: &Transform) -> [f64; 6] {
    let w = rotation_error(&target.r, &current.r);
    [
        target.t[0] - current.t[0],
        target.t[1] - current.t[1],
        target.t[2] - current.t[2],
        w[0],
        w[1],
        w[2],
    ]
}

/// Apply a base-axes twist offset `[Δp; Δθ]` to `pose`: translate by `Δp`,
/// and left-multiply rotation by `exp([Δθ]×)`.
pub fn se3_offset(pose: &Transform, offset: &[f64; 6]) -> Transform {
    let dr = rotation_exp(&[offset[3], offset[4], offset[5]]);
    Transform {
        r: mat3_mul(&dr, &pose.r),
        t: [
            pose.t[0] + offset[0],
            pose.t[1] + offset[1],
            pose.t[2] + offset[2],
        ],
    }
}

/// Straight-line translation and shortest-arc geodesic rotation between `a` and `b`.
pub fn se3_lerp(a: &Transform, b: &Transform, u: f64) -> Transform {
    let w = rotation_error(&b.r, &a.r);
    let dr = rotation_exp(&[w[0] * u, w[1] * u, w[2] * u]);
    Transform {
        r: mat3_mul(&dr, &a.r),
        t: [
            a.t[0] + u * (b.t[0] - a.t[0]),
            a.t[1] + u * (b.t[1] - a.t[1]),
            a.t[2] + u * (b.t[2] - a.t[2]),
        ],
    }
}

/// Euclidean translation distance (m) and geodesic rotation angle (rad) between `a` and `b`.
pub fn se3_distance(a: &Transform, b: &Transform) -> (f64, f64) {
    let dp = [b.t[0] - a.t[0], b.t[1] - a.t[1], b.t[2] - a.t[2]];
    let w = rotation_error(&b.r, &a.r);
    (libm::sqrt(dot3(&dp, &dp)), libm::sqrt(dot3(&w, &w)))
}

/// Transform a 3-D point by `t`: `R · p + t`.
#[inline]
pub fn transform_point(t: &Transform, p: &[f64; 3]) -> [f64; 3] {
    let rp = mat3_vec(&t.r, p);
    [rp[0] + t.t[0], rp[1] + t.t[1], rp[2] + t.t[2]]
}

/// Inverse-transform a 3-D point by `t`: `Rᵀ · (p − t)`.
#[inline]
pub fn inverse_transform_point(t: &Transform, p: &[f64; 3]) -> [f64; 3] {
    let d = [p[0] - t.t[0], p[1] - t.t[1], p[2] - t.t[2]];
    mat3_mul_transpose_vec(&t.r, &d)
}

#[inline]
fn mat3_mul_transpose_vec(r: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[1][0] * v[1] + r[2][0] * v[2],
        r[0][1] * v[0] + r[1][1] * v[1] + r[2][1] * v[2],
        r[0][2] * v[0] + r[1][2] * v[1] + r[2][2] * v[2],
    ]
}

/// Build a [`Transform`] directly from wire pose `[x, y, z, qx, qy, qz, qw]`.
pub fn transform_from_xyz_quat(p: &[f64; 7]) -> Result<Transform, ModelError> {
    pose_from_xyz_quat(p).map(|m| transform_of(&m))
}

/// Convert a [`Transform`] to wire pose `[x, y, z, qx, qy, qz, qw]` (`qw ≥ 0`).
pub fn transform_to_xyz_quat(t: &Transform) -> [f64; 7] {
    pose_to_xyz_quat(&transform_to_pose(t))
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
    pub(crate) origin: [Transform; DOF],
    pub(crate) axis: [[f64; 3]; DOF],
    pub(crate) offset: [f64; DOF],
    pub(crate) flange: Transform,
}

impl<const DOF: usize> KinematicChain<DOF> {
    /// Build a chain from explicit joints and the last-joint → flange transform.
    pub fn new(joints: &[ChainJoint; DOF], flange: &Pose) -> Result<Self, ModelError> {
        let mut chain = KinematicChain {
            origin: [IDENTITY; DOF],
            axis: [[0.0, 0.0, 1.0]; DOF],
            offset: [0.0; DOF],
            flange: transform_from_pose(flange)?,
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
            chain.origin[i] = transform_from_pose(&j.origin)?;
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
            origin: [IDENTITY; DOF],
            axis: [[0.0, 0.0, 1.0]; DOF],
            offset: [0.0; DOF],
            flange: IDENTITY,
        };
        // Rx(alpha) with translation (a, 0, d).
        let link = |a: f64, alpha: f64, d: f64| {
            let mut t = rotation_about(&[1.0, 0.0, 0.0], alpha);
            t.t = [a, 0.0, d];
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
                        chain.flange.t = [0.0, 0.0, row.d];
                    }
                }
            }
        }
        Ok(chain)
    }

    /// Re-express the chain in a different base frame: `base · chain`.
    pub fn with_base(mut self, base: &Pose) -> Result<Self, ModelError> {
        let base = transform_from_pose(base)?;
        if DOF == 0 {
            self.flange = compose(&base, &self.flange);
        } else {
            self.origin[0] = compose(&base, &self.origin[0]);
        }
        Ok(self)
    }

    /// Append a fixed transform after the flange: `chain · flange`.
    pub fn with_flange(mut self, flange: &Pose) -> Result<Self, ModelError> {
        self.flange = compose(&self.flange, &transform_from_pose(flange)?);
        Ok(self)
    }

    /// Joint `i` as stored.
    pub fn joint(&self, i: usize) -> ChainJoint {
        ChainJoint {
            origin: transform_to_pose(&self.origin[i]),
            axis: self.axis[i],
            offset: self.offset[i],
        }
    }

    /// Last-joint → flange transform.
    pub fn flange(&self) -> Pose {
        transform_to_pose(&self.flange)
    }

    /// Joint `i`'s moving rotation at joint value `q`.
    #[inline]
    pub(crate) fn joint_rotation(&self, i: usize, q: f64) -> Transform {
        rotation_about(&self.axis[i], q + self.offset[i])
    }

    pub(crate) fn fk_jacobian_rigid(&self, q: &[f64; DOF]) -> (Transform, [[f64; DOF]; 6]) {
        let mut t = IDENTITY;
        let mut axes = [[0.0f64; 3]; DOF];
        let mut points = [[0.0f64; 3]; DOF];
        for i in 0..DOF {
            t = compose(&t, &self.origin[i]);
            axes[i] = mat3_vec(&t.r, &self.axis[i]);
            points[i] = t.t;
            t = compose(&t, &self.joint_rotation(i, q[i]));
        }
        t = compose(&t, &self.flange);
        let mut jac = [[0.0f64; DOF]; 6];
        for i in 0..DOF {
            let arm = [
                t.t[0] - points[i][0],
                t.t[1] - points[i][1],
                t.t[2] - points[i][2],
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
        let mut t = IDENTITY;
        for i in 0..DOF {
            t = compose(&compose(&t, &self.origin[i]), &self.joint_rotation(i, q[i]));
        }
        transform_to_pose(&compose(&t, &self.flange))
    }

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) of the flange origin, base frame.
    pub fn jacobian(&self, q: &[f64; DOF]) -> [[f64; DOF]; 6] {
        self.fk_jacobian_rigid(q).1
    }

    /// FK and Jacobian from a single pass over the chain.
    pub fn fk_jacobian(&self, q: &[f64; DOF]) -> (Pose, [[f64; DOF]; 6]) {
        let (t, j) = self.fk_jacobian_rigid(q);
        (transform_to_pose(&t), j)
    }

    /// Polish `q` so that `fk(q)` matches `target`, starting from a nearby
    /// solution (typically the nominal analytic IK).
    ///
    /// Damped minimum-norm Newton steps, at most [`CHAIN_IK_MAX_ITERS`]. Returns
    /// the final Frobenius residual `‖fk(q) − target‖`; callers accept the
    /// result against their `fk_atol`. `q` is left untouched if the target is
    /// not a rigid transform (residual `f64::INFINITY`).
    ///
    /// Unlike the finite-difference refiners in [`crate::math::kinematics`]
    /// (`lm_refine`, `dls_ik`), which only need an FK closure, this uses the
    /// chain's analytic Jacobian: one chain evaluation per iteration.
    pub fn refine_ik(&self, target: &Pose, q: &mut [f64; DOF]) -> f64 {
        let target = match transform_from_pose(target) {
            Ok(t) => t,
            Err(_) => return f64::INFINITY,
        };
        let mut cur = *q;
        let mut best = *q;
        let mut best_res = f64::INFINITY;
        for iter in 0..=CHAIN_IK_MAX_ITERS {
            let (tip, jac) = self.fk_jacobian_rigid(&cur);
            let res = frob_diff(&tip, &target);
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
mod tests {
    use super::*;
    use crate::math::testing::{calibrated_chain, nominal_chain, sample_q, SIX_R_DH};
    use core::f64::consts::{FRAC_PI_2, PI};

    fn frob(a: &Pose, b: &Pose) -> f64 {
        (a - b).norm()
    }

    #[test]
    fn standard_dh_matches_the_home_pose() {
        let t = nominal_chain().fk(&[0.0; 6]);
        // (a2 + a3, −(d4 + d6), d1 − d5) for a shoulder / three-parallel / wrist arm.
        let want = [-0.75, -0.19, 0.02];
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
        let mut want = IDENTITY;
        for i in 0..3 {
            let mut tx = IDENTITY;
            tx.t = [rows[i].a, 0.0, 0.0];
            let mut tz = IDENTITY;
            tz.t = [0.0, 0.0, rows[i].d];
            for step in [
                rotation_about(&[1.0, 0.0, 0.0], rows[i].alpha),
                tx,
                rotation_about(&[0.0, 0.0, 1.0], q[i] + rows[i].theta),
                tz,
            ] {
                want = compose(&want, &step);
            }
        }
        assert!(frob(&chain.fk(&q), &transform_to_pose(&want)) < 1e-12);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let chain = calibrated_chain();
        let q = sample_q::<6>(3, 2.0);
        let (pose, jac) = chain.fk_jacobian(&q);
        let base = transform_from_pose(&pose).unwrap();
        let h = 1e-6;
        for c in 0..6 {
            let mut qp = q;
            qp[c] += h;
            let moved = transform_from_pose(&chain.fk(&qp)).unwrap();
            let twist = pose_error(&moved, &base);
            for r in 0..6 {
                assert!((twist[r] / h - jac[r][c]).abs() < 1e-5, "row {r} col {c}");
            }
        }
    }

    #[test]
    fn refine_ik_recovers_calibrated_solution_from_nominal() {
        let nominal = nominal_chain();
        let calibrated = calibrated_chain();
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
        let chain = nominal_chain();
        let mut target = chain.fk(&[0.3; 6]);
        target[(0, 3)] += 5.0;
        let mut q = [0.3; 6];
        assert!(chain.refine_ik(&target, &mut q) > 1.0);
        assert!(q.iter().all(|v| v.is_finite()));
    }

    /// Vendor calibration files express near-parallel axes with DH rows whose
    /// `d` terms are hundreds of metres and cancel (UR `calibration.conf`:
    /// joints 2–4). With the axes exactly parallel the shift is a pure change
    /// of gauge, so the chain must reproduce the nominal kinematics.
    #[test]
    fn large_cancelling_dh_offsets_keep_precision() {
        let mut delta = [DhRow::ZERO; 6];
        delta[1].d = 439.218_734;
        delta[2].d = -445.602_911;
        delta[3].d = 445.602_911 - 439.218_734;
        let shifted =
            KinematicChain::from_dh(DhConvention::Standard, &apply_dh_deltas(&SIX_R_DH, &delta))
                .unwrap();
        let nominal = nominal_chain();
        assert!(transform_to_pose(&shifted.origin[2])[(2, 3)].abs() > 400.0);
        for seed in 0..50 {
            let q = sample_q::<6>(seed, 3.0);
            let (pose, jac) = shifted.fk_jacobian(&q);
            let (want_pose, want_jac) = nominal.fk_jacobian(&q);
            assert!(frob(&pose, &want_pose) < 5e-12, "seed {seed}");
            assert!(transform_from_pose(&pose).is_ok());
            for r in 0..6 {
                for c in 0..6 {
                    assert!((jac[r][c] - want_jac[r][c]).abs() < 5e-12, "seed {seed}");
                }
            }
            // The polish converges as on the nominal chain.
            let mut guess = q;
            for (i, v) in guess.iter_mut().enumerate() {
                *v += 2e-3 * (i as f64 - 2.5);
            }
            assert!(
                shifted.refine_ik(&want_pose, &mut guess) < 1e-9,
                "seed {seed}"
            );
        }
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
        assert_eq!(
            nominal_chain().with_base(&mirror),
            Err(ModelError::NotRigid)
        );
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
            let mut t = rotation_about(&axis, angle);
            t.t = [0.1, -0.2, 0.3];
            let pose = transform_to_pose(&t);
            let back = pose_from_xyz_quat(&pose_to_xyz_quat(&pose)).unwrap();
            assert!(frob(&pose, &back) < 1e-12, "axis {axis:?}");
        }
        assert_eq!(
            pose_from_xyz_quat(&[0.0; 7]),
            Err(ModelError::DegenerateAxis)
        );
    }
}
