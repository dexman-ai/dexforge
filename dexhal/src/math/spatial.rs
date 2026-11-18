//! Arm-agnostic spatial math on a flange pose and geometric Jacobian.
//!
//! Everything a driver's own `fk` / `jacobian` output needs in order to be
//! re-referenced: to a tool-centre point, to a payload's centre of mass, to
//! the tool frame, or differentiated in time. Fixed-size loops only, no
//! allocation.
//!
//! Six-vectors are `[linear; angular]`: twists `[v; ω]`, wrenches `[f; m]`.

use nalgebra::{SMatrix, SVector};

use crate::math::chain::inverse_transform_point;
use crate::math::kinematics::Transform;
use crate::math::model::Pose;
use crate::math::ssik_core::{add3, cross3, mat3_mul, mat3_transpose, mat3_vec, mat3t_vec};

/// Rotation block of a homogeneous pose.
#[inline]
pub fn pose_rotation(pose: &Pose) -> [[f64; 3]; 3] {
    let mut r = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = pose[(i, j)];
        }
    }
    r
}

#[inline]
fn linear_column<const DOF: usize>(jacobian: &[[f64; DOF]; 6], j: usize) -> [f64; 3] {
    [jacobian[0][j], jacobian[1][j], jacobian[2][j]]
}

#[inline]
fn angular_column<const DOF: usize>(jacobian: &[[f64; DOF]; 6], j: usize) -> [f64; 3] {
    [jacobian[3][j], jacobian[4][j], jacobian[5][j]]
}

/// Translational Jacobian (one `[f64; 3]` per joint) of a point fixed in the
/// flange frame at `offset`: `Jv + Jw × (R_flange · offset)`.
pub fn flange_offset_jacobian<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    offset: &[f64; 3],
) -> [[f64; 3]; DOF] {
    let r = mat3_vec(&pose_rotation(flange_pose), offset);
    let mut jvp = [[0.0f64; 3]; DOF];
    for j in 0..DOF {
        jvp[j] = add3(
            &linear_column(jacobian, j),
            &cross3(&angular_column(jacobian, j), &r),
        );
    }
    jvp
}

/// Re-reference a flange geometric Jacobian to the TCP origin.
pub fn tcp_jacobian<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    tcp: &Pose,
) -> [[f64; DOF]; 6] {
    let p = [tcp[(0, 3)], tcp[(1, 3)], tcp[(2, 3)]];
    let lin = flange_offset_jacobian(flange_pose, jacobian, &p);
    let mut j = *jacobian;
    for c in 0..DOF {
        for r in 0..3 {
            j[r][c] = lin[c][r];
        }
    }
    j
}

/// Minimum-norm (Moore–Penrose) pseudo-inverse `J⁺ = Jᵀ (J Jᵀ)⁻¹` of a
/// `6 × DOF` geometric Jacobian, returned as a `DOF × 6` array.
///
/// Exact inverse for `DOF == 6`, least-squares / minimum-norm for `DOF > 6`.
/// `None` at a kinematic singularity.
pub fn pseudo_inverse<const DOF: usize>(j: &[[f64; DOF]; 6]) -> Option<[[f64; 6]; DOF]> {
    let mut a = SMatrix::<f64, 6, 6>::zeros();
    for i in 0..6 {
        for k in 0..6 {
            let mut s = 0.0;
            for c in 0..DOF {
                s += j[i][c] * j[k][c];
            }
            a[(i, k)] = s;
        }
    }
    let ainv = a.try_inverse()?;
    let mut out = [[0.0f64; 6]; DOF];
    for r in 0..DOF {
        for c in 0..6 {
            let mut s = 0.0;
            for i in 0..6 {
                s += j[i][r] * ainv[(i, c)];
            }
            out[r][c] = s;
        }
    }
    Some(out)
}

/// Bias acceleration `J̇(q, q̇) q̇` of the Jacobian's reference point, as
/// `[linear; angular]`.
///
/// Closed form for a serial chain of revolute joints ordered base → tip, the
/// only kind [`crate::math::model::ArmModel`] describes: with columns
/// `(v_i, w_i)`, `∂J_i/∂q_j` is `(w_j × v_i, w_j × w_i)` for `j ≤ i` and
/// `(w_i × v_j, 0)` for `j > i`. One pass, no finite differences.
pub fn jacobian_bias<const DOF: usize>(jacobian: &[[f64; DOF]; 6], qd: &[f64; DOF]) -> [f64; 6] {
    let mut v_total = [0.0f64; 3];
    for i in 0..DOF {
        for r in 0..3 {
            v_total[r] += jacobian[r][i] * qd[i];
        }
    }
    let mut omega = [0.0f64; 3];
    let mut v_cum = [0.0f64; 3];
    let mut bias = [0.0f64; 6];
    for i in 0..DOF {
        let v_i = linear_column(jacobian, i);
        let w_i = angular_column(jacobian, i);
        let spin = cross3(&omega, &w_i);
        for r in 0..3 {
            bias[3 + r] += spin[r] * qd[i];
            omega[r] += w_i[r] * qd[i];
            v_cum[r] += v_i[r] * qd[i];
        }
        let rest = [
            v_total[0] - v_cum[0],
            v_total[1] - v_cum[1],
            v_total[2] - v_cum[2],
        ];
        let lin = add3(&cross3(&omega, &v_i), &cross3(&w_i, &rest));
        for r in 0..3 {
            bias[r] += lin[r] * qd[i];
        }
    }
    bias
}

// ---------------------------------------------------------------------------
// Tool-frame quantities
// ---------------------------------------------------------------------------

/// Express a base-frame geometric Jacobian in the axes of `frame` (normally
/// the TCP pose): `blockdiag(Rᵀ, Rᵀ) · J`. The reference point is unchanged.
pub fn jacobian_in_frame<const DOF: usize>(
    frame: &Pose,
    jacobian: &[[f64; DOF]; 6],
) -> [[f64; DOF]; 6] {
    let r = pose_rotation(frame);
    let mut out = [[0.0f64; DOF]; 6];
    for c in 0..DOF {
        let lin = mat3t_vec(&r, &linear_column(jacobian, c));
        let ang = mat3t_vec(&r, &angular_column(jacobian, c));
        for k in 0..3 {
            out[k][c] = lin[k];
            out[3 + k][c] = ang[k];
        }
    }
    out
}

/// Re-express a wrench (or twist) acting at the tool origin from base axes to
/// the axes of `tool_pose`.
pub fn wrench_base_to_tool(tool_pose: &Pose, wrench: &[f64; 6]) -> [f64; 6] {
    let r = pose_rotation(tool_pose);
    let f = mat3t_vec(&r, &[wrench[0], wrench[1], wrench[2]]);
    let m = mat3t_vec(&r, &[wrench[3], wrench[4], wrench[5]]);
    [f[0], f[1], f[2], m[0], m[1], m[2]]
}

/// Inverse of [`wrench_base_to_tool`].
pub fn wrench_tool_to_base(tool_pose: &Pose, wrench: &[f64; 6]) -> [f64; 6] {
    let r = pose_rotation(tool_pose);
    let f = mat3_vec(&r, &[wrench[0], wrench[1], wrench[2]]);
    let m = mat3_vec(&r, &[wrench[3], wrench[4], wrench[5]]);
    [f[0], f[1], f[2], m[0], m[1], m[2]]
}

fn block_diagonal(lin: &[[f64; 3]; 3], ang: &[[f64; 3]; 3]) -> [[f64; 6]; 6] {
    let mut out = [[0.0f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = lin[i][j];
            out[3 + i][3 + j] = ang[i][j];
        }
    }
    out
}

fn diagonal3(d: [f64; 3]) -> [[f64; 3]; 3] {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}

/// A diagonal stiffness / damping / mass `[kx, ky, kz, krx, kry, krz]` given
/// in the axes of `tool_pose`, as the full 6 × 6 matrix in base axes:
/// `blockdiag(R K_t Rᵀ, R K_r Rᵀ)`.
pub fn diagonal_gain_tool_to_base(tool_pose: &Pose, gain: &[f64; 6]) -> [[f64; 6]; 6] {
    let r = pose_rotation(tool_pose);
    let rt = mat3_transpose(&r);
    let rotate = |d: [f64; 3]| mat3_mul(&mat3_mul(&r, &diagonal3(d)), &rt);
    block_diagonal(
        &rotate([gain[0], gain[1], gain[2]]),
        &rotate([gain[3], gain[4], gain[5]]),
    )
}

/// A diagonal gain given in base axes, as the full 6 × 6 matrix in the axes
/// of `tool_pose`: `blockdiag(Rᵀ K_t R, Rᵀ K_r R)`.
pub fn diagonal_gain_base_to_tool(tool_pose: &Pose, gain: &[f64; 6]) -> [[f64; 6]; 6] {
    let r = pose_rotation(tool_pose);
    let rt = mat3_transpose(&r);
    let rotate = |d: [f64; 3]| mat3_mul(&mat3_mul(&rt, &diagonal3(d)), &r);
    block_diagonal(
        &rotate([gain[0], gain[1], gain[2]]),
        &rotate([gain[3], gain[4], gain[5]]),
    )
}

// ---------------------------------------------------------------------------
// Spatial 6-vector, damped least-squares, and operational-space projections
// ---------------------------------------------------------------------------

/// Rotate both linear and angular 3-blocks of `v` by `r`: `blockdiag(R, R) · v`.
#[inline]
pub fn rotate_spatial(r: &[[f64; 3]; 3], v: &[f64; 6]) -> [f64; 6] {
    let a = mat3_vec(r, &[v[0], v[1], v[2]]);
    let b = mat3_vec(r, &[v[3], v[4], v[5]]);
    [a[0], a[1], a[2], b[0], b[1], b[2]]
}

/// Inverse of [`rotate_spatial`]: `blockdiag(Rᵀ, Rᵀ) · v`.
#[inline]
pub fn rotate_spatial_inv(r: &[[f64; 3]; 3], v: &[f64; 6]) -> [f64; 6] {
    let a = mat3t_vec(r, &[v[0], v[1], v[2]]);
    let b = mat3t_vec(r, &[v[3], v[4], v[5]]);
    [a[0], a[1], a[2], b[0], b[1], b[2]]
}

/// 6 × 6 matrix times 6-vector.
#[inline]
pub fn mat6_vec(m: &[[f64; 6]; 6], v: &[f64; 6]) -> [f64; 6] {
    core::array::from_fn(|r| {
        let mut s = 0.0;
        for c in 0..6 {
            s += m[r][c] * v[c];
        }
        s
    })
}

/// Task-space twist `J q̇` for a `6 × N` geometric Jacobian.
#[inline]
pub fn jacobian_twist<const N: usize>(jac: &[[f64; N]; 6], qd: &[f64; N]) -> [f64; 6] {
    core::array::from_fn(|r| {
        let mut s = 0.0;
        for c in 0..N {
            s += jac[r][c] * qd[c];
        }
        s
    })
}

/// Joint torque `Jᵀ wrench` on the first `dof` columns (remaining entries zero).
#[inline]
pub fn jacobian_transpose_wrench<const N: usize>(
    jac: &[[f64; N]; 6],
    wrench: &[f64; 6],
    dof: usize,
) -> [f64; N] {
    let mut out = [0.0f64; N];
    let n = dof.min(N);
    for c in 0..n {
        let mut s = 0.0;
        for r in 0..6 {
            s += jac[r][c] * wrench[r];
        }
        out[c] = s;
    }
    out
}

/// Task-space wrench `(J⁺)ᵀ τ` from a `N × 6` generalized inverse `j_pinv` and
/// residual joint torque `torque` (`dof` active joints).
#[inline]
pub fn pseudo_inverse_transpose_torque<const N: usize>(
    j_pinv: &[[f64; 6]; N],
    torque: &[f64; N],
    dof: usize,
) -> [f64; 6] {
    let n = dof.min(N);
    core::array::from_fn(|c| {
        let mut s = 0.0;
        for r in 0..n {
            s += j_pinv[r][c] * torque[r];
        }
        s
    })
}

/// Diagonal joint-space projection `diag(Jᵀ K J)` of a symmetric positive
/// semi-definite `6 × 6` task-space gain matrix `gain_base` (`dof` active joints).
pub fn diag_jt_gain_j<const N: usize>(
    jac: &[[f64; N]; 6],
    gain_base: &[[f64; 6]; 6],
    dof: usize,
) -> [f64; N] {
    let mut out = [0.0f64; N];
    let n = dof.min(N);
    for j in 0..n {
        let mut s = 0.0;
        for r in 0..6 {
            for c in 0..6 {
                s += jac[r][j] * gain_base[r][c] * jac[c][j];
            }
        }
        out[j] = s.max(0.0);
    }
    out
}

/// Damped least-squares step `Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ err`.
///
/// Allocation-free and well-defined through kinematic singularities. Returns
/// `[0.0; N]` when `J Jᵀ + λ² I` is not positive definite (`λ == 0` at a
/// singular `J`, or a non-finite `jac`); a non-finite `err` with a
/// finite `jac` propagates `NAN` into the result.
pub fn damped_least_squares<const N: usize>(
    jac: &[[f64; N]; 6],
    err: &[f64; 6],
    lambda: f64,
) -> [f64; N] {
    weighted_damped_least_squares(jac, err, &[1.0; N], lambda)
}

/// Weighted Levenberg–Marquardt / damped least-squares step minimizing
/// `‖J Δq − err‖² + λ² ‖Δq‖_W²` for diagonal joint weights `weights`:
/// `Δq = W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ² I)⁻¹ err`.
pub fn weighted_damped_least_squares<const N: usize>(
    jac: &[[f64; N]; 6],
    err: &[f64; 6],
    weights: &[f64; N],
    lambda: f64,
) -> [f64; N] {
    let winv: [f64; N] = core::array::from_fn(|i| {
        if weights[i].is_finite() && weights[i] > 1e-9 {
            1.0 / weights[i]
        } else {
            1.0
        }
    });
    let mut a = SMatrix::<f64, 6, 6>::zeros();
    let lam2 = lambda * lambda;
    for r in 0..6 {
        for k in 0..6 {
            let mut s = 0.0;
            for c in 0..N {
                s += jac[r][c] * winv[c] * jac[k][c];
            }
            a[(r, k)] = s;
        }
        a[(r, r)] += lam2;
    }
    let Some(chol) = a.cholesky() else {
        return [0.0f64; N];
    };
    let y = chol.solve(&SVector::<f64, 6>::from_column_slice(err));
    core::array::from_fn(|c| {
        let mut jt_y = 0.0;
        for r in 0..6 {
            jt_y += jac[r][c] * y[r];
        }
        winv[c] * jt_y
    })
}

/// Khatib's dynamically consistent generalized inverse
/// `J̄ = M⁻¹ Jᵀ (J M⁻¹ Jᵀ)⁻¹` on the active `dof ≤ N` joints (zero-padded to `N`).
///
/// Guarantees `J M⁻¹ (I − Jᵀ J̄ᵀ) τ = 0` for any joint torque `τ` (null-space
/// torques produce zero operational-space acceleration). Returns `None` if `M`
/// is non-SPD or `Λ⁻¹ = J M⁻¹ Jᵀ` is singular.
pub fn dynamically_consistent_pseudo_inverse<const N: usize>(
    jac: &[[f64; N]; 6],
    mass_matrix: &[[f64; N]; N],
    dof: usize,
) -> Option<[[f64; 6]; N]> {
    let n = dof.min(N);
    if n == 0 {
        return None;
    }
    let mut m = SMatrix::<f64, N, N>::identity();
    let mut j = SMatrix::<f64, 6, N>::zeros();
    for r in 0..n {
        for c in 0..n {
            m[(r, c)] = mass_matrix[r][c];
        }
        for k in 0..6 {
            j[(k, r)] = jac[k][r];
        }
    }
    let m_ch = m.cholesky()?;
    let minv_jt = m_ch.solve(&j.transpose());
    let lambda_inv = j * minv_jt;
    let l_ch = lambda_inv.cholesky()?;
    let lambda = l_ch.inverse();
    let j_bar = minv_jt * lambda;
    let mut out = [[0.0f64; 6]; N];
    for r in 0..n {
        for c in 0..6 {
            out[r][c] = j_bar[(r, c)];
        }
    }
    Some(out)
}

/// Project a candidate joint torque `tau_0` into the operational-space null
/// space `(I − Jᵀ J̄ᵀ) τ_0` using generalized inverse `j_bar` (either
/// [`dynamically_consistent_pseudo_inverse`] or [`pseudo_inverse`]).
pub fn nullspace_project_torque<const N: usize>(
    jac: &[[f64; N]; 6],
    j_bar: &[[f64; 6]; N],
    tau_0: &[f64; N],
    dof: usize,
) -> [f64; N] {
    let n = dof.min(N);
    let u = pseudo_inverse_transpose_torque(j_bar, tau_0, n);
    let proj = jacobian_transpose_wrench(jac, &u, n);
    let mut out = [0.0f64; N];
    for i in 0..n {
        out[i] = tau_0[i] - proj[i];
    }
    out
}

/// Solve `M q̈ = rhs` on the active `dof × dof` sub-block of `mass_matrix`
/// (zero-padded to `N`), using Cholesky with general-inverse fallback.
pub fn solve_mass_matrix<const N: usize>(
    mass_matrix: &[[f64; N]; N],
    rhs: &[f64; N],
    dof: usize,
) -> Option<[f64; N]> {
    let n = dof.min(N);
    let mut m = SMatrix::<f64, N, N>::identity();
    let mut b = SVector::<f64, N>::zeros();
    for r in 0..n {
        b[r] = rhs[r];
        for c in 0..n {
            m[(r, c)] = mass_matrix[r][c];
        }
    }
    let sol = if let Some(ch) = m.cholesky() {
        ch.solve(&b)
    } else {
        m.try_inverse()? * b
    };
    let mut out = [0.0f64; N];
    for i in 0..n {
        out[i] = sol[i];
    }
    Some(out)
}

/// Static wrench `[f; r × f]` that a grasped payload's weight exerts at the
/// TCP origin in base axes (`com_flange` in flange coordinates, `tcp_offset`
/// flange → TCP, `tcp_in_base` base → TCP). Zero when `mass` is not a finite
/// positive number (no payload).
pub fn payload_wrench_at_tcp(
    mass: f64,
    com_flange: &[f64; 3],
    tcp_offset: &Transform,
    tcp_in_base: &Transform,
    gravity: &[f64; 3],
) -> [f64; 6] {
    if !mass.is_finite() || mass <= 0.0 {
        return [0.0; 6];
    }
    let weight = [gravity[0] * mass, gravity[1] * mass, gravity[2] * mass];
    let com_tcp = inverse_transform_point(tcp_offset, com_flange);
    let r = mat3_vec(&tcp_in_base.r, &com_tcp);
    let moment = cross3(&r, &weight);
    [
        weight[0], weight[1], weight[2], moment[0], moment[1], moment[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::chain::pose_from_xyz_quat;
    use crate::math::testing::{calibrated_chain, sample_q};

    #[test]
    fn jacobian_bias_matches_directional_finite_difference() {
        let chain = calibrated_chain();
        let q = sample_q::<6>(4, 2.0);
        let qd = sample_q::<6>(5, 1.5);
        let h = 1e-6;
        let (mut qp, mut qm) = (q, q);
        for j in 0..6 {
            qp[j] += h * qd[j];
            qm[j] -= h * qd[j];
        }
        let (jp, jm) = (chain.jacobian(&qp), chain.jacobian(&qm));
        let bias = jacobian_bias(&chain.jacobian(&q), &qd);
        for r in 0..6 {
            let mut want = 0.0;
            for j in 0..6 {
                want += (jp[r][j] - jm[r][j]) / (2.0 * h) * qd[j];
            }
            assert!(
                (bias[r] - want).abs() < 1e-7,
                "row {r}: {} vs {want}",
                bias[r]
            );
        }
        assert!(bias.iter().any(|v| v.abs() > 1e-2));
    }

    #[test]
    fn pseudo_inverse_inverts_a_regular_jacobian() {
        let jac = calibrated_chain().jacobian(&sample_q::<6>(7, 2.0));
        let jinv = pseudo_inverse(&jac).unwrap();
        for r in 0..6 {
            for c in 0..6 {
                let s: f64 = (0..6).map(|k| jac[r][k] * jinv[k][c]).sum();
                assert!((s - if r == c { 1.0 } else { 0.0 }).abs() < 1e-8);
            }
        }
        assert!(pseudo_inverse(&[[0.0f64; 6]; 6]).is_none());
    }

    #[test]
    fn tool_frame_helpers_round_trip_and_preserve_power() {
        let tool = pose_from_xyz_quat(&[0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.88]).unwrap();
        let wrench = [3.0, -2.0, 5.0, 0.4, 0.1, -0.7];
        let in_tool = wrench_base_to_tool(&tool, &wrench);
        let back = wrench_tool_to_base(&tool, &in_tool);
        for k in 0..6 {
            assert!((back[k] - wrench[k]).abs() < 1e-12);
        }

        // Power is frame independent: wᵀ J q̇ in base axes equals the tool-axes value.
        let chain = calibrated_chain();
        let q = sample_q::<6>(1, 2.0);
        let qd = sample_q::<6>(2, 1.0);
        let jac = chain.jacobian(&q);
        let jac_tool = jacobian_in_frame(&tool, &jac);
        let power = |j: &[[f64; 6]; 6], w: &[f64; 6]| -> f64 {
            (0..6)
                .map(|r| w[r] * (0..6).map(|c| j[r][c] * qd[c]).sum::<f64>())
                .sum()
        };
        assert!((power(&jac, &wrench) - power(&jac_tool, &in_tool)).abs() < 1e-12);

        // A tool-frame diagonal stiffness acts identically in either frame.
        let gain = [800.0, 500.0, 1200.0, 30.0, 20.0, 10.0];
        let k_base = diagonal_gain_tool_to_base(&tool, &gain);
        let twist = [0.01, -0.02, 0.005, 0.1, 0.0, -0.05];
        let mut f_base = [0.0f64; 6];
        for r in 0..6 {
            for c in 0..6 {
                f_base[r] += k_base[r][c] * twist[c];
            }
        }
        let twist_tool = wrench_base_to_tool(&tool, &twist);
        let f_tool = wrench_base_to_tool(&tool, &f_base);
        for k in 0..6 {
            assert!(
                (f_tool[k] - gain[k] * twist_tool[k]).abs() < 1e-9,
                "axis {k}"
            );
        }

        // Base → tool of an isotropic gain stays diagonal.
        let iso = diagonal_gain_base_to_tool(&tool, &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
        for r in 0..6 {
            for c in 0..6 {
                let want = if r != c {
                    0.0
                } else if r < 3 {
                    2.0
                } else {
                    3.0
                };
                assert!((iso[r][c] - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn se3_offset_lerp_distance_and_antipodal_rotation_round_trip() {
        use crate::math::chain::{
            pose_error, se3_distance, se3_lerp, se3_offset, transform_from_xyz_quat,
            transform_point, transform_to_xyz_quat,
        };
        let a = transform_from_xyz_quat(&[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 2.0]).unwrap();
        assert_eq!(
            transform_to_xyz_quat(&a),
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        );
        assert!(transform_from_xyz_quat(&[0.0; 7]).is_err());

        let off = [0.01, -0.02, 0.03, 0.1, 0.0, -0.2];
        let b = se3_offset(&a, &off);
        let e = pose_error(&b, &a);
        for k in 0..6 {
            assert!((e[k] - off[k]).abs() < 1e-12, "axis {k}");
        }
        let mid = se3_lerp(&a, &b, 0.5);
        let (d, ang) = se3_distance(&a, &mid);
        let (d2, ang2) = se3_distance(&a, &b);
        assert!((d - d2 / 2.0).abs() < 1e-12 && (ang - ang2 / 2.0).abs() < 1e-12);

        // Antipodal 180° rotation (cos θ == -1, sin θ == 0) stays well-defined.
        let flip = transform_from_xyz_quat(&[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]).unwrap();
        let (_, flip_ang) = se3_distance(&a, &flip);
        assert!((flip_ang - core::f64::consts::PI).abs() < 1e-12);
        let half_flip = se3_lerp(&a, &flip, 0.5);
        let (_, half_ang) = se3_distance(&a, &half_flip);
        assert!((half_ang - core::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let pt = [0.4, -0.3, 0.8];
        let back = inverse_transform_point(&b, &transform_point(&b, &pt));
        for k in 0..3 {
            assert!((back[k] - pt[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn weighted_dls_and_dynamically_consistent_nullspace_annihilate_task_accel() {
        let mut jac = [[0.0f64; 8]; 6];
        for i in 0..6 {
            jac[i][i] = 1.0 + 0.2 * i as f64;
        }
        // Redundant 7th joint coupled into x and rz.
        jac[0][6] = 0.7;
        jac[5][6] = -0.5;

        let err = [0.05, -0.02, 0.01, 0.03, -0.01, 0.04];
        let uniform = weighted_damped_least_squares(&jac, &err, &[1.0; 8], 1e-4);
        let heavy_base = weighted_damped_least_squares(
            &jac,
            &err,
            &[4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
            1e-4,
        );
        assert!(heavy_base[0].abs() < uniform[0].abs());
        assert!(heavy_base[6].abs() > uniform[6].abs());
        assert_eq!(damped_least_squares(&[[0.0; 8]; 6], &err, 0.02), [0.0; 8]);

        // Non-diagonal SPD mass matrix on 7 active joints.
        let mut m = [[0.0f64; 8]; 8];
        for i in 0..7 {
            m[i][i] = 2.5 - 0.25 * i as f64;
            if i + 1 < 7 {
                m[i][i + 1] = 0.15;
                m[i + 1][i] = 0.15;
            }
        }
        let j_bar = dynamically_consistent_pseudo_inverse(&jac, &m, 7).unwrap();
        let tau_0 = [3.0, -2.0, 1.5, -0.8, 2.2, -1.1, 4.0, 99.0];
        let tau_null = nullspace_project_torque(&jac, &j_bar, &tau_0, 7);
        assert_eq!(tau_null[7], 0.0);

        // Khatib's identity: J M⁻¹ τ_null == 0.
        let qdd_null = solve_mass_matrix(&m, &tau_null, 7).unwrap();
        let xdd = jacobian_twist(&jac, &qdd_null);
        for r in 0..6 {
            assert!(xdd[r].abs() < 1e-11, "task accel {r} leaked: {}", xdd[r]);
        }

        // Non-SPD (indefinite) mass matrix: Khatib inverse refuses, solve falls back.
        let mut indefinite = [[0.0f64; 8]; 8];
        indefinite[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        indefinite[1] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(dynamically_consistent_pseudo_inverse(&jac, &indefinite, 2).is_none());
        let x =
            solve_mass_matrix(&indefinite, &[2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(x[..2], [3.0, 2.0]);
        assert!(solve_mass_matrix(&[[0.0; 8]; 8], &[1.0; 8], 2).is_none());
    }

    const RZ_90: [[f64; 3]; 3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    const RX_90: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
    const I3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_close<const N: usize>(got: &[f64; N], want: &[f64; N], tol: f64) {
        for k in 0..N {
            assert!((got[k] - want[k]).abs() < tol, "[{k}] {got:?} != {want:?}");
        }
    }

    #[test]
    fn spatial_rotation_and_matrix_vector_helpers() {
        let v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let rotated = rotate_spatial(&RZ_90, &v);
        assert_close(&rotated, &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0], 1e-15);
        assert_close(&rotate_spatial_inv(&RZ_90, &rotated), &v, 1e-15);

        let mut m = [[0.0f64; 6]; 6];
        for i in 0..6 {
            m[i][i] = 2.0;
        }
        m[0][5] = 3.0;
        assert_eq!(
            mat6_vec(&m, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [20.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }

    #[test]
    fn jacobian_transpose_and_gain_projection_respect_active_dof() {
        let mut jac = [[0.0f64; 3]; 6];
        jac[0] = [1.0, 0.0, 5.0];
        jac[1] = [0.0, 2.0, 5.0];
        jac[5] = [0.5, 0.0, 5.0];
        let wrench = [1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        assert_eq!(jacobian_transpose_wrench(&jac, &wrench, 2), [2.0, 2.0, 0.0]);
        assert_eq!(
            jacobian_transpose_wrench(&jac, &wrench, 9),
            [2.0, 2.0, 20.0]
        );

        let mut gain = [[0.0f64; 6]; 6];
        gain[0][0] = 10.0;
        gain[1][1] = 20.0;
        gain[5][5] = 4.0;
        gain[0][5] = 1.0;
        gain[5][0] = 1.0;
        // Column 0: 10·1 + 4·0.25 + 2·(1·1·0.5) = 12.
        assert_eq!(diag_jt_gain_j(&jac, &gain, 2), [12.0, 80.0, 0.0]);

        // Round-off negatives are clamped to zero.
        let mut neg = [[0.0f64; 6]; 6];
        neg[0][0] = -1.0;
        assert_eq!(diag_jt_gain_j(&jac, &neg, 1), [0.0, 0.0, 0.0]);

        let mut j_pinv = [[0.0f64; 6]; 3];
        j_pinv[0][0] = 1.0;
        j_pinv[1][0] = 2.0;
        j_pinv[2][0] = 100.0;
        assert_eq!(
            pseudo_inverse_transpose_torque(&j_pinv, &[1.0, 1.0, 1.0], 2)[0],
            3.0
        );
    }

    #[test]
    fn damped_least_squares_non_finite_inputs() {
        let mut jac = [[0.0f64; 6]; 6];
        for i in 0..6 {
            jac[i][i] = 1.0;
        }
        let err = [0.1; 6];
        let mut nan_jac = jac;
        nan_jac[2][2] = f64::NAN;
        assert_eq!(damped_least_squares(&nan_jac, &err, 0.02), [0.0; 6]);
        let mut inf_jac = jac;
        inf_jac[2][2] = f64::INFINITY;
        assert_eq!(damped_least_squares(&inf_jac, &err, 0.02), [0.0; 6]);
        let mut nan_err = err;
        nan_err[0] = f64::NAN;
        assert!(damped_least_squares(&jac, &nan_err, 0.02)[0].is_nan());
        // Non-positive or non-finite weights count as 1.
        assert_eq!(
            weighted_damped_least_squares(&jac, &err, &[0.0, -1.0, f64::NAN, 1.0, 1.0, 1.0], 0.0),
            damped_least_squares(&jac, &err, 0.0)
        );
    }

    #[test]
    fn payload_wrench_at_tcp_moves_weight_through_tool_and_base_frames() {
        let gravity = [0.0, 0.0, -9.81];
        // COM 5 cm beyond a straight 10 cm tool; TCP rotated +90° about base x.
        let tool = Transform {
            r: I3,
            t: [0.0, 0.0, 0.1],
        };
        let tcp = Transform {
            r: RX_90,
            t: [0.3, 0.2, 0.5],
        };
        let w = payload_wrench_at_tcp(2.0, &[0.0, 0.0, 0.15], &tool, &tcp, &gravity);
        // r = Rx · [0, 0, 0.05] = [0, −0.05, 0]; r × [0, 0, −19.62] = [0.981, 0, 0].
        assert_close(&w, &[0.0, 0.0, -19.62, 0.981, 0.0, 0.0], 1e-12);

        // Rotated tool: COM at [0.1, 0.2, 0] in the flange is [0.2, 0, 0] in the TCP.
        let tool = Transform {
            r: RZ_90,
            t: [0.1, 0.0, 0.0],
        };
        let tcp = Transform { r: I3, t: [0.0; 3] };
        let w = payload_wrench_at_tcp(2.0, &[0.1, 0.2, 0.0], &tool, &tcp, &gravity);
        assert_close(&w, &[0.0, 0.0, -19.62, 0.0, 3.924, 0.0], 1e-12);

        for mass in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                payload_wrench_at_tcp(mass, &[0.0, 0.0, 0.15], &tool, &tcp, &gravity),
                [0.0; 6]
            );
        }
    }

    #[test]
    fn rotation_exp_small_angle_and_log_near_pi() {
        use crate::math::chain::{rotation_error, rotation_exp};
        let w = [1e-10, 2e-10, -1e-10];
        let r = rotation_exp(&w);
        assert_close(&[r[2][1], r[0][2], r[1][0]], &w, 1e-19);
        assert_close(&rotation_error(&r, &I3), &w, 1e-22);

        // Just below π the axis must not come from the ill-conditioned `w / sin θ`.
        let n = libm::sqrt(14.0);
        let axis = [1.0 / n, -2.0 / n, 3.0 / n];
        for gap in [1e-3, 1e-6, 1e-8, 1e-10, 0.0] {
            let angle = core::f64::consts::PI - gap;
            let want = [axis[0] * angle, axis[1] * angle, axis[2] * angle];
            let got = rotation_error(&rotation_exp(&want), &I3);
            // At exactly π, `u` and `−u` are the same rotation.
            let sign = if gap == 0.0 && got[0] < 0.0 {
                -1.0
            } else {
                1.0
            };
            let got = [got[0] * sign, got[1] * sign, got[2] * sign];
            assert_close(&got, &want, 1e-13);
        }
        // Obtuse angles take the symmetric-part branch; check continuity across π/2.
        for angle in [1.5, 1.6, 2.5] {
            let want = [axis[0] * angle, axis[1] * angle, axis[2] * angle];
            assert_close(&rotation_error(&rotation_exp(&want), &I3), &want, 1e-14);
        }
    }
}
