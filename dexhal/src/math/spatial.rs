//! Arm-agnostic spatial math on a flange pose and geometric Jacobian.
//!
//! Everything a driver's own `fk` / `jacobian` output needs in order to be
//! re-referenced: to a tool-centre point, to a payload's centre of mass, to
//! the tool frame, or differentiated in time. Fixed-size loops only, no
//! allocation.
//!
//! Six-vectors are `[linear; angular]`: twists `[v; ω]`, wrenches `[f; m]`.

use nalgebra::SMatrix;

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
}
