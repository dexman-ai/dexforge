//! Grasped payload and its joint-space contributions.
//!
//! A payload is rigidly attached to the flange, so its effect on the joints
//! follows from the flange pose and geometric Jacobian alone — the same math
//! for every arm. A driver's generated dynamics describe the arm's own links;
//! these functions supply the payload part to add on top.

use crate::math::chain::ModelError;
use crate::math::model::Pose;
use crate::math::spatial::{flange_offset_jacobian, jacobian_bias, pose_rotation};
use crate::math::ssik_core::{add3, cross3, dot3, mat3_mul, mat3_transpose, mat3_vec};

/// A grasped tool / payload rigidly attached to the flange.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Payload {
    /// Mass in kilograms.
    pub mass: f64,
    /// Centre of mass `[x, y, z]` in the flange frame.
    pub com: [f64; 3],
    /// Rotational inertia (kg·m²) about `com`, in flange-frame axes. Zero for
    /// a point mass; only matters for large tools at high angular rates.
    pub inertia: [[f64; 3]; 3],
}

impl Payload {
    /// No payload.
    pub const NONE: Payload = Payload::new(0.0, [0.0, 0.0, 0.0]);

    /// A point mass at a flange-frame centre of mass.
    #[inline]
    pub const fn new(mass: f64, com: [f64; 3]) -> Payload {
        Payload {
            mass,
            com,
            inertia: [[0.0; 3]; 3],
        }
    }

    /// Add a rotational inertia tensor about the centre of mass.
    #[inline]
    pub const fn with_inertia(mut self, inertia: [[f64; 3]; 3]) -> Payload {
        self.inertia = inertia;
        self
    }

    /// Build and validate a payload from the driver-contract argument shape
    /// (`inertia` row-major 3×3).
    pub fn from_contract(
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<Payload, ModelError> {
        let payload = Payload::new(mass_kg, com).with_inertia([
            [inertia[0], inertia[1], inertia[2]],
            [inertia[3], inertia[4], inertia[5]],
            [inertia[6], inertia[7], inertia[8]],
        ]);
        payload.validate()?;
        Ok(payload)
    }

    /// Reject non-finite values and negative mass.
    pub fn validate(&self) -> Result<(), ModelError> {
        let finite = self.mass.is_finite()
            && self.com.iter().all(|v| v.is_finite())
            && self.inertia.iter().flatten().all(|v| v.is_finite());
        if !finite {
            return Err(ModelError::NonFinite);
        }
        if self.mass < 0.0 {
            return Err(ModelError::InvalidPayload);
        }
        Ok(())
    }

    #[inline]
    fn has_inertia(&self) -> bool {
        self.inertia.iter().flatten().any(|v| *v != 0.0)
    }

    /// Whether the payload contributes nothing to the dynamics.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.mass == 0.0 && !self.has_inertia()
    }
}

/// Joint torque needed to move a grasped `payload` along `(q, qd, qdd)` under
/// `gravity`: its share of `M(q) qdd + C(q, qd) qd + G(q)`.
///
/// `flange_pose` and `jacobian` are the flange FK and geometric Jacobian at
/// `q`. Newton–Euler on the payload body: the force `m (a_com − g)` and the
/// moment `I α + ω × I ω` are mapped to the joints through the Jacobian
/// transpose, with the velocity-product accelerations taken from
/// [`jacobian_bias`]. One pass, no finite differences.
pub fn payload_inverse_dynamics<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    qd: &[f64; DOF],
    qdd: &[f64; DOF],
    payload: &Payload,
    gravity: &[f64; 3],
) -> [f64; DOF] {
    let mut tau = [0.0f64; DOF];
    if payload.is_none() {
        return tau;
    }
    let rot = pose_rotation(flange_pose);
    let r = mat3_vec(&rot, &payload.com);

    let bias = jacobian_bias(jacobian, qd);
    let mut accel = [bias[0], bias[1], bias[2]];
    let mut alpha = [bias[3], bias[4], bias[5]];
    let mut omega = [0.0f64; 3];
    for j in 0..DOF {
        for k in 0..3 {
            accel[k] += jacobian[k][j] * qdd[j];
            alpha[k] += jacobian[3 + k][j] * qdd[j];
            omega[k] += jacobian[3 + k][j] * qd[j];
        }
    }

    // Centre-of-mass acceleration: a + α × r + ω × (ω × r).
    let a_com = add3(
        &add3(&accel, &cross3(&alpha, &r)),
        &cross3(&omega, &cross3(&omega, &r)),
    );
    let force = [
        payload.mass * (a_com[0] - gravity[0]),
        payload.mass * (a_com[1] - gravity[1]),
        payload.mass * (a_com[2] - gravity[2]),
    ];
    // Moment about the flange origin: r × f + I α + ω × I ω.
    let mut moment = cross3(&r, &force);
    if payload.has_inertia() {
        let ib = mat3_mul(&mat3_mul(&rot, &payload.inertia), &mat3_transpose(&rot));
        let euler = add3(
            &mat3_vec(&ib, &alpha),
            &cross3(&omega, &mat3_vec(&ib, &omega)),
        );
        moment = add3(&moment, &euler);
    }
    for j in 0..DOF {
        for k in 0..3 {
            tau[j] += jacobian[k][j] * force[k] + jacobian[3 + k][j] * moment[k];
        }
    }
    tau
}

/// Holding torque contributed by a grasped `payload`: `G(q)` of the payload.
pub fn payload_gravity_torque<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    payload: &Payload,
    gravity: &[f64; 3],
) -> [f64; DOF] {
    let zero = [0.0f64; DOF];
    payload_inverse_dynamics(flange_pose, jacobian, &zero, &zero, payload, gravity)
}

/// Coriolis and centrifugal torque `C(q, qd) qd` contributed by a grasped
/// `payload`.
pub fn payload_coriolis_torque<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    qd: &[f64; DOF],
    payload: &Payload,
) -> [f64; DOF] {
    payload_inverse_dynamics(flange_pose, jacobian, qd, &[0.0; DOF], payload, &[0.0; 3])
}

/// Contribution of a grasped `payload` to the joint-space inertia `M(q)`:
/// `m Jvpᵀ Jvp + Jwᵀ (R I Rᵀ) Jw`.
pub fn payload_inertia_matrix<const DOF: usize>(
    flange_pose: &Pose,
    jacobian: &[[f64; DOF]; 6],
    payload: &Payload,
) -> [[f64; DOF]; DOF] {
    let mut mm = [[0.0f64; DOF]; DOF];
    if payload.mass != 0.0 {
        let jvp = flange_offset_jacobian(flange_pose, jacobian, &payload.com);
        for i in 0..DOF {
            for j in 0..DOF {
                mm[i][j] = payload.mass * dot3(&jvp[i], &jvp[j]);
            }
        }
    }
    if payload.has_inertia() {
        let rot = pose_rotation(flange_pose);
        let ib = mat3_mul(&mat3_mul(&rot, &payload.inertia), &mat3_transpose(&rot));
        let angular = |j: usize| [jacobian[3][j], jacobian[4][j], jacobian[5][j]];
        for j in 0..DOF {
            let iw = mat3_vec(&ib, &angular(j));
            for i in 0..DOF {
                mm[i][j] += dot3(&angular(i), &iw);
            }
        }
    }
    mm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::testing::{calibrated_chain, sample_q};

    fn tool_payload() -> Payload {
        Payload::new(2.5, [0.04, -0.03, 0.08]).with_inertia([
            [0.020, 0.002, 0.000],
            [0.002, 0.030, 0.001],
            [0.000, 0.001, 0.015],
        ])
    }

    fn mass_matrix(q: &[f64; 6]) -> [[f64; 6]; 6] {
        let (pose, jac) = calibrated_chain().fk_jacobian(q);
        payload_inertia_matrix(&pose, &jac, &tool_payload())
    }

    #[test]
    fn gravity_torque_is_the_potential_gradient() {
        let chain = calibrated_chain();
        let payload = tool_payload();
        let gravity = [1.0, -2.0, -9.0];
        let potential = |q: &[f64; 6]| {
            let t = chain.fk(q);
            let com = mat3_vec(&pose_rotation(&t), &payload.com);
            let mut u = 0.0;
            for r in 0..3 {
                u -= payload.mass * gravity[r] * (t[(r, 3)] + com[r]);
            }
            u
        };
        let q = sample_q::<6>(11, 2.0);
        let (pose, jac) = chain.fk_jacobian(&q);
        let tau = payload_gravity_torque(&pose, &jac, &payload, &gravity);
        let h = 1e-6;
        for j in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[j] += h;
            qm[j] -= h;
            let grad = (potential(&qp) - potential(&qm)) / (2.0 * h);
            assert!(
                (tau[j] - grad).abs() < 1e-7,
                "joint {j}: {} vs {grad}",
                tau[j]
            );
        }
    }

    /// `C(q, qd) qd = Ṁ qd − ½ ∂(qdᵀ M qd)/∂q`, for a payload with an offset
    /// centre of mass and a full inertia tensor.
    #[test]
    fn coriolis_is_consistent_with_the_mass_matrix() {
        let q = sample_q::<6>(5, 2.0);
        let qd = sample_q::<6>(6, 1.0);
        let m_qd = |q: &[f64; 6]| {
            let m = mass_matrix(q);
            let mut out = [0.0; 6];
            for i in 0..6 {
                for j in 0..6 {
                    out[i] += m[i][j] * qd[j];
                }
            }
            out
        };
        let kinetic = |q: &[f64; 6]| {
            let v = m_qd(q);
            (0..6).map(|i| 0.5 * qd[i] * v[i]).sum::<f64>()
        };
        let h = 1e-5;
        let (mut qp, mut qm) = (q, q);
        for j in 0..6 {
            qp[j] += h * qd[j];
            qm[j] -= h * qd[j];
        }
        let (mp, mm) = (m_qd(&qp), m_qd(&qm));
        let (pose, jac) = calibrated_chain().fk_jacobian(&q);
        let tau = payload_coriolis_torque(&pose, &jac, &qd, &tool_payload());
        let m = mass_matrix(&q);
        for i in 0..6 {
            let (mut qp, mut qm) = (q, q);
            qp[i] += h;
            qm[i] -= h;
            let want = (mp[i] - mm[i]) / (2.0 * h) - (kinetic(&qp) - kinetic(&qm)) / (2.0 * h);
            assert!(
                (tau[i] - want).abs() < 1e-6,
                "joint {i}: {} vs {want}",
                tau[i]
            );
            for j in 0..6 {
                assert!((m[i][j] - m[j][i]).abs() < 1e-12);
            }
        }
        assert!(tau.iter().any(|t| t.abs() > 1e-2));
    }

    #[test]
    fn inverse_dynamics_is_the_sum_of_its_terms() {
        let q = sample_q::<6>(8, 2.0);
        let qd = sample_q::<6>(9, 1.0);
        let qdd = sample_q::<6>(10, 3.0);
        let gravity = [0.0, 0.0, -9.80665];
        let payload = tool_payload();
        let (pose, jac) = calibrated_chain().fk_jacobian(&q);
        let tau = payload_inverse_dynamics(&pose, &jac, &qd, &qdd, &payload, &gravity);
        let m = payload_inertia_matrix(&pose, &jac, &payload);
        let c = payload_coriolis_torque(&pose, &jac, &qd, &payload);
        let g = payload_gravity_torque(&pose, &jac, &payload, &gravity);
        for i in 0..6 {
            let want = c[i] + g[i] + (0..6).map(|j| m[i][j] * qdd[j]).sum::<f64>();
            assert!(
                (tau[i] - want).abs() < 1e-10,
                "joint {i}: {} vs {want}",
                tau[i]
            );
        }
        let none = payload_inverse_dynamics(&pose, &jac, &qd, &qdd, &Payload::NONE, &gravity);
        assert_eq!(none, [0.0; 6]);
    }

    #[test]
    fn validation_rejects_bad_payloads() {
        assert_eq!(
            Payload::new(-1.0, [0.0; 3]).validate(),
            Err(ModelError::InvalidPayload)
        );
        assert_eq!(
            Payload::from_contract(1.0, [0.0, f64::NAN, 0.0], [0.0; 9]),
            Err(ModelError::NonFinite)
        );
        let p =
            Payload::from_contract(1.0, [0.0; 3], [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        assert_eq!(p.unwrap().inertia[1][1], 2.0);
    }
}
