//! Pinocchio ground-truth parity for the symbolic UR5 dynamics.
//!
//! Compares `generated_dynamics::{gravity, mass_matrix, coriolis,
//! inverse_dynamics}` against the independent Newton-Euler reference in
//! [`dynamics_test_data`] (produced by `codegen/generate_dynamics_test_data.py`).
//!
//! Before treating a mismatch as a dynamics bug, confirm:
//! * joint order matches [`dynamics_test_data::JOINT_NAMES`]
//! * gravity vector matches [`dynamics_test_data::GRAVITY`]

#![cfg(test)]

#[path = "dynamics_test_data.rs"]
mod dynamics_test_data;

use super::generated_dynamics::{coriolis, gravity, inverse_dynamics, mass_matrix};

/// Absolute torque / inertia tolerance. Lagrangian vs. recursive Newton-Euler
/// from the same URDF agree to well below this on commodity `f64`.
const ATOL: f64 = 1e-6;

fn max_abs_diff_vec(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let mut m = 0.0f64;
    for i in 0..6 {
        let d = libm::fabs(a[i] - b[i]);
        if d > m {
            m = d;
        }
    }
    m
}

fn max_abs_diff_mat(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> f64 {
    let mut m = 0.0f64;
    for i in 0..6 {
        for j in 0..6 {
            let d = libm::fabs(a[i][j] - b[i][j]);
            if d > m {
                m = d;
            }
        }
    }
    m
}

#[test]
fn conventions_match_pinocchio_dataset() {
    assert_eq!(dynamics_test_data::DOF, 6);
    assert_eq!(
        dynamics_test_data::GRAVITY,
        [0.0, 0.0, -9.81],
        "gravity sign/convention drift — update generator --gravity or the test"
    );
    assert_eq!(
        dynamics_test_data::JOINT_NAMES,
        [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
    );
    assert!(dynamics_test_data::N_CASES > 0);
    assert_eq!(
        dynamics_test_data::RAW.len(),
        dynamics_test_data::N_CASES * dynamics_test_data::FLOATS_PER_CASE * 8
    );
}

#[test]
fn matches_pinocchio_gravity_mass_coriolis_and_inverse_dynamics() {
    let g_world = dynamics_test_data::GRAVITY;
    let mut worst = [0.0f64; 4]; // g, M, C, tau
    let mut worst_idx = [0usize; 4];

    for idx in 0..dynamics_test_data::N_CASES {
        let c = dynamics_test_data::case(idx);

        let g_got = gravity(&c.q, &g_world);
        let m_got = mass_matrix(&c.q);
        let c_got = coriolis(&c.q, &c.qd);
        let tau_got = inverse_dynamics(&c.q, &c.qd, &c.qdd, &g_world);

        let dg = max_abs_diff_vec(&g_got, &c.gravity_torque);
        let dm = max_abs_diff_mat(&m_got, &c.mass_matrix);
        let dc = max_abs_diff_vec(&c_got, &c.coriolis_torque);
        let dtau = max_abs_diff_vec(&tau_got, &c.inverse_dynamics_torque);

        for (slot, d) in [(0, dg), (1, dm), (2, dc), (3, dtau)] {
            if d > worst[slot] {
                worst[slot] = d;
                worst_idx[slot] = idx;
            }
        }

        assert!(
            dg < ATOL,
            "case {idx}: gravity mismatch max|Δ|={dg:e}\n got={g_got:?}\n ref={:?}",
            c.gravity_torque
        );
        assert!(
            dm < ATOL,
            "case {idx}: mass_matrix mismatch max|Δ|={dm:e}"
        );
        assert!(
            dc < ATOL,
            "case {idx}: coriolis mismatch max|Δ|={dc:e}\n got={c_got:?}\n ref={:?}",
            c.coriolis_torque
        );
        assert!(
            dtau < ATOL,
            "case {idx}: inverse_dynamics mismatch max|Δ|={dtau:e}\n got={tau_got:?}\n ref={:?}",
            c.inverse_dynamics_torque
        );
    }

    assert!(
        worst.iter().all(|&d| d < ATOL),
        "worst Δ: g={:e} (case {}), M={:e} (case {}), C={:e} (case {}), τ={:e} (case {})",
        worst[0],
        worst_idx[0],
        worst[1],
        worst_idx[1],
        worst[2],
        worst_idx[2],
        worst[3],
        worst_idx[3],
    );
}

#[test]
fn inverse_dynamics_equals_m_qdd_plus_c_plus_g() {
    let g_world = dynamics_test_data::GRAVITY;
    let n_check = dynamics_test_data::N_CASES.min(64);
    for idx in 0..n_check {
        let c = dynamics_test_data::case(idx);
        let m = mass_matrix(&c.q);
        let cor = coriolis(&c.q, &c.qd);
        let g = gravity(&c.q, &g_world);
        let tau = inverse_dynamics(&c.q, &c.qd, &c.qdd, &g_world);

        let mut composed = [0.0f64; 6];
        for i in 0..6 {
            let mut mqdd = 0.0;
            for j in 0..6 {
                mqdd += m[i][j] * c.qdd[j];
            }
            composed[i] = mqdd + cor[i] + g[i];
        }
        let d = max_abs_diff_vec(&tau, &composed);
        assert!(
            d < ATOL,
            "case {idx}: τ ≠ M qdd + C qd + g (max|Δ|={d:e})"
        );
    }
}
