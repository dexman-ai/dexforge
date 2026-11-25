//! Shared closed-form IK for the Universal Robots arms (UR5e, UR3e).
//!
//! Both arms use the `ssik` solver `ikgeo.three_parallel` (tier 0): subproblem
//! **SP6** for the coupled wrist/shoulder pair `(q1, q5)`, followed by the
//! inlined `SP1 + SP3 + SP1` chain for the remaining joints. SP6 reduces to
//! intersecting two conics, which produces a quartic solved through the
//! companion-matrix eigenvalue routine in [`dexhal::math::ssik_core`].
//!
//! The generated `_solve_algebraic` blocks of `ur5_ik.py` and `ur3e_ik.py` are
//! identical except for a handful of geometry literals, gathered here in
//! [`Model`]. Every literal is kept verbatim from the artifact and the floating
//! point operation order is unchanged, so each arm reproduces its reference
//! dataset bit for bit. Each arm's `ik_solver` supplies its `const MODEL`.
//!
//! [`solve`] hands the raw candidates to the shared 6-DOF post-process
//! [`finish6`] (FK closure / refinement, dedup, limits, seed sort, truncate).

use nalgebra::SMatrix;

use dexhal::math::closed_form::finish6;
use dexhal::math::ssik_core::sp6_solve;
use dexhal::math::tolerances::TolerancePolicy;
use dexhal::math::ArmModel;

use core::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Capacity of the raw algebraic candidate buffer (up to 4 SP6 roots x 2
/// elbow branches).
pub const MAX_CANDIDATES: usize = 8;

/// Per-arm literals of the generated `_solve_algebraic` block. Names follow
/// the UR DH parameters where the literal is one; the rest are the symbolic
/// rounding residue `ssik build` emits for that arm.
pub struct Model {
    /// Shoulder height `d1` (`p_16_z` offset and the `d_inner_z` constant).
    pub d1: f64,
    /// Shoulder offset `d4` (SP6 `d2` argument and the `q2` SP1 block).
    pub d4: f64,
    /// Wrist-1 offset `d5` (`d_inner_x`, `d_inner_z`).
    pub d5: f64,
    /// Wrist-2 offset `d6` as it appears in the `d_inner` block.
    pub d6: f64,
    /// `y` component of the SP6 input `-p[5]` (the artifact's rendering of `d6`).
    pub neg_p5_y: f64,
    /// Upper-arm length `a2` (constant term of `q2_x3`).
    pub a2: f64,
    /// Forearm length `a3` (`q2_x2`, `q2_x3`).
    pub a3: f64,
    /// Coefficient of `dinr_x10`.
    pub dinr_c10: f64,
    /// Constant `d_inner_y` coefficient (appears twice).
    pub dinr_cy: f64,
    /// SP3 radius squared for `q3`.
    pub q3_r_sq: f64,
    /// Constant term of the SP3 right-hand side for `q3`.
    pub q3_rhs0: f64,
    /// Residue added to `PI` in the SP3 phase for `q3`.
    pub q3_phi0: f64,
    /// `q2` SP1 coefficients (rounding residue), in order of first appearance.
    pub q2_c0: f64,
    /// See [`Model::q2_c0`].
    pub q2_c1: f64,
    /// See [`Model::q2_c0`].
    pub q2_c2: f64,
    /// See [`Model::q2_c0`].
    pub q2_c3: f64,
}

/// Every inverse-kinematics solution of the arm `A` described by `m` — the
/// `ik_all` path: raw candidates from [`solve_algebraic`] through [`finish6`].
/// Never early-exits; single solutions come from `single::six_dof_one`.
#[allow(clippy::too_many_arguments)]
pub fn solve<A: ArmModel<6, MAX>, const MAX: usize>(
    m: &Model,
    t: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 6]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 6]>; MAX] {
    let mut cands = [[0.0f64; 6]; MAX_CANDIDATES];
    let n = solve_algebraic(m, t, &mut cands, policy);
    finish6::<A, MAX>(
        &cands[..n],
        t,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        policy,
        refinement_max_iters,
    )
}

// ---------------------------------------------------------------------------
// Algebraic candidate generation (port of `_solve_algebraic`)
// ---------------------------------------------------------------------------

/// Fills `out` with the raw algebraic IK candidates of the arm described by
/// `m` and returns the count (at most [`MAX_CANDIDATES`]). The body is the
/// artifact's generated code verbatim, hence the lint allowance.
#[allow(clippy::neg_multiply)]
pub fn solve_algebraic(
    m: &Model,
    t: &SMatrix<f64, 4, 4>,
    out: &mut [[f64; 6]; MAX_CANDIDATES],
    policy: TolerancePolicy,
) -> usize {
    let r_00 = t[(0, 0)];
    let r_01 = t[(0, 1)];
    let r_02 = t[(0, 2)];
    let r_10 = t[(1, 0)];
    let r_11 = t[(1, 1)];
    let r_12 = t[(1, 2)];
    let r_20 = t[(2, 0)];
    let r_21 = t[(2, 1)];
    let r_22 = t[(2, 2)];
    let p_x = t[(0, 3)];
    let p_y = t[(1, 3)];
    let p_z = t[(2, 3)];

    let mut count = 0usize;

    // T_target-derived SP6 inputs.
    let p_16_x = p_x;
    let p_16_y = p_y;
    let p_16_z = p_z - m.d1;
    let r_06_axes5_x = -2.15550654822876e-33*r_00 - 2.46519032881566e-32*r_01 + 1.0*r_02;
    let r_06_axes5_y = -2.15550654822876e-33*r_10 - 2.46519032881566e-32*r_11 + 1.0*r_12;
    let r_06_axes5_z = -2.15550654822876e-33*r_20 - 2.46519032881566e-32*r_21 + 1.0*r_22;

    const H_SP_0: [f64; 3] = [0.0, -1.0, 6.123233995736766e-17];
    const K_SP_0: [f64; 3] = [-0.0, -0.0, -1.0];
    const K_SP_1: [f64; 3] = [0.0, -1.2246467991473532e-16, -1.0];
    let neg_p_5: [f64; 3] = [-0.0, m.neg_p5_y, -1.3877787807814457e-17];
    const NEG_AXES_5: [f64; 3] = [-1.2246467991473532e-16, 1.0, -1.8369701987210297e-16];

    let p_16 = [p_16_x, p_16_y, p_16_z];
    let r_06_axes5 = [r_06_axes5_x, r_06_axes5_y, r_06_axes5_z];
    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, neg_p_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, m.d4, 0.0, policy.subproblem_numerical, policy.subproblem_degeneracy, &mut theta15);

    for &(q1, q5) in theta15.iter().take(n15) {

        // NOTE: the literal constants below are the machine-generated symbolic
        // coefficients emitted by `ssik build`, copied verbatim from `<arm>_ik.py`
        // (mechanically translated; most are rounding residue of cos(pi/2)).
        // SP1 for theta14 = q1+q2+q3+q4 (sum of parallel-axis rotations).
        let th14_x0 = libm::sin(q5);
        let th14_x1 = libm::cos(q5);
        let th14_x2 = -th14_x0 + 1.22464679914735e-16*th14_x1;
        let th14_x3 = -2.15550654822876e-33*r_20 - 2.46519032881566e-32*r_21 + 1.0*r_22;
        let th14_x4 = 1.22464679914735e-16*r_00 + 1.0*r_01 + 1.83697019872103e-16*r_02;
        let th14_x5 = libm::sin(q1);
        let th14_x6 = 1.83697019872103e-16*th14_x5;
        let th14_x7 = th14_x4*th14_x6;
        let th14_x8 = 1.22464679914735e-16*r_00 + 1.83697019872103e-16*r_01 - 1.0*r_02;
        let th14_x9 = th14_x5*th14_x8;
        let th14_x10 = 1.0*r_00 - 1.22464679914735e-16*r_01 + 1.22464679914735e-16*r_02;
        let th14_x11 = 1.22464679914735e-16*th14_x5;
        let th14_x12 = th14_x10*th14_x11;
        let th14_x13 = 1.22464679914735e-16*r_10 + 1.0*r_11 + 1.83697019872103e-16*r_12;
        let th14_x14 = libm::cos(q1);
        let th14_x15 = 1.83697019872103e-16*th14_x14;
        let th14_x16 = th14_x13*th14_x15;
        let th14_x17 = 1.22464679914735e-16*r_10 + 1.83697019872103e-16*r_11 - 1.0*r_12;
        let th14_x18 = th14_x14*th14_x17;
        let th14_x19 = 1.0*r_10 - 1.22464679914735e-16*r_11 + 1.22464679914735e-16*r_12;
        let th14_x20 = 1.22464679914735e-16*th14_x14;
        let th14_x21 = th14_x19*th14_x20;
        let th14_x22 = -th14_x12 + th14_x16 - th14_x18 + th14_x21 - th14_x7 + th14_x9;
        let th14_x23 = th14_x10*th14_x20 + th14_x11*th14_x19 + th14_x13*th14_x6 - th14_x14*th14_x8 + th14_x15*th14_x4 - th14_x17*th14_x5;
        let th14_x24 = 1.22464679914735e-16*th14_x0 + 1.0*th14_x1;
        let theta14 = libm::atan2(th14_x2*th14_x3 + th14_x22*(-6.12323399573677e-17*th14_x0 + 7.49879891330929e-33*th14_x1) + th14_x23*(-7.49879891330929e-33*th14_x0 - 6.12323399573677e-17*th14_x1 - 6.12323399573676e-17), th14_x2*th14_x23 + th14_x22*(7.49879891330929e-33 - th14_x24) + th14_x3*(1.49975978266186e-32*th14_x0 + 1.22464679914735e-16*th14_x1 + 6.12323399573676e-17) - (th14_x24 - 3.74939945665464e-33)*(-1.31986709741476e-49*r_20 - 1.50949372273656e-48*r_21 + 6.12323399573677e-17*r_22 + th14_x12 - th14_x16 + th14_x18 - th14_x21 + th14_x7 - th14_x9));
        // SP1 for q6 (wrist roll-2): closed-form atan2.
        let q6_x0 = libm::cos(q5);
        let q6_x1 = 1.0*r_00 - 1.22464679914735e-16*r_01 + 1.22464679914735e-16*r_02;
        let q6_x2 = libm::sin(q1);
        let q6_x3 = 1.0*q6_x2;
        let q6_x4 = 1.0*r_10 - 1.22464679914735e-16*r_11 + 1.22464679914735e-16*r_12;
        let q6_x5 = libm::cos(q1);
        let q6_x6 = 1.0*q6_x5;
        let q6_x7 = q6_x1*q6_x3 - q6_x4*q6_x6 + 6.12323399573677e-17*r_20 - 7.49879891330929e-33*r_21 + 7.49879891330929e-33*r_22;
        let q6_x8 = libm::sin(q5);
        let q6_x9 = 1.0*q6_x8;
        let q6_x10 = 1.22464679914735e-16*q6_x0;
        let q6_x11 = 1.22464679914735e-16*r_00 + 1.0*r_01 + 1.83697019872103e-16*r_02;
        let q6_x12 = 1.22464679914735e-16*r_10 + 1.0*r_11 + 1.83697019872103e-16*r_12;
        let q6_x13 = q6_x11*q6_x3 - q6_x12*q6_x6 + 7.49879891330929e-33*r_20 + 6.12323399573677e-17*r_21 + 1.12481983699639e-32*r_22;
        let q6_x14 = 6.12323399573677e-17*r_22;
        let q6_x15 = 1.22464679914735e-16*r_00 + 1.83697019872103e-16*r_01 - 1.0*r_02;
        let q6_x16 = q6_x6*(1.22464679914735e-16*r_10 + 1.83697019872103e-16*r_11 - 1.0*r_12);
        let q6_x17 = -q6_x14 + q6_x15*q6_x3 - q6_x16 + 7.49879891330929e-33*r_20 + 1.12481983699639e-32*r_21;
        let q6_x18 = 1.0*q6_x0;
        let q6 = libm::atan2(q6_x13*(q6_x10 - q6_x9 + 9.18338008663387e-49) + q6_x17*(1.49975978266186e-32*q6_x0 - 1.83697019872103e-16*q6_x8 - 7.49879891330929e-33) + q6_x7*(-6.12323399573676e-17*q6_x0 - 6.12323399573677e-17), q6_x13*(q6_x10 - 6.12323399573677e-17) + q6_x17*(-q6_x18 - 7.49879891330929e-33) + q6_x7*q6_x9 - (-q6_x18 - 1.22464679914735e-16*q6_x8 + 3.74939945665464e-33)*(-1.22464679914735e-16*q6_x1*q6_x2 - 1.83697019872103e-16*q6_x11*q6_x2 + 1.83697019872103e-16*q6_x12*q6_x5 - q6_x14 + 1.0*q6_x15*q6_x2 - q6_x16 + 1.22464679914735e-16*q6_x4*q6_x5 - 9.05148142787587e-51*r_20 + 1.3684555315672e-48*r_21));
        // d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
        let dinr_x0 = libm::sin(theta14);
        let dinr_x1 = libm::cos(q1);
        let dinr_x2 = 1.0*libm::sin(q1);
        let dinr_x3 = libm::cos(theta14);
        let dinr_x4 = libm::sin(q5);
        let dinr_x5 = m.d6*dinr_x4;
        let dinr_x6 = libm::cos(q5);
        let dinr_x7 = 1.22464679914735e-16 - 1.22464679914735e-16*dinr_x6;
        let dinr_x8 = m.d6*dinr_x7;
        let dinr_x9 = 1.0*dinr_x6 + 1.49975978266186e-32;
        let dinr_x10 = m.dinr_c10*dinr_x0;
        let dinr_x11 = 6.12323399573677e-17*dinr_x3 - 6.12323399573677e-17;
        let d_inner_x = -dinr_x0*dinr_x8 - m.d5*dinr_x0 + dinr_x1*p_x - dinr_x10*dinr_x9 + dinr_x2*p_y + dinr_x3*dinr_x5;
        let d_inner_y = dinr_x1*p_y + dinr_x10*dinr_x4 + dinr_x11*dinr_x8 - dinr_x2*p_x + m.dinr_cy*dinr_x3 + m.d6*dinr_x6 - m.dinr_cy;
        let d_inner_z = dinr_x0*dinr_x5 - 1.38777878078145e-17*dinr_x11*dinr_x7 + m.d6*dinr_x11*dinr_x9 + m.d5*dinr_x3 + dinr_x8*(1.0*dinr_x3 + 3.74939945665464e-33) + 1.0*p_z - m.d1;
        let d_elbow = libm::sqrt(d_inner_x*d_inner_x + d_inner_y*d_inner_y + d_inner_z*d_inner_z);
        // SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        let q3_r_sq = m.q3_r_sq;
        let q3_rhs = m.q3_rhs0 - 0.5 * d_elbow * d_elbow;
        let q3_phi = m.q3_phi0 + PI;
        let theta_q3_plus;
        let theta_q3_minus;
        if q3_r_sq < policy.axis_parallel * policy.axis_parallel {
            theta_q3_plus = 0.0;
            theta_q3_minus = 0.0;
        } else {
            let q3_r = libm::sqrt(q3_r_sq);
            if libm::fabs(q3_rhs) > q3_r + policy.subproblem_feasibility {
                let p = if q3_rhs > 0.0 { q3_phi } else { q3_phi + PI };
                theta_q3_plus = p;
                theta_q3_minus = p;
            } else {
                let clipped = (q3_rhs / q3_r).clamp(-1.0, 1.0);
                let delta = libm::acos(clipped);
                theta_q3_plus = q3_phi + delta;
                theta_q3_minus = q3_phi - delta;
            }
        }
                // LS fallback: theta = phi (or phi + pi if rhs < 0)
        for q3 in [theta_q3_plus, theta_q3_minus] {
            // SP1 for q2 (shoulder pitch): closed-form atan2.
            let q2_x0 = libm::sin(q3);
            let q2_x1 = libm::cos(q3);
            let q2_x2 = m.a3*q2_x0 + m.q2_c0*q2_x1;
            let q2_x3 = m.q2_c0*q2_x0 - m.a3*q2_x1 - m.a2;
            let q2 = libm::atan2(d_inner_x*q2_x2 + d_inner_y*(m.q2_c1*q2_x0 - m.q2_c2*q2_x1 - m.q2_c3) + d_inner_z*q2_x3, d_inner_x*q2_x3 + d_inner_y*(-m.q2_c2*q2_x0 - m.d4) + d_inner_z*(m.q2_c0 - q2_x2) - (m.d4 - m.q2_c1*q2_x1)*(-1.0*d_inner_y + 6.12323399573677e-17*d_inner_z));
            let q4 = libm::remainder(theta14 - q2 - q3, TWO_PI);
            if count < MAX_CANDIDATES {
                out[count] = [q1, q2, q3, q4, q5, q6];
                count += 1;
            }
        }
    }

    count
}
