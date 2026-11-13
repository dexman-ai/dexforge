//! The model surface a driver crate implements.
//!
//! [`ArmModel`] is the whole of what dexhal asks of an arm's generated code:
//! the *nominal*, stateless flange primitives (forward kinematics, geometric
//! Jacobian, the driver's own inverse kinematics and, optionally, rigid-body
//! dynamics) plus a few static facts. The implementation of every one of them
//! lives in the driver crate; this module only fixes their shape.
//!
//! ```
//! use dexhal::math::model::{ArmModel, Pose};
//! use dexhal::math::tolerances::TolerancePolicy;
//!
//! /// Zero-sized marker for one arm, as found in a driver's `arm.rs`.
//! struct DemoArm;
//!
//! impl ArmModel<7, 1> for DemoArm {
//!     const NAME: &'static str = "demo";
//!
//!     fn flange_fk(_q: &[f64; 7]) -> Pose {
//!         Pose::identity() // driver: ik_solver::fk(q)
//!     }
//!
//!     fn flange_jacobian(_q: &[f64; 7]) -> [[f64; 7]; 6] {
//!         [[0.0; 7]; 6] // driver: generated_kinematics::jacobian(q)
//!     }
//!
//!     fn flange_solve(
//!         _target: &Pose,
//!         _max_solutions: Option<usize>,
//!         _q_seed: Option<&[f64; 7]>,
//!         _respect_limits: bool,
//!         _allow_refinement: bool,
//!         _policy: TolerancePolicy,
//!         _refinement_max_iters: usize,
//!     ) -> [Option<[f64; 7]>; 1] {
//!         [None] // driver: ik_solver::solve(..)
//!     }
//! }
//!
//! assert_eq!(DemoArm::flange_fk(&[0.0; 7]), Pose::identity());
//! assert!(!DemoArm::HAS_DYNAMICS);
//! ```
//!
//! Runtime state — TCP, payload, gravity direction, calibration — is not part
//! of the model; it is carried by [`crate::math::arm::Arm`].

use nalgebra::SMatrix;

use crate::math::postprocess::JointLimit;
use crate::math::tolerances::TolerancePolicy;

/// A 4×4 homogeneous SE(3) pose, matching the per-arm `fk` / `solve` type.
pub type Pose = SMatrix<f64, 4, 4>;

/// One arm's generated model: a zero-sized marker type per arm.
///
/// `DOF` is the number of actuated joints and `MAX` the capacity of the IK
/// solution array (both fixed per arm). All joints are revolute and ordered
/// base → flange.
pub trait ArmModel<const DOF: usize, const MAX: usize> {
    /// Human-readable model name reported in the manifest.
    const NAME: &'static str = "";

    /// Position limits per joint (`None` = unlimited / unknown).
    const JOINT_LIMITS: [JointLimit; DOF] = [None; DOF];

    /// Set to `true` by drivers that implement all four dynamics hooks
    /// (`flange_gravity`, `flange_mass_matrix`, `flange_coriolis`,
    /// `flange_inverse_dynamics`). Without it the zero defaults are
    /// indistinguishable from a real model, so [`crate::math::arm::Arm`]
    /// refuses to answer dynamics queries.
    const HAS_DYNAMICS: bool = false;

    /// Forward kinematics of the flange (base → end-effector link).
    fn flange_fk(q: &[f64; DOF]) -> Pose;

    /// Geometric Jacobian `[Jv; Jw]` (6 × DOF) of the flange origin, in the
    /// base frame.
    fn flange_jacobian(q: &[f64; DOF]) -> [[f64; DOF]; 6];

    /// The driver's inverse kinematics for a flange target.
    ///
    /// The driver owns the whole solve: the closed form, FK verification
    /// against `policy.subproblem_numerical`, optional refinement,
    /// de-duplication, joint limits and ordering. dexhal relies on two
    /// properties of the result:
    ///
    /// * with `q_seed`, solutions are ordered nearest-first (wrap-to-π joint
    ///   distance), so the first filled slot is the branch closest to the seed;
    /// * with `respect_limits`, every solution lies inside the arm's limits.
    #[allow(clippy::too_many_arguments)]
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; DOF]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; DOF]>; MAX];

    /// Gravity torque `G(q)` of the arm's own links for a base-frame `gravity`.
    #[inline]
    fn flange_gravity(_q: &[f64; DOF], _gravity: &[f64; 3]) -> [f64; DOF] {
        [0.0; DOF]
    }

    /// Joint-space inertia `M(q)` of the arm's own links.
    #[inline]
    fn flange_mass_matrix(_q: &[f64; DOF]) -> [[f64; DOF]; DOF] {
        [[0.0; DOF]; DOF]
    }

    /// Coriolis and centrifugal torque `C(q, qd) qd` of the arm's own links.
    #[inline]
    fn flange_coriolis(_q: &[f64; DOF], _qd: &[f64; DOF]) -> [f64; DOF] {
        [0.0; DOF]
    }

    /// Inverse dynamics `M(q) qdd + C(q, qd) qd + G(q)` of the arm's own links,
    /// i.e. the driver's generated `inverse_dynamics(q, qd, qdd, gravity)`.
    #[inline]
    fn flange_inverse_dynamics(
        _q: &[f64; DOF],
        _qd: &[f64; DOF],
        _qdd: &[f64; DOF],
        _gravity: &[f64; 3],
    ) -> [f64; DOF] {
        [0.0; DOF]
    }
}
