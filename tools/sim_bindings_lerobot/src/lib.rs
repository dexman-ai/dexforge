//! Python bindings for the LeRobot OEM repo (SO-ARM101).
//!
//! Same code as `tools/sim_bindings` (the shared `common.rs`), built for this vendor's arms, plus
//! `so101.ik_project`: the five-joint arm reaches only a 5-D subset of poses, so a target from a
//! camera or a teleop device needs a nearest-reachable solve, which the exact `solve` /
//! `ik_nearest` do not do.

include!("../../sim_bindings/src/common.rs");

use dexforge_lerobot_core::so101::model::{solve_nearest, DEFAULT_ROT_WEIGHT, NEAREST_MAX_ITERS};
use dexhal::math::{pose_from_xyz_quat, TolerancePolicy as NearestPolicy};

arm_module!(so101_py, "so101", dexforge_lerobot_core, so101, So101, So101Driver);

/// Closest reachable joint vector to `pose` (`[x, y, z, qx, qy, qz, qw]`, base frame), as
/// `(q, position_error_m, orientation_error_rad)`; the errors are what FK says the returned `q`
/// misses the target by, so the caller decides whether that is close enough. `q_seed` selects the
/// branch (nearest to it) and starts the fit. `rot_weight` weighs orientation against position in
/// the least-squares metric (metres per unit of rotation-matrix error). `None` only for a
/// non-finite target. Exact poses come back with zero error.
#[pyfunction]
#[pyo3(signature = (pose, q_seed, respect_limits = true, rot_weight = DEFAULT_ROT_WEIGHT))]
fn ik_project(
    pose: Vec<f64>,
    q_seed: Vec<f64>,
    respect_limits: bool,
    rot_weight: f64,
) -> PyResult<Option<(Vec<f64>, f64, f64)>> {
    let pose: [f64; 7] = pose
        .try_into()
        .map_err(|_| PyValueError::new_err("pose must have 7 elements [x, y, z, qx, qy, qz, qw]"))?;
    let seed: [f64; 5] = q_seed
        .try_into()
        .map_err(|_| PyValueError::new_err("q_seed must have 5 elements"))?;
    if !(rot_weight.is_finite() && rot_weight > 0.0) {
        return Err(PyValueError::new_err("rot_weight must be finite and positive"));
    }
    let target = pose_from_xyz_quat(&pose)
        .map_err(|e| PyValueError::new_err(format!("pose: {e:?}")))?;
    Ok(solve_nearest(
        &target,
        Some(&seed),
        respect_limits,
        rot_weight,
        NearestPolicy::DEFAULT,
        NEAREST_MAX_ITERS,
    )
    .map(|n| (n.q.to_vec(), n.position_error, n.orientation_error)))
}

#[pymodule]
fn dexforge_lerobot(m: &Bound<'_, PyModule>) -> PyResult<()> {
	so101_py::register(m)?;
	let so101 = m.getattr("so101")?.downcast_into::<PyModule>()?;
	so101.add_function(wrap_pyfunction!(ik_project, &so101)?)?;
	m.add_function(wrap_pyfunction!(state_layout, m)?)?;
	m.add_function(wrap_pyfunction!(command_layout, m)?)?;
	Ok(())
}
