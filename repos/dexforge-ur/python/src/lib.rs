//! Python bindings for the Universal Robots OEM repo.

use nalgebra::SMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pyo3::types::PyDict;

use dexforge_ur_core::ur5::generated_dynamics as dyn5;
use dexforge_ur_core::ur5::ik_solver::{fk as ur5_fk_core, solve as ur5_solve_core, TolerancePolicy, DOF};
use dexforge_ur_core::ur5::Ur5Driver;
use dexforge_ur_core::ur5::{Ur5, DOF as UR5_DOF, MAX_SOLUTIONS};
use dexhal::math::{
	Arm, ArmDriver, ArmMath, CoreCommand, CoreCommandViewMut, CoreState, CoreStateViewMut,
	IK_DEFAULT_MAX_JOINT_STEP, MAX_DOF, STANDARD_GRAVITY,
};

fn extract_matrix_4x4(obj: &Bound<'_, PyAny>) -> PyResult<SMatrix<f64, 4, 4>> {
	let rows: Vec<Vec<f64>> = if let Ok(lst) = obj.call_method0("tolist") {
		lst.extract()?
	} else {
		obj.extract()?
	};

	if rows.len() != 4 || rows.iter().any(|r| r.len() != 4) {
		return Err(PyValueError::new_err(
			"T_target must be a 4×4 matrix (list-of-lists or numpy array)",
		));
	}

	let mut data = [0.0f64; 16];
	for i in 0..4 {
		for j in 0..4 {
			data[i * 4 + j] = rows[i][j];
		}
	}
	Ok(SMatrix::<f64, 4, 4>::from_row_slice(&data))
}

fn matrix_to_nested(tf: &SMatrix<f64, 4, 4>) -> Vec<Vec<f64>> {
	(0..4)
		.map(|i| (0..4).map(|j| tf[(i, j)]).collect())
		.collect()
}

#[inline]
fn fk_impl<const DOF: usize, F>(arm_name: &str, q: Vec<f64>, fk_fn: F) -> PyResult<Vec<Vec<f64>>>
where
	F: FnOnce(&[f64; DOF]) -> SMatrix<f64, 4, 4>,
{
	if q.len() != DOF {
		return Err(PyValueError::new_err(format!(
			"{} requires {} joint angles, got {}",
			arm_name,
			DOF,
			q.len()
		)));
	}
	let arr: [f64; DOF] = q.try_into().unwrap();
	let tf = fk_fn(&arr);
	Ok(matrix_to_nested(&tf))
}

#[inline]
#[allow(non_snake_case)]
fn solve_impl<const DOF: usize, const MAX_SOLUTIONS: usize, F>(
	arm_name: &str,
	T_target: Bound<'_, PyAny>,
	max_solutions: Option<usize>,
	q_seed: Option<Vec<f64>>,
	respect_limits: bool,
	allow_refinement: bool,
	fk_atol: f64,
	dedup_atol: f64,
	refinement_max_iters: usize,
	solve_fn: F,
) -> PyResult<Vec<Vec<f64>>>
where
	F: FnOnce(
		&SMatrix<f64, 4, 4>,
		Option<usize>,
		Option<&[f64; DOF]>,
		bool,
		bool,
		f64,
		f64,
		usize,
	) -> [Option<[f64; DOF]>; MAX_SOLUTIONS],
{
	let tf = extract_matrix_4x4(&T_target)?;

	let seed: Option<[f64; DOF]> = match q_seed {
		None => None,
		Some(v) => {
			if v.len() != DOF {
				return Err(PyValueError::new_err(format!(
					"q_seed for {} must have {} elements, got {}",
					arm_name,
					DOF,
					v.len()
				)));
			}
			Some(v.try_into().unwrap())
		}
	};

	let solutions: Vec<Vec<f64>> = solve_fn(
		&tf,
		max_solutions,
		seed.as_ref(),
		respect_limits,
		allow_refinement,
		fk_atol,
		dedup_atol,
		refinement_max_iters,
	)
	.iter()
	.filter_map(|s| s.map(|q| q.to_vec()))
	.collect();

	Ok(solutions)
}

#[pyfunction]
#[pyo3(name = "fk")]
fn ur5_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
	fk_impl::<{ DOF }, _>("ur5", q, ur5_fk_core)
}

#[pyfunction]
#[pyo3(name = "solve")]
#[pyo3(signature = (
	T_target,
	max_solutions = None,
	q_seed = None,
	respect_limits = true,
	allow_refinement = false,
	fk_atol = 1e-5,
	dedup_atol = 1e-3,
	refinement_max_iters = 15,
))]
#[allow(non_snake_case)]
fn ur5_solve(
	T_target: Bound<'_, PyAny>,
	max_solutions: Option<usize>,
	q_seed: Option<Vec<f64>>,
	respect_limits: bool,
	allow_refinement: bool,
	fk_atol: f64,
	dedup_atol: f64,
	refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
	solve_impl::<{ DOF }, 16, _>(
		"ur5",
		T_target,
		max_solutions,
		q_seed,
		respect_limits,
		allow_refinement,
		fk_atol,
		dedup_atol,
		refinement_max_iters,
		|tf, max_sol, seed, respect, refine, fk_a, dedup_a, max_iters| {
			ur5_solve_core(
				tf,
				max_sol,
				seed,
				respect,
				refine,
				TolerancePolicy {
					subproblem_numerical: fk_a,
					subproblem_dedup: dedup_a,
					..TolerancePolicy::DEFAULT
				},
				max_iters,
			)
		},
	)
}

fn joint_vec(name: &str, v: Vec<f64>) -> PyResult<[f64; DOF]> {
	let n = v.len();
	v.try_into().map_err(|_| {
		PyValueError::new_err(format!("{name} must have {DOF} elements, got {n}"))
	})
}

fn nested<const N: usize>(m: &[[f64; N]; N]) -> Vec<Vec<f64>> {
	m.iter().map(|r| r.to_vec()).collect()
}

/// Gravity in the base frame (z up), m/s^2: the same constant the dexhal `Arm` model uses.
const DEFAULT_GRAVITY: [f64; 3] = [0.0, 0.0, -STANDARD_GRAVITY];

fn gravity_vec(g: Option<Vec<f64>>) -> PyResult<[f64; 3]> {
	match g {
		None => Ok(DEFAULT_GRAVITY),
		Some(v) => v
			.try_into()
			.map_err(|_| PyValueError::new_err("g must have 3 elements")),
	}
}

#[pyfunction]
#[pyo3(name = "jacobian")]
fn ur5_jacobian(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
	Ok(nested(&dyn5::jacobian(&joint_vec("q", q)?)))
}

#[pyfunction]
#[pyo3(name = "gravity", signature = (q, g = None))]
fn ur5_gravity(q: Vec<f64>, g: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
	let g = gravity_vec(g)?;
	Ok(dyn5::gravity(&joint_vec("q", q)?, &g).to_vec())
}

#[pyfunction]
#[pyo3(name = "mass_matrix")]
fn ur5_mass_matrix(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
	Ok(nested(&dyn5::mass_matrix(&joint_vec("q", q)?)))
}

#[pyfunction]
#[pyo3(name = "coriolis")]
fn ur5_coriolis(q: Vec<f64>, qd: Vec<f64>) -> PyResult<Vec<f64>> {
	Ok(dyn5::coriolis(&joint_vec("q", q)?, &joint_vec("qd", qd)?).to_vec())
}

#[pyfunction]
#[pyo3(name = "inverse_dynamics", signature = (q, qd, qdd, g = None))]
fn ur5_inverse_dynamics(
	q: Vec<f64>,
	qd: Vec<f64>,
	qdd: Vec<f64>,
	g: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
	let g = gravity_vec(g)?;
	Ok(dyn5::inverse_dynamics(
		&joint_vec("q", q)?,
		&joint_vec("qd", qd)?,
		&joint_vec("qdd", qdd)?,
		&g,
	)
	.to_vec())
}

fn pad_q(q: &[f64; UR5_DOF]) -> [f64; MAX_DOF] {
	let mut out = [0.0; MAX_DOF];
	out[..UR5_DOF].copy_from_slice(q);
	out
}

/// TCP pose `[x, y, z, qx, qy, qz, qw]` (base frame) at `q`.
#[pyfunction]
#[pyo3(name = "fk_pose")]
fn ur5_fk_pose(q: Vec<f64>) -> PyResult<Vec<f64>> {
	let q = joint_vec("q", q)?;
	let arm = Arm::<Ur5, UR5_DOF, MAX_SOLUTIONS>::new();
	arm.forward_kinematics(&pad_q(&q))
		.map(|p| p.to_vec())
		.ok_or_else(|| PyValueError::new_err("q must be finite"))
}

/// Joint vector reaching `pose` (`[x, y, z, qx, qy, qz, qw]`) on the branch of
/// `q_seed`, or `None` when there is no solution or the nearest one is further than
/// the continuity bound `max_joint_step` (rad per joint; the dexhal default when omitted).
#[pyfunction]
#[pyo3(name = "ik_nearest", signature = (pose, q_seed, max_joint_step = None))]
fn ur5_ik_nearest(
	pose: Vec<f64>,
	q_seed: Vec<f64>,
	max_joint_step: Option<f64>,
) -> PyResult<Option<Vec<f64>>> {
	let pose: [f64; 7] = pose
		.try_into()
		.map_err(|_| PyValueError::new_err("pose must have 7 elements [x, y, z, qx, qy, qz, qw]"))?;
	let seed = joint_vec("q_seed", q_seed)?;
	let mut arm = Arm::<Ur5, UR5_DOF, MAX_SOLUTIONS>::new();
	if let Some(step) = max_joint_step {
		arm.set_ik_max_joint_step(step)
			.map_err(|e| PyValueError::new_err(format!("max_joint_step: {e:?}")))?;
	}
	Ok(arm
		.inverse_kinematics(&pose, &pad_q(&seed))
		.map(|q| q[..UR5_DOF].to_vec()))
}

/// Magnitude of the default gravity (m/s^2) used by `gravity`, `inverse_dynamics` and the dexhal `Arm`.
#[pyfunction]
#[pyo3(name = "standard_gravity")]
fn ur5_standard_gravity() -> f64 {
	STANDARD_GRAVITY
}

/// Default continuity bound (rad) of `ik_nearest`.
#[pyfunction]
#[pyo3(name = "ik_default_max_joint_step")]
fn ur5_ik_default_max_joint_step() -> f64 {
	IK_DEFAULT_MAX_JOINT_STEP
}

/// Driver manifest (`manifest.yaml`, validated by dexhal) as a dict.
#[pyfunction]
#[pyo3(name = "manifest")]
fn ur5_manifest(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
	let driver = Ur5Driver::new()
		.map_err(|e| PyValueError::new_err(format!("manifest invalid: {e:?}")))?;
	let m = driver.manifest();
	let n = m.active_dof_count as usize;
	let d = PyDict::new(py);
	d.set_item("name", m.name)?;
	d.set_item("dof", n)?;
	d.set_item("control_freq_hz", m.control_freq_hz)?;
	d.set_item("control_modes", m.control_modes)?;
	d.set_item("is_realtime_required", m.is_realtime_required)?;
	d.set_item("state_staleness_cycles", m.state_staleness_cycles)?;
	d.set_item("command_dispatch_cycles", m.command_dispatch_cycles)?;
	d.set_item("has_torque_sensing", m.has_torque_sensing)?;
	d.set_item("has_static_inertials", m.has_static_inertials)?;
	d.set_item("joint_limits_min", m.joint_limits_min[..n].to_vec())?;
	d.set_item("joint_limits_max", m.joint_limits_max[..n].to_vec())?;
	d.set_item("velocity_limits", m.velocity_limits[..n].to_vec())?;
	d.set_item("acceleration_limits", m.acceleration_limits[..n].to_vec())?;
	d.set_item("effort_limits", m.effort_limits[..n].to_vec())?;
	d.set_item("state_len", m.state_len())?;
	d.set_item("command_len", m.command_len())?;
	d.set_item("max_dof", MAX_DOF)?;
	Ok(d)
}

// Offsets are measured off the contract's own accessors (address arithmetic,
// no hard-coded layout), so they follow any change to the wire structs.
fn slot<const N: usize>(base: usize, field: &[f64; N]) -> (usize, usize) {
	((field.as_ptr() as usize - base) / size_of::<f64>(), N)
}

fn slot_scalar(base: usize, field: &mut f64) -> usize {
	(field as *mut f64 as usize - base) / size_of::<f64>()
}

/// `{field: (offset, length)}` of the state buffer, plus `"len"`.
#[pyfunction]
fn state_layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
	let mut buf = vec![0.0f64; CoreState::LEN];
	let base = buf.as_ptr() as usize;
	let mut v = CoreStateViewMut::new(&mut buf)
		.map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
	let d = PyDict::new(py);
	macro_rules! arr { ($($f:ident),*) => { $( d.set_item(stringify!($f), slot(base, v.$f()))?; )* } }
	macro_rules! sc { ($($f:ident),*) => { $( d.set_item(stringify!($f), (slot_scalar(base, v.$f()), 1usize))?; )* } }
	arr!(q, qd, tcp_pose, tcp_twist, torques, wrench, io_analog_in);
	sc!(
		in_contact, io_digital_in, gripper_width, gripper_force, gripper_has_object, timestamp,
		robot_mode, fault_code, native_reflex_fired
	);
	d.set_item("len", CoreState::LEN)?;
	Ok(d)
}

/// `{field: (offset, length)}` of the command buffer, plus `"len"`.
#[pyfunction]
fn command_layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
	let mut buf = vec![0.0f64; CoreCommand::LEN];
	let base = buf.as_ptr() as usize;
	let mut v = CoreCommandViewMut::new(&mut buf)
		.map_err(|e| PyValueError::new_err(format!("{e:?}")))?;
	let d = PyDict::new(py);
	macro_rules! arr { ($($f:ident),*) => { $( d.set_item(stringify!($f), slot(base, v.$f()))?; )* } }
	macro_rules! sc { ($($f:ident),*) => { $( d.set_item(stringify!($f), (slot_scalar(base, v.$f()), 1usize))?; )* } }
	arr!(
		target_q, target_qd, target_torques, target_pose, target_twist, target_wrench, task_frame,
		task_stiffness_k, task_damping_b, task_mass_m, joint_stiffness_k, joint_damping_b,
		reflex_threshold
	);
	sc!(
		reflex_action_code, io_digital_out_mask, io_digital_out_state, gripper_target_width,
		gripper_target_speed, gripper_max_force
	);
	d.set_item("len", CoreCommand::LEN)?;
	Ok(d)
}

#[pymodule]
fn dexforge_ur(m: &Bound<'_, PyModule>) -> PyResult<()> {
	let ur5_mod = PyModule::new(m.py(), "ur5")?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_fk, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_solve, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_jacobian, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_gravity, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_mass_matrix, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_coriolis, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_inverse_dynamics, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_manifest, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_fk_pose, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_standard_gravity, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_ik_nearest, &ur5_mod)?)?;
	ur5_mod.add_function(wrap_pyfunction!(ur5_ik_default_max_joint_step, &ur5_mod)?)?;
	m.add_submodule(&ur5_mod)?;
	m.add_function(wrap_pyfunction!(state_layout, m)?)?;
	m.add_function(wrap_pyfunction!(command_layout, m)?)?;
	Ok(())
}