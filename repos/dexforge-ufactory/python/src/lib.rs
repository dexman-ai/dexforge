//! Python bindings for the UFactory OEM repo.

use nalgebra::SMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use dexforge_ufactory_core::xarm7::ik_solver::{fk as xarm7_fk_core, solve as xarm7_solve_core, TolerancePolicy, DOF, MAX_SOLUTIONS};

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
fn xarm7_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
	fk_impl::<{ DOF }, _>("xarm7", q, xarm7_fk_core)
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
fn xarm7_solve(
	T_target: Bound<'_, PyAny>,
	max_solutions: Option<usize>,
	q_seed: Option<Vec<f64>>,
	respect_limits: bool,
	allow_refinement: bool,
	fk_atol: f64,
	dedup_atol: f64,
	refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
	solve_impl::<{ DOF }, { MAX_SOLUTIONS }, _>(
		"xarm7",
		T_target,
		max_solutions,
		q_seed,
		respect_limits,
		allow_refinement,
		fk_atol,
		dedup_atol,
		refinement_max_iters,
		|tf, max_sol, seed, respect, refine, fk_a, dedup_a, max_iters| {
			xarm7_solve_core(
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

#[pymodule]
fn dexforge_ufactory(m: &Bound<'_, PyModule>) -> PyResult<()> {
	let xarm7_mod = PyModule::new(m.py(), "xarm7")?;
	xarm7_mod.add_function(wrap_pyfunction!(xarm7_fk, &xarm7_mod)?)?;
	xarm7_mod.add_function(wrap_pyfunction!(xarm7_solve, &xarm7_mod)?)?;
	m.add_submodule(&xarm7_mod)?;
	Ok(())
}