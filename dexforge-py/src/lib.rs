//! Python bindings for `dexforge` analytical inverse kinematics.
//!
//! Exposes per-arm submodules that mirror the Python `ssik` prebuilt artifact
//! interface (e.g. `ssik.prebuilt.ur5_ik`):
//!
//! ```python
//! import numpy as np
//! import dexforge_py
//!
//! T = np.eye(4)           # 4x4 SE(3) target pose
//! q = [0.0] * 6          # joint angles in radians
//!
//! solutions = dexforge_py.ur5.solve(T, respect_limits=False)
//! T_fk      = dexforge_py.ur5.fk(q)
//! ```
//!
//! # Adding a new arm driver
//!
//! 1. Implement the driver in `dexforge/src/drivers/<arm>/ik_solver.rs`
//!    (expose `fk`, `solve`, and `TolerancePolicy::DEFAULT`).
//! 2. Add a `use` statement for the driver alias.
//! 3. Add `#[pyfunction]` + `#[pyo3(name = "fk")]` / `#[pyo3(name = "solve")]`
//!    wrappers calling `fk_impl` and `solve_impl`.
//! 4. In `dexforge_py()`, create a submodule, register both functions, and
//!    call `m.add_submodule`.
//! 5. Create the corresponding Python shim in `ssik/prebuilt/<arm>_ik.py`.

use nalgebra::SMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use dexforge_examples::drivers::franka_panda::ik_solver as franka;
use dexforge_examples::drivers::piper::ik_solver as piper;
use dexforge_examples::drivers::rizon10::ik_solver as rizon10;
use dexforge_examples::drivers::rizon4::ik_solver as rizon4;
use dexforge_examples::drivers::ur5::ik_solver as ur5;
use dexforge_examples::drivers::xarm7::ik_solver as xarm7;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract a 4×4 SE(3) matrix from a Python object.
///
/// Accepts both numpy arrays (via `.tolist()`) and plain Python lists of lists.
fn extract_matrix_4x4(obj: &Bound<'_, PyAny>) -> PyResult<SMatrix<f64, 4, 4>> {
    // numpy ndarrays have `.tolist()`; plain lists do not.
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

/// Convert a `SMatrix<f64, 4, 4>` to a Python-friendly nested `Vec`.
fn matrix_to_nested(tf: &SMatrix<f64, 4, 4>) -> Vec<Vec<f64>> {
    (0..4)
        .map(|i| (0..4).map(|j| tf[(i, j)]).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Generic wrappers
// ---------------------------------------------------------------------------

/// Generic forward kinematics wrapper.
///
/// Templatized on DOF and the FK delegate function.
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

/// Generic inverse kinematics wrapper.
///
/// Templatized on DOF, MAX_SOLUTIONS, and the solve delegate function.
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

// ---------------------------------------------------------------------------
// Per-arm Python functions: UR5
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for UR5.
#[pyfunction]
#[pyo3(name = "fk")]
fn ur5_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    fk_impl::<{ ur5::DOF }, _>("ur5", q, ur5::fk)
}

/// Compute analytical inverse kinematics for UR5.
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
    solve_impl::<{ ur5::DOF }, 16, _>(
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
            ur5::solve(
                tf,
                max_sol,
                seed,
                respect,
                refine,
                ur5::TolerancePolicy {
                    subproblem_numerical: fk_a,
                    subproblem_dedup: dedup_a,
                    ..ur5::TolerancePolicy::DEFAULT
                },
                max_iters,
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Per-arm Python functions: Piper
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for AgileX Piper.
#[pyfunction]
#[pyo3(name = "fk")]
fn piper_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    fk_impl::<{ piper::DOF }, _>("piper", q, piper::fk)
}

/// Compute analytical inverse kinematics for AgileX Piper.
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
fn piper_solve(
    T_target: Bound<'_, PyAny>,
    max_solutions: Option<usize>,
    q_seed: Option<Vec<f64>>,
    respect_limits: bool,
    allow_refinement: bool,
    fk_atol: f64,
    dedup_atol: f64,
    refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
    solve_impl::<{ piper::DOF }, 16, _>(
        "piper",
        T_target,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        fk_atol,
        dedup_atol,
        refinement_max_iters,
        |tf, max_sol, seed, respect, refine, fk_a, dedup_a, max_iters| {
            piper::solve(
                tf,
                max_sol,
                seed,
                respect,
                refine,
                piper::TolerancePolicy {
                    subproblem_numerical: fk_a,
                    subproblem_dedup: dedup_a,
                    ..piper::TolerancePolicy::DEFAULT
                },
                max_iters,
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Per-arm Python functions: Franka Panda
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for Franka Emika Panda.
#[pyfunction]
#[pyo3(name = "fk")]
fn franka_panda_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    fk_impl::<{ franka::DOF }, _>("franka_panda", q, franka::fk)
}

/// Compute analytical inverse kinematics for Franka Emika Panda.
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
fn franka_panda_solve(
    T_target: Bound<'_, PyAny>,
    max_solutions: Option<usize>,
    q_seed: Option<Vec<f64>>,
    respect_limits: bool,
    allow_refinement: bool,
    fk_atol: f64,
    dedup_atol: f64,
    refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
    solve_impl::<{ franka::DOF }, { franka::MAX_SOLUTIONS }, _>(
        "franka_panda",
        T_target,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        fk_atol,
        dedup_atol,
        refinement_max_iters,
        |tf, max_sol, seed, respect, refine, fk_a, dedup_a, max_iters| {
            franka::solve(
                tf,
                max_sol,
                seed,
                respect,
                refine,
                franka::TolerancePolicy {
                    subproblem_numerical: fk_a,
                    subproblem_dedup: dedup_a,
                    ..franka::TolerancePolicy::DEFAULT
                },
                max_iters,
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Per-arm Python functions: xArm7
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for UFactory xArm7.
#[pyfunction]
#[pyo3(name = "fk")]
fn xarm7_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    fk_impl::<{ xarm7::DOF }, _>("xarm7", q, xarm7::fk)
}

/// Compute analytical inverse kinematics for UFactory xArm7.
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
    solve_impl::<{ xarm7::DOF }, { xarm7::MAX_SOLUTIONS }, _>(
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
            xarm7::solve(
                tf,
                max_sol,
                seed,
                respect,
                refine,
                xarm7::TolerancePolicy {
                    subproblem_numerical: fk_a,
                    subproblem_dedup: dedup_a,
                    ..xarm7::TolerancePolicy::DEFAULT
                },
                max_iters,
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Per-arm Python functions: Rizon4
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for Flexiv Rizon 4.
#[pyfunction]
#[pyo3(name = "fk")]
fn rizon4_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    fk_impl::<{ rizon4::DOF }, _>("rizon4", q, rizon4::fk)
}

/// Compute analytical inverse kinematics for Flexiv Rizon 4.
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
fn rizon4_solve(
    T_target: Bound<'_, PyAny>,
    max_solutions: Option<usize>,
    q_seed: Option<Vec<f64>>,
    respect_limits: bool,
    allow_refinement: bool,
    fk_atol: f64,
    dedup_atol: f64,
    refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
    solve_impl::<{ rizon4::DOF }, { rizon4::MAX_SOLUTIONS }, _>(
        "rizon4",
        T_target,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        fk_atol,
        dedup_atol,
        refinement_max_iters,
        |tf, max_sol, seed, respect, refine, fk_a, dedup_a, max_iters| {
            rizon4::solve(
                tf,
                max_sol,
                seed,
                respect,
                refine,
                rizon4::TolerancePolicy {
                    subproblem_numerical: fk_a,
                    subproblem_dedup: dedup_a,
                    ..rizon4::TolerancePolicy::DEFAULT
                },
                max_iters,
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Per-arm Python functions: Rizon10
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for Flexiv Rizon 10.
#[pyfunction]
#[pyo3(name = "fk")]
fn rizon10_fk(q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    fk_impl::<{ rizon10::DOF }, _>("rizon10", q, rizon10::fk)
}

/// Compute analytical inverse kinematics for Flexiv Rizon 10.
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
fn rizon10_solve(
    T_target: Bound<'_, PyAny>,
    max_solutions: Option<usize>,
    q_seed: Option<Vec<f64>>,
    respect_limits: bool,
    allow_refinement: bool,
    fk_atol: f64,
    dedup_atol: f64,
    refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
    solve_impl::<{ rizon10::DOF }, { rizon10::MAX_SOLUTIONS }, _>(
        "rizon10",
        T_target,
        max_solutions,
        q_seed,
        respect_limits,
        allow_refinement,
        fk_atol,
        dedup_atol,
        refinement_max_iters,
        |tf, max_sol, seed, respect, refine, fk_a, dedup_a, max_iters| {
            rizon10::solve(
                tf,
                max_sol,
                seed,
                respect,
                refine,
                rizon10::TolerancePolicy {
                    subproblem_numerical: fk_a,
                    subproblem_dedup: dedup_a,
                    ..rizon10::TolerancePolicy::DEFAULT
                },
                max_iters,
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python module exposing dexforge analytical IK solvers.
///
/// Arms are exposed as submodules so callers use `dexforge_py.<arm>.solve`
/// and `dexforge_py.<arm>.fk`, mirroring the `ssik.prebuilt.<arm>_ik`
/// module structure.
///
/// To add a new arm, create a submodule block below and register it with
/// `m.add_submodule`.
#[pymodule]
fn dexforge_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── UR5 ──────────────────────────────────────────────────────────────────
    let ur5_mod = PyModule::new(m.py(), "ur5")?;
    ur5_mod.add_function(wrap_pyfunction!(ur5_fk, &ur5_mod)?)?;
    ur5_mod.add_function(wrap_pyfunction!(ur5_solve, &ur5_mod)?)?;
    m.add_submodule(&ur5_mod)?;

    // ── Piper ─────────────────────────────────────────────────────────────────
    let piper_mod = PyModule::new(m.py(), "piper")?;
    piper_mod.add_function(wrap_pyfunction!(piper_fk, &piper_mod)?)?;
    piper_mod.add_function(wrap_pyfunction!(piper_solve, &piper_mod)?)?;
    m.add_submodule(&piper_mod)?;

    // ── Franka Panda ──────────────────────────────────────────────────────────
    let franka_mod = PyModule::new(m.py(), "franka_panda")?;
    franka_mod.add_function(wrap_pyfunction!(franka_panda_fk, &franka_mod)?)?;
    franka_mod.add_function(wrap_pyfunction!(franka_panda_solve, &franka_mod)?)?;
    m.add_submodule(&franka_mod)?;

    // ── xArm7 ─────────────────────────────────────────────────────────────────
    let xarm7_mod = PyModule::new(m.py(), "xarm7")?;
    xarm7_mod.add_function(wrap_pyfunction!(xarm7_fk, &xarm7_mod)?)?;
    xarm7_mod.add_function(wrap_pyfunction!(xarm7_solve, &xarm7_mod)?)?;
    m.add_submodule(&xarm7_mod)?;

    // ── Rizon 4 ───────────────────────────────────────────────────────────────
    let rizon4_mod = PyModule::new(m.py(), "rizon4")?;
    rizon4_mod.add_function(wrap_pyfunction!(rizon4_fk, &rizon4_mod)?)?;
    rizon4_mod.add_function(wrap_pyfunction!(rizon4_solve, &rizon4_mod)?)?;
    m.add_submodule(&rizon4_mod)?;

    // ── Rizon 10 ──────────────────────────────────────────────────────────────
    let rizon10_mod = PyModule::new(m.py(), "rizon10")?;
    rizon10_mod.add_function(wrap_pyfunction!(rizon10_fk, &rizon10_mod)?)?;
    rizon10_mod.add_function(wrap_pyfunction!(rizon10_solve, &rizon10_mod)?)?;
    m.add_submodule(&rizon10_mod)?;

    Ok(())
}
