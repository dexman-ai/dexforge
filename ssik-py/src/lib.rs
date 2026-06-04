//! Python bindings for `ssik-rust` analytical inverse kinematics.
//!
//! Exposes two functions whose signatures mirror the Python `ssik` prebuilt
//! artifact interface (e.g. `ssik.prebuilt.ur5_ik`), plus an `arm` name
//! parameter to select the robot:
//!
//! ```python
//! import numpy as np
//! import ssik_py
//!
//! T = np.eye(4)           # 4x4 SE(3) target pose
//! q = [0.0] * 6          # joint angles in radians
//!
//! solutions = ssik_py.solve("ur5", T, respect_limits=False)
//! T_fk      = ssik_py.fk("ur5", q)
//! ```
//!
//! # Adding a new arm driver
//!
//! 1. Implement the driver in `ssik-rust/src/drivers/<arm>/ik_solver.rs`
//!    (expose `fk`, `solve`, and `TolerancePolicy::DEFAULT`).
//! 2. In `solve()` below, add a new `"<arm>" | "<arm>_ik"` match arm.
//! 3. In `fk()` below, add the corresponding match arm.
//! 4. Update the error messages in both functions.

use nalgebra::SMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ssik_rust::drivers::franka_panda::ik_solver as franka;
use ssik_rust::drivers::ur5::ik_solver as ur5;

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
// Public Python functions
// ---------------------------------------------------------------------------

/// Compute analytical forward kinematics for the given robot arm.
///
/// Parameters
/// ----------
/// arm : str
///     Name of the robot arm.  Supported values:
///     ``"ur5"`` / ``"ur5_ik"`` and
///     ``"franka_panda"`` / ``"franka_panda_ik"``.
/// q : list[float]
///     Joint angles in radians.  Length must match the arm's DOF
///     (6 for UR5, 7 for Franka Panda).
///
/// Returns
/// -------
/// list[list[float]]
///     4×4 SE(3) pose of ``ee_link`` in ``base_link`` as a nested list.
///
/// Raises
/// ------
/// ValueError
///     If the arm name is unsupported or ``q`` has the wrong length.
#[pyfunction]
fn fk(arm: &str, q: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
    let tf: SMatrix<f64, 4, 4> = match arm {
        // ── UR5 ──────────────────────────────────────────────────────────────
        // To add a new 6-DOF arm: copy this block and replace the driver alias.
        "ur5" | "ur5_ik" => {
            if q.len() != ur5::DOF {
                return Err(PyValueError::new_err(format!(
                    "ur5 requires {} joint angles, got {}",
                    ur5::DOF,
                    q.len()
                )));
            }
            let arr: [f64; 6] = q.try_into().unwrap();
            ur5::fk(&arr)
        }
        // ── Franka Panda ─────────────────────────────────────────────────────
        // To add a new 7-DOF arm: copy this block and replace the driver alias.
        "franka_panda" | "franka_panda_ik" => {
            if q.len() != franka::DOF {
                return Err(PyValueError::new_err(format!(
                    "franka_panda requires {} joint angles, got {}",
                    franka::DOF,
                    q.len()
                )));
            }
            let arr: [f64; 7] = q.try_into().unwrap();
            franka::fk(&arr)
        }
        // ── Unsupported ───────────────────────────────────────────────────────
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported arm '{arm}'; supported arms: ur5, franka_panda"
            )));
        }
    };

    Ok(matrix_to_nested(&tf))
}

/// Compute analytical inverse kinematics for the given robot arm.
///
/// The signature mirrors the Python ``ssik`` prebuilt artifact interface
/// (e.g. ``ssik.prebuilt.ur5_ik.solve``), with an additional leading ``arm``
/// argument to select the robot.
///
/// Parameters
/// ----------
/// arm : str
///     Name of the robot arm.  Supported values:
///     ``"ur5"`` / ``"ur5_ik"`` and
///     ``"franka_panda"`` / ``"franka_panda_ik"``.
/// T_target : array-like (4×4)
///     SE(3) target pose of ``ee_link`` in ``base_link``.  Accepts a numpy
///     ``ndarray`` or a plain list-of-lists.
/// max_solutions : int or None, optional
///     Cap on the number of returned solutions (post-dedup, post-limits).
///     ``None`` (default) returns the full geometric set.
/// q_seed : list[float] or None, optional
///     Optional seed joint configuration.  When provided, solutions are sorted
///     by wrap-to-π distance from the seed (closest first).  For the Franka
///     Panda the lock-joint sweep also visits samples nearest to the seed
///     first, enabling an efficient trajectory-tracking path when combined
///     with ``max_solutions=1``.
/// respect_limits : bool, optional
///     When ``True`` (default), solutions outside the URDF joint limits are
///     dropped.  Pass ``False`` for the raw geometric set.
/// allow_refinement : bool, optional
///     When ``True``, algebraic candidates are polished with a bounded
///     Levenberg-Marquardt step.  Default ``False``.
/// fk_atol : float, optional
///     FK closure tolerance (Frobenius norm).  Default ``1e-5``, matching
///     ``ssik``'s ``DEFAULT_TOLERANCE_POLICY.subproblem_numerical``.
/// dedup_atol : float, optional
///     Wrap-to-π de-duplication tolerance per joint.  Default ``1e-3``,
///     matching ``ssik``'s ``DEFAULT_TOLERANCE_POLICY.subproblem_dedup``.
/// refinement_max_iters : int, optional
///     Maximum LM iterations per candidate when ``allow_refinement=True``.
///     Default ``15``.
///
/// Returns
/// -------
/// list[list[float]]
///     Each element is a valid joint-angle vector in radians.  Returns an
///     empty list when no solution exists.
///
/// Raises
/// ------
/// ValueError
///     If the arm name is unsupported, ``T_target`` is not 4×4, or
///     ``q_seed`` has the wrong length.
#[pyfunction]
#[pyo3(signature = (
    arm,
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
fn solve(
    arm: &str,
    T_target: Bound<'_, PyAny>,
    max_solutions: Option<usize>,
    q_seed: Option<Vec<f64>>,
    respect_limits: bool,
    allow_refinement: bool,
    fk_atol: f64,
    dedup_atol: f64,
    refinement_max_iters: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let tf = extract_matrix_4x4(&T_target)?;

    let solutions: Vec<Vec<f64>> = match arm {
        // ── UR5 ──────────────────────────────────────────────────────────────
        // To add a new arm: copy this block, replace the driver alias, and
        // adjust the q_seed array size to match the arm's DOF.
        "ur5" | "ur5_ik" => {
            let seed: Option<[f64; 6]> = match q_seed {
                None => None,
                Some(v) => {
                    if v.len() != ur5::DOF {
                        return Err(PyValueError::new_err(format!(
                            "q_seed for ur5 must have {} elements, got {}",
                            ur5::DOF,
                            v.len()
                        )));
                    }
                    Some(v.try_into().unwrap())
                }
            };
            let policy = ur5::TolerancePolicy { fk_atol, dedup_atol };
            ur5::solve(
                &tf, max_solutions, seed.as_ref(), respect_limits,
                allow_refinement, policy, refinement_max_iters,
            )
                .iter()
                .filter_map(|s| s.map(|q| q.to_vec()))
                .collect()
        }
        // ── Franka Panda ─────────────────────────────────────────────────────
        // To add a new arm: copy this block, replace the driver alias, and
        // adjust the q_seed array size to match the arm's DOF.
        "franka_panda" | "franka_panda_ik" => {
            let seed: Option<[f64; 7]> = match q_seed {
                None => None,
                Some(v) => {
                    if v.len() != franka::DOF {
                        return Err(PyValueError::new_err(format!(
                            "q_seed for franka_panda must have {} elements, got {}",
                            franka::DOF,
                            v.len()
                        )));
                    }
                    Some(v.try_into().unwrap())
                }
            };
            let policy = franka::TolerancePolicy { fk_atol, dedup_atol };
            franka::solve(
                &tf, max_solutions, seed.as_ref(), respect_limits,
                allow_refinement, policy, refinement_max_iters,
            )
                .iter()
                .filter_map(|s| s.map(|q| q.to_vec()))
                .collect()
        }
        // ── Unsupported ───────────────────────────────────────────────────────
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported arm '{arm}'; supported arms: ur5, franka_panda"
            )));
        }
    };

    Ok(solutions)
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Python module exposing ssik-rust IK solvers.
///
/// Functions
/// ---------
/// solve(arm, T_target, ...)
///     Inverse kinematics — mirrors ``ssik.prebuilt.<arm>_ik.solve``.
/// fk(arm, q)
///     Forward kinematics — mirrors ``ssik.prebuilt.<arm>_ik.fk``.
#[pymodule]
fn ssik_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(fk, m)?)?;
    Ok(())
}
