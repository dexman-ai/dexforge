//! Python bindings for `ssik-rust` analytical inverse kinematics.
//!
//! Exposes a single function:
//!
//! ```python
//! import ssik_py
//! solutions = ssik_py.compute_ik(x, y, z, rotation, arm)
//! ```
//!
//! where `rotation` is a `scipy.spatial.transform.Rotation` instance (or any
//! object that implements `as_matrix()` returning a 3×3 array-like).

use nalgebra::SMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute analytical inverse kinematics for the given robot arm.
///
/// Parameters
/// ----------
/// x : float
///     X translation of the end-effector in metres.
/// y : float
///     Y translation of the end-effector in metres.
/// z : float
///     Z translation of the end-effector in metres.
/// rotation : scipy.spatial.transform.Rotation
///     Orientation of the end-effector.  Any Python object that exposes an
///     ``as_matrix()`` method returning a 3×3 array-like is accepted.
/// arm : str
///     Name of the robot arm.  Currently supported: ``"ur5"``.
///
/// Returns
/// -------
/// list[list[float]]
///     Each element is a valid joint-angle vector ``[q1, q2, q3, q4, q5, q6]``
///     in radians.  Returns an empty list when no solution exists.
#[pyfunction]
fn compute_ik(
    x: f64,
    y: f64,
    z: f64,
    rotation: Bound<'_, PyAny>,
    arm: &str,
) -> PyResult<Vec<[f64; 6]>> {
    // Call rotation.as_matrix() and then .tolist() to obtain a plain Python
    // nested list that pyo3 can extract without a numpy dependency.
    let rot_matrix_obj = rotation.call_method0("as_matrix")?;
    let rot_list = rot_matrix_obj.call_method0("tolist")?;
    let r: Vec<Vec<f64>> = rot_list.extract()?;

    if r.len() != 3 || r.iter().any(|row| row.len() != 3) {
        return Err(PyValueError::new_err(
            "rotation.as_matrix() must return a 3×3 matrix",
        ));
    }

    // Build the 4×4 SE(3) transform in row-major order.
    //
    // [ R | t ]     row 0: r[0][0] r[0][1] r[0][2] x
    // [ 0 | 1 ]     row 1: r[1][0] r[1][1] r[1][2] y
    //               row 2: r[2][0] r[2][1] r[2][2] z
    //               row 3: 0       0       0       1
    let mut data = [0.0f64; 16];
    for i in 0..3 {
        for j in 0..3 {
            data[i * 4 + j] = r[i][j];
        }
    }
    data[3] = x;   // row 0, col 3
    data[7] = y;   // row 1, col 3
    data[11] = z;  // row 2, col 3
    data[15] = 1.0; // row 3, col 3

    let tf = SMatrix::<f64, 4, 4>::from_row_slice(&data);

    let raw = match arm {
        "ur5" => ssik_rust::drivers::ur5::ik_solver::solve(&tf),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported arm '{arm}'; supported arms: ur5"
            )));
        }
    };

    Ok(raw.iter().filter_map(|s| *s).collect())
}

/// Python module exposing ssik-rust IK solvers.
#[pymodule]
fn ssik_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ik, m)?)?;
    Ok(())
}
