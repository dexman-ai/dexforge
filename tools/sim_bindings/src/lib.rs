//! Python bindings for the Universal Robots OEM repo.

include!("common.rs");

arm_module!(ur5_py, "ur5", dexforge_ur_core, ur5, Ur5, Ur5Driver);
arm_module!(ur3e_py, "ur3e", dexforge_ur_core, ur3e, Ur3e, Ur3eDriver);

#[pymodule]
fn dexforge_ur(m: &Bound<'_, PyModule>) -> PyResult<()> {
	ur5_py::register(m)?;
	ur3e_py::register(m)?;
	m.add_function(wrap_pyfunction!(state_layout, m)?)?;
	m.add_function(wrap_pyfunction!(command_layout, m)?)?;
	Ok(())
}
