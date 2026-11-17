//! The mock exported through `dexhal-plugin` behaves like the mock itself.

use dexforge_mock_core::{dexhal_plugin_entry, MockArm, MockDriver};
use dexhal::math::{
    ActiveMode, Applied, Arm, ArmDriver, ArmError, ArmMath, CoreCommand, CoreState, ModelError,
    WrenchFrame, MAX_DOF,
};
use dexhal_plugin::{LoadedDriver, Plugin, PluginError};

fn plugin() -> Plugin {
    // SAFETY: the entry point is linked into this test binary.
    unsafe { Plugin::from_entry(dexhal_plugin_entry) }.expect("ABI and contract should match")
}

fn command(q: [f64; 6]) -> CoreCommand {
    let mut cmd = CoreCommand::IDLE;
    cmd.control_mode = ActiveMode::JointPos.code();
    cmd.target_q[..6].copy_from_slice(&q);
    cmd
}

fn write(d: &mut dyn ArmDriver, cmd: &CoreCommand) {
    let mut buf = vec![f64::NAN; d.manifest().command_len()];
    cmd.write_to(&mut buf).unwrap();
    d.write_command(&buf).unwrap();
}

fn read(d: &mut dyn ArmDriver) -> Vec<f64> {
    let mut buf = vec![0.0; d.manifest().state_len()];
    d.read_state(&mut buf).unwrap();
    buf
}

fn padded(q: [f64; 6]) -> [f64; MAX_DOF] {
    let mut p = [0.0; MAX_DOF];
    p[..6].copy_from_slice(&q);
    p
}

/// Bitwise equality, so NaN fields compare equal.
fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn manifest_crosses_the_boundary_unchanged() {
    let loaded = plugin().create_driver(b"").unwrap();
    let direct = MockDriver::default_arm().unwrap();
    assert_eq!(
        format!("{:?}", loaded.manifest()),
        format!("{:?}", direct.manifest())
    );
}

#[test]
fn configuration_is_passed_to_the_driver() {
    assert_eq!(
        plugin().create_driver(b"unexpected").err(),
        Some(ArmError::Fault("mock driver takes no configuration"))
    );
}

#[test]
fn data_plane_matches_the_direct_driver() {
    let mut loaded = plugin().create_driver(b"").unwrap();
    let mut direct = MockDriver::default_arm().unwrap();
    loaded.connect().unwrap();
    direct.connect().unwrap();
    for k in 0..8 {
        let cmd = command([0.1 * k as f64; 6]);
        write(&mut loaded, &cmd);
        write(&mut direct, &cmd);
        let (a, b) = (read(&mut loaded), read(&mut direct));
        assert!(same(&a, &b), "state differs after write {k}");
    }
    let state = CoreState::read_from(&read(&mut loaded)).unwrap();
    assert!((state.q[0] - 0.4).abs() < 1e-12);
    loaded.disconnect().unwrap();
}

#[test]
fn errors_cross_the_boundary() {
    let mut d = plugin().create_driver(b"").unwrap();
    assert_eq!(d.read_state(&mut [0.0; 3]), Err(ArmError::BufferTooSmall));
    let mut cmd = command([0.0; 6]);
    cmd.control_mode = 99.0;
    let mut buf = vec![0.0; d.manifest().command_len()];
    cmd.write_to(&mut buf).unwrap();
    assert_eq!(d.write_command(&buf), Err(ArmError::Fault("unknown control_mode")));
    assert_eq!(
        d.set_payload(-1.0, [0.0; 3], [0.0; 9]),
        Err(ArmError::Model(ModelError::InvalidPayload))
    );
}

#[test]
fn configuration_plane_reaches_the_driver_model() {
    let mut d = plugin().create_driver(b"").unwrap();
    let q = padded([0.1, -0.4, 0.7, 0.2, 0.3, -0.5]);
    let before = d.arm().forward_kinematics(&q).unwrap();
    assert_eq!(
        d.set_tcp_offset([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
        Ok(Applied::Host)
    );
    let after = d.arm().forward_kinematics(&q).unwrap();
    assert!((before[2] - after[2]).abs() > 1e-6 || (before[0] - after[0]).abs() > 1e-6);
    assert_eq!(d.set_gravity([0.0, 0.0, -9.81]), Ok(Applied::Host));
    assert_eq!(d.set_ik_max_joint_step(0.5), Ok(()));
    assert_eq!(
        d.clear_faults(),
        MockDriver::default_arm().unwrap().clear_faults()
    );
    assert!(!d.arm().is_calibrated());
    d.clear_calibration();
}

#[test]
fn vendor_commands_cross_the_boundary() {
    let mut d = plugin().create_driver(b"").unwrap();
    let mut response = [0u8; 16];
    assert_eq!(d.handle_vendor_command("echo", b"hello", &mut response), Ok(5));
    assert_eq!(&response[..5], b"hello");
    assert_eq!(
        d.handle_vendor_command("nope", b"", &mut response),
        Err(ArmError::NotSupported)
    );
}

#[test]
fn math_matches_the_direct_model() {
    let math = plugin().create_math().unwrap();
    let direct = Arm::<MockArm, 6, 1>::new();
    let q = padded([0.3, -0.2, 0.5, -0.1, 0.4, 0.2]);
    let qd = padded([0.1; 6]);
    assert_eq!(math.dof(), direct.dof());
    assert_eq!(math.forward_kinematics(&q), ArmMath::forward_kinematics(&direct, &q));
    assert_eq!(math.jacobian_padded(&q), direct.jacobian_padded(&q));
    assert_eq!(math.tool_jacobian_padded(&q), direct.tool_jacobian_padded(&q));
    assert_eq!(
        math.jacobian_pseudo_inverse(&q),
        direct.jacobian_pseudo_inverse(&q)
    );
    let pose = math.forward_kinematics(&q).unwrap();
    assert_eq!(
        math.inverse_kinematics(&pose, &q),
        ArmMath::inverse_kinematics(&direct, &pose, &q)
    );
    let (a, b) = (
        math.dynamics_padded(&q, &qd),
        direct.dynamics_padded(&q, &qd),
    );
    assert_eq!(a.is_some(), b.is_some());
    assert_eq!(
        math.inverse_dynamics_padded(&q, &qd, &qd),
        direct.inverse_dynamics_padded(&q, &qd, &qd)
    );
    assert_eq!(
        math.pose_residual(&q, &pose),
        direct.pose_residual(&q, &pose)
    );
    let wrench = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
    for frame in [WrenchFrame::Base, WrenchFrame::Flange, WrenchFrame::Tcp] {
        assert_eq!(
            math.external_wrench_padded(&q, &wrench, frame, false),
            direct.external_wrench_padded(&q, &wrench, frame, false)
        );
    }
}

#[test]
fn independent_math_is_configurable() {
    let mut math = plugin().create_math().unwrap();
    assert_eq!(math.apply_gravity([0.0, 0.0, -9.81]), Ok(()));
    assert_eq!(math.apply_ik_max_joint_step(0.2), Ok(()));
    assert!(math.apply_payload(f64::NAN, [0.0; 3], [0.0; 9]).is_err());
    math.remove_calibration();
}

#[test]
fn loaded_driver_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<LoadedDriver>();
}

/// Load the mock's `cdylib` from the target directory, as an application
/// would. `cargo test` alone may not build the `cdylib`; run
/// `cargo build -p dexforge-mock-core` first (CI does).
#[test]
fn dynamic_library_loads() {
    let file = format!(
        "{}dexforge_mock_core{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    );
    let exe = std::env::current_exe().unwrap();
    let Some(path) = exe
        .ancestors()
        .flat_map(|dir| [dir.join(&file), dir.join("deps").join(&file)])
        .find(|p| p.is_file())
    else {
        eprintln!("skipped: {file} not built next to {}", exe.display());
        return;
    };
    // SAFETY: the mock is a trusted dexhal plugin.
    let plugin = unsafe { Plugin::load(&path) }.expect("plugin should load");
    let mut d = plugin.create_driver(b"").unwrap();
    assert_eq!(d.manifest().name, "mock-6r");
    write(&mut d, &command([0.5; 6]));
    read(&mut d);
}

#[test]
fn missing_library_is_reported() {
    let err = unsafe { Plugin::load(std::path::Path::new("no-such-dexhal-plugin.bin")) };
    assert!(matches!(err, Err(PluginError::Open(_))));
}
