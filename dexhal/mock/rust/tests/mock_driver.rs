use dexforge_mock_core::{MockArm, MockDriver, MANIFEST};
use dexhal::math::{
    pose_to_xyz_quat, ActiveMode, ArmDriver, ArmError, ArmModel, CoreCommand, CoreState,
};

fn zero_latency() -> &'static str {
    let text = MANIFEST
        .replace("state_staleness_cycles: 1", "state_staleness_cycles: 0")
        .replace("command_dispatch_cycles: 2", "command_dispatch_cycles: 0");
    Box::leak(text.into_boxed_str())
}

fn driver() -> MockDriver<MockArm, 6, 1> {
    MockDriver::default_arm().expect("crate manifest should load")
}

fn zero_latency_driver() -> MockDriver<MockArm, 6, 1> {
    MockDriver::new(zero_latency()).expect("manifest should load")
}

fn command(q: [f64; 6]) -> CoreCommand {
    let mut cmd = CoreCommand::IDLE;
    cmd.control_mode = ActiveMode::JointPos.code();
    cmd.target_q[..6].copy_from_slice(&q);
    cmd
}

fn write(d: &mut MockDriver<MockArm, 6, 1>, cmd: &CoreCommand) {
    let mut buf = vec![f64::NAN; d.manifest().command_len()];
    cmd.write_to(&mut buf).unwrap();
    d.write_command(&buf).unwrap();
}

fn read(d: &mut MockDriver<MockArm, 6, 1>) -> CoreState {
    let mut buf = vec![0.0; d.manifest().state_len()];
    d.read_state(&mut buf).unwrap();
    CoreState::read_from(&buf).unwrap()
}

#[test]
fn default_arm_uses_crate_manifest() {
    let d = driver();
    assert_eq!(d.manifest().name, MockArm::NAME);
    assert_eq!(d.manifest().name, "mock-6r");
    assert_eq!(d.manifest().state_len(), CoreState::LEN);
}

#[test]
fn connect_and_disconnect_are_noops() {
    let mut d = driver();
    assert_eq!(d.connect(), Ok(()));
    assert_eq!(d.disconnect(), Ok(()));
    assert_eq!(d.connect(), Ok(()));
}

#[test]
fn command_becomes_visible_after_dispatch_plus_staleness_writes() {
    let mut d = driver();
    write(&mut d, &command([1.0; 6]));
    assert_eq!(read(&mut d).q[0], 0.0);

    write(&mut d, &command([2.0; 6]));
    assert_eq!(read(&mut d).q[0], 0.0);

    write(&mut d, &command([3.0; 6]));
    assert_eq!(read(&mut d).q[0], 0.0);

    write(&mut d, &command([4.0; 6]));
    assert_eq!(read(&mut d).q[0], 1.0);
}

#[test]
fn oldest_command_is_returned_in_fifo_order() {
    let mut d = driver();
    for k in 1..=10 {
        write(&mut d, &command([k as f64; 6]));
        if k >= 4 {
            assert_eq!(read(&mut d).q[0], (k - 3) as f64, "after write {k}");
        }
    }
}

#[test]
fn read_does_not_consume_queue_entries() {
    let mut d = driver();
    for k in 1..=4 {
        write(&mut d, &command([k as f64; 6]));
    }
    assert_eq!(read(&mut d).q[0], 1.0);
    assert_eq!(read(&mut d).q[0], 1.0);
}

#[test]
fn zero_latency_reflects_the_last_write_immediately() {
    let mut d = zero_latency_driver();
    write(&mut d, &command([0.25; 6]));
    assert_eq!(
        read(&mut d).q,
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
    );
}

#[test]
fn nan_target_holds_last_joint_position() {
    let mut d = zero_latency_driver();
    write(&mut d, &command([0.5, -0.5, 0.1, 0.2, 0.3, 0.4]));
    assert_eq!(read(&mut d).q[1], -0.5);

    write(&mut d, &CoreCommand::IDLE);
    assert_eq!(read(&mut d).q[1], -0.5);
}

#[test]
fn idle_holds_even_when_targets_are_set() {
    let mut d = zero_latency_driver();
    write(&mut d, &command([0.5; 6]));
    assert_eq!(read(&mut d).q[0], 0.5);
    let mut idle = command([1.0; 6]);
    idle.control_mode = ActiveMode::Idle.code();
    idle.target_qd[..6].copy_from_slice(&[0.3; 6]);
    idle.target_wrench = [1.0; 6];
    write(&mut d, &idle);

    let s = read(&mut d);
    assert_eq!(s.q[..6], [0.5; 6]);
    assert_eq!(s.qd[..6], [0.0; 6]);
    assert!(s.wrench.iter().all(|v| v.is_nan()));
}

#[test]
fn unsupported_or_unknown_modes_are_rejected() {
    let mut d = zero_latency_driver();
    let mut buf = vec![f64::NAN; d.manifest().command_len()];
    let mut cmd = command([0.0; 6]);
    cmd.control_mode = ActiveMode::JointImpedance.code();
    cmd.write_to(&mut buf).unwrap();
    assert_eq!(d.write_command(&buf), Err(ArmError::NotSupported));

    cmd.control_mode = 42.0;
    cmd.write_to(&mut buf).unwrap();
    assert!(matches!(d.write_command(&buf), Err(ArmError::Fault(_))));
}

#[test]
fn vendor_echo_fills_the_response() {
    let mut d = driver();
    let mut response = [0u8; 8];
    assert_eq!(d.handle_vendor_command("echo", b"abc", &mut response), Ok(3));
    assert_eq!(&response[..3], b"abc");
    assert_eq!(
        d.handle_vendor_command("echo", &[0; 9], &mut response),
        Err(ArmError::BufferTooSmall)
    );
    assert_eq!(
        d.handle_vendor_command("reboot", &[], &mut response),
        Err(ArmError::NotSupported)
    );
}

#[test]
fn qd_and_torques_pass_through_when_commanded() {
    let mut d = zero_latency_driver();
    let mut cmd = command([0.0; 6]);
    cmd.target_qd[..6].copy_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    cmd.target_torques[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    write(&mut d, &cmd);

    let s = read(&mut d);
    assert_eq!(&s.qd[..6], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    assert_eq!(&s.torques[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tcp_pose_is_forward_kinematics_of_state_q() {
    let mut d = zero_latency_driver();
    let q = [0.3, -0.7, 1.1, 0.2, -0.4, 0.9];
    write(&mut d, &command(q));

    let s = read(&mut d);
    assert_eq!(s.tcp_pose, pose_to_xyz_quat(&MockArm::flange_fk(&q)));
}

#[test]
fn commanded_twist_wrench_io_and_gripper_are_reflected() {
    let mut d = zero_latency_driver();
    let mut cmd = command([0.0; 6]);
    cmd.target_twist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    cmd.target_wrench = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    cmd.io_digital_out_state = 5.0;
    cmd.io_analog_out = [0.5, 1.0, 1.5, 2.0];
    cmd.gripper_target_width = 0.04;
    write(&mut d, &cmd);

    let s = read(&mut d);
    assert_eq!(s.tcp_twist, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    assert_eq!(s.wrench, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
    assert_eq!(s.io_digital_in, 5.0);
    assert_eq!(s.io_analog_in, [0.5, 1.0, 1.5, 2.0]);
    assert_eq!(s.gripper_width, 0.04);
}

#[test]
fn uncommanded_fields_are_nan() {
    let mut d = zero_latency_driver();
    write(&mut d, &command([0.0; 6]));

    let s = read(&mut d);
    assert!(s.tcp_twist.iter().all(|v| v.is_nan()));
    assert!(s.wrench.iter().all(|v| v.is_nan()));
    assert!(s.io_digital_in.is_nan());
    assert!(s.gripper_width.is_nan());
    assert!(s.in_contact.is_nan());
}

#[test]
fn short_buffers_are_rejected() {
    let mut d = driver();
    let mut short_state = vec![0.0; CoreState::LEN - 1];
    assert_eq!(
        d.read_state(&mut short_state),
        Err(ArmError::BufferTooSmall)
    );

    let short_cmd = vec![0.0; CoreCommand::LEN - 1];
    assert_eq!(d.write_command(&short_cmd), Err(ArmError::BufferTooSmall));
}
