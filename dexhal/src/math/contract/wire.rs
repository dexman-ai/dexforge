//! Wire layouts of the real-time data plane.
//!
//! The kernel and a driver exchange two flat `f64` buffers every control
//! cycle. Their fixed prefixes are [`CoreState`] and [`CoreCommand`]:
//! `#[repr(C)]`, all-`f64` structs whose memory layout *is* the wire layout.
//!
//! The data plane reads and writes those buffers in place through the views
//! ([`CoreStateView`], [`CoreStateViewMut`], [`CoreCommandView`],
//! [`CoreCommandViewMut`]): slot offsets are taken from the struct definitions
//! with `offset_of!`, so no `unsafe` cast and no copy is involved. The copying
//! [`CoreState::read_from`] / [`CoreState::write_to`] remain as helpers for
//! code that wants an owned value.

use core::mem::{offset_of, size_of};

use super::{ArmError, MAX_DOF};

const SLOT: usize = size_of::<f64>();

#[inline]
fn array_at<const N: usize>(buffer: &[f64], at: usize) -> &[f64; N] {
    match buffer[at..].first_chunk::<N>() {
        Some(a) => a,
        None => unreachable!("view length is checked on construction"),
    }
}

#[inline]
fn array_at_mut<const N: usize>(buffer: &mut [f64], at: usize) -> &mut [f64; N] {
    match buffer[at..].first_chunk_mut::<N>() {
        Some(a) => a,
        None => unreachable!("view length is checked on construction"),
    }
}

/// Generates the borrowed and mutable views of a wire struct. Arrays are
/// accessed as `&[f64; N]`, scalars by value (and `&mut f64`). `tail` adds
/// accessors for the slots following the fixed prefix; `no_tail` omits them.
macro_rules! wire_views {
    (@tail tail, $strukt:ident) => {
        /// The slots following the fixed prefix (the driver-specific tail).
        pub fn tail(&self) -> &'a [f64] {
            &self.0[$strukt::LEN..]
        }
    };
    (@tail no_tail, $strukt:ident) => {};
    (@tail_mut tail, $strukt:ident) => {
        /// The slots following the fixed prefix (the driver-specific tail).
        pub fn tail(&mut self) -> &mut [f64] {
            &mut self.0[$strukt::LEN..]
        }
    };
    (@tail_mut no_tail, $strukt:ident) => {};
    (
        $strukt:ident, $view:ident, $view_mut:ident, $tail:ident;
        $( $field:ident : $kind:tt ),* $(,)?
    ) => {
        #[doc = concat!("Zero-copy read access to the [`", stringify!($strukt),
            "`] at the front of a buffer.")]
        #[derive(Clone, Copy, Debug)]
        pub struct $view<'a>(&'a [f64]);

        impl<'a> $view<'a> {
            /// View the front of `buffer`; fails if it is too short.
            pub fn new(buffer: &'a [f64]) -> Result<Self, ArmError> {
                if buffer.len() < $strukt::LEN {
                    return Err(ArmError::BufferTooSmall);
                }
                Ok($view(buffer))
            }

            wire_views!(@tail $tail, $strukt);

            $( wire_views!(@get $strukt, $field, $kind); )*
        }

        #[doc = concat!("Zero-copy write access to the [`", stringify!($strukt),
            "`] at the front of a buffer.")]
        #[derive(Debug)]
        pub struct $view_mut<'a>(&'a mut [f64]);

        impl<'a> $view_mut<'a> {
            /// View the front of `buffer`; fails if it is too short.
            pub fn new(buffer: &'a mut [f64]) -> Result<Self, ArmError> {
                if buffer.len() < $strukt::LEN {
                    return Err(ArmError::BufferTooSmall);
                }
                Ok($view_mut(buffer))
            }

            /// Set every slot of the fixed prefix to `NAN` ("not provided").
            pub fn clear(&mut self) {
                self.0[..$strukt::LEN].fill(f64::NAN);
            }

            wire_views!(@tail_mut $tail, $strukt);

            /// Read-only view of the same buffer.
            pub fn as_view(&self) -> $view<'_> {
                $view(self.0)
            }

            $( wire_views!(@get_mut $strukt, $field, $kind); )*
        }
    };
    (@get $strukt:ident, $field:ident, [$n:expr]) => {
        #[doc = concat!("`", stringify!($field), "` slots.")]
        #[inline]
        pub fn $field(&self) -> &'a [f64; $n] {
            array_at(self.0, offset_of!($strukt, $field) / SLOT)
        }
    };
    (@get $strukt:ident, $field:ident, scalar) => {
        #[doc = concat!("`", stringify!($field), "` slot.")]
        #[inline]
        pub fn $field(&self) -> f64 {
            self.0[offset_of!($strukt, $field) / SLOT]
        }
    };
    (@get_mut $strukt:ident, $field:ident, [$n:expr]) => {
        #[doc = concat!("`", stringify!($field), "` slots.")]
        #[inline]
        pub fn $field(&mut self) -> &mut [f64; $n] {
            array_at_mut(self.0, offset_of!($strukt, $field) / SLOT)
        }
    };
    (@get_mut $strukt:ident, $field:ident, scalar) => {
        #[doc = concat!("`", stringify!($field), "` slot.")]
        #[inline]
        pub fn $field(&mut self) -> &mut f64 {
            &mut self.0[offset_of!($strukt, $field) / SLOT]
        }
    };
}

struct Reader<'a>(&'a [f64]);

impl Reader<'_> {
    fn array<const N: usize>(&mut self) -> [f64; N] {
        let (head, tail) = self.0.split_at(N);
        self.0 = tail;
        let mut out = [0.0; N];
        out.copy_from_slice(head);
        out
    }

    fn scalar(&mut self) -> f64 {
        self.array::<1>()[0]
    }
}

struct Writer<'a>(&'a mut [f64]);

impl Writer<'_> {
    fn array(&mut self, v: &[f64]) {
        let (head, tail) = core::mem::take(&mut self.0).split_at_mut(v.len());
        head.copy_from_slice(v);
        self.0 = tail;
    }
}

/// Typed prefix of the driver → kernel state buffer. Fields the hardware does
/// not provide stay `NAN`; the kernel then falls back to the host model.
///
/// Signals are raw and uncompensated: the driver is a pass-through and never
/// runs kinematics, dynamics or frame transforms. Any pre-processing the
/// hardware already did is declared in the manifest instead.
///
/// * `tcp_pose` and `tcp_twist` are at the TCP origin in base axes.
/// * `torques` is the link-side joint effort (N·m) in the positive joint
///   direction, *including* gravity, payload, friction and inertial load — the
///   total effort the joint transmits, not an external-torque estimate. When
///   the manifest sets `sensing.torque_constants`, the driver instead passes
///   the raw motor current / effort through and the kernel scales it
///   ([`ArmManifest::link_torques`](super::ArmManifest::link_torques)).
/// * `wrench` is `[Fx, Fy, Fz, Tx, Ty, Tz]` that the environment and payload
///   apply *onto* the tool, in the frame of `sensing.wrench_frame` (default:
///   TCP origin, base axes; see [`WrenchFrame`](super::WrenchFrame)), and
///   including the tool / payload weight unless
///   `sensing.wrench_is_gravity_compensated` is set.
/// * Digital I/O bitmasks are carried as `f64` and are exact for bits
///   `0..53` only.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreState {
    pub q: [f64; MAX_DOF],
    pub qd: [f64; MAX_DOF],
    /// `[x, y, z, qx, qy, qz, qw]` in base frame, only if the hardware
    /// computes its own TCP.
    pub tcp_pose: [f64; 7],
    /// `[vx, vy, vz, wx, wy, wz]` at the TCP origin, base axes.
    pub tcp_twist: [f64; 6],
    /// Link-side joint effort (N·m), positive joint direction, including
    /// gravity and payload; raw motor current / effort when the manifest
    /// declares `sensing.torque_constants`.
    pub torques: [f64; MAX_DOF],
    /// `[Fx, Fy, Fz, Tx, Ty, Tz]` applied onto the tool, in
    /// `sensing.wrench_frame` (default TCP origin, base axes).
    pub wrench: [f64; 6],
    /// 1.0 = contact, 0.0 = free, NAN = unknown.
    pub in_contact: f64,
    /// Digital input bitmask carried as a float (exact for bits `0..53`).
    pub io_digital_in: f64,
    pub io_analog_in: [f64; 4],
    pub gripper_width: f64,
    pub gripper_force: f64,
    pub gripper_has_object: f64,
    /// Hardware timestamp (s) or monotonic sample counter for dynamic $k$-deep
    /// latency alignment; `NAN` when the driver does not report one.
    pub timestamp: f64,
    /// Hardware operating mode code; `NAN` when not reported.
    pub robot_mode: f64,
    /// Driver / controller status or fault code (`0.0` = healthy, `NAN` = not
    /// reported).
    pub fault_code: f64,
    /// `1.0` when a controller script or native protective reflex fired /
    /// broke contract, `0.0` otherwise, `NAN` if unsupported.
    pub native_reflex_fired: f64,
}

impl CoreState {
    /// Number of `f64` slots.
    pub const LEN: usize = size_of::<CoreState>() / SLOT;

    /// Every field `NAN` ("not provided").
    pub const UNKNOWN: CoreState = CoreState {
        q: [f64::NAN; MAX_DOF],
        qd: [f64::NAN; MAX_DOF],
        tcp_pose: [f64::NAN; 7],
        tcp_twist: [f64::NAN; 6],
        torques: [f64::NAN; MAX_DOF],
        wrench: [f64::NAN; 6],
        in_contact: f64::NAN,
        io_digital_in: f64::NAN,
        io_analog_in: [f64::NAN; 4],
        gripper_width: f64::NAN,
        gripper_force: f64::NAN,
        gripper_has_object: f64::NAN,
        timestamp: f64::NAN,
        robot_mode: f64::NAN,
        fault_code: f64::NAN,
        native_reflex_fired: f64::NAN,
    };

    /// Copy out of the front of `buffer`.
    pub fn read_from(buffer: &[f64]) -> Result<CoreState, ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut r = Reader(buffer);
        Ok(CoreState {
            q: r.array(),
            qd: r.array(),
            tcp_pose: r.array(),
            tcp_twist: r.array(),
            torques: r.array(),
            wrench: r.array(),
            in_contact: r.scalar(),
            io_digital_in: r.scalar(),
            io_analog_in: r.array(),
            gripper_width: r.scalar(),
            gripper_force: r.scalar(),
            gripper_has_object: r.scalar(),
            timestamp: r.scalar(),
            robot_mode: r.scalar(),
            fault_code: r.scalar(),
            native_reflex_fired: r.scalar(),
        })
    }

    /// Copy into the front of `buffer`.
    pub fn write_to(&self, buffer: &mut [f64]) -> Result<(), ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut w = Writer(buffer);
        w.array(&self.q);
        w.array(&self.qd);
        w.array(&self.tcp_pose);
        w.array(&self.tcp_twist);
        w.array(&self.torques);
        w.array(&self.wrench);
        w.array(&[self.in_contact, self.io_digital_in]);
        w.array(&self.io_analog_in);
        w.array(&[
            self.gripper_width,
            self.gripper_force,
            self.gripper_has_object,
            self.timestamp,
            self.robot_mode,
            self.fault_code,
            self.native_reflex_fired,
        ]);
        Ok(())
    }
}

wire_views! {
    CoreState, CoreStateView, CoreStateViewMut, tail;
    q: [MAX_DOF],
    qd: [MAX_DOF],
    tcp_pose: [7],
    tcp_twist: [6],
    torques: [MAX_DOF],
    wrench: [6],
    in_contact: scalar,
    io_digital_in: scalar,
    io_analog_in: [4],
    gripper_width: scalar,
    gripper_force: scalar,
    gripper_has_object: scalar,
    timestamp: scalar,
    robot_mode: scalar,
    fault_code: scalar,
    native_reflex_fired: scalar,
}

/// The control mode a [`CoreCommand`] selects, carried in
/// [`CoreCommand::control_mode`] as a small integer code (`NAN` = idle).
///
/// `write_command` is a single `match` on this tag: each mode names the
/// command fields the driver forwards. Peripheral fields (digital / analog
/// outputs, gripper, reflex thresholds) are independent of the mode and apply
/// whenever they are finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveMode {
    /// No motion target (`NAN`). The driver forwards the hardware's native
    /// hold or zero-damped command (hold the current position, or
    /// `kp = 0, kd = kd_safe, tau = 0` on an MIT bus). [`CoreCommand::IDLE`]
    /// selects it.
    Idle,
    /// `target_q`; `target_qd` / `target_qdd` as feedforward when finite.
    JointPos,
    /// `target_qd`; `target_qdd` as feedforward when finite.
    JointVel,
    /// `target_torques`. Whether the controller adds its own gravity
    /// compensation is declared by `native.gravity_compensation`.
    JointTorque,
    /// `target_pose`; `target_twist` as feedforward when finite.
    CartesianPose,
    /// `target_twist`.
    CartesianTwist,
    /// `target_wrench`.
    CartesianWrench,
    /// Native joint impedance / MIT mode: the 5-tuple `(target_q, target_qd,
    /// joint_stiffness_k, joint_damping_b, target_torques)`.
    JointImpedance,
    /// Native Cartesian impedance: `target_pose`, `target_twist`, `task_frame`,
    /// `task_stiffness_k`, `task_damping_b` and `target_wrench` (`F_virt`);
    /// `target_q` (with `joint_stiffness_k` / `joint_damping_b` when finite)
    /// is the null-space posture when `native.nullspace_posture` is set.
    CartesianImpedance,
    /// Native Cartesian admittance: as [`CartesianImpedance`](Self::CartesianImpedance)
    /// plus `task_mass_m`.
    CartesianAdmittance,
}

impl ActiveMode {
    /// Every mode, in code order (`Idle` first).
    pub const ALL: [ActiveMode; 10] = [
        ActiveMode::Idle,
        ActiveMode::JointPos,
        ActiveMode::JointVel,
        ActiveMode::JointTorque,
        ActiveMode::CartesianPose,
        ActiveMode::CartesianTwist,
        ActiveMode::CartesianWrench,
        ActiveMode::JointImpedance,
        ActiveMode::CartesianImpedance,
        ActiveMode::CartesianAdmittance,
    ];

    /// The wire value: `NAN` for [`Idle`](Self::Idle), else `1.0`, `2.0`, …
    /// in [`ALL`](Self::ALL) order.
    pub const fn code(self) -> f64 {
        match self {
            ActiveMode::Idle => f64::NAN,
            ActiveMode::JointPos => 1.0,
            ActiveMode::JointVel => 2.0,
            ActiveMode::JointTorque => 3.0,
            ActiveMode::CartesianPose => 4.0,
            ActiveMode::CartesianTwist => 5.0,
            ActiveMode::CartesianWrench => 6.0,
            ActiveMode::JointImpedance => 7.0,
            ActiveMode::CartesianImpedance => 8.0,
            ActiveMode::CartesianAdmittance => 9.0,
        }
    }

    /// Decode a wire value; `None` for a value that names no mode (a kernel
    /// bug, which a driver answers like [`Idle`](Self::Idle)).
    pub fn from_code(code: f64) -> Option<ActiveMode> {
        if code.is_nan() {
            return Some(ActiveMode::Idle);
        }
        ActiveMode::ALL.into_iter().find(|m| m.code() == code)
    }
}

/// Typed prefix of the kernel → driver command buffer. Targets the kernel does
/// not drive are `NAN`.
///
/// [`control_mode`](Self::control_mode) selects which targets the driver
/// executes ([`ActiveMode`]); precedence between finite targets is never
/// guessed. Cartesian targets (`target_pose`, `target_twist`, `target_wrench`)
/// are at the TCP origin in base axes; task-space gains (`task_stiffness_k`,
/// `task_damping_b`, `task_mass_m`) are diagonal in `task_frame` (or base axes
/// when `task_frame` is `NAN`).
///
/// There is no command-side tail: [`ArmManifest::command_len`](super::ArmManifest::command_len)
/// is always [`CoreCommand::LEN`]. Vendor features outside this layout go
/// through `ArmDriver::handle_vendor_command`.
///
/// When commands stop arriving the driver does nothing on its own (no timer
/// threads); the hardware's native command-stream watchdog, armed in
/// `connect` and declared by `latency.hardware_watchdog_ms`, is what acts.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreCommand {
    /// [`ActiveMode::code`]; `NAN` = idle / hold.
    pub control_mode: f64,
    pub target_q: [f64; MAX_DOF],
    pub target_qd: [f64; MAX_DOF],
    /// Feedforward joint acceleration (rad/s²) of the reference.
    pub target_qdd: [f64; MAX_DOF],
    pub target_torques: [f64; MAX_DOF],
    pub target_pose: [f64; 7],
    pub target_twist: [f64; 6],
    pub target_wrench: [f64; 6],
    /// Pose `[x, y, z, qx, qy, qz, qw]` of the compliance task frame in the
    /// base frame for delegated native compliance; `NAN` when not commanded.
    pub task_frame: [f64; 7],
    pub task_stiffness_k: [f64; 6],
    pub task_damping_b: [f64; 6],
    pub task_mass_m: [f64; 6],
    pub joint_stiffness_k: [f64; MAX_DOF],
    pub joint_damping_b: [f64; MAX_DOF],
    pub reflex_threshold: [f64; 6],
    /// 0 = stop, 1 = yield / freedrive, 2 = retract.
    pub reflex_action_code: f64,
    /// Digital outputs to drive, as a bitmask (exact for bits `0..53`).
    pub io_digital_out_mask: f64,
    /// Levels of the masked outputs, as a bitmask (exact for bits `0..53`).
    pub io_digital_out_state: f64,
    /// Analog outputs, symmetric with [`CoreState::io_analog_in`].
    pub io_analog_out: [f64; 4],
    pub gripper_target_width: f64,
    pub gripper_target_speed: f64,
    pub gripper_max_force: f64,
}

impl CoreCommand {
    /// Number of `f64` slots.
    pub const LEN: usize = size_of::<CoreCommand>() / SLOT;

    /// Every field `NAN` ("not commanded"); [`ActiveMode::Idle`].
    pub const IDLE: CoreCommand = CoreCommand {
        control_mode: f64::NAN,
        target_q: [f64::NAN; MAX_DOF],
        target_qd: [f64::NAN; MAX_DOF],
        target_qdd: [f64::NAN; MAX_DOF],
        target_torques: [f64::NAN; MAX_DOF],
        target_pose: [f64::NAN; 7],
        target_twist: [f64::NAN; 6],
        target_wrench: [f64::NAN; 6],
        task_frame: [f64::NAN; 7],
        task_stiffness_k: [f64::NAN; 6],
        task_damping_b: [f64::NAN; 6],
        task_mass_m: [f64::NAN; 6],
        joint_stiffness_k: [f64::NAN; MAX_DOF],
        joint_damping_b: [f64::NAN; MAX_DOF],
        reflex_threshold: [f64::NAN; 6],
        reflex_action_code: f64::NAN,
        io_digital_out_mask: f64::NAN,
        io_digital_out_state: f64::NAN,
        io_analog_out: [f64::NAN; 4],
        gripper_target_width: f64::NAN,
        gripper_target_speed: f64::NAN,
        gripper_max_force: f64::NAN,
    };

    /// The decoded [`control_mode`](Self::control_mode).
    pub fn active_mode(&self) -> Option<ActiveMode> {
        ActiveMode::from_code(self.control_mode)
    }

    /// Copy out of the front of `buffer`.
    pub fn read_from(buffer: &[f64]) -> Result<CoreCommand, ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut r = Reader(buffer);
        Ok(CoreCommand {
            control_mode: r.scalar(),
            target_q: r.array(),
            target_qd: r.array(),
            target_qdd: r.array(),
            target_torques: r.array(),
            target_pose: r.array(),
            target_twist: r.array(),
            target_wrench: r.array(),
            task_frame: r.array(),
            task_stiffness_k: r.array(),
            task_damping_b: r.array(),
            task_mass_m: r.array(),
            joint_stiffness_k: r.array(),
            joint_damping_b: r.array(),
            reflex_threshold: r.array(),
            reflex_action_code: r.scalar(),
            io_digital_out_mask: r.scalar(),
            io_digital_out_state: r.scalar(),
            io_analog_out: r.array(),
            gripper_target_width: r.scalar(),
            gripper_target_speed: r.scalar(),
            gripper_max_force: r.scalar(),
        })
    }

    /// Copy into the front of `buffer`.
    pub fn write_to(&self, buffer: &mut [f64]) -> Result<(), ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut w = Writer(buffer);
        w.array(&[self.control_mode]);
        w.array(&self.target_q);
        w.array(&self.target_qd);
        w.array(&self.target_qdd);
        w.array(&self.target_torques);
        w.array(&self.target_pose);
        w.array(&self.target_twist);
        w.array(&self.target_wrench);
        w.array(&self.task_frame);
        w.array(&self.task_stiffness_k);
        w.array(&self.task_damping_b);
        w.array(&self.task_mass_m);
        w.array(&self.joint_stiffness_k);
        w.array(&self.joint_damping_b);
        w.array(&self.reflex_threshold);
        w.array(&[
            self.reflex_action_code,
            self.io_digital_out_mask,
            self.io_digital_out_state,
        ]);
        w.array(&self.io_analog_out);
        w.array(&[
            self.gripper_target_width,
            self.gripper_target_speed,
            self.gripper_max_force,
        ]);
        Ok(())
    }
}

wire_views! {
    CoreCommand, CoreCommandView, CoreCommandViewMut, no_tail;
    control_mode: scalar,
    target_q: [MAX_DOF],
    target_qd: [MAX_DOF],
    target_qdd: [MAX_DOF],
    target_torques: [MAX_DOF],
    target_pose: [7],
    target_twist: [6],
    target_wrench: [6],
    task_frame: [7],
    task_stiffness_k: [6],
    task_damping_b: [6],
    task_mass_m: [6],
    joint_stiffness_k: [MAX_DOF],
    joint_damping_b: [MAX_DOF],
    reflex_threshold: [6],
    reflex_action_code: scalar,
    io_digital_out_mask: scalar,
    io_digital_out_state: scalar,
    io_analog_out: [4],
    gripper_target_width: scalar,
    gripper_target_speed: scalar,
    gripper_max_force: scalar,
}

impl CoreCommandView<'_> {
    /// The decoded `control_mode` slot.
    pub fn active_mode(&self) -> Option<ActiveMode> {
        ActiveMode::from_code(self.control_mode())
    }
}

impl CoreCommandViewMut<'_> {
    /// Write the `control_mode` slot.
    pub fn set_active_mode(&mut self, mode: ActiveMode) {
        *self.control_mode() = mode.code();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> [f64; 128] {
        core::array::from_fn(|i| i as f64)
    }

    #[test]
    fn copying_helpers_round_trip() {
        assert_eq!(CoreState::LEN, 56);
        assert_eq!(CoreCommand::LEN, 109);

        let buf = ramp();
        let state = CoreState::read_from(&buf).unwrap();
        assert_eq!(
            (state.q[0], state.tcp_pose[0], state.in_contact),
            (0.0, 16.0, 43.0)
        );
        assert_eq!(
            (
                state.io_analog_in,
                state.gripper_has_object,
                state.timestamp,
                state.native_reflex_fired
            ),
            ([45.0, 46.0, 47.0, 48.0], 51.0, 52.0, 55.0)
        );
        let mut out = [-1.0f64; 128];
        state.write_to(&mut out).unwrap();
        assert_eq!(out[..CoreState::LEN], buf[..CoreState::LEN]);
        assert_eq!(out[CoreState::LEN], -1.0);

        let cmd = CoreCommand::read_from(&buf).unwrap();
        assert_eq!(
            (cmd.control_mode, cmd.target_q[0], cmd.target_qdd[0]),
            (0.0, 1.0, 17.0)
        );
        assert_eq!(
            (
                cmd.target_pose[0],
                cmd.task_frame[0],
                cmd.reflex_threshold[0],
                cmd.io_analog_out,
                cmd.gripper_max_force
            ),
            (33.0, 52.0, 93.0, [102.0, 103.0, 104.0, 105.0], 108.0)
        );
        let mut out = [-1.0f64; 128];
        cmd.write_to(&mut out).unwrap();
        assert_eq!(out[..CoreCommand::LEN], buf[..CoreCommand::LEN]);
        assert_eq!(out[CoreCommand::LEN], -1.0);

        assert_eq!(
            CoreState::read_from(&buf[..55]),
            Err(ArmError::BufferTooSmall)
        );
        assert_eq!(
            CoreCommand::IDLE.write_to(&mut out[..108]),
            Err(ArmError::BufferTooSmall)
        );
    }

    #[test]
    fn views_agree_with_the_struct_layout() {
        let buf = ramp();
        let state = CoreState::read_from(&buf).unwrap();
        let view = CoreStateView::new(&buf).unwrap();
        assert_eq!(view.q(), &state.q);
        assert_eq!(view.qd(), &state.qd);
        assert_eq!(view.tcp_pose(), &state.tcp_pose);
        assert_eq!(view.tcp_twist(), &state.tcp_twist);
        assert_eq!(view.torques(), &state.torques);
        assert_eq!(view.wrench(), &state.wrench);
        assert_eq!(view.in_contact(), state.in_contact);
        assert_eq!(view.io_digital_in(), state.io_digital_in);
        assert_eq!(view.io_analog_in(), &state.io_analog_in);
        assert_eq!(view.gripper_width(), state.gripper_width);
        assert_eq!(view.gripper_force(), state.gripper_force);
        assert_eq!(view.gripper_has_object(), state.gripper_has_object);
        assert_eq!(view.timestamp(), state.timestamp);
        assert_eq!(view.robot_mode(), state.robot_mode);
        assert_eq!(view.fault_code(), state.fault_code);
        assert_eq!(view.native_reflex_fired(), state.native_reflex_fired);
        assert_eq!(view.tail().len(), 128 - CoreState::LEN);
        assert_eq!(view.tail()[0], 56.0);

        let cmd = CoreCommand::read_from(&buf).unwrap();
        let view = CoreCommandView::new(&buf).unwrap();
        assert_eq!(view.control_mode(), cmd.control_mode);
        assert_eq!(view.target_q(), &cmd.target_q);
        assert_eq!(view.target_qd(), &cmd.target_qd);
        assert_eq!(view.target_qdd(), &cmd.target_qdd);
        assert_eq!(view.target_torques(), &cmd.target_torques);
        assert_eq!(view.target_pose(), &cmd.target_pose);
        assert_eq!(view.target_twist(), &cmd.target_twist);
        assert_eq!(view.target_wrench(), &cmd.target_wrench);
        assert_eq!(view.task_frame(), &cmd.task_frame);
        assert_eq!(view.task_stiffness_k(), &cmd.task_stiffness_k);
        assert_eq!(view.task_damping_b(), &cmd.task_damping_b);
        assert_eq!(view.task_mass_m(), &cmd.task_mass_m);
        assert_eq!(view.joint_stiffness_k(), &cmd.joint_stiffness_k);
        assert_eq!(view.joint_damping_b(), &cmd.joint_damping_b);
        assert_eq!(view.reflex_threshold(), &cmd.reflex_threshold);
        assert_eq!(view.reflex_action_code(), cmd.reflex_action_code);
        assert_eq!(view.io_digital_out_mask(), cmd.io_digital_out_mask);
        assert_eq!(view.io_digital_out_state(), cmd.io_digital_out_state);
        assert_eq!(view.io_analog_out(), &cmd.io_analog_out);
        assert_eq!(view.gripper_target_width(), cmd.gripper_target_width);
        assert_eq!(view.gripper_target_speed(), cmd.gripper_target_speed);
        assert_eq!(view.gripper_max_force(), cmd.gripper_max_force);
    }

    #[test]
    fn mutable_views_write_in_place() {
        let mut buf = [0.0f64; CoreState::LEN + 3];
        let mut view = CoreStateViewMut::new(&mut buf).unwrap();
        view.clear();
        view.q()[..6].copy_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        *view.in_contact() = 1.0;
        *view.timestamp() = 0.042;
        *view.native_reflex_fired() = 0.0;
        view.io_analog_in()[3] = 4.5;
        view.tail().copy_from_slice(&[7.0, 8.0, 9.0]);
        assert_eq!(view.as_view().in_contact(), 1.0);
        assert_eq!(view.as_view().timestamp(), 0.042);

        let state = CoreState::read_from(&buf).unwrap();
        assert_eq!(state.q[..6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert!(state.q[6].is_nan() && state.tcp_pose[0].is_nan());
        assert_eq!(
            (state.in_contact, state.io_analog_in[3], state.timestamp),
            (1.0, 4.5, 0.042)
        );
        assert_eq!(buf[CoreState::LEN..], [7.0, 8.0, 9.0]);

        let mut cmd = [0.0f64; CoreCommand::LEN];
        let mut view = CoreCommandViewMut::new(&mut cmd).unwrap();
        view.clear();
        assert_eq!(view.as_view().active_mode(), Some(ActiveMode::Idle));
        view.set_active_mode(ActiveMode::JointImpedance);
        view.target_q()[0] = 1.5;
        view.target_qdd()[1] = -2.0;
        *view.task_frame() = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        view.io_analog_out()[2] = 3.3;
        *view.gripper_max_force() = 20.0;
        let decoded = CoreCommand::read_from(&cmd).unwrap();
        assert_eq!(decoded.active_mode(), Some(ActiveMode::JointImpedance));
        assert_eq!(
            (
                decoded.target_q[0],
                decoded.target_qdd[1],
                decoded.task_frame[6],
                decoded.io_analog_out[2],
                decoded.gripper_max_force
            ),
            (1.5, -2.0, 1.0, 3.3, 20.0)
        );
        assert!(decoded.target_pose[0].is_nan());

        assert!(CoreStateView::new(&buf[..55]).is_err());
        assert!(CoreCommandViewMut::new(&mut cmd[..108]).is_err());
    }

    #[test]
    fn active_mode_codes_round_trip() {
        for (i, mode) in ActiveMode::ALL.into_iter().enumerate() {
            assert_eq!(ActiveMode::from_code(mode.code()), Some(mode));
            if i > 0 {
                assert_eq!(mode.code(), i as f64);
            }
        }
        assert!(ActiveMode::Idle.code().is_nan());
        assert_eq!(CoreCommand::IDLE.active_mode(), Some(ActiveMode::Idle));
        for bad in [0.0, 1.5, 10.0, -1.0, f64::INFINITY] {
            assert_eq!(ActiveMode::from_code(bad), None, "{bad}");
        }
    }
}