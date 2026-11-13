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
/// accessed as `&[f64; N]`, scalars by value (and `&mut f64`).
macro_rules! wire_views {
    (
        $strukt:ident, $view:ident, $view_mut:ident;
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

            /// The slots following the fixed prefix (the driver-specific tail).
            pub fn tail(&self) -> &'a [f64] {
                &self.0[$strukt::LEN..]
            }

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

            /// The slots following the fixed prefix (the driver-specific tail).
            pub fn tail(&mut self) -> &mut [f64] {
                &mut self.0[$strukt::LEN..]
            }

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
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreState {
    pub q: [f64; MAX_DOF],
    pub qd: [f64; MAX_DOF],
    /// `[x, y, z, qx, qy, qz, qw]`, only if the hardware computes its own TCP.
    pub tcp_pose: [f64; 7],
    pub tcp_twist: [f64; 6],
    pub torques: [f64; MAX_DOF],
    pub wrench: [f64; 6],
    /// 1.0 = contact, 0.0 = free, NAN = unknown.
    pub in_contact: f64,
    /// Digital input bitmask carried as a float.
    pub io_digital_in: f64,
    pub io_analog_in: [f64; 4],
    pub gripper_width: f64,
    pub gripper_force: f64,
    pub gripper_has_object: f64,
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
        ]);
        Ok(())
    }
}

wire_views! {
    CoreState, CoreStateView, CoreStateViewMut;
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
}

/// Typed prefix of the kernel → driver command buffer. Targets the kernel does
/// not drive are `NAN`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreCommand {
    pub target_q: [f64; MAX_DOF],
    pub target_qd: [f64; MAX_DOF],
    pub target_torques: [f64; MAX_DOF],
    pub target_pose: [f64; 7],
    pub target_twist: [f64; 6],
    pub target_wrench: [f64; 6],
    pub task_stiffness_k: [f64; 6],
    pub task_damping_b: [f64; 6],
    pub task_mass_m: [f64; 6],
    pub joint_stiffness_k: [f64; MAX_DOF],
    pub joint_damping_b: [f64; MAX_DOF],
    pub reflex_threshold: [f64; 6],
    /// 0 = stop, 1 = yield / freedrive, 2 = retract.
    pub reflex_action_code: f64,
    pub io_digital_out_mask: f64,
    pub io_digital_out_state: f64,
    pub gripper_target_width: f64,
    pub gripper_target_speed: f64,
    pub gripper_max_force: f64,
}

impl CoreCommand {
    /// Number of `f64` slots.
    pub const LEN: usize = size_of::<CoreCommand>() / SLOT;

    /// Every field `NAN` ("not commanded").
    pub const IDLE: CoreCommand = CoreCommand {
        target_q: [f64::NAN; MAX_DOF],
        target_qd: [f64::NAN; MAX_DOF],
        target_torques: [f64::NAN; MAX_DOF],
        target_pose: [f64::NAN; 7],
        target_twist: [f64::NAN; 6],
        target_wrench: [f64::NAN; 6],
        task_stiffness_k: [f64::NAN; 6],
        task_damping_b: [f64::NAN; 6],
        task_mass_m: [f64::NAN; 6],
        joint_stiffness_k: [f64::NAN; MAX_DOF],
        joint_damping_b: [f64::NAN; MAX_DOF],
        reflex_threshold: [f64::NAN; 6],
        reflex_action_code: f64::NAN,
        io_digital_out_mask: f64::NAN,
        io_digital_out_state: f64::NAN,
        gripper_target_width: f64::NAN,
        gripper_target_speed: f64::NAN,
        gripper_max_force: f64::NAN,
    };

    /// Copy out of the front of `buffer`.
    pub fn read_from(buffer: &[f64]) -> Result<CoreCommand, ArmError> {
        if buffer.len() < Self::LEN {
            return Err(ArmError::BufferTooSmall);
        }
        let mut r = Reader(buffer);
        Ok(CoreCommand {
            target_q: r.array(),
            target_qd: r.array(),
            target_torques: r.array(),
            target_pose: r.array(),
            target_twist: r.array(),
            target_wrench: r.array(),
            task_stiffness_k: r.array(),
            task_damping_b: r.array(),
            task_mass_m: r.array(),
            joint_stiffness_k: r.array(),
            joint_damping_b: r.array(),
            reflex_threshold: r.array(),
            reflex_action_code: r.scalar(),
            io_digital_out_mask: r.scalar(),
            io_digital_out_state: r.scalar(),
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
        w.array(&self.target_q);
        w.array(&self.target_qd);
        w.array(&self.target_torques);
        w.array(&self.target_pose);
        w.array(&self.target_twist);
        w.array(&self.target_wrench);
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
            self.gripper_target_width,
            self.gripper_target_speed,
            self.gripper_max_force,
        ]);
        Ok(())
    }
}

wire_views! {
    CoreCommand, CoreCommandView, CoreCommandViewMut;
    target_q: [MAX_DOF],
    target_qd: [MAX_DOF],
    target_torques: [MAX_DOF],
    target_pose: [7],
    target_twist: [6],
    target_wrench: [6],
    task_stiffness_k: [6],
    task_damping_b: [6],
    task_mass_m: [6],
    joint_stiffness_k: [MAX_DOF],
    joint_damping_b: [MAX_DOF],
    reflex_threshold: [6],
    reflex_action_code: scalar,
    io_digital_out_mask: scalar,
    io_digital_out_state: scalar,
    gripper_target_width: scalar,
    gripper_target_speed: scalar,
    gripper_max_force: scalar,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> [f64; 96] {
        core::array::from_fn(|i| i as f64)
    }

    #[test]
    fn copying_helpers_round_trip() {
        assert_eq!(CoreState::LEN, 52);
        assert_eq!(CoreCommand::LEN, 89);

        let buf = ramp();
        let state = CoreState::read_from(&buf).unwrap();
        assert_eq!(
            (state.q[0], state.tcp_pose[0], state.in_contact),
            (0.0, 16.0, 43.0)
        );
        assert_eq!(
            (state.io_analog_in, state.gripper_has_object),
            ([45.0, 46.0, 47.0, 48.0], 51.0)
        );
        let mut out = [-1.0f64; 96];
        state.write_to(&mut out).unwrap();
        assert_eq!(out[..CoreState::LEN], buf[..CoreState::LEN]);
        assert_eq!(out[CoreState::LEN], -1.0);

        let cmd = CoreCommand::read_from(&buf).unwrap();
        assert_eq!(
            (
                cmd.target_pose[0],
                cmd.reflex_threshold[0],
                cmd.gripper_max_force
            ),
            (24.0, 77.0, 88.0)
        );
        let mut out = [-1.0f64; 96];
        cmd.write_to(&mut out).unwrap();
        assert_eq!(out[..CoreCommand::LEN], buf[..CoreCommand::LEN]);

        assert_eq!(
            CoreState::read_from(&buf[..51]),
            Err(ArmError::BufferTooSmall)
        );
        assert_eq!(
            CoreCommand::IDLE.write_to(&mut out[..88]),
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
        assert_eq!(view.tail().len(), 96 - CoreState::LEN);
        assert_eq!(view.tail()[0], 52.0);

        let cmd = CoreCommand::read_from(&buf).unwrap();
        let view = CoreCommandView::new(&buf).unwrap();
        assert_eq!(view.target_q(), &cmd.target_q);
        assert_eq!(view.target_qd(), &cmd.target_qd);
        assert_eq!(view.target_torques(), &cmd.target_torques);
        assert_eq!(view.target_pose(), &cmd.target_pose);
        assert_eq!(view.target_twist(), &cmd.target_twist);
        assert_eq!(view.target_wrench(), &cmd.target_wrench);
        assert_eq!(view.task_stiffness_k(), &cmd.task_stiffness_k);
        assert_eq!(view.task_damping_b(), &cmd.task_damping_b);
        assert_eq!(view.task_mass_m(), &cmd.task_mass_m);
        assert_eq!(view.joint_stiffness_k(), &cmd.joint_stiffness_k);
        assert_eq!(view.joint_damping_b(), &cmd.joint_damping_b);
        assert_eq!(view.reflex_threshold(), &cmd.reflex_threshold);
        assert_eq!(view.reflex_action_code(), cmd.reflex_action_code);
        assert_eq!(view.io_digital_out_mask(), cmd.io_digital_out_mask);
        assert_eq!(view.io_digital_out_state(), cmd.io_digital_out_state);
        assert_eq!(view.gripper_target_width(), cmd.gripper_target_width);
        assert_eq!(view.gripper_target_speed(), cmd.gripper_target_speed);
        assert_eq!(view.gripper_max_force(), cmd.gripper_max_force);
        assert_eq!(view.tail().len(), 96 - CoreCommand::LEN);
    }

    #[test]
    fn mutable_views_write_in_place() {
        let mut buf = [0.0f64; CoreState::LEN + 3];
        let mut view = CoreStateViewMut::new(&mut buf).unwrap();
        view.clear();
        view.q()[..6].copy_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        *view.in_contact() = 1.0;
        view.io_analog_in()[3] = 4.5;
        view.tail().copy_from_slice(&[7.0, 8.0, 9.0]);
        assert_eq!(view.as_view().in_contact(), 1.0);

        let state = CoreState::read_from(&buf).unwrap();
        assert_eq!(state.q[..6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert!(state.q[6].is_nan() && state.tcp_pose[0].is_nan());
        assert_eq!((state.in_contact, state.io_analog_in[3]), (1.0, 4.5));
        assert_eq!(buf[CoreState::LEN..], [7.0, 8.0, 9.0]);

        let mut cmd = [0.0f64; CoreCommand::LEN];
        let mut view = CoreCommandViewMut::new(&mut cmd).unwrap();
        view.clear();
        view.target_q()[0] = 1.5;
        *view.gripper_max_force() = 20.0;
        let decoded = CoreCommand::read_from(&cmd).unwrap();
        assert_eq!(
            (decoded.target_q[0], decoded.gripper_max_force),
            (1.5, 20.0)
        );
        assert!(decoded.target_pose[0].is_nan());

        assert!(CoreStateView::new(&buf[..51]).is_err());
        assert!(CoreCommandViewMut::new(&mut cmd[..88]).is_err());
    }
}
