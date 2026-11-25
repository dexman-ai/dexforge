//! Stable C-ABI boundary for dexhal drivers built as dynamic libraries.
//!
//! The Rust ABI is unstable, so a driver compiled separately from the kernel
//! cannot hand out a `Box<dyn ArmDriver>`. Instead the driver library exports
//! one symbol, [`ENTRY_SYMBOL`], returning a `#[repr(C)]` [`PluginVTable`] of
//! `extern "C"` functions; the host wraps it back into an [`ArmDriver`] /
//! [`ArmMath`](dexhal::math::ArmMath).
//!
//! # Driver side
//!
//! Implement [`PluginDriver`] (construction from a config blob, plus an
//! independent host-side model for the planner thread) and export it:
//!
//! ```ignore
//! impl dexhal_plugin::PluginDriver for MyDriver {
//!     fn create(config: &[u8]) -> Result<Self, ArmError> { MyDriver::new(config) }
//!     fn create_math() -> Box<dyn ArmMath> { Box::new(Arm::<MyArm, 6, 8>::new()) }
//! }
//! dexhal_plugin::export_plugin!(MyDriver);
//! ```
//!
//! and build the crate with `crate-type = ["cdylib"]`.
//!
//! # Host side
//!
//! ```ignore
//! let plugin = unsafe { Plugin::load(Path::new("libmy_driver.so")) }?;
//! let driver: Box<dyn ArmDriver> = Box::new(plugin.create_driver(b"")?);
//! let math: Box<dyn ArmMath> = Box::new(plugin.create_math()?);
//! ```
//!
//! # Rules of the boundary
//!
//! * [`Plugin::from_entry`] checks [`PluginVTable::abi_version`] and the
//!   contract `major.minor` before any other call.
//! * The data plane passes pointer + length; nothing allocates.
//! * Errors cross as an [`FfiResult`] status code. Fault descriptions and the
//!   manifest's strings are pointers into the plugin's static data, so a
//!   loaded library is never unloaded.
//! * Every call is wrapped in `catch_unwind`; a panic is reported as
//!   [`ArmError::Fault`]. Under `panic = "abort"` it aborts instead.

#![deny(unsafe_op_in_unsafe_fn)]

use core::ffi::c_void;

use dexhal::math::{
    ArmError, ArmManifest, AuxField, AuxSchema, CalibrationData, ComplianceLimits,
    ContractVersion, DynamicsSource, DynamicsState, IkKind, ModelCapabilities, ModelError,
    WrenchFrame, MAX_AUX_FIELDS, MAX_DOF,
};

mod export;
mod host;

pub use export::{vtable, PluginDriver};
pub use host::{LoadedDriver, LoadedMath, Plugin, PluginError};

/// Version of [`PluginVTable`] and the types it passes. Bumped on any change
/// to their layout.
pub const ABI_VERSION: u32 = 4;

/// Name of the entry point a driver library exports (an [`EntryFn`]).
pub const ENTRY_SYMBOL: &str = "dexhal_plugin_entry";

/// Signature of [`ENTRY_SYMBOL`].
pub type EntryFn = unsafe extern "C" fn() -> *const PluginVTable;

/// [`FfiResult::status`] codes.
pub mod status {
    pub const OK: u32 = 0;
    /// [`ArmError::NotSupported`](dexhal::math::ArmError::NotSupported).
    pub const NOT_SUPPORTED: u32 = 1;
    /// [`ArmError::Model`](dexhal::math::ArmError::Model); see `model_error`.
    pub const MODEL: u32 = 2;
    /// [`ArmError::BufferTooSmall`](dexhal::math::ArmError::BufferTooSmall).
    pub const BUFFER_TOO_SMALL: u32 = 3;
    /// [`ArmError::Fault`](dexhal::math::ArmError::Fault); see `fault`.
    pub const FAULT: u32 = 4;
    /// The plugin panicked.
    pub const PANIC: u32 = 5;
}

/// Outcome of a fallible plugin call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiResult {
    /// One of [`status`].
    pub status: u32,
    /// [`ModelError`] code when `status` is [`status::MODEL`].
    pub model_error: u32,
    /// UTF-8 description in the plugin's static data when `status` is
    /// [`status::FAULT`].
    pub fault: FfiStr,
}

impl FfiResult {
    pub const OK: FfiResult = FfiResult::status(status::OK);
    pub const PANIC: FfiResult = FfiResult::status(status::PANIC);

    const fn status(status: u32) -> FfiResult {
        FfiResult {
            status,
            model_error: 0,
            fault: FfiStr::EMPTY,
        }
    }

    pub fn from_result(result: Result<(), ArmError>) -> FfiResult {
        match result {
            Ok(()) => FfiResult::OK,
            Err(ArmError::NotSupported) => FfiResult::status(status::NOT_SUPPORTED),
            Err(ArmError::Model(e)) => FfiResult {
                model_error: model_error_code(e),
                ..FfiResult::status(status::MODEL)
            },
            Err(ArmError::BufferTooSmall) => FfiResult::status(status::BUFFER_TOO_SMALL),
            Err(ArmError::Fault(text)) => FfiResult {
                fault: FfiStr::new(text),
                ..FfiResult::status(status::FAULT)
            },
        }
    }

    /// Back to a [`Result`].
    ///
    /// # Safety
    ///
    /// `fault` must come from a plugin that stays loaded for the rest of the
    /// process.
    pub unsafe fn into_result(self) -> Result<(), ArmError> {
        match self.status {
            status::OK => Ok(()),
            status::NOT_SUPPORTED => Err(ArmError::NotSupported),
            status::MODEL => Err(ArmError::Model(model_error_from_code(self.model_error))),
            status::BUFFER_TOO_SMALL => Err(ArmError::BufferTooSmall),
            // SAFETY: forwarded to the caller.
            status::FAULT => Err(ArmError::Fault(unsafe { self.fault.as_static() })),
            status::PANIC => Err(ArmError::Fault("driver plugin panicked")),
            _ => Err(ArmError::Fault("driver plugin returned an unknown status")),
        }
    }
}

/// A `&'static str` in the plugin's static data.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiStr {
    pub ptr: *const u8,
    pub len: usize,
}

impl FfiStr {
    pub const EMPTY: FfiStr = FfiStr {
        ptr: core::ptr::null(),
        len: 0,
    };

    pub const fn new(s: &'static str) -> FfiStr {
        FfiStr {
            ptr: s.as_ptr(),
            len: s.len(),
        }
    }

    /// The string, or `""` if it is null or not UTF-8.
    ///
    /// # Safety
    ///
    /// `ptr` / `len` must describe memory that stays valid and unchanged for
    /// the rest of the process (a plugin's static data, never unloaded).
    pub unsafe fn as_static(self) -> &'static str {
        if self.ptr.is_null() || self.len == 0 {
            return "";
        }
        // SAFETY: forwarded to the caller.
        let bytes = unsafe { core::slice::from_raw_parts(self.ptr, self.len) };
        core::str::from_utf8(bytes).unwrap_or("")
    }
}

/// Stable code of a [`ModelError`].
pub const fn model_error_code(e: ModelError) -> u32 {
    match e {
        ModelError::NonFinite => 0,
        ModelError::DegenerateAxis => 1,
        ModelError::NotRigid => 2,
        ModelError::InvalidPayload => 3,
        ModelError::NonPositive => 4,
        ModelError::ExcessiveDeviation => 5,
        ModelError::InvalidObserver => 6,
    }
}

/// Inverse of [`model_error_code`]; unknown codes read as `NonFinite`.
pub const fn model_error_from_code(code: u32) -> ModelError {
    match code {
        1 => ModelError::DegenerateAxis,
        2 => ModelError::NotRigid,
        3 => ModelError::InvalidPayload,
        4 => ModelError::NonPositive,
        5 => ModelError::ExcessiveDeviation,
        6 => ModelError::InvalidObserver,
        _ => ModelError::NonFinite,
    }
}

/// [`Applied`](dexhal::math::Applied) codes written by the `set_*` calls.
pub mod applied {
    pub const NATIVE: u8 = 0;
    pub const HOST: u8 = 1;
}

const fn wrench_frame_code(frame: WrenchFrame) -> u8 {
    match frame {
        WrenchFrame::Base => 0,
        WrenchFrame::Flange => 1,
        WrenchFrame::Tcp => 2,
    }
}

const fn wrench_frame_from_code(code: u8) -> Option<WrenchFrame> {
    match code {
        0 => Some(WrenchFrame::Base),
        1 => Some(WrenchFrame::Flange),
        2 => Some(WrenchFrame::Tcp),
        _ => None,
    }
}

const fn ik_kind_code(kind: IkKind) -> u8 {
    match kind {
        IkKind::Fast => 0,
        IkKind::Near => 1,
    }
}

const fn ik_kind_from_code(code: u8) -> Option<IkKind> {
    match code {
        0 => Some(IkKind::Fast),
        1 => Some(IkKind::Near),
        _ => None,
    }
}

/// One [`AuxField`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiAuxField {
    pub label: FfiStr,
    pub size: u32,
}

/// [`ManifestDesc::flags`] bits.
pub mod manifest_flag {
    pub const REALTIME_REQUIRED: u32 = 1 << 0;
    pub const TORQUE_SENSING: u32 = 1 << 1;
    pub const WRENCH_SENSING: u32 = 1 << 2;
    pub const INTEGRATED_GRIPPER: u32 = 1 << 3;
    pub const WRENCH_GRAVITY_COMPENSATED: u32 = 1 << 4;
    pub const NATIVE_CARTESIAN_IMPEDANCE: u32 = 1 << 5;
    pub const NATIVE_CARTESIAN_ADMITTANCE: u32 = 1 << 6;
    pub const NATIVE_JOINT_IMPEDANCE: u32 = 1 << 7;
    pub const NATIVE_CONTACT_DETECTION: u32 = 1 << 8;
    pub const NATIVE_REFLEXES: u32 = 1 << 9;
    pub const NATIVE_PAYLOAD: u32 = 1 << 10;
    pub const NATIVE_TCP_OFFSET: u32 = 1 << 11;
    pub const NATIVE_GRAVITY_COMPENSATION: u32 = 1 << 12;
    pub const NATIVE_REALTIME_MODE_SWITCH: u32 = 1 << 13;
    pub const NATIVE_NULLSPACE_POSTURE: u32 = 1 << 14;
    pub const STATIC_INERTIALS: u32 = 1 << 15;
    pub const MODEL_DYNAMICS: u32 = 1 << 16;
    pub const RESIDUAL_INERTIALS: u32 = 1 << 17;
    pub const VENDOR_DYNAMICS: u32 = 1 << 18;
    pub const LEARNED_DYNAMICS: u32 = 1 << 19;
    pub const TORQUE_OBSERVER: u32 = 1 << 20;
}

/// [`ArmManifest`] in C layout. Strings point into the plugin's static data.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ManifestDesc {
    pub name: FfiStr,
    /// `[major, minor, patch]`.
    pub version: [u32; 3],
    pub control_freq_hz: u32,
    pub control_modes: u32,
    pub active_dof_count: u32,
    /// OR of [`manifest_flag`] bits.
    pub flags: u32,
    pub aux_len: u32,
    pub aux: [FfiAuxField; MAX_AUX_FIELDS],
    pub state_staleness_cycles: u32,
    pub command_dispatch_cycles: u32,
    pub hardware_watchdog_ms: f64,
    pub torque_constants: [f64; MAX_DOF],
    pub wrench_frame: u32,
    pub joint_limits_min: [f64; MAX_DOF],
    pub joint_limits_max: [f64; MAX_DOF],
    pub velocity_limits: [f64; MAX_DOF],
    pub acceleration_limits: [f64; MAX_DOF],
    pub jerk_limits: [f64; MAX_DOF],
    pub effort_limits: [f64; MAX_DOF],
    pub cartesian_velocity_limits: [f64; 2],
    pub cartesian_acceleration_limits: [f64; 2],
    /// `[max_stiffness_k, min_damping_b, min_mass_m, max_wrench]`.
    pub compliance: [[f64; 6]; 4],
    pub gripper_max_width: f64,
    pub gripper_max_force: f64,
    pub model_dof: u32,
    pub model_max_ik_solutions: u32,
}

impl ManifestDesc {
    /// Flatten a manifest. Its strings must be `'static` in the plugin.
    pub fn from_manifest(m: &ArmManifest) -> ManifestDesc {
        // Destructured so that a new manifest field fails to compile here.
        let ArmManifest {
            name,
            version,
            control_freq_hz,
            control_modes,
            is_realtime_required,
            active_dof_count,
            aux,
            state_staleness_cycles,
            command_dispatch_cycles,
            hardware_watchdog_ms,
            has_torque_sensing,
            has_wrench_sensing,
            has_integrated_gripper,
            torque_constants,
            wrench_frame,
            wrench_is_gravity_compensated,
            native_cartesian_impedance,
            native_cartesian_admittance,
            native_joint_impedance,
            native_contact_detection,
            native_reflexes,
            native_payload,
            native_tcp_offset,
            native_gravity_compensation,
            native_realtime_mode_switch,
            native_nullspace_posture,
            joint_limits_min,
            joint_limits_max,
            velocity_limits,
            acceleration_limits,
            jerk_limits,
            effort_limits,
            cartesian_velocity_limits,
            cartesian_acceleration_limits,
            compliance,
            gripper_max_width,
            gripper_max_force,
            has_static_inertials,
            has_residual_inertials,
            dynamics_source,
            has_torque_observer,
            model,
        } = *m;
        use manifest_flag::*;
        let flags = [
            (is_realtime_required, REALTIME_REQUIRED),
            (has_torque_sensing, TORQUE_SENSING),
            (has_wrench_sensing, WRENCH_SENSING),
            (has_integrated_gripper, INTEGRATED_GRIPPER),
            (wrench_is_gravity_compensated, WRENCH_GRAVITY_COMPENSATED),
            (native_cartesian_impedance, NATIVE_CARTESIAN_IMPEDANCE),
            (native_cartesian_admittance, NATIVE_CARTESIAN_ADMITTANCE),
            (native_joint_impedance, NATIVE_JOINT_IMPEDANCE),
            (native_contact_detection, NATIVE_CONTACT_DETECTION),
            (native_reflexes, NATIVE_REFLEXES),
            (native_payload, NATIVE_PAYLOAD),
            (native_tcp_offset, NATIVE_TCP_OFFSET),
            (native_gravity_compensation, NATIVE_GRAVITY_COMPENSATION),
            (native_realtime_mode_switch, NATIVE_REALTIME_MODE_SWITCH),
            (native_nullspace_posture, NATIVE_NULLSPACE_POSTURE),
            (has_static_inertials, STATIC_INERTIALS),
            (model.dynamics, MODEL_DYNAMICS),
            (has_residual_inertials, RESIDUAL_INERTIALS),
            (dynamics_source == DynamicsSource::Vendor, VENDOR_DYNAMICS),
            (dynamics_source == DynamicsSource::Learned, LEARNED_DYNAMICS),
            (has_torque_observer, TORQUE_OBSERVER),
        ]
        .iter()
        .filter(|(set, _)| *set)
        .fold(0, |acc, (_, bit)| acc | bit);
        let mut aux_fields = [FfiAuxField {
            label: FfiStr::EMPTY,
            size: 0,
        }; MAX_AUX_FIELDS];
        for (out, field) in aux_fields.iter_mut().zip(aux.fields()) {
            *out = FfiAuxField {
                label: FfiStr::new(field.label),
                size: field.size,
            };
        }
        ManifestDesc {
            name: FfiStr::new(name),
            version: [version.major, version.minor, version.patch],
            control_freq_hz,
            control_modes,
            active_dof_count: active_dof_count.into(),
            flags,
            aux_len: aux.fields().len() as u32,
            aux: aux_fields,
            state_staleness_cycles,
            command_dispatch_cycles,
            hardware_watchdog_ms,
            torque_constants,
            wrench_frame: wrench_frame_code(wrench_frame).into(),
            joint_limits_min,
            joint_limits_max,
            velocity_limits,
            acceleration_limits,
            jerk_limits,
            effort_limits,
            cartesian_velocity_limits,
            cartesian_acceleration_limits,
            compliance: [
                compliance.max_stiffness_k,
                compliance.min_damping_b,
                compliance.min_mass_m,
                compliance.max_wrench,
            ],
            gripper_max_width,
            gripper_max_force,
            model_dof: model.dof.into(),
            model_max_ik_solutions: model.max_ik_solutions.into(),
        }
    }

    /// Rebuild the manifest; `None` if a field is out of range.
    ///
    /// # Safety
    ///
    /// The strings must come from a plugin that stays loaded for the rest of
    /// the process.
    pub unsafe fn to_manifest(&self) -> Option<ArmManifest> {
        let has = |bit: u32| self.flags & bit != 0;
        let aux_len = usize::try_from(self.aux_len).ok()?;
        let mut fields = [AuxField {
            label: "",
            size: 0,
        }; MAX_AUX_FIELDS];
        for (out, field) in fields.iter_mut().zip(self.aux.get(..aux_len)?) {
            *out = AuxField {
                // SAFETY: forwarded to the caller.
                label: unsafe { field.label.as_static() },
                size: field.size,
            };
        }
        use manifest_flag::*;
        let active_dof_count = u8::try_from(self.active_dof_count)
            .ok()
            .filter(|n| usize::from(*n) <= MAX_DOF)?;
        let aux = AuxSchema::from_slice(&fields[..aux_len])?;
        let has_static_inertials = has(STATIC_INERTIALS);
        let has_residual_inertials = has(RESIDUAL_INERTIALS);
        let model_dynamics = has(MODEL_DYNAMICS);
        if (has_static_inertials || has_residual_inertials) && !model_dynamics {
            return None;
        }
        if has(VENDOR_DYNAMICS) && has(LEARNED_DYNAMICS) {
            return None;
        }
        let dynamics_source = if has(VENDOR_DYNAMICS) {
            DynamicsSource::Vendor
        } else if has(LEARNED_DYNAMICS) {
            DynamicsSource::Learned
        } else {
            DynamicsSource::Model
        };
        if dynamics_source == DynamicsSource::Vendor
            && !aux.has_vendor_dynamics(usize::from(active_dof_count))
        {
            return None;
        }
        let [max_stiffness_k, min_damping_b, min_mass_m, max_wrench] = self.compliance;
        Some(ArmManifest {
            // SAFETY: forwarded to the caller.
            name: unsafe { self.name.as_static() },
            version: ContractVersion {
                major: self.version[0],
                minor: self.version[1],
                patch: self.version[2],
            },
            control_freq_hz: self.control_freq_hz,
            control_modes: self.control_modes,
            is_realtime_required: has(REALTIME_REQUIRED),
            active_dof_count,
            aux,
            state_staleness_cycles: self.state_staleness_cycles,
            command_dispatch_cycles: self.command_dispatch_cycles,
            hardware_watchdog_ms: self.hardware_watchdog_ms,
            has_torque_sensing: has(TORQUE_SENSING),
            has_wrench_sensing: has(WRENCH_SENSING),
            has_integrated_gripper: has(INTEGRATED_GRIPPER),
            torque_constants: self.torque_constants,
            wrench_frame: wrench_frame_from_code(u8::try_from(self.wrench_frame).ok()?)?,
            wrench_is_gravity_compensated: has(WRENCH_GRAVITY_COMPENSATED),
            native_cartesian_impedance: has(NATIVE_CARTESIAN_IMPEDANCE),
            native_cartesian_admittance: has(NATIVE_CARTESIAN_ADMITTANCE),
            native_joint_impedance: has(NATIVE_JOINT_IMPEDANCE),
            native_contact_detection: has(NATIVE_CONTACT_DETECTION),
            native_reflexes: has(NATIVE_REFLEXES),
            native_payload: has(NATIVE_PAYLOAD),
            native_tcp_offset: has(NATIVE_TCP_OFFSET),
            native_gravity_compensation: has(NATIVE_GRAVITY_COMPENSATION),
            native_realtime_mode_switch: has(NATIVE_REALTIME_MODE_SWITCH),
            native_nullspace_posture: has(NATIVE_NULLSPACE_POSTURE),
            joint_limits_min: self.joint_limits_min,
            joint_limits_max: self.joint_limits_max,
            velocity_limits: self.velocity_limits,
            acceleration_limits: self.acceleration_limits,
            jerk_limits: self.jerk_limits,
            effort_limits: self.effort_limits,
            cartesian_velocity_limits: self.cartesian_velocity_limits,
            cartesian_acceleration_limits: self.cartesian_acceleration_limits,
            compliance: ComplianceLimits {
                max_stiffness_k,
                min_damping_b,
                min_mass_m,
                max_wrench,
            },
            gripper_max_width: self.gripper_max_width,
            gripper_max_force: self.gripper_max_force,
            has_static_inertials,
            has_residual_inertials,
            dynamics_source,
            has_torque_observer: has(TORQUE_OBSERVER),
            model: ModelCapabilities {
                dof: u8::try_from(self.model_dof).ok()?,
                max_ik_solutions: u16::try_from(self.model_max_ik_solutions).ok()?,
                dynamics: model_dynamics,
            },
        })
    }
}

/// A host-side model inside the plugin: either the driver's own
/// (`of_driver`, `ptr` is the driver handle) or an independent one from
/// `math_create`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MathHandle {
    pub ptr: *mut c_void,
    pub of_driver: bool,
}

type Q = [f64; MAX_DOF];

/// The function table a driver library exports. All pointers passed in must
/// be valid for the duration of the call; `driver` / `math` handles must come
/// from `driver_create` / `math_create` of the same table.
#[repr(C)]
pub struct PluginVTable {
    /// [`ABI_VERSION`] the plugin was built with.
    pub abi_version: u32,
    /// [`CONTRACT_VERSION`](dexhal::math::CONTRACT_VERSION) the plugin was
    /// built with.
    pub contract_major: u32,
    pub contract_minor: u32,
    pub contract_patch: u32,

    // --- driver lifecycle --------------------------------------------------
    pub driver_create:
        unsafe extern "C" fn(config: *const u8, len: usize, out: *mut *mut c_void) -> FfiResult,
    pub driver_destroy: unsafe extern "C" fn(driver: *mut c_void),
    pub driver_manifest: unsafe extern "C" fn(driver: *const c_void, out: *mut ManifestDesc) -> bool,
    pub connect: unsafe extern "C" fn(driver: *mut c_void) -> FfiResult,
    pub disconnect: unsafe extern "C" fn(driver: *mut c_void) -> FfiResult,

    // --- data plane --------------------------------------------------------
    pub read_state: unsafe extern "C" fn(driver: *mut c_void, buf: *mut f64, len: usize) -> FfiResult,
    pub write_command:
        unsafe extern "C" fn(driver: *mut c_void, buf: *const f64, len: usize) -> FfiResult,

    // --- configuration plane -----------------------------------------------
    /// Writes an [`applied`] code to `out`.
    pub set_payload: unsafe extern "C" fn(
        driver: *mut c_void,
        mass_kg: f64,
        com: *const [f64; 3],
        inertia: *const [f64; 9],
        out: *mut u8,
    ) -> FfiResult,
    pub set_tcp_offset:
        unsafe extern "C" fn(driver: *mut c_void, pose: *const [f64; 7], out: *mut u8) -> FfiResult,
    pub set_gravity:
        unsafe extern "C" fn(driver: *mut c_void, gravity: *const [f64; 3], out: *mut u8) -> FfiResult,
    pub set_calibration:
        unsafe extern "C" fn(driver: *mut c_void, calibration: *const CalibrationData) -> FfiResult,
    pub clear_calibration: unsafe extern "C" fn(driver: *mut c_void) -> FfiResult,
    pub set_ik_max_joint_step: unsafe extern "C" fn(driver: *mut c_void, step: f64) -> FfiResult,
    pub clear_faults: unsafe extern "C" fn(driver: *mut c_void) -> FfiResult,
    /// Writes the reply length to `out_len`.
    pub vendor_command: unsafe extern "C" fn(
        driver: *mut c_void,
        name: *const u8,
        name_len: usize,
        payload: *const u8,
        payload_len: usize,
        response: *mut u8,
        response_len: usize,
        out_len: *mut usize,
    ) -> FfiResult,

    // --- host-side model ---------------------------------------------------
    pub math_create: unsafe extern "C" fn(out: *mut *mut c_void) -> FfiResult,
    pub math_destroy: unsafe extern "C" fn(math: *mut c_void),
    pub math_dof: unsafe extern "C" fn(math: MathHandle) -> usize,
    pub math_is_calibrated: unsafe extern "C" fn(math: MathHandle) -> bool,
    /// The `math_*` queries return `false` where the Rust API returns `None`.
    pub math_forward_kinematics:
        unsafe extern "C" fn(math: MathHandle, q: *const Q, out: *mut [f64; 7]) -> bool,
    pub math_jacobian:
        unsafe extern "C" fn(math: MathHandle, q: *const Q, out: *mut [Q; 6]) -> bool,
    pub math_tool_jacobian:
        unsafe extern "C" fn(math: MathHandle, q: *const Q, out: *mut [Q; 6]) -> bool,
    pub math_jacobian_pseudo_inverse:
        unsafe extern "C" fn(math: MathHandle, q: *const Q, out: *mut [[f64; 6]; MAX_DOF]) -> bool,
    /// `kind` is an `IkKind` code (0 = Fast, 1 = Near); an unknown code
    /// returns `false`.
    pub math_ik_one: unsafe extern "C" fn(
        math: MathHandle,
        target: *const [f64; 7],
        seed: *const Q,
        kind: u8,
        out: *mut Q,
    ) -> bool,
    /// Writes every solution to `out` (sorted nearest-first when `seed` is
    /// non-null, otherwise driver order) and the number written to `count`.
    /// A `cap` below the solution count writes the first `cap` solutions
    /// (the nearest when seeded). Returns `false` on a panic.
    pub math_ik_all: unsafe extern "C" fn(
        math: MathHandle,
        target: *const [f64; 7],
        seed: *const Q,
        out: *mut Q,
        cap: usize,
        count: *mut usize,
    ) -> bool,
    pub math_dynamics: unsafe extern "C" fn(
        math: MathHandle,
        q: *const Q,
        qd: *const Q,
        out: *mut DynamicsState,
    ) -> bool,
    pub math_inverse_dynamics: unsafe extern "C" fn(
        math: MathHandle,
        q: *const Q,
        qd: *const Q,
        qdd: *const Q,
        out: *mut Q,
    ) -> bool,
    /// Writes `[position, rotation]`.
    pub math_pose_residual: unsafe extern "C" fn(
        math: MathHandle,
        q: *const Q,
        reported: *const [f64; 7],
        out: *mut [f64; 2],
    ) -> bool,
    pub math_external_wrench: unsafe extern "C" fn(
        math: MathHandle,
        q: *const Q,
        reported: *const [f64; 6],
        frame: u32,
        gravity_compensated: bool,
        out: *mut [f64; 6],
    ) -> bool,
    pub math_apply_payload: unsafe extern "C" fn(
        math: MathHandle,
        mass_kg: f64,
        com: *const [f64; 3],
        inertia: *const [f64; 9],
    ) -> FfiResult,
    pub math_apply_tcp_offset:
        unsafe extern "C" fn(math: MathHandle, pose: *const [f64; 7]) -> FfiResult,
    pub math_apply_gravity:
        unsafe extern "C" fn(math: MathHandle, gravity: *const [f64; 3]) -> FfiResult,
    pub math_apply_calibration:
        unsafe extern "C" fn(math: MathHandle, calibration: *const CalibrationData) -> FfiResult,
    pub math_remove_calibration: unsafe extern "C" fn(math: MathHandle) -> FfiResult,
    pub math_apply_ik_max_joint_step:
        unsafe extern "C" fn(math: MathHandle, step: f64) -> FfiResult,
    /// Free-space torque for this control tick, advancing the learned
    /// torque observer if any. q_cmd may be null.
    pub math_step_free_torque: unsafe extern "C" fn(
        math: MathHandle,
        q: *const Q,
        qd: *const Q,
        qdd: *const Q,
        q_cmd: *const Q,
        control_freq_hz: u32,
        out: *mut Q,
    ) -> bool,
    pub math_reset_torque_observer: unsafe extern "C" fn(math: MathHandle) -> FfiResult,
    /// Inverse dynamics of the host payload alone.
    pub math_payload_torque: unsafe extern "C" fn(
        math: MathHandle,
        q: *const Q,
        qd: *const Q,
        qdd: *const Q,
        out: *mut Q,
    ) -> bool,
}

/// Export `$driver` (a [`PluginDriver`]) as this library's
/// [`ENTRY_SYMBOL`].
#[macro_export]
macro_rules! export_plugin {
    ($driver:ty) => {
        #[allow(unsafe_code)]
        #[no_mangle]
        pub extern "C" fn dexhal_plugin_entry() -> *const $crate::PluginVTable {
            static VTABLE: $crate::PluginVTable = $crate::vtable::<$driver>();
            &VTABLE
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_round_trip() {
        let errors = [
            ModelError::NonFinite,
            ModelError::DegenerateAxis,
            ModelError::NotRigid,
            ModelError::InvalidPayload,
            ModelError::NonPositive,
            ModelError::ExcessiveDeviation,
            ModelError::InvalidObserver,
        ];
        for e in errors {
            assert_eq!(model_error_from_code(model_error_code(e)), e);
            let r = FfiResult::from_result(Err(ArmError::Model(e)));
            assert_eq!(unsafe { r.into_result() }, Err(ArmError::Model(e)));
        }
        for e in [
            ArmError::NotSupported,
            ArmError::BufferTooSmall,
            ArmError::Fault("bus off"),
        ] {
            let r = FfiResult::from_result(Err(e));
            assert_eq!(unsafe { r.into_result() }, Err(e));
        }
        assert_eq!(unsafe { FfiResult::OK.into_result() }, Ok(()));
        assert!(unsafe { FfiResult::PANIC.into_result() }.is_err());
    }

    #[test]
    fn wrench_frames_round_trip() {
        for f in [WrenchFrame::Base, WrenchFrame::Flange, WrenchFrame::Tcp] {
            assert_eq!(wrench_frame_from_code(wrench_frame_code(f)), Some(f));
        }
        assert_eq!(wrench_frame_from_code(3), None);
    }

    #[test]
    fn manifest_desc_round_trips_and_validates_dynamics_flags() {
        use dexhal::math::{
            aux_dynamics,
            model::{ArmModel, IkKind, Pose},
            tolerances::TolerancePolicy,
            JointRatings,
        };

        struct Demo;
        impl ArmModel<2, 1> for Demo {
            const NAME: &'static str = "demo";
            const HAS_DYNAMICS: bool = true;
            fn flange_fk(_q: &[f64; 2]) -> Pose {
                Pose::identity()
            }
            fn flange_jacobian(_q: &[f64; 2]) -> [[f64; 2]; 6] {
                [[0.0; 2]; 6]
            }
            fn flange_all(
                _t: &Pose,
                _s: Option<&[f64; 2]>,
                _l: bool,
                _r: bool,
                _p: TolerancePolicy,
                _i: usize,
            ) -> [Option<[f64; 2]>; 1] {
                [None]
            }
            fn flange_one(
                _t: &Pose,
                _s: &[f64; 2],
                _k: IkKind,
                _p: TolerancePolicy,
                _i: usize,
            ) -> Option<[f64; 2]> {
                None
            }
        }

        let ratings = JointRatings {
            velocity: [1.0, 1.0],
            acceleration: [2.0, 2.0],
            effort: [3.0, 3.0],
        };
        let mut m = ArmManifest::for_model::<Demo, 2, 1>(500, 1, &ratings).unwrap();
        m.has_static_inertials = false;
        m.has_residual_inertials = true;
        m.dynamics_source = DynamicsSource::Vendor;
        m.aux = AuxSchema::from_slice(&[
            AuxField {
                label: aux_dynamics::GRAVITY_TORQUE,
                size: 2,
            },
            AuxField {
                label: aux_dynamics::CORIOLIS_TORQUE,
                size: 2,
            },
            AuxField {
                label: aux_dynamics::MASS_MATRIX,
                size: 4,
            },
        ])
        .unwrap();

        let desc = ManifestDesc::from_manifest(&m);
        let rebuilt = unsafe { desc.to_manifest() }.unwrap();
        assert!(!rebuilt.has_static_inertials);
        assert!(rebuilt.has_residual_inertials);
        assert_eq!(rebuilt.dynamics_source, DynamicsSource::Vendor);

        // Contradictory flags are rejected across the FFI boundary.
        let mut bad_no_dyn = desc;
        bad_no_dyn.flags &= !manifest_flag::MODEL_DYNAMICS;
        assert!(unsafe { bad_no_dyn.to_manifest() }.is_none());

        let mut bad_no_aux = desc;
        bad_no_aux.aux_len = 0;
        assert!(unsafe { bad_no_aux.to_manifest() }.is_none());

        let mut both = desc;
        both.flags |= manifest_flag::LEARNED_DYNAMICS;
        assert!(unsafe { both.to_manifest() }.is_none());

        // A learned observer survives the round trip and enables Route 6.
        m.dynamics_source = DynamicsSource::Learned;
        m.has_residual_inertials = false;
        m.has_torque_observer = true;
        let rebuilt = unsafe { ManifestDesc::from_manifest(&m).to_manifest() }.unwrap();
        assert_eq!(rebuilt.dynamics_source, DynamicsSource::Learned);
        assert!(rebuilt.has_torque_observer);
        assert!(rebuilt.has_residual_dynamics());
        assert!(!rebuilt.has_impedance_dynamics());
    }
}
