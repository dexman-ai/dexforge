//! Host side: [`Plugin`], [`LoadedDriver`] and [`LoadedMath`].

use core::ffi::c_void;
use core::fmt;
use std::path::Path;

use dexhal::math::{
    Applied, ArmDriver, ArmError, ArmManifest, ArmMath, CalibrationData, DynamicsState, IkKind,
    ModelError, PoseResidual, WrenchFrame, CONTRACT_VERSION, MAX_DOF,
};

use crate::{
    applied, ik_kind_code, wrench_frame_code, EntryFn, FfiResult, ManifestDesc, MathHandle, PluginVTable, Q,
    ABI_VERSION, ENTRY_SYMBOL,
};

/// Why a plugin could not be loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PluginError {
    /// The dynamic library could not be opened.
    Open(String),
    /// The library does not export [`ENTRY_SYMBOL`].
    MissingEntry,
    /// The entry point returned null.
    NullVTable,
    /// The plugin was built against another [`ABI_VERSION`].
    AbiMismatch { found: u32 },
    /// The plugin was built against an incompatible contract `major.minor`.
    ContractMismatch { major: u32, minor: u32 },
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::Open(why) => write!(f, "cannot open driver plugin: {why}"),
            PluginError::MissingEntry => write!(f, "driver plugin does not export {ENTRY_SYMBOL}"),
            PluginError::NullVTable => write!(f, "driver plugin returned a null vtable"),
            PluginError::AbiMismatch { found } => write!(
                f,
                "driver plugin ABI {found}, host expects {ABI_VERSION}"
            ),
            PluginError::ContractMismatch { major, minor } => write!(
                f,
                "driver plugin contract {major}.{minor}, host implements {}.{}",
                CONTRACT_VERSION.major, CONTRACT_VERSION.minor
            ),
        }
    }
}

impl std::error::Error for PluginError {}

/// A driver library whose ABI and contract version have been checked.
#[derive(Clone, Copy)]
pub struct Plugin {
    vtable: &'static PluginVTable,
}

impl fmt::Debug for Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plugin")
            .field("abi_version", &self.vtable.abi_version)
            .finish_non_exhaustive()
    }
}

impl Plugin {
    /// Check and wrap the table returned by `entry`.
    ///
    /// # Safety
    ///
    /// `entry` must be a [`ENTRY_SYMBOL`] produced by
    /// [`export_plugin!`](crate::export_plugin) whose code stays loaded for
    /// the rest of the process.
    pub unsafe fn from_entry(entry: EntryFn) -> Result<Plugin, PluginError> {
        // SAFETY: forwarded to the caller.
        let vtable = unsafe { entry().as_ref() }.ok_or(PluginError::NullVTable)?;
        // Only the leading version fields are read before the check.
        if vtable.abi_version != ABI_VERSION {
            return Err(PluginError::AbiMismatch {
                found: vtable.abi_version,
            });
        }
        if vtable.contract_major != CONTRACT_VERSION.major
            || vtable.contract_minor != CONTRACT_VERSION.minor
        {
            return Err(PluginError::ContractMismatch {
                major: vtable.contract_major,
                minor: vtable.contract_minor,
            });
        }
        Ok(Plugin { vtable })
    }

    /// Open the dynamic library at `path` and check its entry point. The
    /// library is never unloaded.
    ///
    /// # Safety
    ///
    /// Loading a library runs its initialisers; `path` must be a trusted
    /// dexhal driver plugin.
    pub unsafe fn load(path: &Path) -> Result<Plugin, PluginError> {
        // SAFETY: forwarded to the caller.
        let entry = unsafe { sys::open_entry(path) }?;
        // SAFETY: the library is never closed.
        unsafe { Plugin::from_entry(entry) }
    }

    /// The raw function table.
    pub fn vtable(&self) -> &'static PluginVTable {
        self.vtable
    }

    /// Create a driver from its configuration blob.
    pub fn create_driver(&self, config: &[u8]) -> Result<LoadedDriver, ArmError> {
        let vt = self.vtable;
        let mut handle = core::ptr::null_mut();
        // SAFETY: valid config slice and out pointer.
        unsafe { (vt.driver_create)(config.as_ptr(), config.len(), &mut handle).into_result() }?;
        if handle.is_null() {
            return Err(ArmError::Fault("driver plugin returned a null driver"));
        }
        let mut driver = LoadedDriver {
            vtable: vt,
            handle,
            manifest: None,
            math: LoadedMath {
                vtable: vt,
                handle: MathHandle {
                    ptr: handle,
                    of_driver: true,
                },
                owned: false,
            },
        };
        let mut desc = core::mem::MaybeUninit::<ManifestDesc>::uninit();
        // SAFETY: live handle; `desc` is written when `true` is returned, and
        // its strings point into the never-unloaded plugin.
        let manifest = unsafe {
            if !(vt.driver_manifest)(handle, desc.as_mut_ptr()) {
                return Err(ArmError::Fault("driver plugin failed to describe its manifest"));
            }
            desc.assume_init().to_manifest()
        }
        .ok_or(ArmError::Fault("driver plugin manifest is malformed"))?;
        driver.manifest = Some(manifest);
        Ok(driver)
    }

    /// Create an independent host-side model, e.g. for a planner thread.
    pub fn create_math(&self) -> Result<LoadedMath, ArmError> {
        let mut handle = core::ptr::null_mut();
        // SAFETY: valid out pointer.
        unsafe { (self.vtable.math_create)(&mut handle).into_result() }?;
        if handle.is_null() {
            return Err(ArmError::Fault("driver plugin returned a null model"));
        }
        Ok(LoadedMath {
            vtable: self.vtable,
            handle: MathHandle {
                ptr: handle,
                of_driver: false,
            },
            owned: true,
        })
    }
}

/// A driver living in a plugin, as an [`ArmDriver`]. Dropping it destroys the
/// driver.
pub struct LoadedDriver {
    vtable: &'static PluginVTable,
    handle: *mut c_void,
    // `None` only while being constructed.
    manifest: Option<ArmManifest>,
    math: LoadedMath,
}

// SAFETY: the plugin-side driver is `Send` (`PluginDriver: ArmDriver: Send`)
// and the handle is used by one owner at a time.
unsafe impl Send for LoadedDriver {}

impl Drop for LoadedDriver {
    fn drop(&mut self) {
        // SAFETY: live handle, destroyed once.
        unsafe { (self.vtable.driver_destroy)(self.handle) };
    }
}

/// Result of a call that writes an [`applied`] code.
///
/// # Safety
///
/// `call` must be a vtable call of a never-unloaded plugin.
unsafe fn applied_call(call: impl FnOnce(*mut u8) -> FfiResult) -> Result<Applied, ArmError> {
    let mut side = applied::HOST;
    // SAFETY: forwarded to the caller.
    unsafe { call(&mut side).into_result() }?;
    Ok(if side == applied::NATIVE {
        Applied::Native
    } else {
        Applied::Host
    })
}

impl ArmDriver for LoadedDriver {
    fn manifest(&self) -> &ArmManifest {
        self.manifest
            .as_ref()
            .expect("manifest is set at construction")
    }

    fn connect(&mut self) -> Result<(), ArmError> {
        // SAFETY: live handle.
        unsafe { (self.vtable.connect)(self.handle).into_result() }
    }

    fn disconnect(&mut self) -> Result<(), ArmError> {
        // SAFETY: live handle.
        unsafe { (self.vtable.disconnect)(self.handle).into_result() }
    }

    fn read_state(&mut self, state_buffer: &mut [f64]) -> Result<(), ArmError> {
        // SAFETY: live handle and a valid buffer.
        unsafe {
            (self.vtable.read_state)(self.handle, state_buffer.as_mut_ptr(), state_buffer.len())
                .into_result()
        }
    }

    fn write_command(&mut self, cmd_buffer: &[f64]) -> Result<(), ArmError> {
        // SAFETY: live handle and a valid buffer.
        unsafe {
            (self.vtable.write_command)(self.handle, cmd_buffer.as_ptr(), cmd_buffer.len())
                .into_result()
        }
    }

    fn arm(&self) -> &dyn ArmMath {
        &self.math
    }

    fn arm_mut(&mut self) -> &mut dyn ArmMath {
        &mut self.math
    }

    fn set_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<Applied, ArmError> {
        // SAFETY: live handle and valid pointers.
        unsafe {
            applied_call(|out| (self.vtable.set_payload)(self.handle, mass_kg, &com, &inertia, out))
        }
    }

    fn set_tcp_offset(&mut self, pose: [f64; 7]) -> Result<Applied, ArmError> {
        // SAFETY: live handle and valid pointers.
        unsafe { applied_call(|out| (self.vtable.set_tcp_offset)(self.handle, &pose, out)) }
    }

    fn set_gravity(&mut self, gravity: [f64; 3]) -> Result<Applied, ArmError> {
        // SAFETY: live handle and valid pointers.
        unsafe { applied_call(|out| (self.vtable.set_gravity)(self.handle, &gravity, out)) }
    }

    fn set_calibration(&mut self, calibration: &CalibrationData) -> Result<(), ArmError> {
        // SAFETY: live handle and valid pointer.
        unsafe { (self.vtable.set_calibration)(self.handle, calibration).into_result() }
    }

    fn clear_calibration(&mut self) {
        // SAFETY: live handle. A plugin panic leaves the calibration as it was.
        let _ = unsafe { (self.vtable.clear_calibration)(self.handle).into_result() };
    }

    fn set_ik_max_joint_step(&mut self, step: f64) -> Result<(), ArmError> {
        // SAFETY: live handle.
        unsafe { (self.vtable.set_ik_max_joint_step)(self.handle, step).into_result() }
    }

    fn clear_faults(&mut self) -> Result<(), ArmError> {
        // SAFETY: live handle.
        unsafe { (self.vtable.clear_faults)(self.handle).into_result() }
    }

    fn handle_vendor_command(
        &mut self,
        cmd_name: &str,
        payload: &[u8],
        response: &mut [u8],
    ) -> Result<usize, ArmError> {
        let mut n = 0;
        // SAFETY: live handle and valid buffers.
        unsafe {
            (self.vtable.vendor_command)(
                self.handle,
                cmd_name.as_ptr(),
                cmd_name.len(),
                payload.as_ptr(),
                payload.len(),
                response.as_mut_ptr(),
                response.len(),
                &mut n,
            )
            .into_result()
        }?;
        Ok(n.min(response.len()))
    }
}

/// A host-side model living in a plugin, as an [`ArmMath`]: either the
/// driver's own ([`LoadedDriver::arm`]) or an independent one
/// ([`Plugin::create_math`], destroyed on drop).
pub struct LoadedMath {
    vtable: &'static PluginVTable,
    handle: MathHandle,
    owned: bool,
}

// SAFETY: the plugin-side model is `Send` (`ArmMath: Send`) and the handle is
// used by one owner at a time.
unsafe impl Send for LoadedMath {}

impl Drop for LoadedMath {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: live handle, destroyed once.
            unsafe { (self.vtable.math_destroy)(self.handle.ptr) };
        }
    }
}

impl LoadedMath {
    /// Run a query that writes its result to an out pointer.
    fn query<R>(&self, call: impl FnOnce(MathHandle, *mut R) -> bool) -> Option<R> {
        let mut out = core::mem::MaybeUninit::<R>::uninit();
        // SAFETY: the plugin initialises `out` when it returns `true`.
        call(self.handle, out.as_mut_ptr()).then(|| unsafe { out.assume_init() })
    }

    fn model_call(&mut self, call: impl FnOnce(MathHandle) -> FfiResult) -> Result<(), ModelError> {
        // SAFETY: the plugin is never unloaded.
        match unsafe { call(self.handle).into_result() } {
            Ok(()) => Ok(()),
            Err(ArmError::Model(e)) => Err(e),
            // A panic or fault has no `ModelError`; report the closest one.
            Err(_) => Err(ModelError::NonFinite),
        }
    }
}

impl ArmMath for LoadedMath {
    fn dof(&self) -> usize {
        // SAFETY: live handle.
        unsafe { (self.vtable.math_dof)(self.handle) }
    }

    fn is_calibrated(&self) -> bool {
        // SAFETY: live handle.
        unsafe { (self.vtable.math_is_calibrated)(self.handle) }
    }

    fn forward_kinematics(&self, q: &Q) -> Option<[f64; 7]> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_forward_kinematics)(h, q, out) })
    }

    fn jacobian_padded(&self, q: &Q) -> Option<[Q; 6]> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_jacobian)(h, q, out) })
    }

    fn tool_jacobian_padded(&self, q: &Q) -> Option<[Q; 6]> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_tool_jacobian)(h, q, out) })
    }

    fn jacobian_pseudo_inverse(&self, q: &Q) -> Option<[[f64; 6]; MAX_DOF]> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_jacobian_pseudo_inverse)(h, q, out) })
    }

    fn ik_one(&self, target_pose: &[f64; 7], q_seed: &Q, kind: IkKind) -> Option<Q> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe {
            (self.vtable.math_ik_one)(h, target_pose, q_seed, ik_kind_code(kind), out)
        })
    }

    fn ik_all(&self, target_pose: &[f64; 7], q_seed: Option<&Q>, out: &mut [Q]) -> usize {
        let mut count = 0usize;
        let seed = q_seed.map_or(core::ptr::null(), |q| q as *const Q);
        // SAFETY: live handle; `seed` is null or valid for a read; `out` holds
        // `out.len()` writable slots.
        let ok = unsafe {
            (self.vtable.math_ik_all)(
                self.handle,
                target_pose,
                seed,
                out.as_mut_ptr(),
                out.len(),
                &mut count,
            )
        };
        if ok {
            count.min(out.len())
        } else {
            0
        }
    }

    fn dynamics_padded(&self, q: &Q, qd: &Q) -> Option<DynamicsState> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_dynamics)(h, q, qd, out) })
    }

    fn inverse_dynamics_padded(&self, q: &Q, qd: &Q, qdd: &Q) -> Option<Q> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_inverse_dynamics)(h, q, qd, qdd, out) })
    }

    fn pose_residual(&self, q: &Q, reported_tcp: &[f64; 7]) -> Option<PoseResidual> {
        // SAFETY: live handle and valid pointers.
        let [position, rotation] = self.query(|h, out| unsafe {
            (self.vtable.math_pose_residual)(h, q, reported_tcp, out)
        })?;
        Some(PoseResidual { position, rotation })
    }

    fn external_wrench_padded(
        &self,
        q: &Q,
        reported: &[f64; 6],
        frame: WrenchFrame,
        gravity_compensated: bool,
    ) -> Option<[f64; 6]> {
        let frame = wrench_frame_code(frame).into();
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe {
            (self.vtable.math_external_wrench)(h, q, reported, frame, gravity_compensated, out)
        })
    }

    fn apply_payload(
        &mut self,
        mass_kg: f64,
        com: [f64; 3],
        inertia: [f64; 9],
    ) -> Result<(), ModelError> {
        let f = self.vtable.math_apply_payload;
        // SAFETY: live handle and valid pointers.
        self.model_call(|h| unsafe { f(h, mass_kg, &com, &inertia) })
    }

    fn apply_tcp_offset(&mut self, pose: [f64; 7]) -> Result<(), ModelError> {
        let f = self.vtable.math_apply_tcp_offset;
        // SAFETY: live handle and valid pointer.
        self.model_call(|h| unsafe { f(h, &pose) })
    }

    fn apply_gravity(&mut self, gravity: [f64; 3]) -> Result<(), ModelError> {
        let f = self.vtable.math_apply_gravity;
        // SAFETY: live handle and valid pointer.
        self.model_call(|h| unsafe { f(h, &gravity) })
    }

    fn apply_calibration(&mut self, calibration: &CalibrationData) -> Result<(), ModelError> {
        let f = self.vtable.math_apply_calibration;
        // SAFETY: live handle and valid pointer.
        self.model_call(|h| unsafe { f(h, calibration) })
    }

    fn remove_calibration(&mut self) {
        let f = self.vtable.math_remove_calibration;
        // SAFETY: live handle.
        let _ = self.model_call(|h| unsafe { f(h) });
    }

    fn apply_ik_max_joint_step(&mut self, step: f64) -> Result<(), ModelError> {
        let f = self.vtable.math_apply_ik_max_joint_step;
        // SAFETY: live handle.
        self.model_call(|h| unsafe { f(h, step) })
    }

    fn step_free_torque_padded(
        &mut self,
        q: &Q,
        qd: &Q,
        qdd: &Q,
        q_cmd: Option<&Q>,
        control_freq_hz: u32,
    ) -> Option<Q> {
        let f = self.vtable.math_step_free_torque;
        let cmd = q_cmd.map_or(core::ptr::null(), |c| c as *const Q);
        // SAFETY: live handle and valid pointers; `cmd` is null or valid.
        self.query(|h, out| unsafe { f(h, q, qd, qdd, cmd, control_freq_hz, out) })
    }

    fn reset_torque_observer_state(&mut self) {
        let f = self.vtable.math_reset_torque_observer;
        // SAFETY: live handle.
        let _ = self.model_call(|h| unsafe { f(h) });
    }

    fn payload_torque_padded(&self, q: &Q, qd: &Q, qdd: &Q) -> Option<Q> {
        // SAFETY: live handle and valid pointers.
        self.query(|h, out| unsafe { (self.vtable.math_payload_torque)(h, q, qd, qdd, out) })
    }
}

/// Minimal dynamic-library loading, without dependencies. Libraries are never
/// closed.
mod sys {
    use super::{EntryFn, PluginError, ENTRY_SYMBOL};
    use core::ffi::c_void;
    use std::path::Path;

    #[cfg(windows)]
    mod imp {
        use core::ffi::c_void;
        use std::os::windows::ffi::OsStrExt;
        use std::path::Path;

        #[link(name = "kernel32")]
        extern "system" {
            fn LoadLibraryW(name: *const u16) -> *mut c_void;
            fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
            fn GetLastError() -> u32;
        }

        pub unsafe fn open(path: &Path) -> Result<*mut c_void, String> {
            let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
            // SAFETY: NUL-terminated wide string.
            let module = unsafe { LoadLibraryW(wide.as_ptr()) };
            if module.is_null() {
                // SAFETY: plain FFI call.
                return Err(format!("LoadLibraryW failed with error {}", unsafe {
                    GetLastError()
                }));
            }
            Ok(module)
        }

        pub unsafe fn symbol(module: *mut c_void, name: &[u8]) -> *mut c_void {
            // SAFETY: `name` is NUL-terminated; `module` is open.
            unsafe { GetProcAddress(module, name.as_ptr()) }
        }
    }

    #[cfg(unix)]
    mod imp {
        use core::ffi::{c_char, c_int, c_void, CStr};
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::path::Path;

        const RTLD_NOW: c_int = 2;

        #[cfg_attr(any(target_os = "linux", target_os = "android"), link(name = "dl"))]
        extern "C" {
            fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            fn dlerror() -> *const c_char;
        }

        pub unsafe fn open(path: &Path) -> Result<*mut c_void, String> {
            let name = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| "path contains NUL".to_string())?;
            // SAFETY: NUL-terminated path.
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                // SAFETY: dlerror returns null or a NUL-terminated message.
                let why = unsafe { dlerror() };
                return Err(if why.is_null() {
                    "dlopen failed".to_string()
                } else {
                    unsafe { CStr::from_ptr(why) }.to_string_lossy().into_owned()
                });
            }
            Ok(handle)
        }

        pub unsafe fn symbol(handle: *mut c_void, name: &[u8]) -> *mut c_void {
            // SAFETY: `name` is NUL-terminated; `handle` is open.
            unsafe { dlsym(handle, name.as_ptr().cast()) }
        }
    }

    pub unsafe fn open_entry(path: &Path) -> Result<EntryFn, PluginError> {
        // SAFETY: forwarded to the caller.
        let library = unsafe { imp::open(path) }.map_err(PluginError::Open)?;
        let mut name = ENTRY_SYMBOL.as_bytes().to_vec();
        name.push(0);
        // SAFETY: open library, NUL-terminated name.
        let symbol: *mut c_void = unsafe { imp::symbol(library, &name) };
        if symbol.is_null() {
            return Err(PluginError::MissingEntry);
        }
        // SAFETY: the symbol is the plugin's `extern "C" fn` entry point.
        Ok(unsafe { core::mem::transmute::<*mut c_void, EntryFn>(symbol) })
    }
}
