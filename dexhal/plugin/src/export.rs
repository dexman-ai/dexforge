//! Driver side: the generic `extern "C"` functions behind [`vtable`].

use core::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use dexhal::math::{
    Applied, ArmDriver, ArmError, ArmMath, CalibrationData, DynamicsState, CONTRACT_VERSION,
    MAX_DOF,
};

use crate::{
    applied, wrench_frame_from_code, FfiResult, ManifestDesc, MathHandle, PluginVTable, Q,
    ABI_VERSION,
};

/// A driver that can be exported with [`export_plugin!`](crate::export_plugin).
pub trait PluginDriver: ArmDriver + Sized + 'static {
    /// Build the driver from an opaque, driver-defined configuration blob
    /// (e.g. UTF-8 text with the controller address).
    fn create(config: &[u8]) -> Result<Self, ArmError>;

    /// An independent host-side model of the same arm, for a thread other
    /// than the one driving the hardware.
    fn create_math() -> Box<dyn ArmMath>;
}

fn guard(f: impl FnOnce() -> Result<(), ArmError>) -> FfiResult {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => FfiResult::from_result(result),
        Err(_) => FfiResult::PANIC,
    }
}

fn guard_some(f: impl FnOnce() -> Option<()>) -> bool {
    matches!(catch_unwind(AssertUnwindSafe(f)), Ok(Some(())))
}

/// # Safety
///
/// `ptr` must be null or valid for `len` elements; empty when `len == 0`.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: forwarded to the caller.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

/// # Safety
///
/// As [`slice`], for writes.
unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if ptr.is_null() || len == 0 {
        &mut []
    } else {
        // SAFETY: forwarded to the caller.
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }
}

/// # Safety
///
/// `driver` must be a live handle from `driver_create::<T>`.
unsafe fn driver<'a, T>(driver: *mut c_void) -> &'a mut T {
    // SAFETY: forwarded to the caller.
    unsafe { &mut *driver.cast::<T>() }
}

/// # Safety
///
/// `h` must be a live handle of this vtable.
unsafe fn math_ref<'a, T: PluginDriver>(h: MathHandle) -> &'a dyn ArmMath {
    // SAFETY: forwarded to the caller.
    unsafe {
        if h.of_driver {
            (*h.ptr.cast::<T>()).arm()
        } else {
            &**h.ptr.cast::<Box<dyn ArmMath>>()
        }
    }
}

/// # Safety
///
/// `h` must be a live handle of this vtable, not aliased during the call.
unsafe fn math_mut<'a, T: PluginDriver>(h: MathHandle) -> &'a mut dyn ArmMath {
    // SAFETY: forwarded to the caller.
    unsafe {
        if h.of_driver {
            (*h.ptr.cast::<T>()).arm_mut()
        } else {
            &mut **h.ptr.cast::<Box<dyn ArmMath>>()
        }
    }
}

fn applied_code(side: Applied) -> u8 {
    match side {
        Applied::Native => applied::NATIVE,
        Applied::Host => applied::HOST,
    }
}

unsafe extern "C" fn driver_create<T: PluginDriver>(
    config: *const u8,
    len: usize,
    out: *mut *mut c_void,
) -> FfiResult {
    guard(|| {
        // SAFETY: the host passes a valid config slice and out pointer.
        let config = unsafe { slice(config, len) };
        let boxed = Box::new(T::create(config)?);
        unsafe { out.write(Box::into_raw(boxed).cast()) };
        Ok(())
    })
}

unsafe extern "C" fn driver_destroy<T: PluginDriver>(d: *mut c_void) {
    if !d.is_null() {
        // SAFETY: `d` came from `driver_create::<T>` and is destroyed once.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(d.cast::<T>()) })));
    }
}

unsafe extern "C" fn driver_manifest<T: PluginDriver>(
    d: *const c_void,
    out: *mut ManifestDesc,
) -> bool {
    guard_some(|| {
        // SAFETY: live handle and valid out pointer.
        let manifest = unsafe { &*d.cast::<T>() }.manifest();
        unsafe { out.write(ManifestDesc::from_manifest(manifest)) };
        Some(())
    })
}

unsafe extern "C" fn connect<T: PluginDriver>(d: *mut c_void) -> FfiResult {
    // SAFETY: live handle.
    guard(|| unsafe { driver::<T>(d) }.connect())
}

unsafe extern "C" fn disconnect<T: PluginDriver>(d: *mut c_void) -> FfiResult {
    // SAFETY: live handle.
    guard(|| unsafe { driver::<T>(d) }.disconnect())
}

unsafe extern "C" fn read_state<T: PluginDriver>(
    d: *mut c_void,
    buf: *mut f64,
    len: usize,
) -> FfiResult {
    // SAFETY: live handle and a buffer valid for `len` writes.
    guard(|| unsafe { driver::<T>(d).read_state(slice_mut(buf, len)) })
}

unsafe extern "C" fn write_command<T: PluginDriver>(
    d: *mut c_void,
    buf: *const f64,
    len: usize,
) -> FfiResult {
    // SAFETY: live handle and a buffer valid for `len` reads.
    guard(|| unsafe { driver::<T>(d).write_command(slice(buf, len)) })
}

unsafe extern "C" fn set_payload<T: PluginDriver>(
    d: *mut c_void,
    mass_kg: f64,
    com: *const [f64; 3],
    inertia: *const [f64; 9],
    out: *mut u8,
) -> FfiResult {
    guard(|| {
        // SAFETY: live handle and valid pointers.
        let side = unsafe { driver::<T>(d).set_payload(mass_kg, *com, *inertia) }?;
        unsafe { out.write(applied_code(side)) };
        Ok(())
    })
}

unsafe extern "C" fn set_tcp_offset<T: PluginDriver>(
    d: *mut c_void,
    pose: *const [f64; 7],
    out: *mut u8,
) -> FfiResult {
    guard(|| {
        // SAFETY: live handle and valid pointers.
        let side = unsafe { driver::<T>(d).set_tcp_offset(*pose) }?;
        unsafe { out.write(applied_code(side)) };
        Ok(())
    })
}

unsafe extern "C" fn set_gravity<T: PluginDriver>(
    d: *mut c_void,
    gravity: *const [f64; 3],
    out: *mut u8,
) -> FfiResult {
    guard(|| {
        // SAFETY: live handle and valid pointers.
        let side = unsafe { driver::<T>(d).set_gravity(*gravity) }?;
        unsafe { out.write(applied_code(side)) };
        Ok(())
    })
}

unsafe extern "C" fn set_calibration<T: PluginDriver>(
    d: *mut c_void,
    calibration: *const CalibrationData,
) -> FfiResult {
    // SAFETY: live handle and valid pointer.
    guard(|| unsafe { driver::<T>(d).set_calibration(&*calibration) })
}

unsafe extern "C" fn clear_calibration<T: PluginDriver>(d: *mut c_void) -> FfiResult {
    guard(|| {
        // SAFETY: live handle.
        unsafe { driver::<T>(d) }.clear_calibration();
        Ok(())
    })
}

unsafe extern "C" fn set_ik_max_joint_step<T: PluginDriver>(d: *mut c_void, step: f64) -> FfiResult {
    // SAFETY: live handle.
    guard(|| unsafe { driver::<T>(d) }.set_ik_max_joint_step(step))
}

unsafe extern "C" fn clear_faults<T: PluginDriver>(d: *mut c_void) -> FfiResult {
    // SAFETY: live handle.
    guard(|| unsafe { driver::<T>(d) }.clear_faults())
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn vendor_command<T: PluginDriver>(
    d: *mut c_void,
    name: *const u8,
    name_len: usize,
    payload: *const u8,
    payload_len: usize,
    response: *mut u8,
    response_len: usize,
    out_len: *mut usize,
) -> FfiResult {
    guard(|| {
        // SAFETY: live handle and buffers valid for their lengths.
        let name = core::str::from_utf8(unsafe { slice(name, name_len) })
            .map_err(|_| ArmError::Fault("vendor command name is not UTF-8"))?;
        let n = unsafe {
            driver::<T>(d).handle_vendor_command(
                name,
                slice(payload, payload_len),
                slice_mut(response, response_len),
            )
        }?;
        unsafe { out_len.write(n) };
        Ok(())
    })
}

unsafe extern "C" fn math_create<T: PluginDriver>(out: *mut *mut c_void) -> FfiResult {
    guard(|| {
        let boxed: Box<Box<dyn ArmMath>> = Box::new(T::create_math());
        // SAFETY: valid out pointer.
        unsafe { out.write(Box::into_raw(boxed).cast()) };
        Ok(())
    })
}

unsafe extern "C" fn math_destroy(m: *mut c_void) {
    if !m.is_null() {
        // SAFETY: `m` came from `math_create` and is destroyed once.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(m.cast::<Box<dyn ArmMath>>()) })
        }));
    }
}

unsafe extern "C" fn math_dof<T: PluginDriver>(h: MathHandle) -> usize {
    // SAFETY: live handle.
    catch_unwind(AssertUnwindSafe(|| unsafe { math_ref::<T>(h) }.dof())).unwrap_or(0)
}

unsafe extern "C" fn math_is_calibrated<T: PluginDriver>(h: MathHandle) -> bool {
    // SAFETY: live handle.
    catch_unwind(AssertUnwindSafe(|| unsafe { math_ref::<T>(h) }.is_calibrated()))
        .unwrap_or(false)
}

/// Run a query and write its `Some` result to `out`.
///
/// # Safety
///
/// `out` must be valid for a write.
unsafe fn query<R>(out: *mut R, f: impl FnOnce() -> Option<R>) -> bool {
    guard_some(|| {
        let value = f()?;
        // SAFETY: forwarded to the caller.
        unsafe { out.write(value) };
        Some(())
    })
}

unsafe extern "C" fn math_forward_kinematics<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    out: *mut [f64; 7],
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).forward_kinematics(&*q)) }
}

unsafe extern "C" fn math_jacobian<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    out: *mut [Q; 6],
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).jacobian_padded(&*q)) }
}

unsafe extern "C" fn math_tool_jacobian<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    out: *mut [Q; 6],
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).tool_jacobian_padded(&*q)) }
}

unsafe extern "C" fn math_jacobian_pseudo_inverse<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    out: *mut [[f64; 6]; MAX_DOF],
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).jacobian_pseudo_inverse(&*q)) }
}

unsafe extern "C" fn math_inverse_kinematics<T: PluginDriver>(
    h: MathHandle,
    target: *const [f64; 7],
    seed: *const Q,
    out: *mut Q,
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).inverse_kinematics(&*target, &*seed)) }
}

unsafe extern "C" fn math_dynamics<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    qd: *const Q,
    out: *mut DynamicsState,
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).dynamics_padded(&*q, &*qd)) }
}

unsafe extern "C" fn math_inverse_kinematics_all<T: PluginDriver>(
    h: MathHandle,
    target: *const [f64; 7],
    out: *mut Q,
    cap: usize,
    count: *mut usize,
) -> bool {
    guard_some(|| {
        if out.is_null() || count.is_null() {
            return None;
        }
        // SAFETY: the host passes `cap` writable slots starting at `out`.
        let slots = unsafe { core::slice::from_raw_parts_mut(out, cap) };
        // SAFETY: live handle and valid pointers.
        let (target, math) = unsafe { (&*target, math_ref::<T>(h)) };
        let n = math.inverse_kinematics_all(target, slots);
        // SAFETY: `count` is valid for a write.
        unsafe { count.write(n) };
        Some(())
    })
}

unsafe extern "C" fn math_inverse_dynamics<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    qd: *const Q,
    qdd: *const Q,
    out: *mut Q,
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).inverse_dynamics_padded(&*q, &*qd, &*qdd)) }
}

unsafe extern "C" fn math_pose_residual<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    reported: *const [f64; 7],
    out: *mut [f64; 2],
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe {
        query(out, || {
            let r = math_ref::<T>(h).pose_residual(&*q, &*reported)?;
            Some([r.position, r.rotation])
        })
    }
}

unsafe extern "C" fn math_external_wrench<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    reported: *const [f64; 6],
    frame: u32,
    gravity_compensated: bool,
    out: *mut [f64; 6],
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe {
        query(out, || {
            let frame = wrench_frame_from_code(u8::try_from(frame).ok()?)?;
            math_ref::<T>(h).external_wrench_padded(&*q, &*reported, frame, gravity_compensated)
        })
    }
}

unsafe extern "C" fn math_apply_payload<T: PluginDriver>(
    h: MathHandle,
    mass_kg: f64,
    com: *const [f64; 3],
    inertia: *const [f64; 9],
) -> FfiResult {
    // SAFETY: live handle and valid pointers.
    guard(|| Ok(unsafe { math_mut::<T>(h).apply_payload(mass_kg, *com, *inertia) }?))
}

unsafe extern "C" fn math_apply_tcp_offset<T: PluginDriver>(
    h: MathHandle,
    pose: *const [f64; 7],
) -> FfiResult {
    // SAFETY: live handle and valid pointer.
    guard(|| Ok(unsafe { math_mut::<T>(h).apply_tcp_offset(*pose) }?))
}

unsafe extern "C" fn math_apply_gravity<T: PluginDriver>(
    h: MathHandle,
    gravity: *const [f64; 3],
) -> FfiResult {
    // SAFETY: live handle and valid pointer.
    guard(|| Ok(unsafe { math_mut::<T>(h).apply_gravity(*gravity) }?))
}

unsafe extern "C" fn math_apply_calibration<T: PluginDriver>(
    h: MathHandle,
    calibration: *const CalibrationData,
) -> FfiResult {
    // SAFETY: live handle and valid pointer.
    guard(|| Ok(unsafe { math_mut::<T>(h).apply_calibration(&*calibration) }?))
}

unsafe extern "C" fn math_remove_calibration<T: PluginDriver>(h: MathHandle) -> FfiResult {
    guard(|| {
        // SAFETY: live handle.
        unsafe { math_mut::<T>(h) }.remove_calibration();
        Ok(())
    })
}

unsafe extern "C" fn math_apply_ik_max_joint_step<T: PluginDriver>(
    h: MathHandle,
    step: f64,
) -> FfiResult {
    // SAFETY: live handle.
    guard(|| Ok(unsafe { math_mut::<T>(h) }.apply_ik_max_joint_step(step)?))
}

unsafe extern "C" fn math_step_free_torque<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    qd: *const Q,
    qdd: *const Q,
    q_cmd: *const Q,
    control_freq_hz: u32,
    out: *mut Q,
) -> bool {
    // SAFETY: live handle and valid pointers; `q_cmd` is null or valid.
    unsafe {
        query(out, || {
            math_mut::<T>(h).step_free_torque_padded(
                &*q,
                &*qd,
                &*qdd,
                q_cmd.as_ref(),
                control_freq_hz,
            )
        })
    }
}

unsafe extern "C" fn math_reset_torque_observer<T: PluginDriver>(h: MathHandle) -> FfiResult {
    guard(|| {
        // SAFETY: live handle.
        unsafe { math_mut::<T>(h) }.reset_torque_observer_state();
        Ok(())
    })
}

unsafe extern "C" fn math_payload_torque<T: PluginDriver>(
    h: MathHandle,
    q: *const Q,
    qd: *const Q,
    qdd: *const Q,
    out: *mut Q,
) -> bool {
    // SAFETY: live handle and valid pointers.
    unsafe { query(out, || math_ref::<T>(h).payload_torque_padded(&*q, &*qd, &*qdd)) }
}

/// The function table of driver `T`; use through
/// [`export_plugin!`](crate::export_plugin).
pub const fn vtable<T: PluginDriver>() -> PluginVTable {
    PluginVTable {
        abi_version: ABI_VERSION,
        contract_major: CONTRACT_VERSION.major,
        contract_minor: CONTRACT_VERSION.minor,
        contract_patch: CONTRACT_VERSION.patch,
        driver_create: driver_create::<T>,
        driver_destroy: driver_destroy::<T>,
        driver_manifest: driver_manifest::<T>,
        connect: connect::<T>,
        disconnect: disconnect::<T>,
        read_state: read_state::<T>,
        write_command: write_command::<T>,
        set_payload: set_payload::<T>,
        set_tcp_offset: set_tcp_offset::<T>,
        set_gravity: set_gravity::<T>,
        set_calibration: set_calibration::<T>,
        clear_calibration: clear_calibration::<T>,
        set_ik_max_joint_step: set_ik_max_joint_step::<T>,
        clear_faults: clear_faults::<T>,
        vendor_command: vendor_command::<T>,
        math_create: math_create::<T>,
        math_destroy,
        math_dof: math_dof::<T>,
        math_is_calibrated: math_is_calibrated::<T>,
        math_forward_kinematics: math_forward_kinematics::<T>,
        math_jacobian: math_jacobian::<T>,
        math_tool_jacobian: math_tool_jacobian::<T>,
        math_jacobian_pseudo_inverse: math_jacobian_pseudo_inverse::<T>,
        math_inverse_kinematics: math_inverse_kinematics::<T>,
        math_inverse_kinematics_all: math_inverse_kinematics_all::<T>,
        math_dynamics: math_dynamics::<T>,
        math_inverse_dynamics: math_inverse_dynamics::<T>,
        math_pose_residual: math_pose_residual::<T>,
        math_external_wrench: math_external_wrench::<T>,
        math_apply_payload: math_apply_payload::<T>,
        math_apply_tcp_offset: math_apply_tcp_offset::<T>,
        math_apply_gravity: math_apply_gravity::<T>,
        math_apply_calibration: math_apply_calibration::<T>,
        math_remove_calibration: math_remove_calibration::<T>,
        math_apply_ik_max_joint_step: math_apply_ik_max_joint_step::<T>,
        math_step_free_torque: math_step_free_torque::<T>,
        math_reset_torque_observer: math_reset_torque_observer::<T>,
        math_payload_torque: math_payload_torque::<T>,
    }
}
