//! Real-time safety of the learned torque observer (factr2.md §6, criterion
//! 3): no heap allocation in `step_free_torque` or a full driver tick, and a
//! worst-case evaluation time inside the 1 kHz cycle.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use dexforge_mock_core::{dexhal_plugin_entry, MockDriver};
use dexhal::math::{ActiveMode, ArmDriver, CoreCommand, LstmObserver, MAX_DOF};

use dexhal_plugin::Plugin;

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// Serializes the tests so one test's allocations are not counted by another.
static LOCK: Mutex<()> = Mutex::new(());

// SAFETY: defers to `System`; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size().max(1), Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(new_size.max(1), Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Bytes allocated while running `f`.
fn allocated_by(f: impl FnOnce()) -> usize {
    ALLOCATED.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    ALLOCATED.load(Ordering::SeqCst)
}

const DOF: usize = 6;

/// A 6-joint observer (the fixed 2 × 128 LSTM, 256-wide head, 50-sample window).
fn observer() -> &'static [u8] {
    Box::leak(common::dexnext(DOF as u8, 0, [0.0, 0.0, -9.80665]).into_boxed_slice())
}

fn trajectory(
    t: usize,
) -> (
    [f64; MAX_DOF],
    [f64; MAX_DOF],
    [f64; MAX_DOF],
    [f64; MAX_DOF],
) {
    let mut q = [0.0; MAX_DOF];
    let mut qd = [0.0; MAX_DOF];
    let mut qdd = [0.0; MAX_DOF];
    let mut q_cmd = [0.0; MAX_DOF];
    for j in 0..DOF {
        let s = 0.001 * t as f64 * (1.0 + 0.2 * j as f64);
        q[j] = 0.5 * s.sin();
        qd[j] = 0.5 * s.cos();
        qdd[j] = -0.5 * s.sin();
        q_cmd[j] = q[j] + 0.01 * qd[j];
    }
    (q, qd, qdd, q_cmd)
}

#[test]
fn step_free_torque_does_not_allocate() {
    let _guard = LOCK.lock().unwrap();
    let mut driver = MockDriver::default_arm()
        .unwrap()
        .with_torque_observer(observer())
        .unwrap();
    driver.connect().unwrap();
    let mut state = vec![0.0; driver.manifest().state_len()];
    let mut command = vec![f64::NAN; driver.manifest().command_len()];
    let mut cmd = CoreCommand::IDLE;
    cmd.control_mode = ActiveMode::JointPos.code();

    let mut tick = |driver: &mut MockDriver<_, 6, 1>, t: usize| {
        let (q, qd, qdd, q_cmd) = trajectory(t);
        driver.read_state(&mut state).unwrap();
        // None only while warming up after a reset.
        if let Some(tau) = driver.step_free_torque(&state, &q, &qd, &qdd, Some(&q_cmd)) {
            assert!(tau[..DOF].iter().all(|v| v.is_finite()));
        }
        cmd.target_q = q_cmd;
        cmd.write_to(&mut command).unwrap();
        driver.write_command(&command).unwrap();
    };
    for t in 0..10 {
        tick(&mut driver, t);
    }
    let bytes = allocated_by(|| {
        for t in 10..200 {
            tick(&mut driver, t);
            if t == 100 {
                driver.reset_torque_observer();
            }
        }
    });
    assert_eq!(bytes, 0, "the 1 kHz tick allocated {bytes} bytes");
}

#[test]
fn the_plugin_boundary_does_not_allocate() {
    let _guard = LOCK.lock().unwrap();
    let path = std::env::temp_dir().join(format!("dexforge-rt-{}.dexnext", std::process::id()));
    std::fs::write(&path, observer()).unwrap();
    // SAFETY: the entry point is linked into this test binary.
    let plugin = unsafe { Plugin::from_entry(dexhal_plugin_entry) }.unwrap();
    let config = format!("observer={}", path.display());
    let mut driver = plugin.create_driver(config.as_bytes()).unwrap();
    std::fs::remove_file(&path).unwrap();
    driver.connect().unwrap();
    let state = vec![0.0; driver.manifest().state_len()];
    let (q, qd, qdd, q_cmd) = trajectory(0);
    driver.step_free_torque(&state, &q, &qd, &qdd, Some(&q_cmd));
    let bytes = allocated_by(|| {
        for t in 1..100 {
            let (q, qd, qdd, q_cmd) = trajectory(t);
            let tau = driver.step_free_torque(&state, &q, &qd, &qdd, Some(&q_cmd));
            // Warming up at 500 Hz: the first four ticks, and after the reset.
            assert!(tau.is_some() || t < 5 || (51..55).contains(&t), "tick {t}");
            if t == 50 {
                driver.reset_torque_observer();
            }
        }
    });
    assert_eq!(bytes, 0, "the plugin path allocated {bytes} bytes");
}

/// Smallest per-trial maximum of `f(t)` over `trials × ticks` calls, so a
/// preempted sample on a shared machine does not count; and the mean.
fn wcet(trials: usize, ticks: usize, mut reset: impl FnMut(), mut f: impl FnMut(usize)) -> (Duration, Duration) {
    let mut wcet = Duration::MAX;
    let mut total = Duration::ZERO;
    for trial in 0..trials {
        reset();
        let mut worst = Duration::ZERO;
        for t in 0..ticks {
            let start = Instant::now();
            f(trial * ticks + t);
            let elapsed = start.elapsed();
            worst = worst.max(elapsed);
            total += elapsed;
        }
        wcet = wcet.min(worst);
    }
    (wcet, total / (trials * ticks) as u32)
}

/// The window's 50 LSTM frames are spread over the ticks of each 100 Hz
/// period, so the worst tick, warm-up included, fits a 1 kHz cycle. Asserted
/// only in optimized builds (`cargo test --release`).
#[test]
fn worst_case_evaluation_fits_the_1khz_cycle() {
    let _guard = LOCK.lock().unwrap();
    let mut driver = MockDriver::default_arm()
        .unwrap()
        .with_torque_observer(observer())
        .unwrap();
    let freq = driver.manifest().control_freq_hz;
    let state = vec![0.0; driver.manifest().state_len()];
    let cell = std::cell::RefCell::new(&mut driver);
    let (driver_wcet, driver_mean) = wcet(
        5,
        300,
        || cell.borrow_mut().reset_torque_observer(),
        |t| {
            let (q, qd, qdd, q_cmd) = trajectory(t);
            let tau = cell.borrow_mut().step_free_torque(&state, &q, &qd, &qdd, Some(&q_cmd));
            std::hint::black_box(tau);
        },
    );
    println!("step_free_torque at {freq} Hz: mean {driver_mean:?}, WCET {driver_wcet:?}");

    let mut per_stride = Vec::new();
    for control_hz in [1000u32, 100] {
        let stride = LstmObserver::<DOF>::stride_ticks(control_hz);
        let obs = std::cell::RefCell::new(LstmObserver::<DOF>::from_bytes(observer()).unwrap());
        let (w, mean) = wcet(
            5,
            30 * stride,
            || obs.borrow_mut().reset(),
            |t| {
                let (q, qd, _, q_cmd) = trajectory(t);
                let mut u = [[0.0; DOF]; 3];
                for j in 0..DOF {
                    u[0][j] = q[j];
                    u[1][j] = qd[j];
                    u[2][j] = q_cmd[j] - q[j];
                }
                std::hint::black_box(obs.borrow_mut().step(&u, stride));
            },
        );
        println!("LstmObserver::step at {control_hz} Hz (stride {stride}): mean {mean:?}, WCET {w:?}");
        per_stride.push((control_hz, w));
    }

    if !cfg!(debug_assertions) {
        assert!(driver_wcet < Duration::from_millis(1), "driver WCET {driver_wcet:?} exceeds 1 ms");
        for (hz, w) in per_stride {
            let budget = Duration::from_secs(1) / hz;
            assert!(w < budget, "WCET {w:?} at {hz} Hz exceeds {budget:?}");
        }
    }
}