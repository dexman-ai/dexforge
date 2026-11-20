//! IK and dynamics timing for the ur5 arm. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-ur/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

use dexforge_ur_core::Ur5;

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn ur_arm_timing() {
    arm_timing::measure::<Ur5, 6, 16>("ur5");
}

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn ur3e_arm_timing() {
    use dexforge_ur_core::Ur3e;
    arm_timing::measure::<Ur3e, 6, 16>("ur3e");
}
