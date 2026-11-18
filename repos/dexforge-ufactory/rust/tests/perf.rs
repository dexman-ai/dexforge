//! IK and dynamics timing for the xarm7 arm. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-ufactory/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

use dexforge_ufactory_core::Xarm7;

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn ufactory_arm_timing() {
    arm_timing::measure::<Xarm7, 7, 128>("xarm7");
}
