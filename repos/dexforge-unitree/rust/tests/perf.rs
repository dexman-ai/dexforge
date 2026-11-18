//! IK and dynamics timing for the z1 arm. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-unitree/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

use dexforge_unitree_core::Z1;

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn unitree_arm_timing() {
    arm_timing::measure::<Z1, 6, 16>("z1");
}
