//! IK and dynamics timing for the franka_panda arm. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-franka/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

use dexforge_franka_core::FrankaPanda;

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn franka_arm_timing() {
    arm_timing::measure::<FrankaPanda, 7, 128>("franka_panda");
}
