//! IK and dynamics timing for the piper arm. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-agilex/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

use dexforge_agilex_core::Piper;

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn agilex_arm_timing() {
    arm_timing::measure::<Piper, 6, 16>("piper");
}
