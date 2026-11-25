//! IK and dynamics timing for the lerobot arms. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-lerobot/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn so101_6r_arm_timing() {
    use dexforge_lerobot_core::So1016r;
    arm_timing::measure::<So1016r, 6, 16>("so101_6r");
}

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn so101_arm_timing() {
    use dexforge_lerobot_core::So101;
    arm_timing::measure::<So101, 5, 16>("so101");
}
