//! IK and dynamics timing for the rizon4 and rizon10 arms. Shared harness: `dexhal/perf/arm_timing.rs`.
//!
//! Run: `cargo test --release --manifest-path repos/dexforge-flexiv/Cargo.toml --test perf -- --ignored --nocapture`

#[path = "../../../../dexhal/perf/arm_timing.rs"]
mod arm_timing;

use dexforge_flexiv_core::{Rizon10, Rizon4};

#[test]
#[ignore = "perf: run with --release --ignored --nocapture"]
fn flexiv_arm_timing() {
    arm_timing::measure::<Rizon4, 7, 256>("rizon4");
    arm_timing::measure::<Rizon10, 7, 256>("rizon10");
}
