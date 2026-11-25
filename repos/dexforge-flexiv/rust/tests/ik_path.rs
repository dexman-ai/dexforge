//! Seeded path tracking with `flange_one` (Fast and Near). Shared checks:
//! `dexhal/conformance/ik_path.rs`.

#[path = "../../../../dexhal/conformance/ik_path.rs"]
mod ik_path;

use dexhal::math::IkKind;
use dexforge_flexiv_core::{Rizon10, Rizon4};

/// Joint held by the Rizon joint-lock sweep (`rizon_common::Model::lock_idx`).
const LOCKED_JOINT: usize = 2;
const START: [f64; 7] = [0.3, -0.4, 0.2, 1.2, 0.37, 0.8, 0.5];

#[test]
fn rizon4_near_keeps_the_seed_locked_joint_along_a_path() {
    ik_path::track_path::<Rizon4, 256>(START, LOCKED_JOINT, IkKind::Near);
}

#[test]
fn rizon4_fast_tracks_a_path_without_jumps() {
    ik_path::track_path::<Rizon4, 256>(START, LOCKED_JOINT, IkKind::Fast);
}

#[test]
fn rizon10_near_keeps_the_seed_locked_joint_along_a_path() {
    ik_path::track_path::<Rizon10, 256>(START, LOCKED_JOINT, IkKind::Near);
}

#[test]
fn rizon10_fast_tracks_a_path_without_jumps() {
    ik_path::track_path::<Rizon10, 256>(START, LOCKED_JOINT, IkKind::Fast);
}

#[test]
fn rizon4_fast_bounds_the_lock_joint() {
    ik_path::fast_lock_bound::<Rizon4, 256>(START, LOCKED_JOINT);
}

#[test]
fn rizon10_fast_bounds_the_lock_joint() {
    ik_path::fast_lock_bound::<Rizon10, 256>(START, LOCKED_JOINT);
}
