//! Seeded path tracking with `flange_one` (Fast and Near). Shared checks:
//! `dexhal/conformance/ik_path.rs`.

#[path = "../../../../dexhal/conformance/ik_path.rs"]
mod ik_path;

use dexhal::math::IkKind;
use dexforge_ufactory_core::{xarm7::ik_solver::LOCKED_JOINT, Xarm7};

const START: [f64; 7] = [0.3, -0.4, 0.2, 1.2, 0.37, 0.8, 0.5];

#[test]
fn xarm7_near_keeps_the_seed_locked_joint_along_a_path() {
    ik_path::track_path::<Xarm7, 128>(START, LOCKED_JOINT, IkKind::Near);
}

#[test]
fn xarm7_fast_tracks_a_path_without_jumps() {
    ik_path::track_path::<Xarm7, 128>(START, LOCKED_JOINT, IkKind::Fast);
}

#[test]
fn xarm7_fast_bounds_the_lock_joint() {
    ik_path::fast_lock_bound::<Xarm7, 128>(START, LOCKED_JOINT);
}
