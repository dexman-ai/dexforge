//! Seeded path tracking with `flange_one` (Fast and Near). Shared checks:
//! `dexhal/conformance/ik_path.rs`.

#[path = "../../../../dexhal/conformance/ik_path.rs"]
mod ik_path;

use dexhal::math::IkKind;
use dexforge_franka_core::{franka_panda::ik_solver::LOCKED_JOINT, FrankaPanda};

/// In-limit start (Panda joint 4 is limited to about [-3.07, -0.07]).
const START: [f64; 7] = [0.3, -0.4, 0.2, -1.8, 0.37, 1.6, 0.5];

#[test]
fn franka_panda_near_keeps_the_seed_locked_joint_along_a_path() {
    ik_path::track_path::<FrankaPanda, 128>(START, LOCKED_JOINT, IkKind::Near);
}

#[test]
fn franka_panda_fast_tracks_a_path_without_jumps() {
    ik_path::track_path::<FrankaPanda, 128>(START, LOCKED_JOINT, IkKind::Fast);
}

#[test]
fn franka_panda_fast_bounds_the_lock_joint() {
    ik_path::fast_lock_bound::<FrankaPanda, 128>(START, LOCKED_JOINT);
}
