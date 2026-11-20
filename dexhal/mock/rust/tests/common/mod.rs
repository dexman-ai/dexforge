//! Shared helpers for the mock integration tests.

use dexhal::math::{next_param_count, NEXT_HEADER_LEN, NEXT_MAGIC};

/// A synthetic `.dexnext` file for `dof` joints: unit input normalisation,
/// small deterministic weights, `out_mean` 0.5, `out_std` 2. `mode` 0 is
/// FullDynamics, 1 FrictionResidual; no trained payload.
pub fn dexnext(dof: u8, mode: u8, gravity: [f64; 3]) -> Vec<u8> {
    let d = usize::from(dof);
    let mut b = NEXT_MAGIC.to_vec();
    b.extend_from_slice(&[dof, mode, 0, 0, 0, 0, 0, 0]);
    for v in gravity.into_iter().chain([0.0; 13]) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(b.len(), NEXT_HEADER_LEN);
    let weights = next_param_count(d) - 8 * d;
    let params = std::iter::repeat(0.0)
        .take(3 * d)
        .chain(std::iter::repeat(1.0).take(3 * d))
        .chain((0..weights).map(|k| ((k * 37 % 23) as f64 - 11.0) * 0.005))
        .chain(std::iter::repeat(0.5).take(d))
        .chain(std::iter::repeat(2.0).take(d));
    for v in params {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}
