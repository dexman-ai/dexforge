//! Auto-generated ssik IK reference dataset loader for `so101_6r`.
//!
//! Reads the binary dataset `test_data.bin`.
//! Do not edit by hand; regenerate with `codegen/generate_test_data.py`.

#![allow(dead_code, clippy::all)]

extern crate std;
use std::vec::Vec;

/// Degrees of freedom for `so101_6r`.
pub const DOF: usize = 6;

/// Number of reference cases stored in `test_data.bin`.
pub const N_CASES: usize = 500;

/// Raw binary reference dataset (`DXIK` format v1).
pub static RAW: &[u8] = include_bytes!("test_data.bin");

/// A decoded reference case from `test_data.bin`.
#[derive(Clone, Debug)]
pub struct Case {
    pub q_fk: [f64; DOF],
    pub tf: [f64; 16],
    pub solutions: Vec<[f64; DOF]>,
    pub singular: bool,
}

#[inline]
fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[*offset..*offset + 4]);
    *offset += 4;
    u32::from_le_bytes(buf)
}

#[inline]
fn read_f64(bytes: &[u8], offset: &mut usize) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    f64::from_le_bytes(buf)
}

/// Return the `ssik` version string recorded in `test_data.bin`.
pub fn ssik_version() -> &'static str {
    assert!(RAW.len() >= 12 && &RAW[0..4] == b"DXIK", "invalid test_data.bin header");
    let mut offset = 4usize;
    let fmt_ver = read_u32(RAW, &mut offset);
    assert_eq!(fmt_ver, 1, "unsupported test_data.bin format version");
    let ver_len = read_u32(RAW, &mut offset) as usize;
    core::str::from_utf8(&RAW[offset..offset + ver_len])
        .expect("invalid UTF-8 ssik_version in test_data.bin")
}

/// Decode all [`N_CASES`] reference cases from [`RAW`].
pub fn load_cases() -> Vec<Case> {
    assert!(RAW.len() >= 12 && &RAW[0..4] == b"DXIK", "invalid test_data.bin header");
    let mut offset = 4usize;
    let fmt_ver = read_u32(RAW, &mut offset);
    assert_eq!(fmt_ver, 1, "unsupported test_data.bin format version");
    let ver_len = read_u32(RAW, &mut offset) as usize;
    offset += ver_len;

    let dof = read_u32(RAW, &mut offset) as usize;
    let n_cases = read_u32(RAW, &mut offset) as usize;
    let _allow_refinement = read_u32(RAW, &mut offset);
    let _refinement_max_iters = read_u32(RAW, &mut offset);
    assert_eq!(dof, DOF, "test_data.bin DOF mismatch");
    assert_eq!(n_cases, N_CASES, "test_data.bin N_CASES mismatch");

    let mut out = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {
        let mut q_fk = [0.0f64; DOF];
        for i in 0..DOF {
            q_fk[i] = read_f64(RAW, &mut offset);
        }
        let mut tf = [0.0f64; 16];
        for i in 0..16 {
            tf[i] = read_f64(RAW, &mut offset);
        }
        let singular = read_u32(RAW, &mut offset) != 0;
        let n_sols = read_u32(RAW, &mut offset) as usize;
        let mut solutions = Vec::with_capacity(n_sols);
        for _ in 0..n_sols {
            let mut q = [0.0f64; DOF];
            for i in 0..DOF {
                q[i] = read_f64(RAW, &mut offset);
            }
            solutions.push(q);
        }
        out.push(Case {
            q_fk,
            tf,
            solutions,
            singular,
        });
    }
    assert_eq!(offset, RAW.len(), "test_data.bin length mismatch");
    out
}

/// Convenience accessor returning tuple cases for `ik_solver` tests.
pub fn cases() -> Vec<([f64; 16], Vec<[f64; 6]>)> {
    load_cases().into_iter().map(|c| (c.tf, c.solutions)).collect()
}
