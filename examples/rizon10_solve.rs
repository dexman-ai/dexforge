//! Host-side filtering harness used by `scripts/generate_rizon_test_data.py`.
//!
//! Reads target poses from stdin (one per line, 16 space-separated row-major
//! `f64` entries of a 4x4 SE(3) matrix) and prints the rizon10 IK solutions the
//! `#![no_std]` driver computes for each. The generator uses this to retain only
//! poses the Rust solver actually reproduces, keeping the committed regression
//! dataset in sync with the real numeric pipeline.
//!
//! Output format: one line per input pose, `N` solutions joined by `;`, each a
//! space-separated list of seven joint angles (an empty line means no solution).
//!
//! Run: `cargo run --release --example rizon10_solve < poses.txt`

use std::io::{self, BufRead, Write};

use nalgebra::SMatrix;
use ssik_rust::drivers::rizon10::ik_solver::{solve, TolerancePolicy};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let vals: Vec<f64> = trimmed
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("parse f64"))
            .collect();
        assert_eq!(vals.len(), 16, "expected 16 pose entries, got {}", vals.len());

        let t = SMatrix::<f64, 4, 4>::from_fn(|i, j| vals[i * 4 + j]);
        let sols = solve(&t, None, None, false, false, TolerancePolicy::DEFAULT, 15);

        let formatted: Vec<String> = sols
            .iter()
            .flatten()
            .map(|q| {
                q.iter()
                    .map(|v| format!("{v:.17e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        writeln!(out, "{}", formatted.join(";")).expect("write stdout");
    }
}
