//! Auto-generated symbolic kinematics for the UR5 (6-DOF).
//!
//! Forward kinematics and the geometric Jacobian, derived
//! symbolically from the ssik artifact with SymPy and emitted as
//! pure `#![no_std]` Rust. Regenerate with `codegen/generate_dynamics.py`.
//!
//! Do not edit by hand. Every transcendental is a `libm` call and
//! every output is a fixed-size array: no heap, no branches, no loops.

#![allow(clippy::all)]
#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(dead_code)]

/// Degrees of freedom of the UR5.
pub const DOF: usize = 6;

/// Forward kinematics: base -> end-effector SE(3) pose.
///
/// Returns the 4x4 homogeneous transform as a row-major array.
pub fn fk(q: &[f64; 6]) -> [[f64; 4]; 4] {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q4 = q[4];
    let q5 = q[5];
    let x0 = libm::cos(q0);
    let x1 = libm::sin(q5);
    let x2 = libm::sin(q3);
    let x3 = libm::sin(q1);
    let x4 = libm::sin(q2);
    let x5 = x3*x4;
    let x6 = libm::cos(q1);
    let x7 = libm::cos(q2);
    let x8 = x6*x7;
    let x9 = x5 - x8;
    let x10 = x2*x9;
    let x11 = libm::cos(q3);
    let x12 = x3*x7;
    let x13 = x4*x6;
    let x14 = x12 + x13;
    let x15 = x10 - x11*x14;
    let x16 = -x15;
    let x17 = x1*x16;
    let x18 = libm::cos(q5);
    let x19 = libm::sin(q0);
    let x20 = libm::sin(q4);
    let x21 = x11*x9 + x14*x2;
    let x22 = libm::cos(q4);
    let x23 = x0*x22;
    let x24 = x19*x20 - x21*x23;
    let x25 = x11*x14;
    let x26 = x18*(-x2*x9 + x25);
    let x27 = x19*x22;
    let x28 = x0*x20;
    let x29 = x21*x28;
    let x30 = 0.42499999999999999*x6;
    let x31 = 0.39225000000000004*x0;
    let x32 = 0.094649999999999998*x0;
    let x33 = x21*x27 + x28;
    let m: [[f64; 4]; 4] = [
        [-x0*x17 + x18*x24, -x0*x26 - x1*x24, x27 + x29, -x0*x30 - x10*x32 + 0.10915*x19 + x25*x32 + 0.082299999999999998*x27 + 0.082299999999999998*x29 + x31*x5 - x31*x8],
        [-x17*x19 - x18*x33, x1*x33 - x19*x26, x19*x20*x21 - x23, -0.10915*x0 - 0.094649999999999998*x10*x19 + 0.094649999999999998*x11*x14*x19 + 0.082299999999999998*x19*x20*x21 + 0.39225000000000004*x19*x3*x4 - x19*x30 - 0.39225000000000004*x19*x8 - 0.082299999999999998*x23],
        [-x1*x21 - x15*x18*x22, -x17*x22 - x18*x21, x15*x20, -0.39225000000000004*x12 - 0.39225000000000004*x13 - 0.42499999999999999*x3 + 0.094649999999999998*x11*x9 + 0.094649999999999998*x14*x2 - 0.082299999999999998*x16*x20 + 0.089159000000000002],
        [0.0, 0.0, 0.0, 1.0],
    ];
    m
}

/// Geometric Jacobian `[Jv; Jw]` (6 x DOF) of the end-effector.
///
/// Rows 0-2 are linear velocity, rows 3-5 are angular velocity.
pub fn jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q4 = q[4];
    let q5 = q[5];
    let x0 = libm::cos(q0);
    let x1 = 0.10915*x0;
    let x2 = libm::sin(q0);
    let x3 = libm::cos(q1);
    let x4 = 0.42499999999999999*x3;
    let x5 = x2*x4;
    let x6 = libm::sin(q3);
    let x7 = libm::sin(q1);
    let x8 = libm::sin(q2);
    let x9 = x7*x8;
    let x10 = libm::cos(q2);
    let x11 = x10*x3;
    let x12 = -x11 + x9;
    let x13 = x12*x6;
    let x14 = 0.094649999999999998*x2;
    let x15 = x10*x7;
    let x16 = x3*x8;
    let x17 = x15 + x16;
    let x18 = x17*x6;
    let x19 = libm::cos(q3);
    let x20 = x12*x19;
    let x21 = x18 + x20;
    let x22 = libm::sin(q4);
    let x23 = 0.082299999999999998*x22;
    let x24 = x2*x23;
    let x25 = libm::cos(q4);
    let x26 = x0*x25;
    let x27 = x17*x19;
    let x28 = -x14*x27 + 0.082299999999999998*x26;
    let x29 = 0.39225000000000004*x2;
    let x30 = x11*x29 - x29*x9;
    let x31 = 0.094649999999999998*x18;
    let x32 = -x31;
    let x33 = 0.094649999999999998*x20;
    let x34 = x13 - x17*x19;
    let x35 = -x34;
    let x36 = 0.39225000000000004*x15 + 0.39225000000000004*x16 + x23*x35 + x32 - x33;
    let x37 = x36 + 0.42499999999999999*x7;
    let x38 = 0.082299999999999998*x22*x35 - x31 - x33;
    let x39 = x0*x38;
    let x40 = -x12;
    let x41 = x19*x40;
    let x42 = x40*x6;
    let x43 = x2*x35;
    let x44 = -x18 + x41;
    let x45 = -x14*x42 + x24*x44 + x28;
    let x46 = -x0*x4;
    let x47 = 0.10915*x2;
    let x48 = 0.094649999999999998*x0;
    let x49 = x2*x25;
    let x50 = x27*x48 + 0.082299999999999998*x49;
    let x51 = x0*x21*x23 - x13*x48 + x50;
    let x52 = x47 + x51;
    let x53 = 0.39225000000000004*x0;
    let x54 = -x11*x53 + x53*x9;
    let x55 = x52 + x54;
    let x56 = x1 + x45;
    let x57 = x30 + x56;
    let x58 = -0.082299999999999998*x0*x22*x44 + x42*x48 + x50;
    let x59 = x21*x22;
    let x60 = -x0;
    let j: [[f64; 6]; 6] = [
        [x1 + x13*x14 - x21*x24 + x28 + x30 + x5, x0*x37, x0*x36, x39, -x43*(x23*(x27 + x42) + x32 + 0.094649999999999998*x41) - x44*x45, 0.0],
        [x46 + x55, x2*x37, x2*x36, x2*x38, x21*x51 - x34*x39, 0.0],
        [0.0, -x0*(-x46 - x47 - x54 - x58) - x2*(x5 + x57), x0*x55 - x2*x57, x0*x52 - x2*x56, x35*(-x0*x45 - x2*x58), 0.0],
        [0.0, x2, x2, x2, x0*x35, x0*x59 + x49],
        [0.0, x60, x60, x60, x43, x2*x59 - x26],
        [1.0, 0.0, 0.0, 0.0, x21, x22*x34],
    ];
    j
}
