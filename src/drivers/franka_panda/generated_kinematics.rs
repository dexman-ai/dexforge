//! Auto-generated symbolic kinematics for the Franka Panda (7-DOF).
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

/// Degrees of freedom of the Franka Panda.
pub const DOF: usize = 7;

/// Forward kinematics: base -> end-effector SE(3) pose.
///
/// Returns the 4x4 homogeneous transform as a row-major array.
pub fn fk(q: &[f64; 7]) -> [[f64; 4]; 4] {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q4 = q[4];
    let q5 = q[5];
    let q6 = q[6];
    let x0 = libm::cos(q4);
    let x1 = libm::sin(q0);
    let x2 = libm::cos(q2);
    let x3 = x1*x2;
    let x4 = libm::cos(q1);
    let x5 = libm::sin(q2);
    let x6 = libm::cos(q0);
    let x7 = x5*x6;
    let x8 = x3 + x4*x7;
    let x9 = libm::sin(q4);
    let x10 = libm::sin(q3);
    let x11 = libm::sin(q1);
    let x12 = x11*x6;
    let x13 = x10*x12;
    let x14 = libm::cos(q3);
    let x15 = x1*x5;
    let x16 = x15 - x2*x4*x6;
    let x17 = -x16;
    let x18 = x0*x8 + x9*(x13 + x14*x17);
    let x19 = libm::sin(q6);
    let x20 = x18*x19;
    let x21 = libm::cos(q6);
    let x22 = libm::sin(q5);
    let x23 = x12*x14;
    let x24 = libm::cos(q5);
    let x25 = x8*x9;
    let x26 = x14*x16;
    let x27 = -x10*x11*x6 + x26;
    let x28 = -x22*(x10*x17 - x23) + x24*(-x0*x27 - x25);
    let x29 = x21*x28;
    let x30 = x10*x16;
    let x31 = x23 + x30;
    let x32 = x24*x31;
    let x33 = x0*x27 + x25;
    let x34 = x22*x33;
    let x35 = 0.087999999999999995*x24;
    let x36 = x15*x4 - x2*x6;
    let x37 = x1*x11;
    let x38 = x10*x37;
    let x39 = x3*x4;
    let x40 = x39 + x7;
    let x41 = x14*x40;
    let x42 = x38 + x41;
    let x43 = x0*x36 + x42*x9;
    let x44 = x19*x43;
    let x45 = x14*x37;
    let x46 = x10*x40;
    let x47 = -x45 + x46;
    let x48 = x22*x47;
    let x49 = x0*x42 - x36*x9;
    let x50 = x24*x49;
    let x51 = -x48 + x50;
    let x52 = x21*x51;
    let x53 = x24*x47;
    let x54 = x22*x49;
    let x55 = x11*x2;
    let x56 = x14*x55;
    let x57 = -x10*x4 + x56;
    let x58 = -x0*x11*x5 - x57*x9;
    let x59 = x19*x58;
    let x60 = x14*x4;
    let x61 = x10*x55;
    let x62 = x60 + x61;
    let x63 = x22*x62;
    let x64 = x0*x57 - x11*x5*x9;
    let x65 = -x24*x64 + x63;
    let x66 = x21*x65;
    let x67 = x24*x62;
    let x68 = x22*x64;
    let m: [[f64; 4]; 4] = [
        [0.70710675649449928*x18*x21 - 0.70710675649449928*x19*x28 - 0.70710680587859431*x20 - 0.70710680587859431*x29, -0.70710680587859343*x18*x21 + 0.70710680587859343*x19*x28 - 0.70710675649450017*x20 - 0.70710675649450017*x29, -x32 - x34, 0.316*x12 - 0.082500000000000004*x13 - 0.082500000000000004*x15 + 0.082500000000000004*x2*x4*x6 + 0.087999999999999995*x22*x31 + 0.38399999999999967*x23 + 0.082500000000000004*x26 + 0.38399999999999967*x30 - 0.10699999999999987*x32 - x33*x35 - 0.10699999999999987*x34],
        [-0.70710675649449928*x19*x51 + 0.70710675649449928*x21*x43 - 0.70710680587859431*x44 - 0.70710680587859431*x52, 0.70710680587859343*x19*x51 - 0.70710680587859343*x21*x43 - 0.70710675649450017*x44 - 0.70710675649450017*x52, x53 + x54, 0.316*x37 - 0.082500000000000004*x38 + 0.082500000000000004*x39 - 0.082500000000000004*x41 + 0.38399999999999967*x45 - 0.38399999999999967*x46 - 0.087999999999999995*x48 + 0.087999999999999995*x50 + 0.10699999999999987*x53 + 0.10699999999999987*x54 + 0.082500000000000004*x7],
        [-0.70710675649449928*x19*x65 + 0.70710675649449928*x21*x58 - 0.70710680587859431*x59 - 0.70710680587859431*x66, 0.70710680587859343*x19*x65 - 0.70710680587859343*x21*x58 - 0.70710675649450017*x59 - 0.70710675649450017*x66, -x67 - x68, 0.316*x4 - 0.082500000000000004*x55 + 0.082500000000000004*x56 + 0.38399999999999967*x60 + 0.38399999999999967*x61 + 0.087999999999999995*x63 - 0.10699999999999987*x67 - 0.10699999999999987*x68 - 0.082500000000000004*x10*x4 - x35*x64 + 0.33300000000000002],
        [0.0, 0.0, 0.0, 1.0],
    ];
    m
}

/// Geometric Jacobian `[Jv; Jw]` (6 x DOF) of the end-effector.
///
/// Rows 0-2 are linear velocity, rows 3-5 are angular velocity.
pub fn jacobian(q: &[f64; 7]) -> [[f64; 7]; 6] {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q4 = q[4];
    let q5 = q[5];
    let q6 = q[6];
    let x0 = libm::sin(q0);
    let x1 = libm::sin(q1);
    let x2 = x0*x1;
    let x3 = libm::sin(q2);
    let x4 = libm::cos(q0);
    let x5 = x3*x4;
    let x6 = libm::cos(q1);
    let x7 = libm::cos(q2);
    let x8 = x0*x7;
    let x9 = x6*x8;
    let x10 = libm::sin(q3);
    let x11 = x5 + x9;
    let x12 = x10*x11;
    let x13 = libm::cos(q3);
    let x14 = x11*x13;
    let x15 = x13*x2;
    let x16 = x10*x2;
    let x17 = x12 - x15;
    let x18 = libm::sin(q5);
    let x19 = 0.087999999999999995*x18;
    let x20 = libm::cos(q5);
    let x21 = x17*x20;
    let x22 = libm::sin(q4);
    let x23 = x4*x7;
    let x24 = x0*x3;
    let x25 = -x23 + x24*x6;
    let x26 = libm::cos(q4);
    let x27 = x14 + x16;
    let x28 = -x22*x25 + x26*x27;
    let x29 = x18*x28;
    let x30 = 0.087999999999999995*x20;
    let x31 = -x17*x19 + 0.10699999999999987*x21 + x28*x30 + 0.10699999999999987*x29;
    let x32 = -0.38399999999999967*x12 - 0.082500000000000004*x14 + 0.38399999999999967*x15 - 0.082500000000000004*x16 + x31;
    let x33 = x32 + 0.082500000000000004*x5 + 0.082500000000000004*x9;
    let x34 = 0.316*x2 + x33;
    let x35 = x1*x7;
    let x36 = 0.082500000000000004*x35;
    let x37 = x10*x6;
    let x38 = 0.082500000000000004*x37;
    let x39 = x13*x6;
    let x40 = x10*x35;
    let x41 = x39 + x40;
    let x42 = x20*x41;
    let x43 = 0.10699999999999987*x42;
    let x44 = x1*x3;
    let x45 = x13*x35;
    let x46 = -x37 + x45;
    let x47 = -x22*x44 + x26*x46;
    let x48 = x18*x47;
    let x49 = -0.087999999999999995*x18*x41 + x30*x47 + x43 + 0.10699999999999987*x48;
    let x50 = -0.38399999999999967*x1*x10*x7 - 0.082500000000000004*x1*x13*x7 - 0.38399999999999967*x13*x6 + x38 + x49;
    let x51 = x36 + x50;
    let x52 = -x51 + 0.316*x6;
    let x53 = -x47;
    let x54 = 0.10699999999999987*x18;
    let x55 = x19*x41 + x30*x53 - x43 + x53*x54;
    let x56 = x22*x27 + x25*x26;
    let x57 = x22*x46 + x26*x44;
    let x58 = x1*x4;
    let x59 = x13*x58;
    let x60 = x23*x6;
    let x61 = x24 - x60;
    let x62 = x10*x61;
    let x63 = x59 + x62;
    let x64 = 0.316*x58;
    let x65 = x5*x6 + x8;
    let x66 = x22*x65;
    let x67 = x10*x58;
    let x68 = x13*x61;
    let x69 = -x67 + x68;
    let x70 = x26*x69 + x66;
    let x71 = x18*x70;
    let x72 = 0.10699999999999987*x71;
    let x73 = x20*x63;
    let x74 = 0.10699999999999987*x73;
    let x75 = x30*x70;
    let x76 = 0.082500000000000004*x67;
    let x77 = 0.38399999999999967*x59 + 0.38399999999999967*x62 + 0.082500000000000004*x68 - x76;
    let x78 = -0.082500000000000004*x24 + 0.082500000000000004*x60;
    let x79 = -x38 + 0.38399999999999967*x39 + 0.38399999999999967*x40 + 0.082500000000000004*x45 + x55;
    let x80 = -x10*x61 - x59;
    let x81 = -x69;
    let x82 = x26*x81 - x66;
    let x83 = -x19*x80 + 0.10699999999999987*x20*x80 + x30*x82 + x54*x82;
    let x84 = x77 + x83;
    let x85 = x78 + x84;
    let x86 = -0.087999999999999995*x18*x63 + x72 + x74 + x75;
    let x87 = -x22*x69 + x26*x65;
    let j: [[f64; 7]; 6] = [
        [-x34, x4*x52, -x2*x51 - x33*x6, -x25*x50 + x32*x44, -x17*x55 - x31*x41, x31*x57 - x49*x56, 0.0],
        [x19*x63 + x64 - x72 - x74 - x75 + x77 + x78, x0*x52, -x58*(-x36 + x79) + x6*x85, -x44*x84 - x65*x79, x41*x83 + x55*x80, -x55*(x22*x81 + x26*x65) - x57*x83, 0.0],
        [0.0, -x0*x34 - x4*(x64 + x85), x1*(-x0*x85 + x33*x4), -x25*(0.38399999999999967*x1*x13*x4 + 0.38399999999999967*x10*x61 + 0.082500000000000004*x13*x61 - x76 - x86) + x32*x65, x17*x83 - x31*x80, x31*x87 + x56*x86, 0.0],
        [0.0, -x0, x58, x65, x63, x87, -x71 - x73],
        [0.0, x4, x2, x25, -x17, x56, x21 + x29],
        [1.0, 0.0, x6, -x44, x41, -x57, -x42 - x48],
    ];
    j
}
