//! Auto-generated symbolic kinematics for the xArm7 (7-DOF).
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

/// Degrees of freedom of the xArm7.
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
    let x0 = libm::sin(q6);
    let x1 = libm::cos(q4);
    let x2 = libm::sin(q0);
    let x3 = libm::cos(q2);
    let x4 = x2*x3;
    let x5 = libm::cos(q1);
    let x6 = libm::sin(q2);
    let x7 = libm::cos(q0);
    let x8 = x6*x7;
    let x9 = x4 + x5*x8;
    let x10 = x1*x9;
    let x11 = libm::sin(q4);
    let x12 = libm::sin(q3);
    let x13 = libm::sin(q1);
    let x14 = x13*x7;
    let x15 = libm::cos(q3);
    let x16 = x2*x6;
    let x17 = x3*x7;
    let x18 = x16 - x17*x5;
    let x19 = x15*x18;
    let x20 = -x12*x14 + x19;
    let x21 = libm::cos(q6);
    let x22 = libm::sin(q5);
    let x23 = x14*x15;
    let x24 = x12*x18;
    let x25 = x23 + x24;
    let x26 = x22*x25;
    let x27 = libm::cos(q5);
    let x28 = -x1*x20 + x11*x9;
    let x29 = -x20;
    let x30 = -x12*x18 - x23;
    let x31 = x1*x29 + x11*x9;
    let x32 = x16*x5 - x17;
    let x33 = x13*x2;
    let x34 = x12*x33;
    let x35 = x4*x5;
    let x36 = x35 + x8;
    let x37 = x15*x36;
    let x38 = x34 + x37;
    let x39 = -x1*x32 + x11*x38;
    let x40 = x15*x33;
    let x41 = x12*x36;
    let x42 = -x40 + x41;
    let x43 = x22*x42;
    let x44 = x1*x38 + x11*x32;
    let x45 = x27*x44;
    let x46 = x43 + x45;
    let x47 = x22*x44;
    let x48 = x13*x6;
    let x49 = x13*x3;
    let x50 = x15*x49;
    let x51 = -x12*x5 + x50;
    let x52 = -x1*x48 + x11*x51;
    let x53 = x15*x5;
    let x54 = x12*x49;
    let x55 = x53 + x54;
    let x56 = x22*x55;
    let x57 = x1*x51 + x11*x48;
    let x58 = x27*x57;
    let x59 = x56 + x58;
    let x60 = x27*x55;
    let m: [[f64; 4]; 4] = [
        [x0*(x10 + x11*x20) + x21*(-x26 + x27*x28), -x0*(x22*x30 + x27*x31) - x21*(-x10 + x11*x29), -x22*x31 + x27*x30, 0.077500000000000013*x12*x13*x7 + 0.29299999999999993*x13*x7 - 0.052499999999999998*x16 - 0.077500000000000013*x19 - 0.096999999999999892*x22*x28 - 0.3424999999999998*x23 - 0.3424999999999998*x24 - 0.096999999999999892*x25*x27 - 0.076000000000000012*x26 + 0.076000000000000012*x27*x28 + 0.052499999999999998*x3*x5*x7],
        [-x0*x39 + x21*x46, -x0*x46 - x21*x39, x27*x42 - x47, 0.096999999999999892*x27*x42 + 0.29299999999999993*x33 + 0.077500000000000013*x34 + 0.052499999999999998*x35 + 0.077500000000000013*x37 - 0.3424999999999998*x40 + 0.3424999999999998*x41 + 0.076000000000000012*x43 + 0.076000000000000012*x45 - 0.096999999999999892*x47 + 0.052499999999999998*x8],
        [x0*x52 - x21*x59, x0*x59 + x21*x52, x22*x57 - x60, -0.052499999999999998*x49 + 0.29299999999999993*x5 - 0.077500000000000013*x50 - 0.3424999999999998*x53 - 0.3424999999999998*x54 - 0.076000000000000012*x56 - 0.076000000000000012*x58 - 0.096999999999999892*x60 + 0.077500000000000013*x12*x5 + 0.096999999999999892*x22*x57 + 0.26700000000000002],
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
    let x6 = libm::sin(q3);
    let x7 = x2*x6;
    let x8 = 0.077500000000000013*x7;
    let x9 = libm::cos(q3);
    let x10 = x2*x9;
    let x11 = 0.3424999999999998*x10;
    let x12 = libm::cos(q1);
    let x13 = libm::cos(q2);
    let x14 = x0*x13;
    let x15 = x12*x14;
    let x16 = x15 + x5;
    let x17 = x16*x6;
    let x18 = 0.3424999999999998*x17;
    let x19 = x16*x9;
    let x20 = 0.077500000000000013*x19;
    let x21 = -x10 + x17;
    let x22 = libm::sin(q5);
    let x23 = 0.076000000000000012*x22;
    let x24 = libm::cos(q5);
    let x25 = libm::sin(q4);
    let x26 = x13*x4;
    let x27 = x0*x3;
    let x28 = x12*x27 - x26;
    let x29 = libm::cos(q4);
    let x30 = x19 + x7;
    let x31 = x29*x30;
    let x32 = x25*x28 + x31;
    let x33 = x22*x32;
    let x34 = 0.076000000000000012*x24;
    let x35 = -x11 + 0.052499999999999998*x15 + x18 + x20 + x21*x23 + 0.096999999999999892*x21*x24 + x32*x34 - 0.096999999999999892*x33 + 0.052499999999999998*x5 + x8;
    let x36 = 0.29299999999999993*x2 + x35;
    let x37 = x1*x13;
    let x38 = 0.052499999999999998*x37;
    let x39 = x1*x3;
    let x40 = x25*x39;
    let x41 = x12*x6;
    let x42 = x37*x9;
    let x43 = -x41 + x42;
    let x44 = x29*x43 + x40;
    let x45 = x22*x44;
    let x46 = x12*x9;
    let x47 = x37*x6;
    let x48 = x46 + x47;
    let x49 = x24*x48;
    let x50 = x23*x48 + 0.096999999999999892*x49;
    let x51 = -0.077500000000000013*x41 + 0.077500000000000013*x42 + 0.3424999999999998*x46 + 0.3424999999999998*x47;
    let x52 = -0.29299999999999993*x12 + x34*x44 + x38 - 0.096999999999999892*x45 + x50 + x51;
    let x53 = -x29*x43 - x40;
    let x54 = 0.096999999999999892*x22;
    let x55 = -x34*x53 + x50 + x53*x54;
    let x56 = x51 + x55;
    let x57 = x38 + x56;
    let x58 = -x21;
    let x59 = 0.096999999999999892*x24;
    let x60 = x25*x28 + x31;
    let x61 = x23*x58 - x34*x60 + x54*x60 + x58*x59;
    let x62 = x11 - x18 - x20 + x61 - x8;
    let x63 = x25*x30 - x28*x29;
    let x64 = x1*x29*x3 - x25*x43;
    let x65 = x12*x26;
    let x66 = x27 - x65;
    let x67 = x66*x9;
    let x68 = 0.077500000000000013*x67;
    let x69 = x6*x66;
    let x70 = 0.3424999999999998*x69;
    let x71 = x1*x4;
    let x72 = x71*x9;
    let x73 = x69 + x72;
    let x74 = x12*x5 + x14;
    let x75 = x6*x71;
    let x76 = x67 - x75;
    let x77 = x25*x74 - x29*x76;
    let x78 = 0.3424999999999998*x72;
    let x79 = 0.077500000000000013*x75;
    let x80 = x78 - x79;
    let x81 = 0.052499999999999998*x27;
    let x82 = 0.052499999999999998*x65;
    let x83 = x81 - x82;
    let x84 = -x66;
    let x85 = x84*x9;
    let x86 = x6*x84;
    let x87 = -x1*x4*x9 + x86;
    let x88 = -x87;
    let x89 = x25*x74;
    let x90 = x29*(x75 + x85) + x89;
    let x91 = x23*x88 - x34*x90 + x54*x90 + x59*x88;
    let x92 = x80 - 0.077500000000000013*x85 - 0.3424999999999998*x86 + x91;
    let x93 = x83 + x92;
    let x94 = x29*x74;
    let x95 = -x76;
    let x96 = x25*x95 - x94;
    let x97 = x29*x95 + x89;
    let x98 = x22*x97;
    let j: [[f64; 7]; 6] = [
        [-x36, -x4*x52, -x12*x35 - x2*x57, -x28*x56 - x39*x62, -x21*x55 - x48*x61, -x55*x63 + x61*x64, 0.0],
        [0.29299999999999993*x1*x4 - x23*x73 + 0.076000000000000012*x24*x77 - x54*x77 - x59*x73 - x68 - x70 - x80 - x83, -x0*x52, -x12*x93 + x57*x71, x39*x92 + x56*x74, x48*x91 - x55*x73, x55*x96 - x64*x91, 0.0],
        [0.0, -x0*x36 - x4*(x23*x87 + 0.096999999999999892*x24*x87 + x34*x97 - x68 - x70 + 0.29299999999999993*x71 - x78 + x79 - x81 + x82 - 0.096999999999999892*x98), x1*(x0*x93 + x35*x4), x28*x92 - x62*x74, x21*x91 + x61*x73, -x61*x96 + x63*x91, 0.0],
        [0.0, -x0, x71, x74, -x73, -x25*x76 - x94, x24*x87 - x98],
        [0.0, x4, x2, x28, x21, x63, x21*x24 - x33],
        [1.0, 0.0, x12, -x39, -x48, x64, x45 - x49],
    ];
    j
}
