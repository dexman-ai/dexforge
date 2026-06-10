//! Auto-generated symbolic kinematics for the Rizon 10 (7-DOF).
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

/// Degrees of freedom of the Rizon 10.
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
    let x9 = x5*x8;
    let x10 = x4 + x9;
    let x11 = libm::sin(q4);
    let x12 = libm::sin(q3);
    let x13 = libm::sin(q1);
    let x14 = x13*x7;
    let x15 = x12*x14;
    let x16 = libm::cos(q3);
    let x17 = x2*x6;
    let x18 = x17 - x3*x5*x7;
    let x19 = -x18;
    let x20 = x15 + x16*x19;
    let x21 = x1*x10 + x11*x20;
    let x22 = libm::cos(q6);
    let x23 = libm::cos(q5);
    let x24 = x14*x16;
    let x25 = libm::sin(q5);
    let x26 = x10*x11;
    let x27 = -x26;
    let x28 = -x23*(x12*x19 - x24) + x25*(x1*x20 + x27);
    let x29 = x12*x18;
    let x30 = x24 + x29;
    let x31 = x25*x30;
    let x32 = -x15 + x16*x18;
    let x33 = -x32;
    let x34 = x1*x33 + x27;
    let x35 = 0.1100000000000001*x25;
    let x36 = x17*x5;
    let x37 = -x3*x7 + x36;
    let x38 = -x37;
    let x39 = x13*x2;
    let x40 = x4*x5 + x8;
    let x41 = x12*x39 + x16*x40;
    let x42 = x11*x41;
    let x43 = -x1*x38 + x42;
    let x44 = x16*x39;
    let x45 = x12*x40;
    let x46 = -x44 + x45;
    let x47 = x23*x46;
    let x48 = x1*x41;
    let x49 = x25*(x11*x38 + x48) - x47;
    let x50 = x25*x46;
    let x51 = -x11*x37 + x48;
    let x52 = x23*x51;
    let x53 = 0.10299999999999998*x1;
    let x54 = x13*x6;
    let x55 = x13*x3;
    let x56 = -x12*x5 + x16*x55;
    let x57 = x11*x56;
    let x58 = x1*x54 + x57;
    let x59 = x16*x5;
    let x60 = x12*x55;
    let x61 = x59 + x60;
    let x62 = x23*x61;
    let x63 = x1*x56 - x11*x54;
    let x64 = x25*x63;
    let x65 = -x62 + x64;
    let x66 = x25*x61;
    let x67 = x23*x63;
    let m: [[f64; 4]; 4] = [
        [-x0*x21 + x22*x28, -x0*x28 - x21*x22, -x23*(x1*x32 + x26) - x31, 0.10299999999999998*x1*x10 + 0.10299999999999998*x11*x33 - 0.45000000000000007*x14 + 0.13*x2 - 0.1100000000000001*x23*x30 + 0.095999999999999976*x23*x34 - 0.39499999999999991*x24 - 0.39499999999999991*x29 - 0.095999999999999976*x31 - x34*x35 - 0.12*x4 - 0.12*x9],
        [-x0*x43 + x22*x49, -x0*x49 - x22*x43, x50 + x52, 0.12*x3*x7 - x35*x51 - 0.12*x36 + x37*x53 - 0.45000000000000007*x39 + 0.10299999999999998*x42 - 0.39499999999999991*x44 + 0.39499999999999991*x45 + 0.1100000000000001*x47 + 0.095999999999999976*x50 + 0.095999999999999976*x52 - 0.13*x7],
        [-x0*x58 + x22*x65, -x0*x65 - x22*x58, x66 + x67, 0.45000000000000007*x5 - 0.12*x54 + 0.10299999999999998*x57 + 0.39499999999999991*x59 + 0.39499999999999991*x60 + 0.1100000000000001*x62 - 0.1100000000000001*x64 + 0.095999999999999976*x66 + 0.095999999999999976*x67 + x53*x54 + 0.44999999999999996],
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
    let x0 = libm::cos(q0);
    let x1 = libm::sin(q0);
    let x2 = libm::sin(q1);
    let x3 = libm::cos(q2);
    let x4 = x0*x3;
    let x5 = 0.12*x4;
    let x6 = libm::cos(q1);
    let x7 = libm::sin(q2);
    let x8 = x1*x7;
    let x9 = x6*x8;
    let x10 = 0.12*x9;
    let x11 = libm::sin(q3);
    let x12 = x0*x7;
    let x13 = x1*x3;
    let x14 = x12 + x13*x6;
    let x15 = x11*x14;
    let x16 = 0.39499999999999991*x15;
    let x17 = libm::cos(q3);
    let x18 = x1*x2;
    let x19 = x17*x18;
    let x20 = 0.39499999999999991*x19;
    let x21 = libm::sin(q4);
    let x22 = x11*x18 + x14*x17;
    let x23 = x21*x22;
    let x24 = 0.10299999999999998*x23;
    let x25 = libm::cos(q4);
    let x26 = -x4 + x9;
    let x27 = x25*x26;
    let x28 = x15 - x19;
    let x29 = libm::cos(q5);
    let x30 = 0.1100000000000001*x29;
    let x31 = x28*x30;
    let x32 = x22*x25;
    let x33 = -x21*x26 + x32;
    let x34 = x29*x33;
    let x35 = libm::sin(q5);
    let x36 = x28*x35;
    let x37 = 0.095999999999999976*x36;
    let x38 = 0.1100000000000001*x35;
    let x39 = x31 - x33*x38 + 0.095999999999999976*x34 + x37;
    let x40 = x24 + 0.10299999999999998*x27 + x39;
    let x41 = x16 - x20 + x40;
    let x42 = -x10 + x41 + x5;
    let x43 = x2*x7;
    let x44 = x17*x6;
    let x45 = x2*x3;
    let x46 = x11*x45;
    let x47 = -x11*x6 + x17*x45;
    let x48 = x21*x47;
    let x49 = x25*x43;
    let x50 = x44 + x46;
    let x51 = -x21*x43 + x25*x47;
    let x52 = x29*x51;
    let x53 = x35*x50;
    let x54 = x30*x50 - x38*x51 + 0.095999999999999976*x52 + 0.095999999999999976*x53;
    let x55 = 0.10299999999999998*x48 + 0.10299999999999998*x49 + x54;
    let x56 = 0.39499999999999991*x44 + 0.39499999999999991*x46 + x55;
    let x57 = -0.12*x43 + x56;
    let x58 = x57 + 0.45000000000000007*x6;
    let x59 = x57 + 0.22250000000000003*x6;
    let x60 = -0.22250000000000003*x18 + x42;
    let x61 = -0.057500000000000002*x43 + x56;
    let x62 = 0.057500000000000002*x4 + x41 - 0.057500000000000002*x9;
    let x63 = 0.18999999999999995*x44 + 0.18999999999999995*x46 + x55;
    let x64 = 0.18999999999999995*x15 - 0.18999999999999995*x19 + x40;
    let x65 = x23 + x27;
    let x66 = 0.072999999999999982*x48 + 0.072999999999999982*x49 + x54;
    let x67 = x48 + x49;
    let x68 = 0.072999999999999982*x23 + 0.072999999999999982*x27 + x39;
    let x69 = x0*x2;
    let x70 = 0.45000000000000007*x69;
    let x71 = x12*x6;
    let x72 = x13 + x71;
    let x73 = x25*x72;
    let x74 = 0.10299999999999998*x73;
    let x75 = -x74;
    let x76 = -x4*x6 + x8;
    let x77 = x11*x76;
    let x78 = 0.12*x13;
    let x79 = x17*x69;
    let x80 = 0.39499999999999991*x79;
    let x81 = 0.12*x71;
    let x82 = x77 + x79;
    let x83 = x35*x82;
    let x84 = x30*x82 + 0.095999999999999976*x83;
    let x85 = -x0*x11*x2 + x17*x76;
    let x86 = -x85;
    let x87 = x21*x86;
    let x88 = x21*x72;
    let x89 = -x88;
    let x90 = x25*x86 + x89;
    let x91 = 0.095999999999999976*x29;
    let x92 = x38*x90 - x90*x91;
    let x93 = -0.10299999999999998*x87 + x92;
    let x94 = x75 + 0.39499999999999991*x77 + x78 + x80 + x81 + x84 + x93;
    let x95 = -x76;
    let x96 = x11*x69 + x17*x95;
    let x97 = x11*x95;
    let x98 = -x79 + x97;
    let x99 = x25*x96 + x89;
    let x100 = 0.10299999999999998*x21*x96 + x30*x98 + 0.095999999999999976*x35*x98 - 0.1100000000000001*x35*x99 + x74 - x80 + x91*x99 + 0.39499999999999991*x97;
    let x101 = 0.057500000000000002*x0*x6*x7 + 0.057500000000000002*x1*x3 - x100;
    let x102 = x75 + 0.18999999999999995*x77 + 0.18999999999999995*x79 + x84;
    let x103 = -0.072999999999999982*x25*x72 + x84;
    let x104 = -x26;
    let x105 = x104*x21 + x32;
    let x106 = x100 - x78 - x81;
    let x107 = x21*x85;
    let x108 = x25*x85 + x88;
    let x109 = x108*x29;
    let x110 = -0.1100000000000001*x108*x35 + 0.095999999999999976*x109;
    let x111 = -x107 + x25*x72;
    let j: [[f64; 7]; 6] = [
        [0.13*x0 + 0.45000000000000007*x1*x2 - x42, -x0*x58, -x18*x59 - x6*x60, x2*x62*x7 - x26*x61, x28*x63 - x50*x64, x65*x66 - x67*x68, 0.0],
        [0.13*x1 - x70 - x94, -x1*x58, x59*x69 - x6*(0.22250000000000003*x69 + x94), x101*x43 + x61*x72, x50*(-x102 - x93) + x63*x82, -x66*(x73 + x87) + x67*(-x103 + 0.072999999999999982*x21*x86 - x92), 0.0],
        [0.0, -x0*(-0.0625*x1 - x106 + x70) - x1*(0.0625*x0 + x10 + 0.10299999999999998*x104*x25 + x105*x38 - x105*x91 - x16 + 0.45000000000000007*x18 + x20 - x24 - x31 - x37 - x5), -x2*(x0*x60 + x1*(0.22250000000000003*x0*x2 - x106)), -x101*x26 - x62*x72, -x28*(-x102 - 0.10299999999999998*x107 - x110) - x64*x82, x111*x68 - x65*(-x103 - 0.072999999999999982*x107 - x110), 0.0],
        [0.0, x1, -x69, -x72, -x82, x111, -x109 - x83],
        [0.0, -x0, -x18, x104, x28, x65, x34 + x36],
        [1.0, 0.0, x6, -x43, x50, x67, x52 + x53],
    ];
    j
}
