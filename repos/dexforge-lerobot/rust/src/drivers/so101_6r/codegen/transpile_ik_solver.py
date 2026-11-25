"""Translate the ssik `three_parallel` artifact for so101_6r into ik_solver.rs.

Mechanical, deterministic translation (what `codegen/translate_ssik_artifact.md` asks for):
the arm-specific constants and straight-line algebra are lifted from the artifact text;
the wrapper (dedup, limits, seed sort, LM polish, FK, tests) is the reference arm's
(`repos/dexforge-ur/.../ur5/ik_solver.rs`) with SO-101 names, limits, T_HOME and a generic
reference configuration. The UR5-specific wrist-singularity test is not carried over.

Usage (from the repository root):
    python repos/dexforge-lerobot/rust/src/drivers/so101_6r/codegen/transpile_ik_solver.py \\
        repos/dexforge-lerobot/rust/src/drivers/so101_6r/codegen/so101_6r_ik.py \\
        repos/dexforge-lerobot/rust/src/drivers/so101_6r/ik_solver.rs

ik_solver.rs is *written by this script*, not by hand: rerun it after the artifact changes. It
needs `ssik` (for the artifact's imports) and the UR5 reference driver in `repos/dexforge-ur`.
"""
import importlib.util, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]  # repos/dexforge-lerobot/rust/src/drivers/so101_6r/codegen -> repo root
art_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
ARM = "so101_6r"
REF_Q = "[0.3, -0.8, 1.0, -0.4, 0.8, 0.2]"
src = art_path.read_text()
lines = src.splitlines()
spec = importlib.util.spec_from_file_location("art", art_path)
art = importlib.util.module_from_spec(spec); spec.loader.exec_module(art)

def conv(e):
    e = e.strip()
    for fn in ("sin", "cos", "atan2", "sqrt", "acos"):
        e = e.replace(f"math.{fn}(", f"libm::{fn}(")
    e = e.replace("math.pi", "PI")
    assert "math." not in e and "**" not in e and "np." not in e, e
    stripped = re.sub(r"\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\.\d+(?:[eE][-+]?\d+)?", "F", e)
    assert not re.search(r"(?<![\w.])\d+(?![\w.])", stripped), f"bare integer literal in: {e[:120]}"
    return e

def find(pat, start=0):
    for i in range(start, len(lines)):
        if re.match(pat, lines[i]):
            return i
    raise KeyError(pat)

i0 = find(r"def _solve_algebraic")
i_p16 = find(r"\s+p_16_x =", i0)
i_loop = find(r"\s+for q1, q5 in theta15_solutions", i0)
i_dinr_end = find(r"\s+d_elbow =", i0)
i_q2 = find(r"\s+q2_x0 =", i0)
i_end = find(r"\s+q4 =", i0)
assign = re.compile(r"^\s+(\w+) = (.+)$")

def block(a, b):
    return [f"let {m.group(1)} = {conv(m.group(2))};" for ln in lines[a:b] if (m := assign.match(ln))]

pre, consts = [], []
for ln in lines[i_p16:i_loop]:
    m = assign.match(ln)
    if not m:
        continue
    name, rhs = m.groups()
    if rhs.startswith("np.array(") and name.startswith("_"):
        vals = re.search(r"\[(.*)\]", rhs).group(1)
        consts.append(f"const {name.lstrip('_').upper()}: [f64; 3] = [{vals}];")
    elif name.startswith(("p_16_", "r_06_axes5_")):
        pre.append(f"let {name} = {conv(rhs)};")
sp6 = re.search(r"_sp6_runtime\.solve\(h_sp, k_sp, p_sp, ([^,]+), ([^)]+)\)", src)
sp6_a, sp6_b = sp6.group(1).strip(), sp6.group(2).strip()

loop_body = [l for l in block(i_loop, i_dinr_end + 1) if not re.match(r"let (s1|c1|s5|c5|s14|c14) =", l)]
q3_r_sq = re.search(r"_q3_R_sq = (\S+)", src).group(1)
q3_rhs = re.search(r"_q3_rhs = (.+)", src).group(1).replace("1/2*d_elbow**2", "0.5 * (d_elbow * d_elbow)")
q3_phi = conv(re.search(r"_q3_phi = (.+)", src).group(1))
q2_body = block(i_q2, i_end)
ind = lambda ls, n: "\n".join(" " * n + l for l in ls)

algebraic = f"""\
    let p_16 = [p_16_x, p_16_y, p_16_z];
    let r_06_axes5 = [r_06_axes5_x, r_06_axes5_y, r_06_axes5_z];

{ind(consts, 4)}

    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];
    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, {sp6_a}, {sp6_b}, policy.subproblem_numerical, policy.subproblem_degeneracy, &mut theta15);

    for si in 0..n15 {{
        let (q1, q5) = theta15[si];

        // NOTE: the literal constants below are the machine-generated symbolic
        // coefficients emitted by `ssik build`, copied verbatim from `{ARM}_ik.py`
        // (mechanically translated by `codegen/transpile_ik_solver.py`; most small ones
        // are rounding residue of the URDF's rotations).
{ind(loop_body, 8)}

        // SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        let q3_r_sq = {q3_r_sq}_f64;
        let q3_rhs = {q3_rhs};
        let q3_phi = {q3_phi};
        let theta_q3_plus;
        let theta_q3_minus;
        if q3_r_sq < policy.axis_parallel * policy.axis_parallel {{
            theta_q3_plus = 0.0;
            theta_q3_minus = 0.0;
        }} else {{
            let q3_r = libm::sqrt(q3_r_sq);
            if libm::fabs(q3_rhs) > q3_r + policy.subproblem_feasibility {{
                let p = if q3_rhs > 0.0 {{ q3_phi }} else {{ q3_phi + PI }};
                theta_q3_plus = p;
                theta_q3_minus = p;
            }} else {{
                let clipped = (q3_rhs / q3_r).clamp(-1.0, 1.0);
                let delta = libm::acos(clipped);
                theta_q3_plus = q3_phi + delta;
                theta_q3_minus = q3_phi - delta;
            }}
        }}

        for q3 in [theta_q3_plus, theta_q3_minus] {{
{ind(q2_body, 12)}
            let q4 = libm::remainder(theta14 - q2 - q3, TWO_PI);

            if count < 8 {{
                out[count] = [q1, q2, q3, q4, q5, q6];
                count += 1;
            }}
        }}
    }}

    count
}}
"""

ur5 = (REPO / "repos/dexforge-ur/rust/src/drivers/ur5/ik_solver.rs").read_text()

# --- algebraic region ---
a = ur5.index("    // T_target-derived SP6 inputs.")
b = ur5.index("// ---------------------------------------------------------------------------\n// Forward kinematics")
rs = ur5[:a] + "    // T_target-derived SP6 inputs.\n" + ind(pre, 4) + "\n\n" + algebraic + "\n\n" + ur5[b:]

# --- module docs ---
doc_end = rs.index("use nalgebra::SMatrix;")
new_doc = f"""//! Analytical inverse kinematics for the **SO-ARM101 embedded as a 6R chain**
//! (arm name `{ARM}`).
//!
//! The physical SO-ARM101 arm has five actuated joints. This driver models it as
//! a six-joint chain: the five real joints plus one zero-offset *virtual* wrist
//! joint (`virtual_wrist`, axis perpendicular to the wrist roll) appended after
//! the wrist roll. At `q6 = 0` the chain is exactly the physical arm. The
//! five-joint arm that callers should use is [`crate::drivers::so101`], which
//! pins the virtual joint to zero.
//!
//! This is a line-by-line translation of the Python `ssik` artifact
//! `codegen/{ARM}_ik.py`, built from `{ARM}.urdf` (solver `ikgeo.three_parallel`,
//! tier 0), into a deterministic, `#![no_std]`, zero-allocation Rust
//! implementation.
//!
//! The closed form is: subproblem **SP6** for the coupled wrist/shoulder angle
//! pair `(q1, q5)`, followed by the inlined `SP1 + SP3 + SP1` chain for the
//! remaining joints. SP6 reduces to intersecting two conics, which produces a
//! quartic solved through the companion-matrix eigenvalue routine in
//! [`dexhal::math::ssik_core`].
//!
//! ## Equivalence to the Python reference
//!
//! [`solve`] mirrors `{ARM}_ik.solve` with the same argument list:
//! `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`, `policy`,
//! and `refinement_max_iters`. The algebraic solution set is FK-verified to
//! [`TolerancePolicy::subproblem_numerical`] and de-duplicated modulo `2*pi`.
//!
//! [`fk`] mirrors `{ARM}_ik.fk` (alias of `_fk`): product-of-exponentials forward
//! kinematics using the baked chain constants.

"""
rs = new_doc + rs[doc_end:]

# --- names / consts ---
rs = rs.replace("/// Degrees of freedom of the UR5.", "/// Degrees of freedom (five SO-101 joints plus the virtual wrist joint).")
rs = rs.replace('pub const BASE_LINK: &str = "base_link";', f'pub const BASE_LINK: &str = "{art.BASE_LINK}";')
rs = rs.replace('pub const EE_LINK: &str = "ee_link";', f'pub const EE_LINK: &str = "{art.EE_LINK}";')
lim = "\n".join(f"    ({float(lo)!r}, {float(hi)!r})," for lo, hi in art._JOINT_LIMITS)
rs, n = re.subn(r"const JOINT_LIMITS: \[\(f64, f64\); 6\] = \[.*?\n\];", f"const JOINT_LIMITS: [(f64, f64); 6] = [\n{lim}\n];", rs, flags=re.S)
assert n == 1
rs = rs.replace("ur5_ik.py", f"{ARM}_ik.py").replace("ur5_ik", f"{ARM}_ik")

# --- analytic refinement (master prompt: no finite-difference Jacobians in ik_solver.rs) ---
old_imp = "use dexhal::math::kinematics::{lm_refine6, mat4_frob_diff};"
assert old_imp in rs
rs = rs.replace(old_imp, "use dexhal::math::kinematics::{lm_refine_masked_analytic, mat4_frob_diff, residual_columns};")
old_fn_start = rs.index("fn refine6(")
old_fn_end = rs.index("}\n", rs.index("lm_refine6(q_in", old_fn_start)) + 2
new_fn = """fn refine6(
    q_in: &[f64; 6],
    target: &SMatrix<f64, 4, 4>,
    max_iters: usize,
    fk_atol: f64,
) -> Option<[f64; 6]> {
    const ALL: [usize; 6] = [0, 1, 2, 3, 4, 5];
    let q = lm_refine_masked_analytic::<6, 6, _, _>(
        q_in,
        &ALL,
        |q| pose_residual12(q, target),
        |q| {
            let f = super::generated_kinematics::fk(q);
            let rot = [
                [f[0][0], f[0][1], f[0][2]],
                [f[1][0], f[1][1], f[1][2]],
                [f[2][0], f[2][1], f[2][2]],
            ];
            residual_columns(&rot, &super::generated_kinematics::jacobian(q))
        },
        max_iters,
    );
    if q.iter().all(|v| v.is_finite()) && fk_residual(&q, target) <= fk_atol {
        Some(q)
    } else {
        None
    }
}
"""
rs = rs[:old_fn_start] + new_fn + rs[old_fn_end:]
rs = rs.replace("Levenberg-Marquardt polish of a 6-DOF  configuration", "Levenberg-Marquardt polish (analytic Jacobian) of a 6-DOF configuration")

# --- tests ---
th = ",\n            ".join(", ".join(repr(float(v)) for v in row) for row in art.T_HOME)
rs, n = re.subn(
    r"        // UR5e at q = 0.*?\n        let t_home = \[.*?\];",
    "        // T_HOME at q = zeros(6), baked from the ssik artifact (the URDF's home pose).\n"
    f"        let t_home = [\n            {th},\n        ];", rs, count=1, flags=re.S)
assert n == 1
rs, n = re.subn(
    r"    /// A generic \(non-singular\) configuration\..*?const REF_Q: \[f64; 6\] = \[[^\]]*\];",
    "    /// A generic (non-singular) configuration inside every joint limit; round-trip tests use it.\n"
    f"    const REF_Q: [f64; 6] = {REF_Q};", rs, count=1, flags=re.S)
assert n == 1
# drop the UR5-specific wrist-singularity test (doc comment through end of test fn)
s = rs.index("    /// Wrist singularity (q5 = 0 or pi)")
e = rs.rindex("\n}\n")                                  # closing brace of `mod tests`
rs = rs[:s].rstrip() + "\n" + rs[e + 1:]
assert "wrist_singularity" not in rs
for old, new in (("for the UR5.", "for this arm."), ("for the UR5 (tier-0 arm)", "for this tier-0 arm"),
                 ("at most 8 for the UR5", "at most 8 here"), ("a 6-DOF UR5 configuration", "a 6-DOF configuration"),
                 ("for the UR5", "for this arm"), ("the UR5", "this arm"), ("UR5e", "SO-101"), ("UR5", "this arm")):
    rs = rs.replace(old, new)
leftover = [l for l in rs.splitlines() if re.search(r"ur5|UR5|Universal", l)]
assert not leftover, leftover[:5]
out_path.write_text(rs)
print("wrote", out_path, len(rs.splitlines()), "lines")
