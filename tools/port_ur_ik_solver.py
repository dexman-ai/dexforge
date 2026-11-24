#!/usr/bin/env python3
"""Re-port the algebraic solve block of a UR arm's ``ik_solver.rs`` from its ssik artifact.

The UR arms share one solver family (``ikgeo.three_parallel``: SP6 for ``(q1, q5)``, then an
inlined ``SP1 + SP3 + SP1`` chain). Only the machine-generated constants and the straight-line
algebra differ between arms, so this rewrites exactly two regions of
``drivers/<arm>/ik_solver.rs`` from ``drivers/<arm>/codegen/<arm>_ik.py``:

* the body of ``solve_algebraic`` (from ``// T_target-derived SP6 inputs.`` to the forward
  kinematics section), and
* the ``JOINT_LIMITS`` table.

Everything else in the file (kernels, wrappers, doc comments, tests) is left alone. The expressions
are translated verbatim (``math.sin`` -> ``libm::sin`` and so on); anything the translator does not
understand stops the run instead of being guessed.

Run it after ``ssik_artifact.py build`` has refreshed the artifact. Re-running it on an up to date
file changes nothing.

Usage::

    python tools/port_ur_ik_solver.py --arm ur5
    python tools/port_ur_ik_solver.py --arm ur3e
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOLVE_START = "    // T_target-derived SP6 inputs."
FK_SECTION = "\n\n// ---------------------------------------------------------------------------\n// Forward kinematics"


def conv(expr: str) -> str:
    e = expr.strip()
    e = (e.replace("math.sin", "libm::sin").replace("math.cos", "libm::cos")
          .replace("math.atan2", "libm::atan2").replace("math.sqrt", "libm::sqrt").replace("math.pi", "PI"))
    e = e.replace("1/2*d_elbow**2", "0.5 * d_elbow * d_elbow")
    assert "**" not in e and "%" not in e, f"untranslated expression: {e}"
    return e


def build_solve_block(py: str) -> str:
    body = py[py.index("def _solve_algebraic"):py.index("    return candidates")]
    lines = body.splitlines()
    skip = re.compile(r"^\s*(s1|c1|s5|c5|s14|c14|s3|c3) = ")
    consts: dict[str, str] = {}
    for ln in lines:
        m = re.match(r"_(H_SP_0|K_SP_0|K_SP_1|NEG_P_5|NEG_AXES_5) = np.array\(\[(.*)\]\)", ln.strip())
        if m:
            consts[m.group(1)] = m.group(2)
    missing = {"H_SP_0", "K_SP_0", "K_SP_1", "NEG_P_5", "NEG_AXES_5"} - consts.keys()
    assert not missing, f"artifact lacks SP6 constants {sorted(missing)}"

    res = [SOLVE_START]
    for ln in lines:
        m = re.match(r"(p_16_[xyz]|r_06_axes5_[xyz]) = (.*)", ln.strip())
        if m:
            res.append(f"    let {m.group(1)} = {conv(m.group(2))};")
    res.append("")
    for n in ("H_SP_0", "K_SP_0", "K_SP_1", "NEG_P_5", "NEG_AXES_5"):
        res.append(f"    const {n}: [f64; 3] = [{consts[n]}];")
    res.append("""
    let p_16 = [p_16_x, p_16_y, p_16_z];
    let r_06_axes5 = [r_06_axes5_x, r_06_axes5_y, r_06_axes5_z];
    let h_sp = [H_SP_0, H_SP_0, H_SP_0, H_SP_0];
    let k_sp = [K_SP_0, K_SP_1, K_SP_0, K_SP_1];
    let p_sp = [p_16, NEG_P_5, r_06_axes5, NEG_AXES_5];

    let mut theta15 = [(0.0f64, 0.0f64); 4];""")
    d = re.search(r"_sp6_runtime\.solve\(h_sp, k_sp, p_sp, ([0-9.e-]+), 0\.0\)", body).group(1)
    res.append(f"    let n15 = sp6_solve(&h_sp, &k_sp, &p_sp, {d}, 0.0, policy.subproblem_numerical, "
               "policy.subproblem_degeneracy, &mut theta15);")
    res.append("""
    for si in 0..n15 {
        let (q1, q5) = theta15[si];

        // NOTE: the literal constants below are the machine-generated symbolic
        // coefficients emitted by `ssik build`, copied verbatim from `ur5_ik.py`
        // (mechanically translated; most are rounding residue of cos(pi/2)).""")

    inner = body[body.index("        # SP1 for theta14"):]
    in_q3 = False
    for ln in inner.splitlines():
        s = ln.strip()
        if not s or skip.match(ln):
            continue
        ind = " " * (len(ln) - len(ln.lstrip()))
        if s.startswith("#"):
            res.append(ind + "//" + s[1:])
            continue
        if s.startswith("for q3 in"):
            res.append(ind + "for q3 in [theta_q3_plus, theta_q3_minus] {")
            in_q3 = False
            continue
        if s.startswith("candidates.append"):
            res.append(ind + "if count < 8 {\n" + ind + "    out[count] = [q1, q2, q3, q4, q5, q6];\n"
                       + ind + "    count += 1;\n" + ind + "}")
            continue
        if s.startswith("q4 = "):
            res.append(ind + "let q4 = libm::remainder(theta14 - q2 - q3, TWO_PI);")
            continue
        if s.startswith("_q3_R_sq"):
            res.append(ind + "let q3_r_sq = " + s.split("= ")[1] + "_f64;")
            continue
        if s.startswith("_q3_rhs = "):
            res.append(ind + "let q3_rhs = " + conv(s.split("= ", 1)[1]) + ";")
            continue
        if s.startswith("_q3_phi = "):
            res.append(ind + "let q3_phi = " + conv(s.split("= ", 1)[1]) + ";\n" + ind
                       + "let theta_q3_plus;\n" + ind + "let theta_q3_minus;")
            in_q3 = True
            continue
        if in_q3:
            if s.startswith("if _q3_R_sq"):
                res.append(
                    ind + "if q3_r_sq < policy.axis_parallel * policy.axis_parallel {\n"
                    + ind + "    theta_q3_plus = 0.0;\n" + ind + "    theta_q3_minus = 0.0;\n"
                    + ind + "} else {\n" + ind + "    let q3_r = libm::sqrt(q3_r_sq);\n"
                    + ind + "    if libm::fabs(q3_rhs) > q3_r + policy.subproblem_feasibility {\n"
                    + ind + "        let p = if q3_rhs > 0.0 { q3_phi } else { q3_phi + PI };\n"
                    + ind + "        theta_q3_plus = p;\n" + ind + "        theta_q3_minus = p;\n"
                    + ind + "    } else {\n"
                    + ind + "        let clipped = (q3_rhs / q3_r).clamp(-1.0, 1.0);\n"
                    + ind + "        let delta = libm::acos(clipped);\n"
                    + ind + "        theta_q3_plus = q3_phi + delta;\n"
                    + ind + "        theta_q3_minus = q3_phi - delta;\n"
                    + ind + "    }\n" + ind + "}")
                continue
            if s.startswith("for q3 in"):
                in_q3 = False
            else:
                continue
        m = re.match(r"([A-Za-z_0-9]+) = (.*)", s)
        assert m, f"unhandled line in the artifact: {s!r}"
        res.append(f"{ind}let {m.group(1)} = {conv(m.group(2))};")
    res.append("        }\n    }\n\n    count\n}")
    return "\n".join(res) + "\n"


def port(arm: str) -> bool:
    d = REPO / "repos/dexforge-ur/rust/src/drivers" / arm
    py = (d / "codegen" / f"{arm}_ik.py").read_text()
    path = d / "ik_solver.rs"
    rs = path.read_text()
    block = build_solve_block(py).replace(f"`ur5_ik.py`", f"`{arm}_ik.py`")
    a = rs.index(SOLVE_START)
    b = rs.index(FK_SECTION)
    new = rs[:a] + block + rs[b:]
    lim = re.search(r"_JOINT_LIMITS = \[(.*?)\n\]", py, re.S).group(1)
    rows = re.findall(r"\(([^)]*)\)", lim)
    table = "const JOINT_LIMITS: [(f64, f64); 6] = [\n" + "\n".join(f"    ({r})," for r in rows) + "\n];"
    new, n = re.subn(r"const JOINT_LIMITS: \[\(f64, f64\); 6\] = \[.*?\n\];", lambda _: table, new, flags=re.S)
    assert n == 1, "JOINT_LIMITS table not found"
    changed = new != rs
    if changed:
        path.write_text(new)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["ur5", "ur3e"])
    args = ap.parse_args()
    print(("rewrote" if port(args.arm) else "unchanged:"), f"drivers/{args.arm}/ik_solver.rs")


if __name__ == "__main__":
    main()
