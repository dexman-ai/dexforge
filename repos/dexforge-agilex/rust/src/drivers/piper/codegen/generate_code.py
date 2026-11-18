import ast, importlib.util, inspect, os, re

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("piper_ik", os.path.join(_HERE, "piper_ik.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

src = inspect.getsource(P._build_pq_matrices)
mod = ast.parse(src)
fn = mod.body[0]

def rust_num(v):
    # v is a Python int or float
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, int):
        return f"{v}.0"
    r = repr(float(v))
    # ensure it has a . or e
    if ('.' not in r) and ('e' not in r) and ('E' not in r) and ('inf' not in r) and ('nan' not in r):
        r += '.0'
    return r

def name_map(nid):
    m = re.fullmatch(r'T_(\d+)', nid)
    if m:
        return f"t{m.group(1)}"
    m = re.fullmatch(r'_pq_x(\d+)', nid)
    if m:
        return f"pq_x{m.group(1)}"
    raise ValueError("unknown name "+nid)

def emit(node):
    if isinstance(node, ast.BinOp):
        op = node.op
        l = emit(node.left); r = emit(node.right)
        if isinstance(op, ast.Add): return f"({l} + {r})"
        if isinstance(op, ast.Sub): return f"({l} - {r})"
        if isinstance(op, ast.Mult): return f"({l} * {r})"
        if isinstance(op, ast.Div): return f"({l} / {r})"
        if isinstance(op, ast.Pow):
            # exponent must be constant int
            assert isinstance(node.right, ast.Constant), ast.dump(node)
            e = node.right.value
            base = emit(node.left)
            if e == 2: return f"({base} * {base})"
            if e == 3: return f"({base} * {base} * {base})"
            raise ValueError("pow "+str(e))
        raise ValueError("op "+ast.dump(op))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"(-{emit(node.operand)})"
        if isinstance(node.op, ast.UAdd):
            return emit(node.operand)
        raise ValueError
    if isinstance(node, ast.Constant):
        return rust_num(node.value)
    if isinstance(node, ast.Name):
        return name_map(node.id)
    if isinstance(node, ast.Call):
        raise ValueError("unexpected call "+ast.dump(node))
    raise ValueError("node "+ast.dump(node))

lets = []
arrays = {}
for stmt in fn.body:
    if isinstance(stmt, ast.Assign):
        tgt = stmt.targets[0]
        if isinstance(tgt, ast.Name):
            nid = tgt.id
            if nid.startswith('_pq_x'):
                lets.append((name_map(nid), emit(stmt.value)))
            elif nid in ('p_sin','p_cos','p_one','q'):
                # value is a Call np.array([...], dtype=...) -> first arg is List of Lists
                call = stmt.value
                lst = call.args[0]
                rows = []
                for row in lst.elts:
                    cols = [emit(e) for e in row.elts]
                    rows.append(cols)
                arrays[nid] = rows
        else:
            pass
    elif isinstance(stmt, ast.Return):
        pass

# sanity: shapes
for k,v in arrays.items():
    print("//", k, len(v), "x", len(v[0]))

dest = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "pq.rs"))
with open(dest, "w") as f:
    f.write("//! Per-arm Raghavan-Roth `(P_sin, P_cos, P_one, Q)` matrix builder for the\n")
    f.write("//! AgileX Piper, translated verbatim from\n")
    f.write("//! `codegen/piper_ik.py::_build_pq_matrices`.\n")
    f.write("//!\n")
    f.write("//! The body is machine-generated (sympy `cse`) and is reproduced here without\n")
    f.write("//! manual edits so the floating-point accumulation order matches the Python\n")
    f.write("//! reference bit-for-bit. Regenerate with `codegen/generate_code.py`.\n\n")
    f.write("// Do not edit by hand. See codegen/generate_code.py.\n")
    f.write("#[allow(clippy::all)]\n")
    f.write("#[allow(unused_parens)]\n")
    f.write("#[allow(clippy::too_many_lines)]\n")
    f.write("pub(crate) fn build_pq_matrices(\n")
    f.write("    t0: f64, t1: f64, t2: f64, t3: f64, t4: f64, t5: f64,\n")
    f.write("    t6: f64, t7: f64, t8: f64, t9: f64, t10: f64, t11: f64,\n")
    f.write(") -> ([[f64; 9]; 14], [[f64; 9]; 14], [[f64; 9]; 14], [[f64; 8]; 14]) {\n")
    for name, expr in lets:
        f.write(f"    let {name} = {expr};\n")
    def emit_array(name, rows, ncol):
        f.write(f"    let {name}: [[f64; {ncol}]; 14] = [\n")
        for row in rows:
            assert len(row)==ncol, (name, len(row), ncol)
            f.write("        [" + ", ".join(row) + "],\n")
        f.write("    ];\n")
    emit_array("p_sin", arrays['p_sin'], 9)
    emit_array("p_cos", arrays['p_cos'], 9)
    emit_array("p_one", arrays['p_one'], 9)
    emit_array("q", arrays['q'], 8)
    f.write("    (p_sin, p_cos, p_one, q)\n")
    f.write("}\n")
print("wrote src/drivers/piper/pq.rs")

# ---- consts.rs: baked geometric constants from the artifact -----------------

def _num(x):
    r = repr(float(x))
    return r if any(c in r for c in ".e") else r + ".0"

def _vec(a):
    return "[" + ", ".join(_num(x) for x in a) + "]"

def _mat(a):
    return "[\n" + "".join("    " + _vec(r) + ",\n" for r in a) + "]"

_W_TRIG_TO_X = """[
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
]"""

_limits = "[\n" + "".join(f"    ({_num(lo)}, {_num(hi)}),\n" for lo, hi in P._JOINT_LIMITS) + "]"

consts_dest = os.path.normpath(os.path.join(_HERE, "..", "consts.rs"))
with open(consts_dest, "w") as f:
    f.write("//! Baked geometric constants for the AgileX Piper 6-DOF arm,\n")
    f.write("//! extracted verbatim from `codegen/piper_ik.py`. Regenerate with\n")
    f.write("//! `codegen/generate_code.py`.\n\n")
    f.write("#![allow(clippy::unreadable_literal)]\n")
    f.write("#![allow(clippy::approx_constant)]\n\n")
    f.write(f"pub(crate) const DH_ALPHA: [f64; 6] = {_vec(P._DH_ALPHA)};\n\n")
    f.write(f"pub(crate) const DH_A: [f64; 6] = {_vec(P._DH_A)};\n\n")
    f.write(f"pub(crate) const DH_D: [f64; 6] = {_vec(P._DH_D)};\n\n")
    f.write(f"pub(crate) const DH_THETA_OFFSET: [f64; 6] = {_vec(P._DH_THETA_OFFSET)};\n\n")
    f.write(f"pub(crate) const T_PRE_INV: [[f64; 4]; 4] = {_mat(P._T_PRE_INV)};\n\n")
    f.write(f"pub(crate) const T_POST_INV: [[f64; 4]; 4] = {_mat(P._T_POST_INV)};\n\n")
    f.write(f"pub(crate) const JOINT_LIMITS: [(f64, f64); 6] = {_limits};\n\n")
    f.write(f"pub(crate) const W_TRIG_TO_X: [[f64; 9]; 9] = {_W_TRIG_TO_X};\n\n")
    f.write("#[cfg_attr(not(test), allow(dead_code))]\n")
    f.write(f"pub(crate) const T_HOME: [[f64; 4]; 4] = {_mat(P.T_HOME)};\n")
