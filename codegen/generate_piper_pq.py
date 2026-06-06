import ast, inspect, re
from ssik.prebuilt import piper_ik as P

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

import os

dest = os.path.join(os.path.dirname(__file__), "..", "src", "drivers", "piper", "pq.rs")
with open(os.path.normpath(dest), "w") as f:
    f.write("//! Per-arm Raghavan-Roth `(P_sin, P_cos, P_one, Q)` matrix builder for the\n")
    f.write("//! AgileX Piper, translated verbatim from\n")
    f.write("//! `ssik.prebuilt.piper_ik._build_pq_matrices`.\n")
    f.write("//!\n")
    f.write("//! The body is machine-generated (sympy `cse`) and is reproduced here without\n")
    f.write("//! manual edits so the floating-point accumulation order matches the Python\n")
    f.write("//! reference bit-for-bit. Regenerate with `codegen/generate_piper_pq.py`.\n\n")
    f.write("// Do not edit by hand. See codegen/generate_piper_pq.py.\n")
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
