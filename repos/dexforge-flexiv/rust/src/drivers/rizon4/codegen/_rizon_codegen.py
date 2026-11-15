"""Shared codegen helpers for the Flexiv Rizon (Rizon4 / Rizon10) IK ports.

The Rizon arms are ``jointlock.seven_r`` (tier 1) 7R arms: locking joint
``_LOCK_IDX`` (= 2, **no chain reversal**) collapses each arm to a series of
6R sub-chains, one per lock sample. Each 6R sub-chain is solved with the
deterministic **Raghavan-Roth** kernel that already backs the Piper driver
(``ssik.solvers.ikgeo._raghavan_roth`` / ``ssik_core::rr_*`` on the Rust side).

This module composes the per-sample RR derivation with the *same* build-time
codegen ssik uses for its tier-2 artifacts
(``ssik.codegen._compose.general_6r``), then re-emits the per-sample
``_build_pq_matrices`` body and DH/RR-meta constants as Rust. The reused
Piper AST walker guarantees the floating-point accumulation order matches the
Python reference bit-for-bit.

Composing one 6R sub-chain costs ~45 s (sympy derivation + ``cse``), so the
16 compositions per arm are cached on disk under ``$TMPDIR`` keyed by the
sub-chain DH, making repeated regenerations cheap.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import re
import tempfile

import numpy as np

from ssik.codegen._compose import general_6r as _G6
from ssik.solvers.jointlock.seven_r import _lock_joint

# Arm name -> ssik prebuilt module name.
ARMS = {
    "rizon4": "ssik.prebuilt.rizon4_ik",
    "rizon10": "ssik.prebuilt.rizon10_ik",
}

DRIVERS_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "rizon_compose_cache")


def load_ref(arm: str):
    return importlib.import_module(ARMS[arm])


def _sub_kb(ref, qlock: float):
    return _lock_joint(ref._build_kb(), ref._LOCK_IDX, float(qlock))


def composed_source(ref, qlock: float) -> str:
    """Return the composed RR sub-chain source for ``qlock``.

    The composed source is cached on disk keyed by a hash of the sub-chain DH
    geometry so repeated runs avoid the ~45 s sympy derivation.
    """
    sub = _sub_kb(ref, qlock)
    # Hash the sub-chain's POE geometry for a stable cache key.
    h = hashlib.sha256()
    for j in sub.joints:
        h.update(np.asarray(j.axis, dtype=np.float64).tobytes())
        h.update(np.asarray(j.T_left, dtype=np.float64).tobytes())
        h.update(np.asarray(j.T_right, dtype=np.float64).tobytes())
    key = h.hexdigest()[:16]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = os.path.join(_CACHE_DIR, f"{key}.py")
    if os.path.exists(cache):
        with open(cache) as fh:
            return fh.read()
    src = _G6.compose(sub)
    with open(cache, "w") as fh:
        fh.write(src)
    return src


def composed_namespace(ref, qlock: float) -> dict:
    """Compose the RR sub-chain for ``qlock`` and exec it into a namespace."""
    src = composed_source(ref, qlock)
    key = hashlib.sha256(src.encode()).hexdigest()[:16]
    ns: dict = {"np": np}
    exec(compile(src, f"<composed {key}>", "exec"), ns)  # noqa: S102
    return ns


def lock_samples(ref) -> np.ndarray:
    return np.asarray(ref._LOCK_SAMPLES, dtype=np.float64)


# ---------------------------------------------------------------------------
# Rust literal helpers
# ---------------------------------------------------------------------------


def rfloat(x) -> str:
    if isinstance(x, bool):
        raise ValueError
    r = repr(float(x))
    if ("." not in r) and ("e" not in r) and ("E" not in r) and ("inf" not in r) and ("nan" not in r):
        r += ".0"
    return r


def rvec(a) -> str:
    return "[" + ", ".join(rfloat(x) for x in a) + "]"


def rmat4(m) -> str:
    m = np.asarray(m, dtype=np.float64)
    rows = ["[" + ", ".join(rfloat(x) for x in m[i]) + "]" for i in range(4)]
    return "[" + ", ".join(rows) + "]"


# ---------------------------------------------------------------------------
# PQ builder AST -> Rust (identical accumulation order to generate_code.py)
# ---------------------------------------------------------------------------


def _name_map(nid: str) -> str:
    m = re.fullmatch(r"T_(\d+)", nid)
    if m:
        return f"t{m.group(1)}"
    m = re.fullmatch(r"_pq_x(\d+)", nid)
    if m:
        return f"pq_x{m.group(1)}"
    raise ValueError("unknown name " + nid)


def _emit(node) -> str:
    if isinstance(node, ast.BinOp):
        op = node.op
        l = _emit(node.left)
        r = _emit(node.right)
        if isinstance(op, ast.Add):
            return f"({l} + {r})"
        if isinstance(op, ast.Sub):
            return f"({l} - {r})"
        if isinstance(op, ast.Mult):
            return f"({l} * {r})"
        if isinstance(op, ast.Div):
            return f"({l} / {r})"
        if isinstance(op, ast.Pow):
            assert isinstance(node.right, ast.Constant), ast.dump(node)
            e = node.right.value
            base = _emit(node.left)
            if e == 2:
                return f"({base} * {base})"
            if e == 3:
                return f"({base} * {base} * {base})"
            raise ValueError("pow " + str(e))
        raise ValueError("op " + ast.dump(op))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"(-{_emit(node.operand)})"
        if isinstance(node.op, ast.UAdd):
            return _emit(node.operand)
        raise ValueError
    if isinstance(node, ast.Constant):
        return rfloat(node.value)
    if isinstance(node, ast.Name):
        return _name_map(node.id)
    if isinstance(node, ast.Call):
        raise ValueError("unexpected call " + ast.dump(node))
    raise ValueError("node " + ast.dump(node))


def render_pq_builder(src: str, fn_name: str) -> str:
    """Render the composed ``_build_pq_matrices`` body as a Rust function.

    ``src`` is the full composed-module source text (the function cannot be
    recovered with ``inspect`` because it is produced via ``exec``).
    """
    mod = ast.parse(src)
    fn = next(
        node
        for node in mod.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_pq_matrices"
    )

    lets: list[tuple[str, str]] = []
    arrays: dict[str, list[list[str]]] = {}
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name):
                nid = tgt.id
                if nid.startswith("_pq_x"):
                    lets.append((_name_map(nid), _emit(stmt.value)))
                elif nid in ("p_sin", "p_cos", "p_one", "q"):
                    call = stmt.value
                    lst = call.args[0]
                    rows = []
                    for row in lst.elts:
                        rows.append([_emit(e) for e in row.elts])
                    arrays[nid] = rows
        elif isinstance(stmt, ast.Return):
            pass

    out: list[str] = []
    out.append("#[allow(clippy::all)]")
    out.append("#[allow(unused_parens)]")
    out.append("#[allow(clippy::too_many_lines)]")
    out.append(f"pub(crate) fn {fn_name}(")
    out.append("    t0: f64, t1: f64, t2: f64, t3: f64, t4: f64, t5: f64,")
    out.append("    t6: f64, t7: f64, t8: f64, t9: f64, t10: f64, t11: f64,")
    out.append(") -> ([[f64; 9]; 14], [[f64; 9]; 14], [[f64; 9]; 14], [[f64; 8]; 14]) {")
    for name, expr in lets:
        out.append(f"    let {name} = {expr};")

    def emit_array(name: str, rows: list[list[str]], ncol: int) -> None:
        out.append(f"    let {name}: [[f64; {ncol}]; 14] = [")
        for row in rows:
            assert len(row) == ncol, (name, len(row), ncol)
            out.append("        [" + ", ".join(row) + "],")
        out.append("    ];")

    emit_array("p_sin", arrays["p_sin"], 9)
    emit_array("p_cos", arrays["p_cos"], 9)
    emit_array("p_one", arrays["p_one"], 9)
    emit_array("q", arrays["q"], 8)
    out.append("    (p_sin, p_cos, p_one, q)")
    out.append("}")
    return "\n".join(out)
