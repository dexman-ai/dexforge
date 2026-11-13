#!/usr/bin/env python3
"""Shared binary format and Rust loader generator for ssik IK reference data.

Each arm's reference dataset lives in ``test_data.bin`` alongside a thin
``#[cfg(test)]`` Rust loader ``test_data.rs`` in the arm's driver directory
(``repos/dexforge-<oem>/rust/src/drivers/<arm>/``). Both Rust unit tests and
the shared Python test suite (``ssik_shared_tests/``) read the same binary
blob, ensuring a single source of truth for FK and IK cross-validation.

Binary layout of ``test_data.bin`` (all integers and floats little-endian):
  - ``magic``: 4 bytes ``b"DXIK"``
  - ``format_version``: ``u32`` (``1``)
  - ``ver_len``: ``u32`` byte length of UTF-8 ``ssik_version`` string
  - ``ssik_version``: ``ver_len`` UTF-8 bytes (e.g. ``b"1.2.3"``)
  - ``dof``: ``u32``
  - ``n_cases``: ``u32``
  - ``allow_refinement``: ``u32`` (``0`` or ``1``)
  - ``refinement_max_iters``: ``u32``
  - ``cases[n_cases]``:
      - ``q_fk[dof]``: ``f64`` configuration used to generate ``tf = fk(q_fk)``
      - ``tf[16]``: ``f64`` row-major 4x4 SE(3) target pose
      - ``singular``: ``u32`` (``0`` for strict/non-singular, ``1`` for singular)
      - ``n_sols``: ``u32`` number of reference IK solutions
      - ``solutions[n_sols * dof]``: ``f64`` flattened reference joint solutions
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_CODEGEN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODEGEN_DIR.parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from repo_paths import list_known_arms, resolve_arm_driver_dir  # noqa: E402

MAGIC = b"DXIK"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class IKCase:
    """Single FK/IK reference test case."""

    q_fk: List[float]
    tf: List[float]
    solutions: List[List[float]]
    singular: bool = False


@dataclass(frozen=True)
class IKDataset:
    """Unpacked IK reference dataset for an arm."""

    ssik_version: str
    dof: int
    allow_refinement: bool
    refinement_max_iters: int
    cases: List[IKCase]


def get_locked_ssik_version(repo_root: Optional[Path] = None) -> str:
    """Read the pinned ssik version from ``SSIK_VERSION.txt``."""
    root = repo_root or _REPO_ROOT
    version_file = root / "SSIK_VERSION.txt"
    if not version_file.is_file():
        raise FileNotFoundError(f"SSIK_VERSION.txt not found at {version_file}")
    content = version_file.read_text(encoding="utf-8").strip()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ssik=="):
            return line[len("ssik==") :].strip()
        if line.startswith("ssik"):
            return line[len("ssik") :].strip().lstrip("=").strip()
        return line
    raise ValueError(f"SSIK_VERSION.txt at {version_file} is empty")


def get_installed_ssik_version(repo_root: Optional[Path] = None) -> str:
    """Return the installed ``ssik.__version__`` and verify it matches ``SSIK_VERSION.txt``."""
    locked = get_locked_ssik_version(repo_root)
    try:
        import ssik
    except ImportError as exc:
        raise RuntimeError("Cannot verify ssik version: ssik is not installed") from exc
    installed = getattr(ssik, "__version__", None)
    if not installed or installed == "unknown":
        raise RuntimeError("Cannot verify ssik version: ssik.__version__ is unavailable")
    if installed != locked:
        raise RuntimeError(
            f"Installed ssik version ({installed}) does not match "
            f"SSIK_VERSION.txt ({locked}). Install with: pip install ssik=={locked}"
        )
    return installed


def pack_ik_test_data(
    ssik_version: str,
    dof: int,
    cases: Sequence[IKCase],
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> bytes:
    """Serialize an IK reference dataset into the ``DXIK`` binary format."""
    ver_bytes = ssik_version.encode("utf-8")
    parts: List[bytes] = [
        MAGIC,
        struct.pack("<II", FORMAT_VERSION, len(ver_bytes)),
        ver_bytes,
        struct.pack(
            "<IIII",
            int(dof),
            len(cases),
            1 if allow_refinement else 0,
            int(refinement_max_iters),
        ),
    ]
    for idx, c in enumerate(cases):
        q_arr = np.asarray(c.q_fk, dtype="<f8").reshape(-1)
        tf_arr = np.asarray(c.tf, dtype="<f8").reshape(-1)
        if q_arr.size != dof:
            raise ValueError(f"case {idx}: q_fk len {q_arr.size} != dof {dof}")
        if tf_arr.size != 16:
            raise ValueError(f"case {idx}: tf len {tf_arr.size} != 16")
        n_sols = len(c.solutions)
        if n_sols > 0:
            sols_arr = np.asarray(c.solutions, dtype="<f8").reshape(-1)
            if sols_arr.size != n_sols * dof:
                raise ValueError(
                    f"case {idx}: solutions size {sols_arr.size} != {n_sols} * {dof}"
                )
            sols_bytes = sols_arr.tobytes()
        else:
            sols_bytes = b""
        parts.append(q_arr.tobytes())
        parts.append(tf_arr.tobytes())
        parts.append(struct.pack("<II", 1 if c.singular else 0, n_sols))
        parts.append(sols_bytes)
    return b"".join(parts)


def unpack_ik_test_data(blob: bytes) -> IKDataset:
    """Deserialize a ``DXIK`` binary blob into an :class:`IKDataset`."""
    if len(blob) < 12 or blob[:4] != MAGIC:
        raise ValueError("Invalid test_data.bin magic header (expected b'DXIK')")
    fmt_ver, ver_len = struct.unpack_from("<II", blob, 4)
    if fmt_ver != FORMAT_VERSION:
        raise ValueError(f"Unsupported test_data.bin format version: {fmt_ver}")
    offset = 12
    ssik_version = blob[offset : offset + ver_len].decode("utf-8")
    offset += ver_len
    dof, n_cases, allow_ref, max_iters = struct.unpack_from("<IIII", blob, offset)
    offset += 16

    cases: List[IKCase] = []
    q_bytes = dof * 8
    tf_bytes = 16 * 8
    for _ in range(n_cases):
        q_fk = list(np.frombuffer(blob, dtype="<f8", count=dof, offset=offset).astype(float))
        offset += q_bytes
        tf = list(np.frombuffer(blob, dtype="<f8", count=16, offset=offset).astype(float))
        offset += tf_bytes
        singular_u32, n_sols = struct.unpack_from("<II", blob, offset)
        offset += 8
        if n_sols > 0:
            raw_sols = np.frombuffer(
                blob, dtype="<f8", count=n_sols * dof, offset=offset
            ).reshape(n_sols, dof)
            solutions = [list(map(float, row)) for row in raw_sols]
            offset += n_sols * dof * 8
        else:
            solutions = []
        cases.append(
            IKCase(
                q_fk=q_fk,
                tf=tf,
                solutions=solutions,
                singular=bool(singular_u32),
            )
        )

    if offset != len(blob):
        raise ValueError(
            f"Trailing bytes in test_data.bin: consumed {offset} of {len(blob)}"
        )

    return IKDataset(
        ssik_version=ssik_version,
        dof=dof,
        allow_refinement=bool(allow_ref),
        refinement_max_iters=int(max_iters),
        cases=cases,
    )


def load_arm_test_data(arm: str, repo_root: Optional[Path] = None) -> IKDataset:
    """Locate and load ``test_data.bin`` for *arm*."""
    root = repo_root or _REPO_ROOT
    base = arm[:-3] if arm.endswith("_ik") else arm
    driver_dir = resolve_arm_driver_dir(root, base)
    if driver_dir is None:
        raise FileNotFoundError(f"Could not locate driver directory for arm {base!r}")
    bin_path = driver_dir / "test_data.bin"
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing reference data file: {bin_path}")
    return unpack_ik_test_data(bin_path.read_bytes())


def emit_ik_test_data_rs(
    arm: str,
    dof: int,
    n_cases: int,
    has_singular_tuple: bool,
    bin_filename: str = "test_data.bin",
) -> str:
    """Emit the ``#[cfg(test)]`` Rust loader module ``test_data.rs``."""
    if has_singular_tuple:
        tuple_type = f"([f64; 16], Vec<[f64; {dof}]>, bool)"
        tuple_expr = "(c.tf, c.solutions, c.singular)"
    else:
        tuple_type = f"([f64; 16], Vec<[f64; {dof}]>)"
        tuple_expr = "(c.tf, c.solutions)"

    return f"""\
//! Auto-generated ssik IK reference dataset loader for `{arm}`.
//!
//! Reads the shared binary dataset `{bin_filename}` (also consumed by the
//! Python cross-validation suite in `ssik_shared_tests/`).
//! Do not edit by hand; regenerate with `codegen/generate_test_data.py`.

#![allow(dead_code, clippy::all)]

extern crate std;
use std::vec::Vec;

/// Degrees of freedom for `{arm}`.
pub const DOF: usize = {dof};

/// Number of reference cases stored in `{bin_filename}`.
pub const N_CASES: usize = {n_cases};

/// Raw binary reference dataset (`DXIK` format v1).
pub static RAW: &[u8] = include_bytes!("{bin_filename}");

/// A decoded reference case from `{bin_filename}`.
#[derive(Clone, Debug)]
pub struct Case {{
    pub q_fk: [f64; DOF],
    pub tf: [f64; 16],
    pub solutions: Vec<[f64; DOF]>,
    pub singular: bool,
}}

#[inline]
fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {{
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[*offset..*offset + 4]);
    *offset += 4;
    u32::from_le_bytes(buf)
}}

#[inline]
fn read_f64(bytes: &[u8], offset: &mut usize) -> f64 {{
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[*offset..*offset + 8]);
    *offset += 8;
    f64::from_le_bytes(buf)
}}

/// Return the `ssik` version string recorded in `{bin_filename}`.
pub fn ssik_version() -> &'static str {{
    assert!(RAW.len() >= 12 && &RAW[0..4] == b"DXIK", "invalid test_data.bin header");
    let mut offset = 4usize;
    let fmt_ver = read_u32(RAW, &mut offset);
    assert_eq!(fmt_ver, 1, "unsupported test_data.bin format version");
    let ver_len = read_u32(RAW, &mut offset) as usize;
    core::str::from_utf8(&RAW[offset..offset + ver_len])
        .expect("invalid UTF-8 ssik_version in test_data.bin")
}}

/// Decode all [`N_CASES`] reference cases from [`RAW`].
pub fn load_cases() -> Vec<Case> {{
    assert!(RAW.len() >= 12 && &RAW[0..4] == b"DXIK", "invalid test_data.bin header");
    let mut offset = 4usize;
    let fmt_ver = read_u32(RAW, &mut offset);
    assert_eq!(fmt_ver, 1, "unsupported test_data.bin format version");
    let ver_len = read_u32(RAW, &mut offset) as usize;
    offset += ver_len;

    let dof = read_u32(RAW, &mut offset) as usize;
    let n_cases = read_u32(RAW, &mut offset) as usize;
    let _allow_refinement = read_u32(RAW, &mut offset);
    let _refinement_max_iters = read_u32(RAW, &mut offset);
    assert_eq!(dof, DOF, "test_data.bin DOF mismatch");
    assert_eq!(n_cases, N_CASES, "test_data.bin N_CASES mismatch");

    let mut out = Vec::with_capacity(n_cases);
    for _ in 0..n_cases {{
        let mut q_fk = [0.0f64; DOF];
        for i in 0..DOF {{
            q_fk[i] = read_f64(RAW, &mut offset);
        }}
        let mut tf = [0.0f64; 16];
        for i in 0..16 {{
            tf[i] = read_f64(RAW, &mut offset);
        }}
        let singular = read_u32(RAW, &mut offset) != 0;
        let n_sols = read_u32(RAW, &mut offset) as usize;
        let mut solutions = Vec::with_capacity(n_sols);
        for _ in 0..n_sols {{
            let mut q = [0.0f64; DOF];
            for i in 0..DOF {{
                q[i] = read_f64(RAW, &mut offset);
            }}
            solutions.push(q);
        }}
        out.push(Case {{
            q_fk,
            tf,
            solutions,
            singular,
        }});
    }}
    assert_eq!(offset, RAW.len(), "test_data.bin length mismatch");
    out
}}

/// Convenience accessor returning tuple cases for `ik_solver` tests.
pub fn cases() -> Vec<{tuple_type}> {{
    load_cases().into_iter().map(|c| {tuple_expr}).collect()
}}
"""


def write_ik_test_data(
    driver_dir: Path,
    arm: str,
    dof: int,
    cases: Sequence[IKCase],
    ssik_version: Optional[str] = None,
    has_singular_tuple: bool = False,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> Tuple[Path, Path]:
    """Write ``test_data.bin`` and ``test_data.rs`` into *driver_dir*."""
    version = ssik_version or get_installed_ssik_version()
    driver_dir = Path(driver_dir)
    driver_dir.mkdir(parents=True, exist_ok=True)
    bin_path = driver_dir / "test_data.bin"
    rs_path = driver_dir / "test_data.rs"

    blob = pack_ik_test_data(
        ssik_version=version,
        dof=dof,
        cases=cases,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )
    bin_path.write_bytes(blob)
    rs_text = emit_ik_test_data_rs(
        arm=arm,
        dof=dof,
        n_cases=len(cases),
        has_singular_tuple=has_singular_tuple,
        bin_filename=bin_path.name,
    )
    rs_path.write_text(rs_text, encoding="utf-8")
    return bin_path, rs_path


def verify_all_arms_ssik_version(repo_root: Optional[Path] = None) -> List[str]:
    """Verify every known arm's ``test_data.bin`` matches ``SSIK_VERSION.txt``.

    :returns: List of verified arm names.
    :raises RuntimeError: If any arm is missing ``test_data.bin`` or has a
        mismatched ``ssik_version``.
    """
    root = repo_root or _REPO_ROOT
    expected = get_locked_ssik_version(root)
    arms = list_known_arms(root)
    if not arms:
        raise RuntimeError(f"No arm drivers found under {root}")
    for arm in arms:
        dataset = load_arm_test_data(arm, root)
        if dataset.ssik_version != expected:
            raise RuntimeError(
                f"{arm}: test_data.bin ssik_version={dataset.ssik_version!r} "
                f"does not match SSIK_VERSION.txt ({expected!r})"
            )
        if not dataset.cases:
            raise RuntimeError(f"{arm}: test_data.bin contains 0 cases")
    return arms


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that every arm's test_data.bin matches SSIK_VERSION.txt.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the dexforge repository root.",
    )
    args = parser.parse_args(argv)
    if args.check:
        arms = verify_all_arms_ssik_version(args.repo_root)
        locked = get_locked_ssik_version(args.repo_root)
        print(
            f"[ik_test_data] Verified ssik_version=={locked} across "
            f"{len(arms)} arm(s): {', '.join(arms)}"
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
