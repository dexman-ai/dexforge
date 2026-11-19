"""Tests for ``codegen/scaffold.py`` (template rendering into a temp vendor tree)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sympy")
ssik = pytest.importorskip("ssik")

_CODEGEN = Path(__file__).resolve().parent.parent
_REPO = _CODEGEN.parent
sys.path.insert(0, str(_CODEGEN))
import scaffold as S  # noqa: E402

UR5_URDF = _REPO / "repos" / "dexforge-ur" / "rust" / "src" / "drivers" / "ur5" / "ur5.urdf"
UR5_ARTIFACT = Path(ssik.__file__).resolve().parent / "prebuilt" / "ur5_ik.py"


def test_render_requires_every_placeholder():
    assert S.render("a {{x}} b {{{y}}}", {"x": 1, "y": "Z"}) == "a 1 b {Z}"
    with pytest.raises(KeyError):
        S.render("{{missing}}", {})


@pytest.mark.parametrize(
    "value, literal",
    [(1.0, "1.0"), (-0.425, "-0.425"), (6.123233995736766e-17, "6.123233995736766e-17"), (3, "3.0")],
)
def test_rust_f64(value, literal):
    assert S.rust_f64(value) == literal
    assert float(literal) == float(value)


def test_struct_name():
    assert S.struct_name("ur5") == "Ur5"
    assert S.struct_name("franka_panda") == "FrankaPanda"


@pytest.fixture(scope="module")
def ur5_scaffold(tmp_path_factory):
    if not UR5_URDF.is_file() or not UR5_ARTIFACT.is_file():
        pytest.skip("ur5 URDF / ssik prebuilt not available")
    repos = tmp_path_factory.mktemp("repos")
    res = S.scaffold("ur", "ur5", UR5_URDF, "base_link", "ee_link", artifact=UR5_ARTIFACT, repos_dir=repos)
    return res


def test_scaffold_writes_the_crate_layout(ur5_scaffold):
    lay = ur5_scaffold.layout
    expected = [
        lay.vendor_dir / "Cargo.toml",
        lay.vendor_dir / "README.md",
        lay.crate_dir / "Cargo.toml",
        lay.crate_dir / "manifest.yaml",
        lay.crate_dir / "src" / "lib.rs",
        lay.drivers_dir / "mod.rs",
        lay.driver_dir / "mod.rs",
        lay.driver_dir / "arm.rs",
        lay.driver_dir / "driver.rs",
        lay.driver_dir / "ur5.urdf",
        lay.codegen_dir / "ur5_ik.py",
        lay.codegen_dir / "generate_test_data.py",
    ]
    for path in expected:
        assert path.is_file(), path
    assert not (lay.codegen_dir / "__pycache__").exists()
    for path in expected:
        if path.suffix in (".rs", ".toml", ".yaml", ".md") or path.name == "generate_test_data.py":
            assert not re.search(r"\{\{\w+\}\}", path.read_text(encoding="utf-8")), path


def test_scaffold_arm_rs_implements_arm_model(ur5_scaffold):
    arm = (ur5_scaffold.layout.driver_dir / "arm.rs").read_text(encoding="utf-8")
    assert "impl ArmModel<DOF, MAX_SOLUTIONS> for Ur5" in arm
    assert 'const NAME: &\'static str = "ur5";' in arm
    assert "pub const DOF: usize = 6;" in arm
    assert "pub const MAX_SOLUTIONS: usize = 16;" in arm
    assert "const HAS_DYNAMICS: bool = true;" in arm
    for hook in ("flange_fk", "flange_jacobian", "flange_solve", "flange_gravity",
                 "flange_mass_matrix", "flange_coriolis", "flange_inverse_dynamics"):
        assert f"fn {hook}(" in arm
    assert "use super::{generated_dynamics, generated_kinematics, ik_solver};" in arm
    assert len(re.findall(r"^        Some\(\(", arm, re.M)) == 6


def test_scaffold_registers_the_arm(ur5_scaffold):
    lay = ur5_scaffold.layout
    assert "pub mod ur5;" in (lay.drivers_dir / "mod.rs").read_text(encoding="utf-8")
    assert "pub use drivers::ur5::{Ur5, Ur5Driver};" in (lay.crate_dir / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    mod_rs = (lay.driver_dir / "mod.rs").read_text(encoding="utf-8")
    assert "mod dynamics_parity;" in mod_rs
    driver = (lay.driver_dir / "driver.rs").read_text(encoding="utf-8")
    assert 'include_str!("../../../manifest.yaml")' in driver


def test_scaffold_manifest_draft(ur5_scaffold):
    text = ur5_scaffold.layout.manifest.read_text(encoding="utf-8")
    assert re.search(r"^driver: ur5\b", text, re.M)
    assert 'version: "0.2"' in text
    assert "velocity: [" + ", ".join(["3.141592653589793"] * 6) + "]" in text  # UR5e: pi rad/s
    assert "effort: [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]" in text
    assert "has_static_inertials: true" in text
    assert "TODO" in text  # agent-supplied fields stay explicit


def test_scaffold_keeps_existing_files_unless_forced(ur5_scaffold, tmp_path):
    lay = ur5_scaffold.layout
    driver = lay.driver_dir / "driver.rs"
    driver.write_text("// agent edits\n", encoding="utf-8")
    arm = lay.driver_dir / "arm.rs"
    arm.write_text("// stale\n", encoding="utf-8")
    res = S.scaffold(
        "ur", "ur5", lay.urdf, "base_link", "ee_link", repos_dir=lay.vendor_dir.parent
    )
    assert driver.read_text(encoding="utf-8") == "// agent edits\n"
    assert driver in res.skipped
    assert "impl ArmModel" in arm.read_text(encoding="utf-8")  # always regenerated
    lib = (lay.crate_dir / "src" / "lib.rs").read_text(encoding="utf-8")
    assert lib.count("pub use drivers::ur5::") == 1
    assert (lay.drivers_dir / "mod.rs").read_text(encoding="utf-8").count("pub mod ur5;") == 1


def test_scaffold_rejects_mismatched_links(tmp_path):
    if not UR5_URDF.is_file() or not UR5_ARTIFACT.is_file():
        pytest.skip("ur5 URDF / ssik prebuilt not available")
    with pytest.raises(ValueError, match="does not match"):
        S.scaffold("ur", "ur5", UR5_URDF, "base_link", "tool0", artifact=UR5_ARTIFACT, repos_dir=tmp_path)
