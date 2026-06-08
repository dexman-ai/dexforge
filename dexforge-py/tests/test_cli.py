from __future__ import annotations

from pathlib import Path

from dexforge import cli


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    codegen = repo / "codegen"
    drivers = repo / "src" / "drivers"
    codegen.mkdir(parents=True)
    drivers.mkdir(parents=True)

    for script_name in cli._IK_CODEGEN_SCRIPTS + ("generate_dynamics.py",):
        (codegen / script_name).write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    for arm in cli._DYNAMICS_ARMS:
        arm_dir = drivers / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / f"{arm}_ik.py").write_text("# artifact\n", encoding="utf-8")

    return repo


def test_build_runs_full_codegen_pipeline(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    calls: list[list[str]] = []

    def _fake_run(command, cwd):
        calls.append(command)
        return 0

    monkeypatch.setattr(cli, "_run_command", _fake_run)
    rc = cli.main(["build", "--repo-root", str(repo), "--python", "python3"])

    assert rc == 0
    assert len(calls) == len(cli._IK_CODEGEN_SCRIPTS) + len(cli._DYNAMICS_ARMS)
    assert calls[0] == ["python3", str(repo / "codegen" / cli._IK_CODEGEN_SCRIPTS[0])]
    dynamics_calls = calls[-len(cli._DYNAMICS_ARMS) :]
    assert [Path(cmd[3]).parent.name for cmd in dynamics_calls] == list(cli._DYNAMICS_ARMS)
    assert calls[-1] == [
        "python3",
        str(repo / "codegen" / "generate_dynamics.py"),
        "--artifact",
        str(repo / "src" / "drivers" / "rizon10" / "rizon10_ik.py"),
        "--output",
        str(repo / "src" / "drivers" / "rizon10" / "generated_kinematics.rs"),
    ]


def test_build_stops_after_first_failure(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path)
    calls: list[list[str]] = []

    def _fake_run(command, cwd):
        calls.append(command)
        return 9 if len(calls) == 2 else 0

    monkeypatch.setattr(cli, "_run_command", _fake_run)
    rc = cli.main(["build", "--repo-root", str(repo), "--python", "python3"])

    assert rc == 9
    assert len(calls) == 2
