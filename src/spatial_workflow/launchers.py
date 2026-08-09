"""Render small, reproducible shell launchers for workflow stages."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .config import load_config


def _require_stage(config: dict[str, Any], stage: str) -> None:
    if not isinstance(config.get(stage), dict):
        raise ValueError(f"Configuration section '{stage}' must be a mapping")


def _python_bin(config: dict[str, Any]) -> str:
    python_bin = config.get("runtime", {}).get("python_bin", "python3")
    if not isinstance(python_bin, str) or not python_bin.strip():
        raise ValueError("runtime.python_bin must be a non-empty string")
    return python_bin


def _write_script(output_path: str | Path, content: str) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _job_script(commands: list[str]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            *commands,
            "",
        ]
    )


def write_conversion_script(
    config_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write an executable AnnData conversion job."""

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    _require_stage(config, "conversion")
    command = (
        f"{shlex.quote(_python_bin(config))} "
        "-m spatial_workflow.ingest "
        f"--config {shlex.quote(str(config_file))}"
    )
    return _write_script(output_path, _job_script([command]))


def write_spatial_overview_script(
    config_path: str | Path,
    output_path: str | Path,
    *,
    overlap_only: bool = False,
) -> Path:
    """Write a full overview job or an overlap-only backfill job."""

    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    _require_stage(config, "nncomp")
    python_bin = shlex.quote(_python_bin(config))
    quoted_config = shlex.quote(str(config_file))
    if overlap_only:
        commands = [
            f"{python_bin} -m spatial_workflow.nncomp "
            f"--config {quoted_config} --overlap-only"
        ]
    else:
        _require_stage(config, "cellcharter")
        commands = [
            f"{python_bin} -m spatial_workflow.overview --config {quoted_config}",
        ]
    return _write_script(output_path, _job_script(commands))


def write_tmux_launcher(
    job_script: str | Path,
    output_path: str | Path,
    session_name: str,
    log_path: str | Path,
) -> Path:
    """Write an executable launcher for a detached tmux job."""

    if not session_name:
        raise ValueError("session_name must be non-empty")

    job = Path(job_script).expanduser().resolve()
    log = Path(log_path).expanduser().resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    tmux_command = (
        f"exec {shlex.quote(str(job))} "
        f"> {shlex.quote(str(log))} 2>&1"
    )
    command = (
        "tmux new-session -d "
        f"-s {shlex.quote(session_name)} "
        f"bash -lc {shlex.quote(tmux_command)}"
    )
    return _write_script(output_path, _job_script([command]))
