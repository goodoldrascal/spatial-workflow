import os
import shlex
from pathlib import Path

import pytest
import yaml

from spatial_workflow.launchers import (
    write_conversion_script,
    write_spatial_overview_script,
    write_tmux_launcher,
)


def _write_config(
    path: Path,
    *,
    runtime: dict | None = None,
    include_conversion: bool = True,
    include_overview: bool = True,
) -> None:
    config = {
        "project": {"name": "demo"},
        "paths": {"data_root": "data", "results_root": "results"},
        "schema": {"sample_key": "sample_id"},
    }
    if runtime is not None:
        config["runtime"] = runtime
    if include_conversion:
        config["conversion"] = {"input_format": "xenium"}
    if include_overview:
        config["cellcharter"] = {"input_h5ad": "sample.h5ad"}
        config["nncomp"] = {"input_h5ad": "cellcharter.h5ad"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _assert_executable(path: Path) -> None:
    assert path.stat().st_mode & os.X_OK


def test_write_conversion_script_uses_default_python_and_quotes_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config dir" / "workflow.yaml"
    _write_config(config_path)

    output = write_conversion_script(config_path, tmp_path / "jobs" / "convert.sh")
    text = output.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert (
        "python3 -m spatial_workflow.ingest "
        f"--config {shlex.quote(str(config_path.resolve()))}"
    ) in text
    _assert_executable(output)


def test_write_spatial_overview_script_uses_resume_aware_entrypoint(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_config(config_path, runtime={"python_bin": "/envs/spatial python"})

    output = write_spatial_overview_script(
        config_path,
        tmp_path / "spatial-overview.sh",
    )
    text = output.read_text(encoding="utf-8")
    python_bin = shlex.quote("/envs/spatial python")
    quoted_config = shlex.quote(str(config_path.resolve()))
    overview = (
        f"{python_bin} -m spatial_workflow.overview --config {quoted_config}"
    )

    assert overview in text
    assert "spatial_workflow.cellcharter" not in text
    _assert_executable(output)


def test_write_spatial_overview_script_can_backfill_overlap_only(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_config(config_path, runtime={"python_bin": "/envs/spatial"})

    output = write_spatial_overview_script(
        config_path,
        tmp_path / "overlap-only.sh",
        overlap_only=True,
    )
    text = output.read_text(encoding="utf-8")

    assert "spatial_workflow.cellcharter" not in text
    assert (
        "/envs/spatial -m spatial_workflow.nncomp "
        f"--config {config_path.resolve()} --overlap-only"
    ) in text
    _assert_executable(output)


def test_conversion_script_requires_conversion_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.yaml"
    _write_config(config_path, include_conversion=False)

    with pytest.raises(ValueError, match="conversion"):
        write_conversion_script(config_path, tmp_path / "convert.sh")


def test_write_tmux_launcher_quotes_session_job_and_log(tmp_path: Path) -> None:
    job = tmp_path / "job scripts" / "overview.sh"
    job.parent.mkdir()
    job.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    output = write_tmux_launcher(
        job,
        tmp_path / "launch.sh",
        "sample overview",
        tmp_path / "log dir" / "overview.log",
    )
    text = output.read_text(encoding="utf-8")
    inner = (
        f"exec {shlex.quote(str(job.resolve()))} "
        f"> {shlex.quote(str((tmp_path / 'log dir' / 'overview.log').resolve()))} "
        "2>&1"
    )
    expected = (
        "tmux new-session -d "
        f"-s {shlex.quote('sample overview')} "
        f"bash -lc {shlex.quote(inner)}"
    )

    assert expected in text
    _assert_executable(output)
