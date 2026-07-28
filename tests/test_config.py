from pathlib import Path

import pytest
import yaml

from spatial_workflow.config import load_config, resolve_path


def _write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_load_config_reads_required_sections(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    expected = {
        "project": {"name": "demo"},
        "paths": {"data_root": "data", "results_root": "results"},
        "schema": {"sample_key": "sample_id"},
    }
    _write_config(path, expected)

    assert load_config(path) == expected


def test_load_config_rejects_missing_shared_section(tmp_path: Path) -> None:
    path = tmp_path / "workflow.yaml"
    _write_config(path, {"project": {}, "paths": {}})

    with pytest.raises(ValueError, match="schema"):
        load_config(path)


def test_resolve_path_uses_config_directory_and_optional_root(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "workflow.yaml"
    config_path.parent.mkdir()

    assert resolve_path(config_path, "samples.csv") == (
        config_path.parent / "samples.csv"
    ).resolve()
    assert resolve_path(config_path, "sample", root="../data") == (
        tmp_path / "data" / "sample"
    ).resolve()


def test_resolve_path_keeps_absolute_values(tmp_path: Path) -> None:
    absolute = tmp_path / "data" / "sample"

    assert resolve_path(tmp_path / "workflow.yaml", absolute) == absolute.resolve()
