from pathlib import Path

import yaml

import spatial_workflow.overview as overview


def _write_config(tmp_path: Path) -> Path:
    config = {
        "project": {"id": "test"},
        "paths": {"results_root": "../results"},
        "schema": {"sample_key": "sample"},
        "cellcharter": {
            "input_h5ad": "01_anndata/input.h5ad",
            "output_dir": "02_cellcharter",
            "output_name": "sample_cellcharter",
        },
        "nncomp": {
            "input_h5ad": "02_cellcharter/sample_cellcharter.h5ad",
            "output_dir": "03_nncomp",
            "overlap": {
                "enabled": True,
                "reciprocal_cell_types": {
                    "enabled": True,
                    "analyses": ["within_compartment", "whole_sample"],
                },
            },
        },
    }
    config_path = tmp_path / "configs" / "local.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _touch_all(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_overview_runs_fresh_stages_when_outputs_are_absent(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cellcharter, nncomp, overlap, plan, reciprocal = overview._overview_paths(
        config_path, config
    )

    monkeypatch.setattr(overview, "run_cellcharter", lambda _: _touch_all(cellcharter))
    monkeypatch.setattr(
        overview,
        "run_nncomp",
        lambda _: (
            _touch_all(nncomp),
            _touch_all(overlap),
            _touch_all(plan),
            _touch_all(reciprocal),
        ),
    )

    result = overview.run_spatial_overview(config_path)

    assert result["actions"] == ["ran_cellcharter", "ran_nncomp"]


def test_overview_skips_local_stages_and_backfills_overlap_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cellcharter, nncomp, overlap, plan, reciprocal = overview._overview_paths(
        config_path, config
    )
    _touch_all(cellcharter)
    _touch_all(nncomp)
    called = []

    def backfill(_):
        called.append("overlap")
        _touch_all(overlap)

    def backfill_reciprocal(_):
        called.append("reciprocal")
        _touch_all(reciprocal)

    def backfill_plan(_):
        called.append("plan")
        _touch_all(plan)

    monkeypatch.setattr(overview, "run_directional_overlap", backfill)
    monkeypatch.setattr(
        overview,
        "run_reciprocal_celltype_permutation",
        backfill_reciprocal,
    )
    monkeypatch.setattr(overview, "run_permutation_plan", backfill_plan)
    monkeypatch.setattr(
        overview,
        "run_cellcharter",
        lambda _: (_ for _ in ()).throw(AssertionError("must skip CellCharter")),
    )

    result = overview.run_spatial_overview(config_path)

    assert called == ["overlap", "plan", "reciprocal"]
    assert result["actions"] == [
        "used_existing_cellcharter",
        "used_existing_nncomp",
        "ran_directional_overlap",
        "ran_permutation_plan",
        "ran_reciprocal_celltype_permutation",
    ]


def test_overview_backfills_missing_whole_sample_without_rejecting_existing_within(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cellcharter, nncomp, overlap, plan, reciprocal = overview._overview_paths(
        config_path, config
    )
    _touch_all(cellcharter)
    _touch_all(nncomp)
    _touch_all(overlap)
    _touch_all(plan)
    reciprocal["within_compartment_celltype_permutation"].touch()
    called = []

    def backfill(_):
        called.append("reciprocal")
        _touch_all(reciprocal)

    monkeypatch.setattr(
        overview,
        "run_reciprocal_celltype_permutation",
        backfill,
    )

    result = overview.run_spatial_overview(config_path)

    assert called == ["reciprocal"]
    assert "ran_reciprocal_celltype_permutation" in result["actions"]
