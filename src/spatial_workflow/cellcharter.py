"""Run scVI, a within-sample spatial graph, and CellCharter AutoK."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .config import load_config, resolve_path


def _software_versions(include_scvi: bool) -> dict[str, str]:
    packages = ["anndata", "cellcharter", "squidpy"]
    if include_scvi:
        packages.append("scvi-tools")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def _train_scvi(
    data: ad.AnnData,
    use_rep: str,
    settings: dict[str, Any],
    *,
    counts_layer: str,
    sample_key: str,
    seed: int,
) -> None:
    try:
        import scvi
    except ImportError as exc:
        raise ImportError("scvi-tools is required to create the configured representation") from exc

    if counts_layer not in data.layers:
        raise KeyError(f"Missing raw-count layer {counts_layer!r} required by scVI")
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(data, layer=counts_layer, batch_key=sample_key)
    model = scvi.model.SCVI(
        data,
        n_latent=int(settings.get("n_latent", 30)),
        n_hidden=int(settings.get("n_hidden", 128)),
        n_layers=int(settings.get("n_layers", 2)),
        dropout_rate=float(settings.get("dropout_rate", 0.1)),
        gene_likelihood=str(settings.get("gene_likelihood", "nb")),
    )
    model.train(
        max_epochs=int(settings.get("max_epochs", 60)),
        batch_size=int(settings.get("batch_size", 1024)),
        accelerator=settings.get("accelerator", "auto"),
        devices=settings.get("devices", "auto"),
        early_stopping=bool(settings.get("early_stopping", True)),
    )
    data.obsm[use_rep] = model.get_latent_representation().astype(np.float32)


def _build_graph(
    data: ad.AnnData,
    settings: dict[str, Any],
    sample_key: str,
    spatial_key: str,
) -> None:
    try:
        import squidpy as sq
    except ImportError as exc:
        raise ImportError("squidpy is required for the CellCharter spatial graph") from exc

    mode = str(settings.get("mode", "delaunay")).lower()
    kwargs: dict[str, Any] = {
        "library_key": sample_key,
        "coord_type": "generic",
        "spatial_key": spatial_key,
    }
    if mode == "delaunay":
        kwargs["delaunay"] = True
    elif mode == "knn":
        n_neighs = int(settings.get("n_neighs", settings.get("n_neighbors", 6)))
        if n_neighs < 1:
            raise ValueError("cellcharter.graph.n_neighs must be positive")
        kwargs.update(delaunay=False, n_neighs=n_neighs)
    elif mode == "radius":
        radius = settings.get("radius")
        if radius is None or float(radius) <= 0:
            raise ValueError("cellcharter.graph.radius must be positive in radius mode")
        kwargs.update(delaunay=False, radius=float(radius))
    else:
        raise ValueError("cellcharter.graph.mode must be delaunay, knn, or radius")
    sq.gr.spatial_neighbors(data, **kwargs)


def _stability_table(model: Any) -> pd.DataFrame:
    values = np.asarray(model.stability)
    candidate_ks = np.asarray(model.n_clusters[1:-1], dtype=int)
    if values.ndim != 2 or values.shape[0] != len(candidate_ks):
        raise ValueError(f"Unexpected AutoK stability shape: {values.shape}")
    return pd.DataFrame(
        {
            "k": candidate_ks,
            "mean_stability": values.mean(axis=1),
            "sd_stability": values.std(axis=1),
            "n_comparisons": values.shape[1],
        }
    )


def _abundance_table(
    obs: pd.DataFrame,
    *,
    domain_key: str,
    sample_key: str,
    condition_key: str,
) -> pd.DataFrame:
    abundance = (
        obs.groupby([sample_key, condition_key, domain_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = obs.groupby(sample_key, observed=True).size().rename("sample_cells")
    abundance["sample_cells"] = abundance[sample_key].map(totals)
    abundance["fraction"] = abundance["n_cells"] / abundance["sample_cells"]
    return abundance


def run_cellcharter(config_path: str | Path) -> Path:
    """Run the configured CellCharter analysis and return the annotated H5AD."""

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    settings = config["cellcharter"]
    schema = config["schema"]
    results_root = resolve_path(config_path, config["paths"]["results_root"])
    input_path = resolve_path(config_path, settings["input_h5ad"], root=results_root)
    output_dir = resolve_path(config_path, settings["output_dir"], root=results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = str(settings["output_name"])
    stem = output_name.removesuffix(".h5ad")
    output_path = output_dir / f"{stem}.h5ad"
    summary_path = output_dir / f"{stem}.summary.json"
    abundance_path = output_dir / f"{stem}.domain_abundance.csv"
    stability_path = output_dir / f"{stem}.autok_stability.csv"
    existing = [
        path
        for path in (output_path, summary_path, abundance_path, stability_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"CellCharter outputs already exist: {', '.join(map(str, existing))}")

    data = ad.read_h5ad(input_path)
    sample_key = schema["sample_key"]
    condition_key = schema["condition_key"]
    cell_type_key = schema["cell_type_key"]
    domain_key = schema["spatial_domain_key"]
    spatial_key = schema["spatial_key"]
    counts_layer = schema["counts_layer"]
    missing = [key for key in (sample_key, condition_key, cell_type_key) if key not in data.obs]
    if missing:
        raise KeyError(f"Input is missing required obs columns: {missing}")
    label_keys = (sample_key, condition_key, cell_type_key)
    null_labels = [key for key in label_keys if data.obs[key].isna().any()]
    if null_labels:
        raise ValueError(f"Required obs labels contain null values: {null_labels}")
    condition_counts = data.obs.groupby(sample_key, observed=True)[condition_key].nunique()
    inconsistent = condition_counts[condition_counts.ne(1)]
    if not inconsistent.empty:
        raise ValueError(
            f"Each sample must map to one condition: {inconsistent.index.astype(str).tolist()}"
        )
    if spatial_key not in data.obsm:
        raise KeyError(f"Input is missing obsm[{spatial_key!r}]")
    coordinates = np.asarray(data.obsm[spatial_key])
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] < 2
        or not np.isfinite(coordinates[:, :2]).all()
    ):
        raise ValueError("CellCharter requires finite two-dimensional spatial coordinates")
    data.obs[sample_key] = pd.Categorical(data.obs[sample_key].astype(str))
    data.obs[condition_key] = pd.Categorical(data.obs[condition_key].astype(str))

    clustering = settings.get("clustering", {})
    seed = int(clustering.get("seed", 0))
    use_rep = settings.get("use_rep", "X_scVI")
    use_rep = None if use_rep in (None, "X") else str(use_rep)
    trained_scvi = False
    if use_rep is not None and use_rep not in data.obsm:
        scvi_settings = settings.get("scvi", {})
        if not scvi_settings.get("enabled", True):
            raise KeyError(f"Configured representation {use_rep!r} is absent and scVI is disabled")
        _train_scvi(
            data,
            use_rep,
            scvi_settings,
            counts_layer=counts_layer,
            sample_key=sample_key,
            seed=seed,
        )
        trained_scvi = True

    try:
        import cellcharter as cc
    except ImportError as exc:
        raise ImportError("cellcharter is required for compartment discovery") from exc

    graph = settings.get("graph", {})
    _build_graph(data, graph, sample_key, spatial_key)
    distance_percentile = graph.get("distance_percentile", 99)
    if distance_percentile is not None:
        cc.gr.remove_long_links(data, distance_percentile=float(distance_percentile))

    aggregation = settings.get("aggregation", {})
    functions = aggregation.get("functions", ["mean"])
    if isinstance(functions, list) and len(functions) == 1:
        functions = functions[0]
    aggregate_key = "X_cellcharter"
    cc.gr.aggregate_neighbors(
        data,
        n_layers=int(aggregation.get("n_layers", 3)),
        aggregations=functions,
        use_rep=use_rep,
        out_key=aggregate_key,
        sample_key=sample_key,
    )

    min_k = int(clustering.get("min_k", 2))
    max_k = int(clustering.get("max_k", 16))
    max_runs = int(clustering.get("max_runs", 10))
    if min_k < 2 or max_k < min_k or max_runs < 2:
        raise ValueError("AutoK requires min_k >= 2, max_k >= min_k, and max_runs >= 2")
    model = cc.tl.ClusterAutoK(
        n_clusters=(min_k, max_k),
        max_runs=max_runs,
        convergence_tol=float(clustering.get("convergence_tolerance", 0.001)),
        model_class=cc.tl.GaussianMixture,
        model_params={
            "random_state": seed,
            "trainer_params": {
                "accelerator": clustering.get("accelerator", "cpu"),
                "enable_progress_bar": bool(clustering.get("progress_bar", True)),
            },
        },
    )
    model.fit(data, use_rep=aggregate_key)
    best_k = int(model.best_k)
    data.obs[domain_key] = pd.Categorical(
        model.predict(data, use_rep=aggregate_key, k=best_k).astype(str)
    )

    abundance = _abundance_table(
        data.obs,
        domain_key=domain_key,
        sample_key=sample_key,
        condition_key=condition_key,
    )
    stability = _stability_table(model)
    abundance.to_csv(abundance_path, index=False)
    stability.to_csv(stability_path, index=False)

    summary = {
        "config_path": str(config_path),
        "configuration": config,
        "software": _software_versions(trained_scvi),
        "input_h5ad": str(input_path),
        "output_h5ad": str(output_path),
        "shape": list(data.shape),
        "use_rep": use_rep or "X",
        "trained_scvi": trained_scvi,
        "spatial_domain_key": domain_key,
        "best_k": best_k,
        "peaks": np.asarray(model.peaks).astype(int).tolist(),
        "domain_counts": data.obs[domain_key].astype(str).value_counts().sort_index().to_dict(),
        "graph": graph,
        "aggregation": aggregation,
        "clustering": clustering,
        "abundance_csv": str(abundance_path),
        "stability_csv": str(stability_path),
    }
    data.uns["spatial_workflow_cellcharter"] = json.dumps(summary, sort_keys=True)
    data.write_h5ad(output_path, compression="gzip")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(run_cellcharter(args.config))


if __name__ == "__main__":
    main()
