from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from spatial_workflow.cnmf_program_review import (
    REVIEW_COLUMNS,
    blind_sample_labels,
    build_review_decision,
    high_usage_gene_support,
    high_usage_neighbor_context,
    prepare_blinded_usage_obs,
    program_spatial_knn_enrichment,
    program_usage_diagnostic_metrics,
    program_usage_qc_summary,
    select_high_usage_cells,
    top_usage_spatial_knn_enrichment,
    upsert_review_decision,
)


def _usage_obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["vap_sample", "vap_sample", "naive_sample", "naive_sample"],
            "condition": ["vap_igg", "vap_igg", "naive_igg", "naive_igg"],
            "cluster_sub": ["AST", "AST", "AST", "AST"],
            "celltype_short": ["AST", "AST", "AST", "AST"],
            "celltype_full": ["AST", "AST", "AST", "AST"],
            "spatial_domain": ["0", "0", "1", "1"],
            "nCount_Xenium": [10, 20, 30, 40],
            "nFeature_Xenium": [5, 6, 7, 8],
            "Usage_1": [0.9, 0.6, 0.3, 0.1],
            "Usage_2": [0.1, 0.4, 0.7, 0.9],
        },
        index=["c0", "c1", "c2", "c3"],
    )


def test_blinded_review_omits_experimental_labels() -> None:
    obs = _usage_obs()
    mapping = blind_sample_labels(obs["sample_id"])
    assert set(mapping) == {"vap_sample", "naive_sample"}
    assert set(mapping.values()) == {"Section_01", "Section_02"}
    assert mapping == blind_sample_labels(obs["sample_id"])

    review, returned_mapping = prepare_blinded_usage_obs(
        obs,
        sample_key="sample_id",
    )
    assert returned_mapping == mapping
    assert "sample_id" not in review
    assert "condition" not in review
    assert not {"vap_igg", "naive_igg"}.intersection(
        review.astype(str).to_numpy().reshape(-1)
    )
    assert set(review["blinded_sample"]) == {"Section_01", "Section_02"}


def test_high_usage_selection_and_summary_are_exact() -> None:
    review, _ = prepare_blinded_usage_obs(
        _usage_obs(),
        sample_key="sample_id",
    )
    high = select_high_usage_cells(
        review,
        program="Usage_1",
        top_fraction=0.5,
    )
    assert high["cell_id"].tolist() == ["c0", "c1"]
    assert high["usage"].tolist() == [0.9, 0.6]
    summary = program_usage_qc_summary(
        review,
        usage_cols=["Usage_1"],
        top_fraction=0.5,
    ).iloc[0]
    assert summary["n_top_cells"] == 2
    assert summary["largest_section_fraction"] == 1.0
    assert summary["effective_top_sections"] == 1.0


def test_diagnostic_metrics_are_condition_blind_and_find_redundancy() -> None:
    review, _ = prepare_blinded_usage_obs(
        _usage_obs(),
        sample_key="sample_id",
    )
    metrics = program_usage_diagnostic_metrics(
        review,
        usage_cols=["Usage_1", "Usage_2"],
        top_fraction=0.5,
    ).set_index("program")
    assert set(metrics.index) == {"Usage_1", "Usage_2"}
    assert metrics.loc["Usage_1", "sd_usage"] > 0
    assert metrics.loc[
        "Usage_1", "max_abs_pairwise_spearman_program"
    ] == "Usage_2"
    assert metrics.loc["Usage_1", "max_abs_pairwise_spearman"] == -1.0
    assert not set(metrics.columns).intersection({"condition", "sample_id"})


def test_spatial_knn_enrichment_is_section_scoped() -> None:
    obs = pd.DataFrame(
        {
            "sample_id": ["s1"] * 6,
            "cluster_sub": ["AST"] * 6,
            "celltype_short": ["AST"] * 6,
            "celltype_full": ["AST"] * 6,
            "spatial_domain": ["0"] * 6,
            "nCount_Xenium": [10] * 6,
            "nFeature_Xenium": [5] * 6,
            "Usage_1": [0.9, 0.8, 0.1, 0.1, 0.1, 0.1],
        },
        index=[f"c{index}" for index in range(6)],
    )
    review, _ = prepare_blinded_usage_obs(obs, sample_key="sample_id")
    coords = np.array([[0, 0], [1, 0], [100, 0], [200, 0], [300, 0], [400, 0]])
    metric = top_usage_spatial_knn_enrichment(
        review,
        coords,
        program="Usage_1",
        top_fraction=1 / 3,
        n_neighbors=1,
    )
    assert metric["spatial_knn_observed_top_fraction"] == 1.0
    assert metric["spatial_knn_expected_top_fraction"] == 0.2
    assert metric["spatial_knn_enrichment"] == 5.0
    combined = program_spatial_knn_enrichment(
        review,
        coords,
        usage_cols=["Usage_1"],
        top_fraction=1 / 3,
        n_neighbors=1,
    ).iloc[0]
    assert combined["spatial_knn_observed_top_fraction"] == 1.0
    assert combined["spatial_knn_expected_top_fraction"] == 0.2
    assert combined["spatial_knn_enrichment"] == 5.0


def test_gene_support_compares_high_cells_to_background() -> None:
    matrix = sp.csr_matrix(
        np.array(
            [
                [5, 0, 1],
                [4, 0, 0],
                [0, 3, 0],
                [0, 2, 0],
            ],
            dtype=np.int32,
        )
    )
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["expected", "other", "third"]),
    )
    result = high_usage_gene_support(
        adata,
        high_cell_ids=["c0", "c1"],
        genes=["expected", "other", "missing"],
    ).set_index("gene")
    assert result.loc["expected", "high_usage_detection_fraction"] == 1.0
    assert result.loc["expected", "background_detection_fraction"] == 0.0
    assert result.loc["expected", "log2_mean_count_ratio"] > 0
    assert result.attrs.get("missing_genes", []) == ["missing"]


def test_neighbor_context_is_sample_scoped_and_excludes_self() -> None:
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s2"],
            "cluster_sub": ["focal", "near_a", "near_b", "other_sample"],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    coords = np.array([[0, 0], [1, 0], [5, 0], [0, 0]], dtype=float)
    result = high_usage_neighbor_context(
        obs,
        coords,
        focal_cell_ids=["c0"],
        sample_key="sample_id",
        cell_type_key="cluster_sub",
        radius=10,
    ).set_index("neighbor_cell_type")
    assert set(result.index) == {"near_a", "near_b"}
    assert result.loc["near_a", "n_neighbor_occurrences"] == 1
    assert result.loc["near_b", "fraction_focal_cells_with_neighbor"] == 1.0


def test_review_decisions_are_upserted_atomically(tmp_path) -> None:
    path = tmp_path / "decisions.tsv"
    first = build_review_decision(
        lineage="astrocyte",
        program="Usage_1",
        review_decision="keep_primary",
        review_confidence="high",
        review_notes="coherent",
        reviewer="reviewer",
        top_fraction=0.05,
        neighbor_radius_um=30,
        whitelist_sha256="abc",
    )
    upsert_review_decision(path, first)
    second = dict(first)
    second["review_decision"] = "needs_followup"
    second["review_notes"] = "inspect another section"
    upsert_review_decision(path, second)
    result = pd.read_csv(path, sep="\t", dtype=str)
    assert list(result.columns) == REVIEW_COLUMNS
    assert len(result) == 1
    assert result.loc[0, "review_decision"] == "needs_followup"
