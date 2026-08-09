#!/usr/bin/env python3
"""Apply the approved condition-blind cNMF reviews and freeze the whitelist.

This script deliberately preserves the original draft whitelist. Human review
records continue to reference that source whitelist's SHA-256. The derived
human-reviewed whitelist is frozen only after all programs have a final
decision and records the complete review contract in a hash-bearing manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_REVIEW_ROOT = (
    PROJECT_ROOT / "results" / "ab_xenium" / "05_cnmf" / "program_review"
)
DEFAULT_WHITELIST = PROGRAM_REVIEW_ROOT / "cnmf_program_whitelist_draft.tsv"
DEFAULT_QUEUE = PROGRAM_REVIEW_ROOT / "cnmf_program_review_queue_draft.tsv"
DEFAULT_DECISIONS = PROGRAM_REVIEW_ROOT / "human_review_decisions_draft.tsv"
DEFAULT_OUTPUT_DECISIONS = PROGRAM_REVIEW_ROOT / "human_review_decisions.tsv"
DEFAULT_OUTPUT = (
    PROGRAM_REVIEW_ROOT / "cnmf_program_whitelist_human_reviewed.tsv"
)

REVIEWER = "Nicholas Rhyan"
TOP_FRACTION = 0.05
NEIGHBOR_RADIUS_UM = 30.0

REVIEW_COLUMNS = [
    "lineage",
    "program",
    "review_decision",
    "review_confidence",
    "review_notes",
    "reviewer",
    "top_fraction",
    "neighbor_radius_um",
    "whitelist_sha256",
    "reviewed_at_utc",
]

WHITELIST_COLUMNS = [
    "lineage",
    "selected_k",
    "program",
    "proposed_label",
    "category",
    "confidence",
    "primary_include",
    "sensitivity_include",
    "decision",
    "rationale",
    "top_genes",
]

DRAFT_TO_REVIEW = {
    "include_primary": "keep_primary",
    "include_sensitivity": "keep_sensitivity",
    "exclude": "exclude",
}

REVIEW_TO_WHITELIST = {
    "keep_primary": ("TRUE", "TRUE", "include_primary"),
    "keep_sensitivity": ("FALSE", "TRUE", "include_sensitivity"),
    "exclude": ("FALSE", "FALSE", "exclude"),
    "needs_followup": ("FALSE", "FALSE", "needs_followup"),
}

# The reviewed proposal raised confidence for this strongly supported thalamic
# excitatory program. Other remaining confidence calls match the queue draft.
CONFIDENCE_OVERRIDES_BY_ORDER = {42: "high"}


FINAL_REVIEW_OVERRIDES = {
    ("astrocyte", "Usage_16"): {
        "review_decision": "exclude",
        "review_confidence": "medium",
        "review_notes": (
            "Exclude this spatially reproducible but mixed factor. C3, "
            "Col23a1, Gpr50, Frzb, and Slit2 provide partial astroglial "
            "support, but Tbx3, Agrp, and Nkx2-1 add hypothalamic or "
            "mesenchymal identity without a clean astrocyte-intrinsic "
            "interpretation. Direct programs from the appropriate neighboring "
            "lineages are preferable in the neighborhood model."
        ),
    },
    ("oligodendrocyte", "Usage_9"): {
        "review_decision": "exclude",
        "review_confidence": "medium",
        "review_notes": (
            "Exclude this neuronal-signaling-heavy mixed factor. Gria2 and "
            "Gria3 can occur in genuine OPC biology, but Celsr3, Ryr2, "
            "Cacna1d, Igsf9b, and Abcc8 do not establish a secure intrinsic "
            "oligodendroglial program. Neuronal programs are represented "
            "directly from neighboring neuronal lineages."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--output-decisions", type=Path, default=DEFAULT_OUTPUT_DECISIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer", default=REVIEWER)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write decisions, reviewed whitelist, and manifest after validation.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def metric_text(row: pd.Series) -> str:
    genes = ", ".join(str(row["top_genes"]).split("|")[:8])
    qc_max = max(
        abs(float(row["spearman_nCount_Xenium"])),
        abs(float(row["spearman_nFeature_Xenium"])),
    )
    return (
        f"Defining genes are {genes}. Condition-blind support: mean top-gene "
        f"detection {float(row['mean_top_gene_detection_fraction']):.3f}, "
        f"median top-gene log2 mean-count enrichment "
        f"{float(row['median_top_gene_log2_mean_count_ratio']):.3f}, spatial "
        f"KNN enrichment {float(row['spatial_knn_enrichment']):.3f}, effective "
        f"top-section breadth {float(row['effective_top_sections']):.2f}, "
        f"largest-section fraction {float(row['largest_section_fraction']):.3f}, "
        f"and maximum absolute QC correlation {qc_max:.3f}."
    )


def decision_note(row: pd.Series, decision: str) -> str:
    evidence = metric_text(row)
    lineage = str(row["lineage"])
    program = str(row["program"])
    label = str(row["proposed_label"])

    special = {
        ("excitatory_neuron", "Usage_15"): (
            "Retain only for sensitivity analysis because the coherent neuronal "
            "signal mixes extracellular-matrix and stress-response features."
        ),
        ("excitatory_neuron", "Usage_25"): (
            "Retain only for sensitivity analysis because the IFN response is "
            "plausible but moderately supported and lacks a lineage-specific "
            "anchor among the defining genes."
        ),
        ("epd", "Usage_4"): (
            "Retain as a primary interferon-responsive ependymal activity "
            "program. Its section concentration is acceptable because EPD cells "
            "are intrinsically spatially restricted."
        ),
        ("chp", "Usage_3"): (
            "Retain only for sensitivity analysis because this plausible choroid "
            "plexus epithelial state is concentrated in a small number of "
            "sections and has modest spatial separation."
        ),
        ("chp", "Usage_4"): (
            "Retain only for sensitivity analysis: the IFN program is very "
            "strong molecularly, but 91% of top-usage cells occur in one blinded "
            "section."
        ),
    }
    if (lineage, program) in special:
        return f"{label}. {evidence} {special[(lineage, program)]}"

    if decision == "keep_primary":
        if lineage == "perivascular":
            conclusion = (
                "Retain as a primary vascular or stromal program in the combined "
                "EC/PVF/PERICYTE source. The current PVF annotation is "
                "heterogeneous, and downstream subclustering should refine the "
                "subtype label."
            )
        else:
            conclusion = (
                f"Retain as a primary {str(row['category']).replace('_', ' ')} "
                "program because the biological interpretation is coherent "
                "within the source lineage."
            )
    elif decision == "keep_sensitivity":
        conclusion = (
            "Retain only for sensitivity analysis because the biological signal "
            "is plausible but not sufficiently secure for the primary model."
        )
    else:
        weak = (
            float(row["mean_top_gene_detection_fraction"]) < 0.10
            or float(row["median_top_gene_log2_mean_count_ratio"]) < 1.0
        )
        if weak:
            conclusion = (
                "Exclude because the proposed signal is incompatible with the "
                "source lineage and also lacks robust raw-count support."
            )
        else:
            conclusion = (
                "Exclude because this is a coherent but source-lineage-"
                "incompatible program; the matching neighboring lineage is the "
                "appropriate representation in the neighborhood model."
            )
    return f"{label}. {evidence} {conclusion}"


def validate_inputs(
    whitelist: pd.DataFrame,
    queue: pd.DataFrame,
    decisions: pd.DataFrame,
    whitelist_hash: str,
) -> None:
    if list(whitelist.columns) != WHITELIST_COLUMNS:
        raise ValueError("Unexpected source whitelist columns")
    if list(decisions.columns) != REVIEW_COLUMNS:
        raise ValueError("Unexpected human decision columns")
    for name, frame in {
        "whitelist": whitelist,
        "queue": queue,
        "decisions": decisions,
    }.items():
        if frame.duplicated(["lineage", "program"]).any():
            raise ValueError(f"Duplicate lineage-program rows in {name}")
    if len(whitelist) != 150 or len(queue) != 150:
        raise ValueError("Expected 150 whitelist and queue rows")
    if set(queue["review_order"].astype(int)) != set(range(1, 151)):
        raise ValueError("Review queue orders are not exactly 1 through 150")
    if not decisions["whitelist_sha256"].eq(whitelist_hash).all():
        raise ValueError("Existing decisions do not match the source whitelist hash")
    if len(decisions) not in {41, 150}:
        raise ValueError(
            f"Expected 41 pre-approval or 150 completed decisions; found {len(decisions)}"
        )


def approved_decisions(
    queue: pd.DataFrame,
    current: pd.DataFrame,
    reviewer: str,
    whitelist_hash: str,
) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    completed = current.copy()
    completed["reviewer"] = reviewer

    # Nicholas explicitly promoted the clean astrocyte IFN program to primary.
    astro_mask = completed["lineage"].eq("astrocyte") & completed["program"].eq(
        "Usage_13"
    )
    if int(astro_mask.sum()) != 1:
        raise ValueError("Expected one existing astrocyte Usage_13 review")
    astro_metrics = queue.loc[
        queue["lineage"].eq("astrocyte") & queue["program"].eq("Usage_13")
    ]
    if len(astro_metrics) != 1:
        raise ValueError("Expected one astrocyte Usage_13 queue row")
    astro = astro_metrics.iloc[0]
    completed.loc[astro_mask, "review_decision"] = "keep_primary"
    completed.loc[astro_mask, "review_confidence"] = "medium"
    completed.loc[astro_mask, "review_notes"] = (
        "Interferon and inflammatory response. "
        + metric_text(astro)
        + " Retain as a primary astrocyte activity program: it is strongly "
        "supported, spans all 12 blinded sections, and is not associated with "
        "transcript depth or detected-gene count."
    )
    completed.loc[astro_mask, "reviewed_at_utc"] = now

    for key, override in FINAL_REVIEW_OVERRIDES.items():
        mask = completed["lineage"].eq(key[0]) & completed["program"].eq(key[1])
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected one existing {key[0]} {key[1]} review")
        completed.loc[mask, "review_decision"] = override["review_decision"]
        completed.loc[mask, "review_confidence"] = override["review_confidence"]
        completed.loc[mask, "review_notes"] = override["review_notes"]
        completed.loc[mask, "reviewer"] = reviewer
        completed.loc[mask, "top_fraction"] = TOP_FRACTION
        completed.loc[mask, "neighbor_radius_um"] = NEIGHBOR_RADIUS_UM
        completed.loc[mask, "whitelist_sha256"] = whitelist_hash
        completed.loc[mask, "reviewed_at_utc"] = now

    existing_keys = set(zip(completed["lineage"], completed["program"]))
    new_records: list[dict] = []
    for _, row in queue.loc[queue["review_order"].astype(int).ge(42)].iterrows():
        key = (str(row["lineage"]), str(row["program"]))
        decision = DRAFT_TO_REVIEW[str(row["decision"])]
        confidence = CONFIDENCE_OVERRIDES_BY_ORDER.get(
            int(row["review_order"]), str(row["confidence"])
        )
        record = {
            "lineage": key[0],
            "program": key[1],
            "review_decision": decision,
            "review_confidence": confidence,
            "review_notes": decision_note(row, decision),
            "reviewer": reviewer,
            "top_fraction": TOP_FRACTION,
            "neighbor_radius_um": NEIGHBOR_RADIUS_UM,
            "whitelist_sha256": whitelist_hash,
            "reviewed_at_utc": now,
        }
        if key in existing_keys:
            mask = completed["lineage"].eq(key[0]) & completed["program"].eq(key[1])
            for column in REVIEW_COLUMNS:
                completed.loc[mask, column] = record[column]
        else:
            new_records.append(record)

    if new_records:
        completed = pd.concat(
            [completed, pd.DataFrame(new_records, columns=REVIEW_COLUMNS)],
            ignore_index=True,
        )
    completed = completed.loc[:, REVIEW_COLUMNS].sort_values(
        ["lineage", "program"], kind="stable"
    )
    if len(completed) != 150:
        raise ValueError(f"Expected 150 completed decisions; found {len(completed)}")
    if completed.duplicated(["lineage", "program"]).any():
        raise ValueError("Completed decisions contain duplicate keys")
    return completed.reset_index(drop=True)


def build_reviewed_whitelist(
    whitelist: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    reviewed = whitelist.merge(
        decisions[
            [
                "lineage",
                "program",
                "review_decision",
                "review_confidence",
                "review_notes",
            ]
        ],
        on=["lineage", "program"],
        how="left",
        validate="one_to_one",
    )
    if reviewed["review_decision"].isna().any():
        raise ValueError("Some whitelist rows lack human decisions")
    mapped = reviewed["review_decision"].map(REVIEW_TO_WHITELIST)
    reviewed["primary_include"] = mapped.map(lambda values: values[0])
    reviewed["sensitivity_include"] = mapped.map(lambda values: values[1])
    reviewed["decision"] = mapped.map(lambda values: values[2])
    reviewed["confidence"] = reviewed["review_confidence"]
    reviewed["rationale"] = reviewed["review_notes"]
    return reviewed.loc[:, WHITELIST_COLUMNS]


def build_manifest(
    *,
    source_whitelist: Path,
    source_whitelist_hash: str,
    queue_path: Path,
    decisions_path: Path,
    decisions_hash: str,
    output_path: Path,
    output_hash: str,
    decisions: pd.DataFrame,
    reviewed: pd.DataFrame,
) -> dict:
    counts = decisions["review_decision"].value_counts().to_dict()
    unresolved = decisions.loc[
        decisions["review_decision"].eq("needs_followup"),
        ["lineage", "program", "review_confidence", "review_notes"],
    ].to_dict(orient="records")
    frozen = len(unresolved) == 0
    return {
        "status": (
            "human_reviewed_frozen" if frozen else "human_reviewed_draft"
        ),
        "frozen": frozen,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": REVIEWER,
        "source_whitelist": str(source_whitelist.resolve()),
        "source_whitelist_sha256": source_whitelist_hash,
        "review_queue": str(queue_path.resolve()),
        "review_queue_sha256": sha256(queue_path),
        "human_decisions": str(decisions_path.resolve()),
        "human_decisions_sha256": decisions_hash,
        "output_whitelist": str(output_path.resolve()),
        "output_whitelist_sha256": output_hash,
        "n_programs": int(len(reviewed)),
        "review_decision_counts": {str(k): int(v) for k, v in counts.items()},
        "n_primary_programs": int(reviewed["primary_include"].eq("TRUE").sum()),
        "n_sensitivity_programs_total": int(
            reviewed["sensitivity_include"].eq("TRUE").sum()
        ),
        "n_sensitivity_only_programs": int(
            (
                reviewed["primary_include"].eq("FALSE")
                & reviewed["sensitivity_include"].eq("TRUE")
            ).sum()
        ),
        "n_unresolved": int(len(unresolved)),
        "unresolved_programs": unresolved,
        "notes": [
            "The original draft whitelist is preserved unchanged.",
            "All programs have a final condition-blind human decision."
            if frozen
            else "needs_followup programs are unadmitted in both feature sets.",
            "This artifact is the frozen model whitelist."
            if frozen
            else "This artifact is not the frozen model whitelist.",
            "EPD Usage_4 remains primary because EPD is intrinsically spatially restricted.",
            "Astrocyte Usage_13 and oligodendrocyte Usage_17 are primary IFN programs.",
        ],
    }


def main() -> None:
    args = parse_args()
    whitelist = pd.read_csv(
        args.whitelist, sep="\t", dtype=str, keep_default_na=False
    )
    queue = pd.read_csv(args.queue, sep="\t", keep_default_na=False)
    decisions = pd.read_csv(
        args.decisions, sep="\t", dtype=str, keep_default_na=False
    )
    whitelist_hash = sha256(args.whitelist)
    validate_inputs(whitelist, queue, decisions, whitelist_hash)
    completed = approved_decisions(
        queue, decisions, args.reviewer, whitelist_hash
    )
    reviewed = build_reviewed_whitelist(whitelist, completed)

    decision_counts = completed["review_decision"].value_counts()
    expected_counts = {
        "keep_primary": 55,
        "keep_sensitivity": 7,
        "exclude": 88,
        "needs_followup": 0,
    }
    observed_counts = {
        key: int(decision_counts.get(key, 0)) for key in expected_counts
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Unexpected approved decision counts: {observed_counts}; "
            f"expected {expected_counts}"
        )
    if reviewed.loc[
        reviewed["lineage"].eq("epd") & reviewed["program"].eq("Usage_4"),
        "decision",
    ].tolist() != ["include_primary"]:
        raise ValueError("EPD Usage_4 must remain primary")

    print("approved decision counts:")
    print(decision_counts.to_string())
    print(
        "selected programs:",
        int(reviewed["primary_include"].eq("TRUE").sum()),
        "primary;",
        int(reviewed["sensitivity_include"].eq("TRUE").sum()),
        "total sensitivity feature set",
    )
    unresolved = completed.loc[
        completed["review_decision"].eq("needs_followup"),
        ["lineage", "program"],
    ]
    print("unresolved:")
    print(unresolved.to_string(index=False))

    if not args.apply:
        print("Dry run complete; no files written. Pass --apply to write outputs.")
        return

    atomic_write_tsv(completed, args.output_decisions)
    atomic_write_tsv(reviewed, args.output)
    decisions_hash = sha256(args.output_decisions)
    output_hash = sha256(args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = build_manifest(
        source_whitelist=args.whitelist,
        source_whitelist_hash=whitelist_hash,
        queue_path=args.queue,
        decisions_path=args.output_decisions,
        decisions_hash=decisions_hash,
        output_path=args.output,
        output_hash=output_hash,
        decisions=completed,
        reviewed=reviewed,
    )
    manifest["reviewer"] = args.reviewer
    atomic_write_json(manifest, manifest_path)
    print("wrote:", args.output_decisions)
    print("wrote:", args.output)
    print("wrote:", manifest_path)


if __name__ == "__main__":
    main()
