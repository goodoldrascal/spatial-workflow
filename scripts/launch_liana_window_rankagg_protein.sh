#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${SPATIAL_WORKFLOW_PYTHON:-python3}"
output_dir="${repo_dir}/results/ab_xenium/07_liana_window_rankagg"
cellchat_csv="${CELLCHAT_CSV:?Set CELLCHAT_CSV to the CellChat interaction CSV}"
log_file="${output_dir}/logs/liana_window_rankagg_protein.log"

mkdir -p "${output_dir}/logs"
cd "${repo_dir}"

exec > >(tee -a "${log_file}") 2>&1

echo "[$(date --iso-8601=seconds)] Starting protein-signaling CellChat window RankAgg"

"${python_bin}" scripts/run_liana_window_rankagg.py --config configs/local.yaml run \
  --input-h5ad results/ab_xenium/02_cellcharter/ab_xenium_cellcharter.h5ad \
  --output-dir "${output_dir}" \
  --resource-mode cellchat \
  --liana-resource-name mouseconsensus \
  --adaptive-k 60 \
  --grid-stride 100.0 \
  --expr-prop 0.1 \
  --min-cells 5 \
  --min-required-groups 2 \
  --spatial-bandwidth 250.0 \
  --spatial-kernel gaussian \
  --spatial-trim-fraction 0.1 \
  --n-perms 200 \
  --seed 1337 \
  --n-jobs 8 \
  --anchor-k-target 60 \
  --anchor-min-cells 2 \
  --anchor-max-radius 150.0 \
  --anchor-dedup-distance 60.0 \
  --cellchat-csv "${cellchat_csv}"

echo "[$(date --iso-8601=seconds)] Window RankAgg complete; starting pairwise analysis"

"${python_bin}" scripts/run_liana_window_rankagg.py --config configs/local.yaml analyze \
  --output-dir "${output_dir}" \
  --cellchat-csv "${cellchat_csv}" \
  --condition naive_igg \
  --condition vap_igg \
  --condition vap_ab

echo "[$(date --iso-8601=seconds)] Protein-signaling CellChat workflow complete"
