# Roadmap

The workflow will grow one verified stage at a time. Later stages are recorded
here to keep interfaces consistent without adding empty modules in advance.

## 1. Spatial overview

The current stage converts Seurat exports, bundles, or raw Xenium output into
the standard AnnData contract, runs CellCharter, and then runs
`spatial-nncomp`. The review notebooks summarize samples, spatial domains,
cell-type composition, and neighboring cell types, with optional per-domain
dives.

## 2. Differential expression

One replicate-aware pseudobulk runner will support distinct questions:

- whole-sample pseudobulk, pooling cell types within each biological sample;
- cell-type-stratified pseudobulk, testing conditions within each cell type;
- cell-type-stratified pseudobulk within an optional spatial domain.

Cluster marker discovery is a separate analysis and will not be labeled
pseudobulk differential expression.

## 3. cNMF

cNMF will run on selected broad or fine cell types, including user-defined
merged labels, with optional restriction to one or more spatial domains.

## 4. Windowed cell-cell communication

Adaptive spatial windows will feed LIANA rank aggregation and pairwise sample
tables. Configuration will make window geometry, significance rules, minimum
support, edge filtering, missing-edge zero filling, condition contrasts, and
multiple-testing scope explicit.

## 5. Pathway analysis

Pathway effects will be assessed with sample-label permutation and RMS distance
to a reference-condition centroid. Driver tables will rank ligand-receptor
edges by their contribution to a pathway effect without implying causality.

## 6. Single-cell communication

Single-cell edge scoring will support contact and secreted modes, retain sender
cell IDs for secreted edges, and provide explicit, configurable permutation
null contexts. The consolidated entrypoint will also provide the
endpoint-enrichment loop for questions of the form: when A signals B, is B more
likely to signal C? Channel inclusion for condition modeling will be independent
of significance in any one reference sample. Adaptive A-to-B-to-C searches will
remain experimental until their full selection procedure is represented in the
null model.

## Stage acceptance gate

A stage is accepted only when it has a documented configuration, a runnable
CLI, a read-only review notebook, focused tests, and a parity check against a
known Mayfield analysis artifact. Only then does implementation move to the
next stage.
