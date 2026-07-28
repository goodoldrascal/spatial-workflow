# CellCharter and neighbor composition

This stage performs three ordered operations:

1. train or reuse a transcript representation;
2. build a within-sample spatial graph and fit CellCharter AutoK;
3. calculate nearest-neighbor composition using `spatial-nncomp`.

The generated job remains an ordinary bash script. A separate tmux wrapper can
launch it on whichever compute node the user chooses; the repository does not
encode SSH behavior or a host name.

CellCharter labels are written to the configured spatial-domain key. They are
data-driven contexts rather than curated anatomical region names.

The nearest-neighbor stage writes tidy composition and neighbor-fraction tables
for the read-only compartment review notebook.
Sample-domain-cell-type combinations are zero-filled, and outputs report both
total and supporting sample counts so condition means have an explicit support
interpretation.

